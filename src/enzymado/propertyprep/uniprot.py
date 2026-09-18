"""UniProt sequence lookup, metadata matching, and checkpoint support."""

# Preserve positional compatibility for the existing property-preparation API.
# ruff: noqa: FBT001, FBT002

from __future__ import annotations

import collections.abc
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
import typing

import requests

import enzymado.paths_and_constants
import enzymado.types

_SEQUENCE_INDEX_CACHE: dict[str, enzymado.types.SwissProtSequenceIndex] = {}
_DEFAULT_SWISSPROT_PATH = (
    enzymado.paths_and_constants.Paths.data_raw_swissprot_dir / "swissprot_ec.tsv"
)


@dataclass(frozen=True)
class _RemoteLookupOptions:
    """Settings for the optional EMBL-EBI sequence-similarity fallback."""

    enabled: bool
    email: str | None
    database: str
    timeout: float
    poll_interval: float
    max_polls: int


def _load_sequence_checkpoint(path: Path) -> enzymado.types.SequenceLookupResult:
    """Load completed sequences and failures from a checkpoint if present."""
    if not path.exists():
        return {}, {}
    with path.open("r", encoding="utf-8") as handle:
        state = typing.cast(dict[str, object], json.load(handle))
    done = typing.cast(dict[str, str], state.get("done", {}))
    failed = typing.cast(dict[str, str], state.get("failed", {}))
    return done, failed


def _load_metadata_checkpoint(path: Path) -> enzymado.types.EnzymeLookupResult:
    """Load completed enzyme metadata and failures from a checkpoint."""
    if not path.exists():
        return {}, {}
    with path.open("r", encoding="utf-8") as handle:
        state = typing.cast(dict[str, object], json.load(handle))
    done = typing.cast(
        dict[str, enzymado.types.EnzymeMetadata],
        state.get("done", {}),
    )
    failed = typing.cast(dict[str, str], state.get("failed", {}))
    return done, failed


def _save_lookup_checkpoint(
    path: Path,
    done: collections.abc.Mapping[str, object],
    failed: collections.abc.Mapping[str, str],
) -> None:
    """Atomically save successful and failed lookup state."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"done": done, "failed": failed},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    temporary_path.replace(path)


def _normalize_uniprot_ids(
    uniprot_ids: collections.abc.Iterable[object],
) -> list[str]:
    """Return non-empty unique identifiers while preserving input order."""
    identifiers: list[str] = []
    seen: set[str] = set()
    for value in uniprot_ids:
        identifier = str(value).strip()
        if identifier and identifier.lower() != "nan" and identifier not in seen:
            identifiers.append(identifier)
            seen.add(identifier)
    return identifiers


def _prepare_pending_keys(
    keys: list[str],
    done: collections.abc.Mapping[str, object],
    failed: dict[str, str],
    *,
    retry_failed: bool,
) -> list[str]:
    """Apply retry policy and return keys absent from checkpoint state."""
    if retry_failed:
        for key in keys:
            failed.pop(key, None)
    return [key for key in keys if key not in done and key not in failed]


def _search_uniprot_sequence(
    identifier: str,
    session: requests.Session,
    timeout: float,
) -> str:
    """Resolve a sequence through UniProt's accession/ID search endpoint."""
    response = session.get(
        "https://rest.uniprot.org/uniprotkb/search",
        params={
            "query": (
                f"(accession:{identifier}) OR (id:{identifier}) OR "
                f"(sec_acc:{identifier})"
            ),
            "format": "json",
            "size": 5,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = typing.cast(dict[str, object], response.json())
    results = typing.cast(list[dict[str, object]], payload.get("results", []))

    chosen: dict[str, object] | None = None
    for entry in results:
        accession = entry.get("primaryAccession", "")
        entry_id = entry.get("uniProtkbId", "")
        secondary = set(
            typing.cast(list[str], entry.get("secondaryAccessions", []))
        )
        if identifier == accession or identifier == entry_id or identifier in secondary:
            chosen = entry
            break
    if chosen is None and results:
        chosen = results[0]
    if chosen is None:
        return ""
    sequence = typing.cast(dict[str, str], chosen.get("sequence", {}))
    return sequence.get("value", "").upper()


def _fetch_uniprot_sequence(
    identifier: str,
    session: requests.Session,
    timeout: float,
) -> str:
    """Try UniProt FASTA, JSON, then identifier search for one sequence."""
    response = session.get(
        f"https://rest.uniprot.org/uniprotkb/{identifier}.fasta",
        timeout=timeout,
    )
    if response.status_code == 200:
        lines = response.text.strip().splitlines()
        if lines and lines[0].startswith(">"):
            sequence = "".join(lines[1:]).replace(" ", "").upper()
            if sequence:
                return sequence

    response = session.get(
        f"https://rest.uniprot.org/uniprotkb/{identifier}.json",
        timeout=timeout,
    )
    if response.status_code == 200:
        payload = typing.cast(dict[str, object], response.json())
        sequence_payload = typing.cast(
            dict[str, str],
            payload.get("sequence", {}),
        )
        sequence = sequence_payload.get("value", "").upper()
        if sequence:
            return sequence
    return _search_uniprot_sequence(identifier, session, timeout)


def _print_lookup_progress(
    completed: int,
    pending_total: int,
    overall_done: int,
    total: int,
) -> None:
    """Print pending and overall progress with empty-input-safe percentages."""
    pending_percent = completed / pending_total * 100 if pending_total else 100
    overall_completed = overall_done + completed
    overall_percent = overall_completed / total * 100 if total else 100
    print(
        f"Progress pending: {completed}/{pending_total} ({pending_percent:.2f}%) | "
        f"overall: {overall_completed}/{total} ({overall_percent:.2f}%)"
    )


def sequence_getter(
    uniprot_ids: collections.abc.Iterable[object],
    checkpoint_path: str | Path = "uniprot_checkpoint.json",
    save_every: int = 50,
    timeout: float = 15,
    sleep_s: float = 0.0,
    retry_failed: bool = True,
    show_progress: bool = True,
) -> enzymado.types.SequenceLookupResult:
    ckpt = Path(checkpoint_path)
    done, failed = _load_sequence_checkpoint(ckpt)
    identifiers = _normalize_uniprot_ids(uniprot_ids)
    pending = _prepare_pending_keys(
        identifiers,
        done,
        failed,
        retry_failed=retry_failed,
    )

    total = len(identifiers)
    pending_total = len(pending)
    overall_done = total - pending_total

    print(
        f"Resume status -> done: {len(done)}, failed: {len(failed)}, "
        f"pending: {pending_total}, total: {total}"
    )

    with requests.Session() as session:
        for i, uid in enumerate(pending, start=1):
            try:
                seq = _fetch_uniprot_sequence(uid, session, timeout)
                if not seq:
                    raise ValueError(f"No sequence found for: {uid}")
                done[uid] = seq
                failed.pop(uid, None)
            except Exception as exc:  # noqa: BLE001 - checkpoint per-ID failures.
                failed[uid] = str(exc)

            if i % save_every == 0:
                _save_lookup_checkpoint(ckpt, done, failed)

            if show_progress and (i % save_every == 0 or i == pending_total):
                _print_lookup_progress(i, pending_total, overall_done, total)

            if sleep_s > 0:
                time.sleep(sleep_s)

    _save_lookup_checkpoint(ckpt, done, failed)
    return done, failed


def _normalize_protein_sequence(sequence: object) -> str:
    seq = str(sequence).strip().upper().replace(" ", "")
    if not seq or seq == "NAN":
        raise ValueError("Empty sequence")
    return seq


def _split_ec_numbers(ec_value: object) -> list[str]:
    if ec_value is None:
        return []
    text = str(ec_value).strip()
    if not text or text.lower() == "nan":
        return []
    ecs = []
    for piece in text.replace(",", ";").split(";"):
        value = piece.strip()
        if value:
            ecs.append(value)
    return ecs


def _load_swissprot_sequence_index(
    swissprot_tsv_path: str | Path,
) -> enzymado.types.SwissProtSequenceIndex:
    swissprot_tsv_path = str(Path(swissprot_tsv_path).resolve())
    cached = _SEQUENCE_INDEX_CACHE.get(swissprot_tsv_path)
    if cached is not None:
        return cached

    index: enzymado.types.SwissProtSequenceIndex = {}
    with Path(swissprot_tsv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            seq = (row.get("Sequence") or "").strip().upper()
            if not seq:
                continue
            record = {
                "accession": (row.get("Entry") or "").strip() or None,
                "ec_numbers": _split_ec_numbers(row.get("EC number")),
            }
            index.setdefault(seq, []).append(record)

    _SEQUENCE_INDEX_CACHE[swissprot_tsv_path] = index
    return index


def _pick_best_local_match(
    records: list[enzymado.types.UniProtMatch],
) -> enzymado.types.UniProtMatch | None:
    if not records:
        return None

    def sort_key(record: enzymado.types.UniProtMatch) -> tuple[int, str]:
        accession = record.get("accession") or ""
        ec_numbers = record.get("ec_numbers") or []
        return (0 if ec_numbers else 1, accession)

    return sorted(records, key=sort_key)[0]


def _extract_ec_numbers_from_uniprot_json(payload: object) -> list[str]:
    ec_numbers: list[str] = []
    seen: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "ecNumbers" and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            ec = (item.get("value") or "").strip()
                        else:
                            ec = str(item).strip()
                        if ec and ec not in seen:
                            seen.add(ec)
                            ec_numbers.append(ec)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return ec_numbers


def _get_uniprot_entry_metadata(
    accession: str,
    session: requests.Session,
    timeout: float = 20,
) -> enzymado.types.UniProtMatch:
    response = session.get(
        f"https://rest.uniprot.org/uniprotkb/{accession}.json",
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    ec_numbers = _extract_ec_numbers_from_uniprot_json(payload)
    return {
        "accession": payload.get("primaryAccession") or accession,
        "ec_numbers": ec_numbers,
    }


def _submit_ebi_fasta_job(
    sequence: str,
    email: str,
    session: requests.Session,
    timeout: float = 30,
    database: str = "uniprotkb_swissprot",
) -> str:
    response = session.post(
        "https://www.ebi.ac.uk/Tools/services/rest/fasta/run",
        data={
            "email": email,
            "program": "fasta",
            "stype": "protein",
            "database": database,
            "sequence": sequence,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    job_id = response.text.strip()
    if not job_id:
        raise ValueError("EBI FASTA returned an empty job id")
    return job_id


def _poll_ebi_fasta_job(
    job_id: str,
    session: requests.Session,
    timeout: float = 20,
    poll_interval: float = 3.0,
    max_polls: int = 40,
) -> None:
    status_url = f"https://www.ebi.ac.uk/Tools/services/rest/fasta/status/{job_id}"
    for _ in range(max_polls):
        response = session.get(status_url, timeout=timeout)
        response.raise_for_status()
        status = response.text.strip()
        if status == "FINISHED":
            return
        if status in {"ERROR", "FAILURE", "NOT_FOUND"}:
            raise ValueError(f"EBI FASTA job {job_id} ended with status: {status}")
        time.sleep(poll_interval)
    raise TimeoutError(f"EBI FASTA job {job_id} did not finish after {max_polls} polls")


def _parse_best_accession_from_ebi_output(output_text: str) -> str | None:
    for line in output_text.splitlines():
        line = line.strip()
        if not line or not line.startswith(("SP:", "TR:")):
            continue
        match = re.match(r"^(?:SP|TR):([A-Z0-9]+)\b", line)
        if match:
            return match.group(1)
    return None


def _find_accession_by_sequence_similarity(
    sequence: str,
    email: str | None,
    session: requests.Session,
    timeout: float = 30,
    poll_interval: float = 3.0,
    max_polls: int = 40,
    database: str = "uniprotkb_swissprot",
) -> str:
    if not email:
        raise ValueError(
            "email is required for remote sequence similarity lookup via EMBL-EBI FASTA"
        )

    job_id = _submit_ebi_fasta_job(sequence, email, session, timeout, database)
    _poll_ebi_fasta_job(job_id, session, timeout, poll_interval, max_polls)

    result = session.get(
        f"https://www.ebi.ac.uk/Tools/services/rest/fasta/result/{job_id}/out",
        timeout=timeout,
    )
    result.raise_for_status()
    accession = _parse_best_accession_from_ebi_output(result.text)
    if not accession:
        raise ValueError("Could not parse a UniProt accession from EBI FASTA output")
    return accession


def _normalize_unique_sequences(
    sequences: collections.abc.Iterable[object],
) -> list[str]:
    """Normalize valid protein sequences and preserve first-seen order."""
    normalized: list[str] = []
    seen: set[str] = set()
    for sequence in sequences:
        try:
            value = _normalize_protein_sequence(sequence)
        except ValueError:
            continue
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _resolve_enzyme_metadata(
    sequence: str,
    swissprot_index: enzymado.types.SwissProtSequenceIndex,
    session: requests.Session,
    options: _RemoteLookupOptions,
) -> enzymado.types.EnzymeMetadata:
    """Prefer an exact Swiss-Prot match, then optionally query EMBL-EBI."""
    local_match = _pick_best_local_match(swissprot_index.get(sequence, []))
    if local_match is not None:
        accession = local_match["accession"]
        ec_numbers = local_match["ec_numbers"]
        match_source = "local_swissprot_exact"
    else:
        if not options.enabled:
            raise ValueError("No exact Swiss-Prot match found for sequence")
        remote_accession = _find_accession_by_sequence_similarity(
            sequence=sequence,
            email=options.email,
            session=session,
            timeout=options.timeout,
            poll_interval=options.poll_interval,
            max_polls=options.max_polls,
            database=options.database,
        )
        remote_match = _get_uniprot_entry_metadata(
            accession=remote_accession,
            session=session,
            timeout=options.timeout,
        )
        accession = remote_match["accession"]
        ec_numbers = remote_match["ec_numbers"]
        match_source = "ebi_fasta_similarity"

    return {
        "accession": accession,
        "ec_number": "; ".join(ec_numbers) if ec_numbers else None,
        "is_enzyme": bool(ec_numbers),
        "match_source": match_source,
    }


def sequence_to_enzyme_metadata(
    sequences: collections.abc.Iterable[object],
    checkpoint_path: str | Path = "sequence_to_enzyme_checkpoint.json",
    swissprot_tsv_path: str | Path | None = None,
    save_every: int = 25,
    timeout: float = 20,
    sleep_s: float = 0.0,
    retry_failed: bool = True,
    show_progress: bool = True,
    use_remote_fallback: bool = True,
    email: str | None = None,
    remote_database: str = "uniprotkb_swissprot",
    poll_interval: float = 3.0,
    max_polls: int = 40,
) -> enzymado.types.EnzymeLookupResult:
    if swissprot_tsv_path is None:
        swissprot_tsv_path = _DEFAULT_SWISSPROT_PATH

    ckpt = Path(checkpoint_path)
    done, failed = _load_metadata_checkpoint(ckpt)
    unique_sequences = _normalize_unique_sequences(sequences)
    pending = _prepare_pending_keys(
        unique_sequences,
        done,
        failed,
        retry_failed=retry_failed,
    )

    total = len(unique_sequences)
    pending_total = len(pending)
    swissprot_index = _load_swissprot_sequence_index(swissprot_tsv_path)
    remote_options = _RemoteLookupOptions(
        enabled=use_remote_fallback,
        email=email,
        database=remote_database,
        timeout=timeout,
        poll_interval=poll_interval,
        max_polls=max_polls,
    )

    print(
        f"Resume status -> done: {len(done)}, failed: {len(failed)}, "
        f"pending: {pending_total}, total: {total}"
    )

    with requests.Session() as session:
        for i, seq in enumerate(pending, start=1):
            try:
                done[seq] = _resolve_enzyme_metadata(
                    seq,
                    swissprot_index,
                    session,
                    remote_options,
                )
                failed.pop(seq, None)
            except Exception as exc:  # noqa: BLE001 - checkpoint per-ID failures.
                failed[seq] = str(exc)

            if i % save_every == 0:
                _save_lookup_checkpoint(ckpt, done, failed)

            if sleep_s > 0:
                time.sleep(sleep_s)

    _save_lookup_checkpoint(ckpt, done, failed)
    return done, failed
