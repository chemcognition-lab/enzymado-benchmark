"""Shared preprocessing utilities for real-case enzyme benchmark datasets."""

from __future__ import annotations

import argparse
import collections
import csv
import math
from pathlib import Path
import re

from rdkit import Chem, RDLogger

from enzymado import paths_and_constants

RDLogger.DisableLog("rdApp.*")

_C = paths_and_constants.Constants
_P = paths_and_constants.Paths

RAW_REAL_CASE_DIR = _P.data_raw_real_case
DOCS_DATASETS_DIR = _P.docs_datasets_dir
PROCESSED_REAL_CASE_DIR = _P.data_processed_real_case

CANONICAL_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

AA3_TO_AA1: dict[str, str] = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}

STANDARD_CODON_TABLE: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

_MUT_TOKEN_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_COLON_MUT_RE = re.compile(r"^[A-Z]:(\d+):([A-Z])$")
_HGVS_PRO_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)$")


def clean_text(value: object) -> str:
    """Strip whitespace and normalize null-like values to an empty string."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_sequence(sequence: object) -> str:
    """Normalize a protein sequence by removing whitespace and uppercasing."""
    return re.sub(r"\s+", "", clean_text(sequence)).upper()


def clean_protein_sequence(sequence: object) -> str:
    """Clean a protein sequence by stripping gaps, stop symbols, and whitespace."""
    return re.sub(r"[-.*]", "", normalize_sequence(sequence))


def is_valid_sequence(sequence: str, *, min_len: int = 30) -> bool:
    """Return True if sequence meets minimum length and uses 20 canonical AAs."""
    return len(sequence) >= min_len and set(sequence).issubset(CANONICAL_AA)


def slugify(text: str) -> str:
    """Convert free text into a lowercase underscore-delimited identifier."""
    return re.sub(r"[^a-z0-9]+", "_", clean_text(text).lower()).strip("_")


def resolve_raw_or_docs(
    docs_rel: str | Path,
    *,
    raw_rel: str | Path | None = None,
) -> Path:
    """Resolve a dataset file from RAW_REAL_CASE_DIR or DOCS_DATASETS_DIR."""
    raw_candidate = RAW_REAL_CASE_DIR / (raw_rel or docs_rel)
    return raw_candidate if raw_candidate.exists() else DOCS_DATASETS_DIR / docs_rel


def parse_output_root(description: str | None = None) -> Path:
    """Parse the standard --output-root CLI argument for real-case scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROCESSED_REAL_CASE_DIR,
        help="Root directory for processed real-case datasets.",
    )
    return parser.parse_args().output_root


def read_fasta(path: Path) -> dict[str, str]:
    """Read a FASTA file into a mapping from record identifier to sequence."""
    records: dict[str, list[str]] = {}
    current_id: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].strip().split()[0]
                records[current_id] = []
            elif current_id is not None:
                records[current_id].append(clean_protein_sequence(line))
    return {rec_id: "".join(parts) for rec_id, parts in records.items()}


def read_single_fasta(path: Path) -> str:
    """Read a single-sequence FASTA file and return the cleaned protein sequence."""
    return next(iter(read_fasta(path).values()))


def translate_dna(dna_seq: str) -> str:
    """Translate a coding DNA sequence into a protein sequence without stop codon."""
    clean_dna = re.sub(r"\s+", "", dna_seq).upper()
    aa_list: list[str] = []
    for idx in range(0, len(clean_dna), 3):
        aa = STANDARD_CODON_TABLE[clean_dna[idx : idx + 3]]
        if aa == "*":
            break
        aa_list.append(aa)
    return "".join(aa_list)


def _apply_single_token(
    residues: list[str],
    token: str,
    *,
    offset: int,
    verify_wt: bool,
) -> None:
    """Mutate a single residue in-place from a standard or colon mutation token."""
    colon_match = _COLON_MUT_RE.match(token)
    if colon_match is not None:
        pos_1based = int(colon_match.group(1)) + offset
        residues[pos_1based - 1] = colon_match.group(2)
        return

    std_match = _MUT_TOKEN_RE.match(token)
    if std_match is None:
        raise ValueError(f"Invalid mutation token: {token!r}")
    wt_aa, pos_raw, mut_aa = std_match.groups()
    pos_1based = int(pos_raw) + offset
    actual_wt = residues[pos_1based - 1]
    if verify_wt and actual_wt != wt_aa:
        raise ValueError(
            f"Wild-type mismatch for {token!r}: expected {wt_aa!r} at "
            f"position {pos_1based}, found {actual_wt!r}"
        )
    residues[pos_1based - 1] = mut_aa


def apply_mutations(
    wt_seq: str,
    mutation_str: str | None,
    *,
    offset: int = 0,
    verify_wt: bool = True,
) -> str:
    """Apply 1-based point mutations onto a wild-type protein sequence."""
    text = clean_text(mutation_str)
    if not text or text.upper() in {"WT", "WILDTYPE", "WILD-TYPE", "WILD_TYPE", "-"}:
        return wt_seq

    residues = list(wt_seq)
    tokens = [tok.strip() for tok in re.split(r"[/,+;_|\s]+", text) if tok.strip()]
    for token in tokens:
        if token.upper() != "WT":
            _apply_single_token(residues, token, offset=offset, verify_wt=verify_wt)

    mutated = "".join(residues)
    if not is_valid_sequence(mutated):
        raise ValueError(f"Mutated sequence for {mutation_str!r} is invalid")
    return mutated


def parse_hgvs_missense(hgvs_pro: str, wt_seq: str) -> str | None:
    """Resolve an HGVS protein token (e.g., p.Pro489Asn) to a protein sequence."""
    text = clean_text(hgvs_pro)
    if text in {"WT", "p.="}:
        return wt_seq
    match = _HGVS_PRO_RE.match(text)
    if match is None:
        return None
    wt_3, pos_str, mut_3 = match.groups()
    if mut_3 in {"Ter", "="} or wt_3 == mut_3:
        return None
    wt_aa = AA3_TO_AA1.get(wt_3)
    mut_aa = AA3_TO_AA1.get(mut_3)
    if wt_aa is None or mut_aa is None:
        return None
    return apply_mutations(wt_seq, f"{wt_aa}{pos_str}{mut_aa}")


def canonicalize_smiles(
    smiles: object,
    *,
    neutralize_alkoxide: bool = False,
) -> str:
    """Validate and canonicalize a SMILES string using RDKit."""
    raw = clean_text(smiles)
    if neutralize_alkoxide:
        raw = raw.replace("[O-]", "O")
    mol = Chem.MolFromSmiles(raw)
    if mol is None or not raw:
        raise ValueError(f"Invalid SMILES string: {raw!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_multi_smiles(smiles_list: list[str]) -> str:
    """Canonicalize multiple reactant/product SMILES and join them with dots."""
    return ".".join(canonicalize_smiles(s) for s in smiles_list)


def parse_float(value: object) -> float | None:
    """Parse a finite floating-point number, returning None for missing tokens."""
    text = clean_text(value)
    if not text or text.upper() in {"ND", "NS", "NA", "N/A", "-"}:
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    return num if math.isfinite(num) else None


def compute_log10(value: float, *, offset: float = 0.0) -> float:
    """Compute base-10 logarithm with an optional additive offset."""
    return math.log10(value + offset)


def format_float(value: float, *, decimals: int = 6) -> str:
    """Format a float to a fixed maximum number of decimals."""
    return f"{value:.{decimals}f}"


def aggregate_replicates(
    records: list[tuple[tuple[str, ...], float]],
) -> dict[tuple[str, ...], float]:
    """Compute arithmetic mean across technical/biological replicates per key."""
    grouped: dict[tuple[str, ...], list[float]] = collections.defaultdict(list)
    for key, val in records:
        grouped[key].append(val)
    return {key: sum(vals) / len(vals) for key, vals in grouped.items()}


def build_layout_a_rows(
    task_name: str,
    records: list[tuple[tuple[str, ...], float]],
    *,
    target_col: str,
    raw_col: str | None = None,
    log10_offset: float | None = None,
) -> list[dict[str, object]]:
    """Aggregate replicates and format Layout A (protein-only) benchmark rows."""
    aggregated = aggregate_replicates(records)
    rows: list[dict[str, object]] = []
    for idx, ((seq,), val) in enumerate(sorted(aggregated.items()), start=1):
        row: dict[str, object] = {
            _C.dataset_id: f"{task_name}_{idx:06d}",
            _C.protein_sequence: seq,
        }
        if raw_col is not None:
            row[raw_col] = format_float(val)
        target_val = (
            compute_log10(val, offset=log10_offset) if log10_offset is not None else val
        )
        row[target_col] = format_float(target_val)
        rows.append(row)
    return rows


def build_layout_b_rows(
    task_name: str,
    records: list[tuple[tuple[str, ...], float]],
    *,
    target_col: str,
    raw_col: str | None = None,
    log10_offset: float | None = None,
) -> list[dict[str, object]]:
    """Aggregate replicates and format Layout B (protein + SMILES) benchmark rows."""
    aggregated = aggregate_replicates(records)
    rows: list[dict[str, object]] = []
    for idx, ((seq, smi), val) in enumerate(sorted(aggregated.items()), start=1):
        row: dict[str, object] = {
            _C.dataset_id: f"{task_name}_{idx:06d}",
            _C.protein_sequence: seq,
            _C.smiles: smi,
        }
        if raw_col is not None:
            row[raw_col] = format_float(val)
        target_val = (
            compute_log10(val, offset=log10_offset) if log10_offset is not None else val
        )
        row[target_col] = format_float(target_val)
        rows.append(row)
    return rows


def validate_dataset_rows(
    rows: list[dict[str, object]],
    *,
    task_name: str,
    min_rows: int = 300,
    max_rows: int = 20000,
    min_unique_seqs: int = 10,
) -> None:
    """Validate row count, unique protein count, and sequence integrity."""
    n_rows = len(rows)
    if not (min_rows <= n_rows <= max_rows):
        raise ValueError(
            f"Task {task_name!r} has {n_rows} rows, outside bounds "
            f"[{min_rows}, {max_rows}]"
        )

    dataset_ids = {str(row.get(_C.dataset_id, "")) for row in rows}
    if len(dataset_ids) != n_rows:
        raise ValueError(f"Task {task_name!r} has non-unique dataset_id values")

    unique_seqs = {str(row.get(_C.protein_sequence, "")) for row in rows}
    if not all(is_valid_sequence(seq) for seq in unique_seqs):
        raise ValueError(f"Task {task_name!r} contains invalid protein_sequence")
    if len(unique_seqs) < min_unique_seqs:
        raise ValueError(
            f"Task {task_name!r} has only {len(unique_seqs)} unique sequences "
            f"(minimum {min_unique_seqs})"
        )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    """Write dictionary rows to a CSV file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_processed_dataset(
    task_name: str,
    fieldnames: list[str],
    rows: list[dict[str, object]],
    *,
    output_root: Path = PROCESSED_REAL_CASE_DIR,
    min_rows: int = 300,
    max_rows: int = 20000,
    min_unique_seqs: int = 10,
) -> Path:
    """Validate and write a processed real-case dataset CSV."""
    validate_dataset_rows(
        rows,
        task_name=task_name,
        min_rows=min_rows,
        max_rows=max_rows,
        min_unique_seqs=min_unique_seqs,
    )
    out_path = output_root / task_name / f"{task_name}.csv"
    write_csv(out_path, fieldnames, rows)
    return out_path
