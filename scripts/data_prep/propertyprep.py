import numpy as np
import pandas as pd
import sys
from pathlib import Path
import csv
import json
import time
import requests
import re

# Resolve root path correctly for a script inside the scripts/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Wait to import paths and constants until sys.path is configured
from src.paths_and_constants import Paths, Constants
_P = Paths()
_C = Constants()

_UNIPROT_SEQUENCE_INDEX_CACHE = {}


def sequence_getter(
    uniprot_ids,
    checkpoint_path="uniprot_checkpoint.json",
    save_every=50,
    timeout=15,
    sleep_s=0.0,
    retry_failed=True,
    show_progress=True,
):
    ckpt = Path(checkpoint_path)

    if ckpt.exists():
        with ckpt.open("r", encoding="utf-8") as f:
            state = json.load(f)
        done = state.get("done", {})
        failed = state.get("failed", {})
    else:
        done, failed = {}, {}

    ids = []
    seen = set()
    for x in uniprot_ids:
        uid = str(x).strip()
        if uid and uid.lower() != "nan" and uid not in seen:
            ids.append(uid)
            seen.add(uid)

    if retry_failed:
        for uid in ids:
            failed.pop(uid, None)

    pending = [uid for uid in ids if uid not in done and uid not in failed]

    total = len(ids)
    pending_total = len(pending)
    overall_done = total - pending_total

    def save_ckpt():
        tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"done": done, "failed": failed}, f, ensure_ascii=False, indent=2)
        tmp.replace(ckpt)

    print(
        f"Resume status -> done: {len(done)}, failed: {len(failed)}, "
        f"pending: {pending_total}, total: {total}"
    )

    with requests.Session() as session:
        for i, uid in enumerate(pending, start=1):
            try:
                seq = ""
                r = session.get(f"https://rest.uniprot.org/uniprotkb/{uid}.fasta", timeout=timeout)
                if r.status_code == 200:
                    lines = r.text.strip().splitlines()
                    if lines and lines[0].startswith(">"):
                        seq = "".join(lines[1:]).replace(" ", "").upper()

                if not seq:
                    rj = session.get(f"https://rest.uniprot.org/uniprotkb/{uid}.json", timeout=timeout)
                    if rj.status_code == 200:
                        seq = rj.json().get("sequence", {}).get("value", "").upper()

                if not seq:
                    rs = session.get(
                        "https://rest.uniprot.org/uniprotkb/search",
                        params={
                            "query": f"(accession:{uid}) OR (id:{uid}) OR (sec_acc:{uid})",
                            "format": "json",
                            "size": 5,
                        },
                        timeout=timeout,
                    )
                    rs.raise_for_status()
                    results = rs.json().get("results", [])

                    chosen = None
                    for entry in results:
                        acc = entry.get("primaryAccession", "")
                        kb_id = entry.get("uniProtkbId", "")
                        sec = set(entry.get("secondaryAccessions", []))
                        if uid == acc or uid == kb_id or uid in sec:
                            chosen = entry
                            break
                    if chosen is None and results:
                        chosen = results[0]

                    if chosen:
                        seq = chosen.get("sequence", {}).get("value", "").upper()

                if not seq:
                    raise ValueError(f"No sequence found for: {uid}")

                done[uid] = seq
                failed.pop(uid, None)

            except Exception as e:
                failed[uid] = str(e)

            if i % save_every == 0:
                save_ckpt()

            if show_progress and (i % save_every == 0 or i == pending_total):
                print(
                    f"Progress pending: {i}/{pending_total} ({(i/pending_total*100 if pending_total else 100):.2f}%) | "
                    f"overall: {overall_done + i}/{total} ({((overall_done + i)/total*100 if total else 100):.2f}%)"
                )

            if sleep_s > 0:
                time.sleep(sleep_s)

    save_ckpt()
    return done, failed


def _normalize_protein_sequence(sequence):
    seq = str(sequence).strip().upper().replace(" ", "")
    if not seq or seq == "NAN":
        raise ValueError("Empty sequence")
    return seq


def _split_ec_numbers(ec_value):
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


def _load_swissprot_sequence_index(swissprot_tsv_path):
    swissprot_tsv_path = str(Path(swissprot_tsv_path).resolve())
    cached = _UNIPROT_SEQUENCE_INDEX_CACHE.get(swissprot_tsv_path)
    if cached is not None:
        return cached

    index = {}
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

    _UNIPROT_SEQUENCE_INDEX_CACHE[swissprot_tsv_path] = index
    return index


def _pick_best_local_match(records):
    if not records:
        return None

    def sort_key(record):
        accession = record.get("accession") or ""
        ec_numbers = record.get("ec_numbers") or []
        return (0 if ec_numbers else 1, accession)

    return sorted(records, key=sort_key)[0]


def _extract_ec_numbers_from_uniprot_json(payload):
    ec_numbers = []
    seen = set()

    def walk(node):
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


def _get_uniprot_entry_metadata(accession, session, timeout=20):
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


def _submit_ebi_fasta_job(sequence, email, session, timeout=30, database="uniprotkb_swissprot"):
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


def _poll_ebi_fasta_job(job_id, session, timeout=20, poll_interval=3.0, max_polls=40):
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


def _parse_best_accession_from_ebi_output(output_text):
    for line in output_text.splitlines():
        line = line.strip()
        if not line or not line.startswith(("SP:", "TR:")):
            continue
        match = re.match(r"^(?:SP|TR):([A-Z0-9]+)\b", line)
        if match:
            return match.group(1)
    return None


def _find_accession_by_sequence_similarity(
    sequence, email, session, timeout=30, poll_interval=3.0, max_polls=40, database="uniprotkb_swissprot"
):
    if not email:
        raise ValueError("email is required for remote sequence similarity lookup via EMBL-EBI FASTA")

    job_id = _submit_ebi_fasta_job(sequence, email, session, timeout, database)
    _poll_ebi_fasta_job(job_id, session, timeout, poll_interval, max_polls)

    result = session.get(f"https://www.ebi.ac.uk/Tools/services/rest/fasta/result/{job_id}/out", timeout=timeout)
    result.raise_for_status()
    accession = _parse_best_accession_from_ebi_output(result.text)
    if not accession:
        raise ValueError("Could not parse a UniProt accession from EBI FASTA output")
    return accession


def sequence_to_enzyme_metadata(
    sequences,
    checkpoint_path="sequence_to_enzyme_checkpoint.json",
    swissprot_tsv_path=None,
    save_every=25,
    timeout=20,
    sleep_s=0.0,
    retry_failed=True,
    show_progress=True,
    use_remote_fallback=True,
    email=None,
    remote_database="uniprotkb_swissprot",
    poll_interval=3.0,
    max_polls=40,
):
    if swissprot_tsv_path is None:
        swissprot_tsv_path = _P.data_raw_swissprot_dir / "swissprot_ec.tsv"

    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        with ckpt.open("r", encoding="utf-8") as f:
            state = json.load(f)
        done = state.get("done", {})
        failed = state.get("failed", {})
    else:
        done, failed = {}, {}

    unique_sequences = []
    seen = set()
    for sequence in sequences:
        try:
            seq = _normalize_protein_sequence(sequence)
        except ValueError:
            continue
        if seq not in seen:
            unique_sequences.append(seq)
            seen.add(seq)

    if retry_failed:
        for seq in unique_sequences:
            failed.pop(seq, None)

    pending = [seq for seq in unique_sequences if seq not in done and seq not in failed]

    total = len(unique_sequences)
    pending_total = len(pending)
    overall_done = total - pending_total

    swissprot_index = _load_swissprot_sequence_index(swissprot_tsv_path)

    def save_ckpt():
        tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"done": done, "failed": failed}, f, ensure_ascii=False, indent=2)
        tmp.replace(ckpt)

    print(
        f"Resume status -> done: {len(done)}, failed: {len(failed)}, "
        f"pending: {pending_total}, total: {total}"
    )

    with requests.Session() as session:
        for i, seq in enumerate(pending, start=1):
            try:
                match = _pick_best_local_match(swissprot_index.get(seq, []))
                match_source = None

                if match is not None:
                    accession = match.get("accession")
                    ec_numbers = match.get("ec_numbers") or []
                    match_source = "local_swissprot_exact"
                else:
                    if not use_remote_fallback:
                        raise ValueError("No exact Swiss-Prot match found for sequence")
                    accession = _find_accession_by_sequence_similarity(
                        sequence=seq, email=email, session=session, timeout=timeout,
                        poll_interval=poll_interval, max_polls=max_polls, database=remote_database
                    )
                    metadata = _get_uniprot_entry_metadata(accession=accession, session=session, timeout=timeout)
                    accession = metadata.get("accession")
                    ec_numbers = metadata.get("ec_numbers") or []
                    match_source = "ebi_fasta_similarity"

                ec_number = "; ".join(ec_numbers) if ec_numbers else None
                done[seq] = {
                    "accession": accession,
                    "ec_number": ec_number,
                    "is_enzyme": bool(ec_numbers),
                    "match_source": match_source,
                }
                failed.pop(seq, None)

            except Exception as e:
                failed[seq] = str(e)

            if i % save_every == 0:
                save_ckpt()

            if sleep_s > 0:
                time.sleep(sleep_s)

    save_ckpt()
    return done, failed


MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

def parse_substitution(sub):
    m = MUT_RE.match(sub.strip().upper())
    if not m:
        raise ValueError(f"Substitution format invalid: {sub}")
    wt, pos, mut = m.groups()
    return wt, int(pos), mut

def apply_substitutions(seq, substitutions, sep=r"[;,]\s*"):
    seq_list = list(seq)
    muts = re.split(sep, substitutions.strip())

    for sub in muts:
        if not sub:
            continue
        wt, pos, mut = parse_substitution(sub)
        idx = pos - 1  # 1-based -> 0-based

        if idx < 0 or idx >= len(seq_list):
            raise IndexError(f"Position out of range: {pos} for sequence of length {len(seq_list)}")
        if seq_list[idx] != wt:
            raise ValueError(f"Mismatch in {sub}: expected '{wt}' at position {pos}, but found '{seq_list[idx]}'")

        seq_list[idx] = mut

    return "".join(seq_list)


active_pattern = re.compile(r"ACT_SITE\s+(\d+)")

def parse_active_sites(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, (list, tuple, set)):
        out = []
        for x in v:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                pass
        return sorted(set(out))
    if isinstance(v, (int, float)) and not pd.isna(v):
        return [int(v)]
    return [int(x) for x in active_pattern.findall(str(v))]


seg_re = re.compile(r'BINDING\s+([^;]+);\s*(.*?)(?=BINDING\s+|$)', re.DOTALL)
lig_re = re.compile(r'/ligand="([^"]*)"')
chebi_re = re.compile(r'/ligand_id="([^"]*)"')
int_re = re.compile(r'^\d+$')
range_re = re.compile(r'^(\d+)\.\.(\d+)$')

def parse_loc_safe(loc: str):
    loc = loc.strip()
    isoform = None
    if ":" in loc:
        isoform, loc = loc.split(":", 1)
        loc = loc.strip()

    m = range_re.match(loc)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start <= end:
            return isoform, list(range(start, end + 1))
        return isoform, None

    if int_re.match(loc):
        return isoform, [int(loc)]

    return isoform, None

def parse_binding(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return [], [], [], 0

    s = str(v)
    positions = []
    ligands = []
    ligand_ids = []
    ignored = 0

    for m in seg_re.finditer(s):
        loc_raw = m.group(1).strip()
        attrs = m.group(2)

        _, pos_list = parse_loc_safe(loc_raw)
        if not pos_list:
            ignored += 1
            continue

        ligand = lig_re.search(attrs)
        chebi = chebi_re.search(attrs)

        ligand_val = ligand.group(1) if ligand else None
        chebi_val = chebi.group(1) if chebi else None

        for p in pos_list:
            positions.append(p)
            ligands.append(ligand_val)
            ligand_ids.append(chebi_val)

    return positions, ligands, ligand_ids, ignored



def main():
    print("Running Propertyprep Script...")
    
    stats = {}
    
    # ---------------------------
    # Binding (BioLiP)
    # ---------------------------
    print("\nProcessing BioLip Binding...")
    biol_path = _P.data_processed_datasets / 'biolip_binding' / 'biolip_cleaned_with_negatives.csv'
    if biol_path.exists():
        biol = pd.read_csv(biol_path, index_col=0)
        len_before = len(biol)
        st = {"datapoints_before_cleaning": len_before, "removed_by_reason": {"missing_values": 0, "duplicates": 0, "invalid_sequence": 0, "was_not_enzyme": 0, "other": 0}}

        biol["protein_sequence"] = biol["protein_sequence"].astype(str).str.strip()
        biol["uniprot_id"] = biol["uniprot_id"].replace("", pd.NA)

        
        len_b = len(biol)
        seq_to_uniprot = (
            biol.dropna(subset=["uniprot_id"])
              .drop_duplicates(subset=["protein_sequence"])
              .set_index("protein_sequence")["uniprot_id"]
        )
        biol["uniprot_id"] = biol["uniprot_id"].fillna(biol["protein_sequence"].map(seq_to_uniprot))

        biol['smiles'] = biol['smiles'].astype(str).str.strip()
        biol['ligand_id_raw'] = biol['ligand_id_raw'].replace("", pd.NA)
        biol['ligand_name'] = biol['ligand_name'].replace("", pd.NA)

        smiles_to_id = (
            biol.dropna(subset=['ligand_id_raw']).drop_duplicates(subset=['smiles']).set_index('smiles')['ligand_id_raw']
        )
        smiles_to_name = (
            biol.dropna(subset=['ligand_name']).drop_duplicates(subset=['smiles']).set_index('smiles')['ligand_name']
        )

        biol['ligand_id_raw'] = biol['ligand_id_raw'].fillna(biol['smiles'].map(smiles_to_id))
        biol['ligand_name'] = biol['ligand_name'].fillna(biol['smiles'].map(smiles_to_name))

        final_cols = ['uniprot_id', 'protein_sequence', 'ligand_id', 'ligand_name', 'smiles', 'label']
        if all(c in biol.columns for c in final_cols):
            biol_final = biol[final_cols].rename(columns={'label': 'binding'}).reset_index(drop=True)
            biol_final.to_csv(_P.data_processed_properties / 'binding' / 'biolip_binding.csv', index=False)
            st["datapoints_after_cleaning"] = len(biol_final)
            st["datapoints_removed_total"] = st["datapoints_before_cleaning"] - st["datapoints_after_cleaning"]
            st["removed_percentage"] = (st["datapoints_removed_total"] / st["datapoints_before_cleaning"] * 100) if st["datapoints_before_cleaning"] else 0
            stats["biolip_binding"] = {"amount_of_removed_data": st}
            print("BioLip Binding saved.")


    # ---------------------------
    # DDG (FireProtDB)
    # ---------------------------
    print("\nProcessing FireProtDB DDG...")
    ddg_path = _P.data_interim_dir / 'fireprotdb' / 'fireprotdb_ddg.csv'
    if ddg_path.exists():
        ddg = pd.read_csv(ddg_path, index_col=0).reset_index()
        st = {"datapoints_before_cleaning": len(ddg), "removed_by_reason": {"missing_values": 0, "duplicates": 0, "invalid_sequence": 0, "was_not_enzyme": 0, "other": 0}}
        fire_cols = ['EXPERIMENT_ID', 'UNIPROTKB', 'SUBSTITUTION', 'DDG']
        ddg = ddg[fire_cols]
        unique_prot = ddg['UNIPROTKB'].unique()

        done, failed = sequence_getter(
            unique_prot,
            checkpoint_path=str(_P.data_interim_dir / "sequence_checkpoint" / "uniprot_checkpoint.json"),
            save_every=10
        )
        ddg["sequence"] = ddg['UNIPROTKB'].map(done)
        ddg_f = ddg.rename(columns={
            'EXPERIMENT_ID': _C.dataset_id,
            'UNIPROTKB': _C.uniprot_id,
            'sequence': _C.protein_sequence,
            'SUBSTITUTION': 'mutation',
            'DDG': 'ddg'
        })
        len_b = len(ddg_f)
        ddg_f = ddg_f.dropna(subset=[_C.protein_sequence, 'mutation'])
        st['removed_by_reason']['missing_values'] += len_b - len(ddg_f)
        ddg_f['mut_protein_sequence'] = ddg_f.apply(lambda row: apply_substitutions(row['protein_sequence'], row['mutation']), axis=1)
        ddg_f.rename(columns={'protein_sequence': 'wildtype_sequence', 'mut_protein_sequence': 'protein_sequence'}, inplace=True)
        ddg_f = ddg_f[["dataset_id", "uniprot_id", "wildtype_sequence", "mutation", "protein_sequence", "ddg"]]
        ddg_f = ddg_f.dropna(subset=[_C.protein_sequence, 'ddg'])
        ddg_f = ddg_f.drop_duplicates(subset=[_C.protein_sequence, 'ddg'])
        
        out_dir = _P.data_processed_properties / 'ddg_mut'
        out_dir.mkdir(parents=True, exist_ok=True)
        ddg_f.to_csv(out_dir / 'fireprotdb_ddg.csv', index=False)
        st["datapoints_after_cleaning"] = len(ddg_f)
        st["datapoints_removed_total"] = st["datapoints_before_cleaning"] - st["datapoints_after_cleaning"]
        st["removed_percentage"] = (st["datapoints_removed_total"] / st["datapoints_before_cleaning"] * 100) if st["datapoints_before_cleaning"] else 0
        stats["fireprotdb_ddg"] = {"amount_of_removed_data": st}
        print("FireProtDB DDG saved.")


    # ---------------------------
    # DTM (FireProtDB)
    # ---------------------------
    print("\nProcessing FireProtDB DTM...")
    dtm_path = _P.data_interim_dir / 'fireprotdb' / 'fireprotdb_dtm.csv'
    if dtm_path.exists():
        dtm = pd.read_csv(dtm_path, index_col=0).reset_index()
        st = {"datapoints_before_cleaning": len(dtm), "removed_by_reason": {"missing_values": 0, "duplicates": 0, "invalid_sequence": 0, "was_not_enzyme": 0, "other": 0}}
        fire_cols = ['EXPERIMENT_ID', 'UNIPROTKB', 'SUBSTITUTION', 'DTM']
        dtm = dtm[fire_cols]
        unique_prot_dtm = dtm['UNIPROTKB'].unique()

        done, failed = sequence_getter(
            unique_prot_dtm,
            checkpoint_path=str(_P.data_interim_dir / "sequence_checkpoint" / "uniprot_checkpoint.json"),
            save_every=10
        )
        dtm["sequence"] = dtm['UNIPROTKB'].map(done)
        dtm_f = dtm.rename(columns={
            'EXPERIMENT_ID': _C.dataset_id,
            'UNIPROTKB': _C.uniprot_id,
            'sequence': _C.protein_sequence,
            'SUBSTITUTION': 'mutation',
            'DTM': 'dtm'
        })
        len_b = len(dtm_f)
        dtm_f = dtm_f.dropna(subset=['mutation', _C.protein_sequence])
        st['removed_by_reason']['missing_values'] += len_b - len(dtm_f)
        dtm_f['mut_protein_sequence'] = dtm_f.apply(lambda row: apply_substitutions(row['protein_sequence'], row['mutation']), axis=1)
        dtm_f.rename(columns={'protein_sequence': 'wildtype_sequence', 'mut_protein_sequence': 'protein_sequence'}, inplace=True)
        dtm_f = dtm_f[["dataset_id", "uniprot_id", "wildtype_sequence", "mutation", "protein_sequence", "dtm"]]
        dtm_f = dtm_f.drop_duplicates(subset=[_C.protein_sequence, 'dtm'])
        
        out_dir = _P.data_processed_properties / 'dtm'
        out_dir.mkdir(parents=True, exist_ok=True)
        dtm_f.to_csv(out_dir / 'fireprotdb_dtm.csv', index=False)
        st["datapoints_after_cleaning"] = len(dtm_f)
        st["datapoints_removed_total"] = st["datapoints_before_cleaning"] - st["datapoints_after_cleaning"]
        st["removed_percentage"] = (st["datapoints_removed_total"] / st["datapoints_before_cleaning"] * 100) if st["datapoints_before_cleaning"] else 0
        stats["fireprotdb_dtm"] = {"amount_of_removed_data": st}
        print("FireProtDB DTM saved.")


    # ---------------------------
    # pH Optimum (BRENDA)
    # ---------------------------
    print("\nProcessing BRENDA phopt...")
    brenda_path = _P.data_interim_dir / 'phopt' / 'brenda_phopt.csv'
    if brenda_path.exists():
        brenda_raw = pd.read_csv(brenda_path, index_col=0)
        st = {"datapoints_before_cleaning": len(brenda_raw), "removed_by_reason": {"missing_values": 0, "duplicates": 0, "invalid_sequence": 0, "was_not_enzyme": 0, "other": 0}}
        unique_brenda = brenda_raw['uniprot_from_brenda'].unique()

        done, failed = sequence_getter(
            unique_brenda,
            checkpoint_path=str(_P.data_interim_dir / "sequence_checkpoint" / "uniprot_checkpoint.json"),
            save_every=10
        )
        cols = ['uniprot_from_brenda', 'phopt_median', 'n_measurements', 'phopt_std']
        brenda_opt = brenda_raw[cols]
        brenda_opt["sequence"] = brenda_opt['uniprot_from_brenda'].map(done)
        brenda_opt = brenda_opt.rename(columns={
            'uniprot_from_brenda': _C.uniprot_id,
            'sequence': _C.protein_sequence,
            'phopt': 'phopt'
        })
        brenda_opt = brenda_opt[[_C.uniprot_id, _C.protein_sequence, 'phopt_median', 'n_measurements', 'phopt_std']]
        len_b = len(brenda_opt)
        brenda_opt = brenda_opt.dropna(subset=[_C.protein_sequence])
        st['removed_by_reason']['missing_values'] += len_b - len(brenda_opt)
        len_b = len(brenda_opt)
        brenda_opt = brenda_opt.drop_duplicates(subset=[_C.protein_sequence, 'phopt_median', 'n_measurements', 'phopt_std'])
        st['removed_by_reason']['duplicates'] += len_b - len(brenda_opt)
        
        out_dir = _P.data_processed_properties / 'ph_opt'
        out_dir.mkdir(parents=True, exist_ok=True)
        brenda_opt.to_csv(out_dir / 'brenda_phopt.csv', index=False)
        st["datapoints_after_cleaning"] = len(brenda_opt)
        st["datapoints_removed_total"] = st["datapoints_before_cleaning"] - st["datapoints_after_cleaning"]
        st["removed_percentage"] = (st["datapoints_removed_total"] / st["datapoints_before_cleaning"] * 100) if st["datapoints_before_cleaning"] else 0
        stats["brenda_phopt"] = {"amount_of_removed_data": st}
        print("BRENDA phopt saved.")


    # ---------------------------
    # kcat (CatPred)
    # ---------------------------
    print("\nProcessing CatPred kcat...")
    kcat_path = _P.data_raw_catpred_dir / 'CatPred_kcat.csv'
    if kcat_path.exists():
        kcat_raw = pd.read_csv(kcat_path, index_col=0)
        st = {"datapoints_before_cleaning": len(kcat_raw), "removed_by_reason": {"missing_values": 0, "duplicates": 0, "invalid_sequence": 0, "was_not_enzyme": 0, "other": 0}}
        kcat_columns = ['uniprot', 'sequence', 'reactant_smiles', 'temperature', 'ph', 'log10kcat_max']
        if all(c in kcat_raw.columns for c in kcat_columns):
            kcat = kcat_raw[kcat_columns].rename(columns={
                'uniprot': _C.uniprot_id,
                'sequence': _C.protein_sequence,
                'reactant_smiles': _C.smiles,
                'temperature': 'temperature',
                'ph': 'ph',
                'log10kcat_max': 'kcat'
            })
            out_dir = _P.data_processed_properties / 'kcat'
            out_dir.mkdir(parents=True, exist_ok=True)
            kcat.to_csv(out_dir / 'catpred_kcat.csv', index=False)
            st["datapoints_after_cleaning"] = len(kcat)
            st["datapoints_removed_total"] = st["datapoints_before_cleaning"] - st["datapoints_after_cleaning"]
            st["removed_percentage"] = (st["datapoints_removed_total"] / st["datapoints_before_cleaning"] * 100) if st["datapoints_before_cleaning"] else 0
            stats["catpred_kcat"] = {"amount_of_removed_data": st}
            print("CatPred kcat saved.")


    # ---------------------------
    # KM (CatPred)
    # ---------------------------
    print("\nProcessing CatPred KM...")
    km_path = _P.data_raw_catpred_dir / 'CatPred_km.csv'
    if km_path.exists():
        km_raw = pd.read_csv(km_path, index_col=0)
        st = {"datapoints_before_cleaning": len(km_raw), "removed_by_reason": {"missing_values": 0, "duplicates": 0, "invalid_sequence": 0, "was_not_enzyme": 0, "other": 0}}
        km_columns = ['uniprot', 'sequence', 'substrate_smiles', 'temperature', 'ph', 'log10km_mean']
        if all(c in km_raw.columns for c in km_columns):
            km = km_raw[km_columns].rename(columns={
                'uniprot': _C.uniprot_id,
                'sequence': _C.protein_sequence,
                'substrate_smiles': _C.smiles,
                'temperature': 'temperature',
                'ph': 'ph',
                'log10km_mean': 'km'
            })
            out_dir = _P.data_processed_properties / 'km'
            out_dir.mkdir(parents=True, exist_ok=True)
            km.to_csv(out_dir / 'catpred_km.csv', index=False)
            st["datapoints_after_cleaning"] = len(km)
            st["datapoints_removed_total"] = st["datapoints_before_cleaning"] - st["datapoints_after_cleaning"]
            st["removed_percentage"] = (st["datapoints_removed_total"] / st["datapoints_before_cleaning"] * 100) if st["datapoints_before_cleaning"] else 0
            stats["catpred_km"] = {"amount_of_removed_data": st}
            print("CatPred KM saved.")


    # ---------------------------
    # ProteinEA solubility
    # ---------------------------
    print("\nProcessing ProteinEA Solubility...")
    proteinea_configs = [
        {
            "property_name": "solubility",
            "target_col": "solubility",
            "input_path": _P.data_processed_properties / 'solubility' / 'proteinea_solubility.csv',
            "output_path": _P.data_processed_properties / 'solubility' / 'proteinea_solubility.csv',
        },
    ]

    proteinea_remote_email = None
    checkpoint_dir = _P.data_interim_dir / "sequence_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for cfg in proteinea_configs:
        if not cfg["input_path"].exists():
            continue
        df = pd.read_csv(cfg["input_path"])
        st = {"datapoints_before_cleaning": len(df), "removed_by_reason": {"missing_values": 0, "duplicates": 0, "invalid_sequence": 0, "was_not_enzyme": 0, "other": 0}}
        df[_C.protein_sequence] = df[_C.protein_sequence].astype(str).str.strip().str.upper()
        df["split"] = df["split"].astype(str).str.strip()
        df = df.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        len_b = len(df)
        df = df.dropna(subset=[_C.protein_sequence, cfg["target_col"], "split"]).copy()
        st['removed_by_reason']['missing_values'] += len_b - len(df)
        len_b = len(df)
        df = df.drop_duplicates(subset=[_C.protein_sequence, cfg["target_col"], "split"]).copy()
        st['removed_by_reason']['duplicates'] += len_b - len(df)

        unique_sequences = df[_C.protein_sequence].dropna().astype(str).unique()
        done, failed = sequence_to_enzyme_metadata(
            unique_sequences,
            checkpoint_path=str(checkpoint_dir / f"proteinea_{cfg['property_name']}_to_enzyme_checkpoint.json"),
            save_every=25,
            use_remote_fallback=bool(proteinea_remote_email),
            email=proteinea_remote_email,
        )

        metadata_by_sequence = {seq: meta for seq, meta in done.items() if isinstance(meta, dict)}
        df[_C.uniprot_id] = df[_C.protein_sequence].map(lambda seq: (metadata_by_sequence.get(seq) or {}).get("accession"))
        df["ec_number"] = df[_C.protein_sequence].map(lambda seq: (metadata_by_sequence.get(seq) or {}).get("ec_number"))
        df["is_enzyme"] = df[_C.protein_sequence].map(lambda seq: bool((metadata_by_sequence.get(seq) or {}).get("is_enzyme")))

        len_b = len(df)
        df = df[df["is_enzyme"]].copy()
        st['removed_by_reason']['was_not_enzyme'] += len_b - len(df)
        cols_to_keep = ["dataset_id", _C.uniprot_id, _C.protein_sequence, "ec_number", cfg["target_col"], "split"]
        if all(c in df.columns for c in cols_to_keep):
            df = df[cols_to_keep].reset_index(drop=True)
            cfg["output_path"].parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cfg["output_path"], index=False)
            print(f"ProteinEA {cfg['property_name']} saved.")
            st["datapoints_after_cleaning"] = len(df)
            st["datapoints_removed_total"] = st["datapoints_before_cleaning"] - st["datapoints_after_cleaning"]
            st["removed_percentage"] = (st["datapoints_removed_total"] / st["datapoints_before_cleaning"] * 100) if st["datapoints_before_cleaning"] else 0
            stats[f"proteinea_{cfg['property_name']}"] = {"amount_of_removed_data": st}


    # ---------------------------
    # EC Number (UniProt)
    # ---------------------------
    print("\nProcessing UniProt EC numbers...")
    swissprot_path = _P.data_raw_swissprot_dir / 'swissprot_ec.tsv'
    if swissprot_path.exists():
        # Do not use index_col=1 to retain standard columns correctly
        ec = pd.read_csv(swissprot_path, sep='\t')
        st = {"datapoints_before_cleaning": len(ec), "removed_by_reason": {"missing_values": 0, "duplicates": 0, "invalid_sequence": 0, "was_not_enzyme": 0, "other": 0}}
        ec_columns = ['Entry', 'Sequence', 'EC number']
        if all(c in ec.columns for c in ec_columns):
            ec = ec[ec_columns].rename(columns={
                'Entry': _C.uniprot_id,
                'Sequence': _C.protein_sequence,
                'EC number': 'ec_number'
            })
            len_b = len(ec)
            ec = ec.dropna(subset=['ec_number'])
            st['removed_by_reason']['missing_values'] += len_b - len(ec)
            len_b = len(ec)
            ec = ec.drop_duplicates(subset=[_C.protein_sequence, 'ec_number'])
            st['removed_by_reason']['duplicates'] += len_b - len(ec)
            
            out_dir = _P.data_processed_properties / 'ec_number'
            out_dir.mkdir(parents=True, exist_ok=True)
            ec.to_csv(out_dir / 'swissprot_ec.csv', index=False)
            print("UniProt EC numbers saved.")
            st["datapoints_after_cleaning"] = len(ec)
            st["datapoints_removed_total"] = st["datapoints_before_cleaning"] - st["datapoints_after_cleaning"]
            st["removed_percentage"] = (st["datapoints_removed_total"] / st["datapoints_before_cleaning"] * 100) if st["datapoints_before_cleaning"] else 0
            stats["swissprot_ec"] = {"amount_of_removed_data": st}


    # ---------------------------
    # Active sites (UniProt)
    # ---------------------------
    print("\nProcessing UniProt Active sites...")
    if swissprot_path.exists():
        active = pd.read_csv(swissprot_path, sep='\t')
        st = {"datapoints_before_cleaning": len(active), "removed_by_reason": {"missing_values": 0, "duplicates": 0, "invalid_sequence": 0, "was_not_enzyme": 0, "other": 0}}
        active_columns = ['Entry', 'EC number', 'Sequence', 'Active site']
        if all(c in active.columns for c in active_columns):
            active = active[active_columns].rename(columns={
                'Entry': _C.uniprot_id,
                'EC number': 'ec_number',
                'Sequence': _C.protein_sequence,
                'Active site': 'active_site'
            })
            len_b = len(active)
            active.dropna(subset=['active_site'], inplace=True)
            st['removed_by_reason']['missing_values'] += len_b - len(active)
            len_b = len(active)
            active.drop_duplicates(subset=[_C.protein_sequence, 'active_site'], inplace=True)
            st['removed_by_reason']['duplicates'] += len_b - len(active)

            out_dir_interim = _P.data_interim_dir / 'swissprot'
            out_dir_interim.mkdir(parents=True, exist_ok=True)
            active.to_csv(out_dir_interim / 'swissprot_active_site.csv', index=False)

            active['active_site'] = active['active_site'].apply(parse_active_sites)
            
            out_dir = _P.data_processed_properties / 'active_sites'
            out_dir.mkdir(parents=True, exist_ok=True)
            active.to_csv(out_dir / 'swissprot_active_site.csv', index=False)
            print("UniProt Active sites saved.")
            st["datapoints_after_cleaning"] = len(active)
            st["datapoints_removed_total"] = st["datapoints_before_cleaning"] - st["datapoints_after_cleaning"]
            st["removed_percentage"] = (st["datapoints_removed_total"] / st["datapoints_before_cleaning"] * 100) if st["datapoints_before_cleaning"] else 0
            stats["swissprot_active_site"] = {"amount_of_removed_data": st}


    # ---------------------------
    # Binding sites (UniProt)
    # ---------------------------
    print("\nProcessing UniProt Binding sites...")
    if swissprot_path.exists():
        binding_site = pd.read_csv(swissprot_path, sep='\t')
        st = {"datapoints_before_cleaning": len(binding_site), "removed_by_reason": {"missing_values": 0, "duplicates": 0, "invalid_sequence": 0, "was_not_enzyme": 0, "other": 0}}
        binding_columns = ['Entry', 'EC number', 'Sequence', 'Binding site']
        if all(c in binding_site.columns for c in binding_columns):
            binding_site = binding_site[binding_columns].rename(columns={
                'Entry': _C.uniprot_id,
                'EC number': 'ec_number',
                'Sequence': _C.protein_sequence,
                'Binding site': 'binding_site'
            })
            len_b = len(binding_site)
            binding_site.dropna(subset=['binding_site'], inplace=True)
            st['removed_by_reason']['missing_values'] += len_b - len(binding_site)
            len_b = len(binding_site)
            binding_site.drop_duplicates(subset=[_C.protein_sequence, 'binding_site'], inplace=True)
            st['removed_by_reason']['duplicates'] += len_b - len(binding_site)
            binding_site.to_csv(out_dir_interim / 'swissprot_binding_site.csv', index=False)

            parsed = binding_site["binding_site"].apply(parse_binding)
            binding_site["binding_positions"] = parsed.apply(lambda x: x[0])
            binding_site["binding_ligand"] = parsed.apply(lambda x: x[1])
            binding_site["binding_ligand_id"] = parsed.apply(lambda x: x[2])
            binding_site["binding_ignored_count"] = parsed.apply(lambda x: x[3])
            
            binding_site.to_csv(out_dir_interim / 'swissprot_binding_site2.csv', index=False)

            binding_site.drop(columns=["binding_ignored_count", "binding_site", "binding_ligand", "binding_ligand_id", "ec_number"], inplace=True)
            
            out_dir = _P.data_processed_properties / 'binding_site'
            out_dir.mkdir(parents=True, exist_ok=True)
            binding_site.to_csv(out_dir / 'swissprot_binding_site.csv', index=False)
            print("UniProt Binding sites saved.")
            st["datapoints_after_cleaning"] = len(binding_site)
            st["datapoints_removed_total"] = st["datapoints_before_cleaning"] - st["datapoints_after_cleaning"]
            st["removed_percentage"] = (st["datapoints_removed_total"] / st["datapoints_before_cleaning"] * 100) if st["datapoints_before_cleaning"] else 0
            stats["swissprot_binding_site"] = {"amount_of_removed_data": st}

    print("\nDone!")
    
    # Save the stats dict
    results_dir = _P.results_dir / "dataprep_datacards" / "propertyprep_datacards"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "propertyprep_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()