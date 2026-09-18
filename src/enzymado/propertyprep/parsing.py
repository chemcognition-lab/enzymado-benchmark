"""Mutation and Swiss-Prot feature parsers."""

import re

import pandas as pd

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
            raise IndexError(
                f"Position out of range: {pos} for sequence of length {len(seq_list)}"
            )
        if seq_list[idx] != wt:
            raise ValueError(
                f"Mismatch in {sub}: expected '{wt}' at position {pos}, "
                f"but found '{seq_list[idx]}'"
            )

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


seg_re = re.compile(r"BINDING\s+([^;]+);\s*(.*?)(?=BINDING\s+|$)", re.DOTALL)
lig_re = re.compile(r'/ligand="([^"]*)"')
chebi_re = re.compile(r'/ligand_id="([^"]*)"')
int_re = re.compile(r"^\d+$")
range_re = re.compile(r"^(\d+)\.\.(\d+)$")


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
