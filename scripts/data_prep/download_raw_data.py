#!/usr/bin/env python3
"""
Unified script to download raw dataset files.
"""

import argparse
from pathlib import Path
import shutil
import sys
import urllib.parse
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths_and_constants import Paths  # noqa: E402


def download_biolip(p: Paths):
    out_path = p.data_raw_biolip_dir / "BioLiP_nr.txt.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://aideepmed.com/BioLiP/download/BioLiP_nr.txt.gz"
    
    print(f"Downloading BioLiP from {url} to {out_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(out_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"Downloaded BioLiP to {out_path}")


def download_swissprot(p: Paths):
    out_path = p.data_raw_swissprot_dir / "swissprot_ec.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    query = "reviewed:true AND ec:*"
    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "format": "tsv",
        "fields": "accession,id,protein_name,ec,go_id,go,organism_id,organism_name,protein_existence,gene_names,sequence,length,cc_function,cc_catalytic_activity,cc_pathway,cc_cofactor,cc_activity_regulation,ft_act_site,ft_binding,xref_pdb,xref_kegg",
        "query": query,
    }
    
    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"
    print(f"Downloading Swiss-Prot to {out_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as response, open(out_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"Downloaded Swiss-Prot to {out_path}")


def download_fireprotdb(p: Paths):
    out_path = p.data_raw_fireprotdb_dir / "FireProtDB.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # According to docs, 'query' is usually expected. 
    # An empty query string or '{}' often triggers a 'match all'.
    params = {
        "format": "tsv",
        "query": "{}" # Request all records
    }
    query_string = urllib.parse.urlencode(params)
    url = f"https://loschmidt.chemi.muni.cz/fireprotdb/api/search?{query_string}"
    
    print(f"Downloading FireProtDB from {url} to {out_path}...")
    
    # Using a browser-like User-Agent to avoid being blocked
    req = urllib.request.Request(
        url, 
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Enzymado/1.0"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            with open(out_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        
        file_size = out_path.stat().st_size
        
        # If the file is smaller than 5KB, it's almost certainly an error message, 
        # not the database (which is >10MB).
        if file_size < 5000:
            print(f"Warning: Downloaded file is very small ({file_size} bytes).")
            with open(out_path, 'r') as f:
                peek = f.read(500)
            print(f"Server Response Peek: {peek}")
            print("Check if the 'query' parameter needs to be adjusted.")
        else:
            print(f"Successfully downloaded FireProtDB TSV: {file_size // 1024} KB")
            
    except Exception as e:
        print(f"Failed to download FireProtDB: {e}")


def download_proteinea(p: Paths):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency 'datasets'. Skipping ProteinEA download.")
        return

    for ds_name in ["fluorescence", "solubility"]:
        out_path = str(p.data_raw_dir / "proteinea" / ds_name)
        print(f"Downloading proteinea/{ds_name} via HuggingFace datasets...")
        dataset = load_dataset(f"proteinea/{ds_name}")
        dataset.save_to_disk(out_path)
        print(f"Saved to {out_path}")


def main():
    p = Paths()
    parser = argparse.ArgumentParser(description="Download all raw datasets for Enzymado.")
    parser.add_argument("--biolip", action="store_true", help="Download BioLiP")
    parser.add_argument("--swissprot", action="store_true", help="Download SwissProt")
    parser.add_argument("--fireprotdb", action="store_true", help="Download FireProtDB")
    parser.add_argument("--proteinea", action="store_true", help="Download ProteinEA")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    args = parser.parse_args()

    if args.all or args.biolip:
        download_biolip(p)
    if args.all or args.swissprot:
        download_swissprot(p)
    if args.all or args.fireprotdb:
        download_fireprotdb(p)
    if args.all or args.proteinea:
        download_proteinea(p)

if __name__ == "__main__":
    main()
