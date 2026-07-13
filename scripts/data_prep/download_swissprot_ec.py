#!/usr/bin/env python3
import argparse
import urllib.request
import urllib.parse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download reviewed Swiss-Prot proteins with EC annotations from UniProt.")
    parser.add_argument("--out-path", type=Path, default=Path("data/raw/swissprot/swissprot_ec.tsv"))
    parser.add_argument("--format", type=str, default="tsv")
    parser.add_argument(
        "--fields", 
        type=str, 
        default="accession,id,protein_name,ec,go_id,go,organism_id,organism_name,protein_existence,gene_names,sequence,length,cc_function,cc_catalytic_activity,cc_pathway,cc_cofactor,cc_activity_regulation,ft_act_site,ft_binding,xref_pdb,xref_kegg"
    )
    parser.add_argument("--compress", action="store_true")
    args = parser.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    query = "reviewed:true AND ec:*"
    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    
    params = {
        "format": args.format,
        "fields": args.fields,
        "query": query,
    }
    if args.compress:
        params["compressed"] = "true"

    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"

    print(f"Downloading Swiss-Prot entries to {args.out_path}...")
    
    # Use a user-agent to prevent 403 Forbidden errors
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as response, open(args.out_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        
    print(f"Downloaded Swiss-Prot entries with EC numbers to {args.out_path}")

if __name__ == "__main__":
    main()
