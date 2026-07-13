#!/usr/bin/env python3
import argparse
import urllib.request
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download the raw BioLiP export expected by the cleaning pipeline.")
    parser.add_argument("--out-path", type=Path, default=Path("data/raw/biolip/BioLiP_nr.txt.gz"))
    parser.add_argument("--url", type=str, default="https://aideepmed.com/BioLiP/download/BioLiP_nr.txt.gz")
    args = parser.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading BioLiP from {args.url} to {args.out_path}...")
    
    # Use a user-agent to prevent 403 Forbidden errors
    req = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(args.out_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        
    print(f"Downloaded BioLiP to {args.out_path}")

if __name__ == "__main__":
    main()
