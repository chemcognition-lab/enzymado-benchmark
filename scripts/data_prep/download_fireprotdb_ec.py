#!/usr/bin/env python3
"""Download FireProtDB exports and keep only rows with usable EC annotations."""

import argparse
import csv
import gzip
import io
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile

DEFAULT_URLS = [
    "https://loschmidt.chemi.muni.cz/fireprotdb/api/search?format=csv&sort=",
    "https://loschmidt.chemi.muni.cz/fireprotdb/api/search?format=tsv&sort=",
    "https://loschmidt.chemi.muni.cz/fireprotdb/download/FireProtDB.tsv",
    "https://loschmidt.chemi.muni.cz/fireprotdb/download/FireProtDB.csv",
    "https://loschmidt.chemi.muni.cz/fireprotdb/download/FireProtDB.tsv.gz",
    "https://loschmidt.chemi.muni.cz/fireprotdb/download/FireProtDB.csv.gz",
]


def download_to_temp(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "codex-fireprotdb-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if getattr(resp, "status", 200) != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(resp, tmp)
            return tmp.name


def try_download(urls, timeout):
    last_err = None
    for url in urls:
        try:
            path = download_to_temp(url, timeout)
            return url, path
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to download from any URL: {last_err}")


def open_text_stream(path):
    with open(path, "rb") as fh:
        sig = fh.read(4)
    if sig.startswith(b"\x1f\x8b"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    if sig.startswith(b"PK\x03\x04"):
        zf = zipfile.ZipFile(path)
        members = zf.namelist()
        if not members:
            raise RuntimeError("Zip archive is empty")
        member = members[0]
        return io.TextIOWrapper(zf.open(member, "r"), encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def detect_delimiter(header_line):
    if header_line.count("\t") >= header_line.count(","):
        return "\t"
    return ","


def find_ec_index(header):
    normalized = [h.strip().lower() for h in header]
    for idx, name in enumerate(normalized):
        if name in {"ec", "ec number", "ec_number", "ec-number"}:
            return idx
    for idx, name in enumerate(normalized):
        if "ec" in name and "number" in name:
            return idx
    for idx, name in enumerate(normalized):
        if name.startswith("ec"):
            return idx
    return None


def split_ec_values(value):
    parts = re.split(r"[;,]\s*", value.strip())
    return [p for p in parts if p]


def ec_matches(value, ec_filter):
    if not value or value.strip().lower() in {"na", "n/a", "none"}:
        return False
    if ec_filter is None:
        return True
    if ec_filter.endswith("*"):
        prefix = ec_filter[:-1]
        return any(v.startswith(prefix) for v in split_ec_values(value))
    return any(v == ec_filter for v in split_ec_values(value))


def main():
    parser = argparse.ArgumentParser(
        description="Download FireProtDB and filter rows by EC number."
    )
    parser.add_argument(
        "--url",
        action="append",
        help="Download URL (can be specified multiple times).",
    )
    parser.add_argument(
        "--out",
        default="data/raw/fireprotdb/fireprotdb_ec.tsv",
        help="Output TSV path (default: data/raw/fireprotdb/fireprotdb_ec.tsv).",
    )
    parser.add_argument(
        "--ec",
        default=None,
        help="EC number filter (exact match or prefix with *). If omitted, keeps rows with any EC.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout seconds (default: 60).",
    )
    args = parser.parse_args()

    urls = args.url if args.url else DEFAULT_URLS
    url_used, tmp_path = try_download(urls, args.timeout)

    try:
        with open_text_stream(tmp_path) as stream:
            header_line = stream.readline()
            if not header_line:
                raise RuntimeError("Downloaded file is empty.")
            delimiter = detect_delimiter(header_line)
            header = next(csv.reader([header_line], delimiter=delimiter))
            ec_idx = find_ec_index(header)
            if ec_idx is None:
                raise RuntimeError("Could not find an EC column in the header.")

            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8", newline="") as out_fh:
                writer = csv.writer(out_fh, delimiter="\t")
                writer.writerow(header)

                reader = csv.reader(stream, delimiter=delimiter)
                total = 0
                kept = 0
                for row in reader:
                    if not row:
                        continue
                    total += 1
                    ec_value = row[ec_idx] if ec_idx < len(row) else ""
                    if ec_matches(ec_value, args.ec):
                        writer.writerow(row)
                        kept += 1

        print(f"Downloaded from: {url_used}")
        print(f"Rows kept: {kept} of {total}")
        print(f"Output: {args.out}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
