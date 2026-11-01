#!/usr/bin/env python3
"""Generate per-sequence or pooled ESM-2 embeddings as safetensors shards."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import time

from shard_writer import SafetensorsPooledWriter, SafetensorsShardWriter
import torch
from transformers import EsmModel, EsmTokenizer

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device: {}".format(device))


def get_esm2_model(model_name):
    print("Loading ESM2 from: {}".format(model_name))
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()
    return model, tokenizer


def read_fasta(fasta_path, split_char, id_field):
    sequences = dict()
    with Path(fasta_path).open("r", encoding="utf-8") as fasta_f:
        for line in fasta_f:
            if line.startswith(">"):
                uniprot_id = line.replace(">", "").strip().split(split_char)[id_field]
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                sequences[uniprot_id] = ""
            else:
                sequences[uniprot_id] += "".join(line.split()).replace("-", "")
    return sequences


def get_embeddings(
    seq_path,
    emb_path,
    model_name,
    per_protein,
    split_char,
    id_field,
    mean_pooled_path=None,
    max_residues=4000,
    max_seq_len=1000,
    max_batch=100,
):

    seq_dict = read_fasta(seq_path, split_char, id_field)
    model, tokenizer = get_esm2_model(model_name)

    avg_length = sum(len(seq) for seq in seq_dict.values()) / len(seq_dict)
    n_long = sum(1 for seq in seq_dict.values() if len(seq) > max_seq_len)
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)

    print("Total number of sequences: {}".format(len(seq_dict)))
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    start = time.time()
    batch = []

    with ExitStack() as stack:
        hf = stack.enter_context(SafetensorsShardWriter(emb_path))
        hf_pooled = (
            stack.enter_context(SafetensorsPooledWriter(mean_pooled_path))
            if mean_pooled_path is not None
            else None
        )

        n_embeddings = sum(1 for pid, _ in seq_dict if pid in hf)
        seq_dict = [(pid, seq) for pid, seq in seq_dict if pid not in hf]
        if not seq_dict:
            print("All sequences already embedded.")
            return True
        print(f"Resuming: {n_embeddings} already done, {len(seq_dict)} remaining.")
        for seq_idx, (identifier, seq) in enumerate(seq_dict, 1):
            seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
            seq_len = len(seq)
            batch.append((identifier, seq, seq_len))

            n_res_batch = sum(s_len for _, _, s_len in batch) + seq_len
            if (
                len(batch) >= max_batch
                or n_res_batch >= max_residues
                or seq_idx == len(seq_dict)
                or seq_len > max_seq_len
            ):
                ids, seqs, seq_lens = zip(*batch)
                batch = []

                token_encoding = tokenizer(
                    list(seqs),
                    add_special_tokens=True,
                    padding="longest",
                    return_tensors="pt",
                ).to(device)

                try:
                    with torch.no_grad():
                        outputs = model(**token_encoding)
                except RuntimeError:
                    print(
                        "RuntimeError during embedding for {} (L={})".format(
                            identifier, seq_len
                        )
                    )
                    continue

                for batch_idx, bid in enumerate(ids):
                    s_len = seq_lens[batch_idx]
                    # ESM2 adds BOS at pos 0 and EOS at pos s_len+1; slice only residues
                    emb = outputs.last_hidden_state[batch_idx, 1 : s_len + 1]

                    if mean_pooled_path is not None:
                        hf.add(bid, emb)
                        hf_pooled.add(bid, emb.mean(dim=0))
                    else:
                        if per_protein:
                            emb = emb.mean(dim=0)
                        hf.add(bid, emb)
                    n_embeddings += 1

                    if n_embeddings == 1:
                        print(
                            "Example: embedded {} with length {} to shape: {}".format(
                                bid, s_len, emb.shape
                            )
                        )

    end = time.time()

    print("\n############# STATS #############")
    print("Total number of embeddings: {}".format(n_embeddings))
    if n_embeddings == 0:
        print(
            "No embeddings were generated. Check for GPU compatibility or "
            "RuntimeErrors above."
        )
        return False
    print(
        "Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}".format(
            end - start, (end - start) / n_embeddings, avg_length
        )
    )
    return True


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "esm2_embedding_script.py creates ESM2 embeddings for a given FASTA file "
            "and saves them as bf16 safetensors shards. "
            "Example: python esm2_embedding_script.py --input sequences.fasta "
            "--output embeddings_shards --model facebook/esm2_t33_650M_UR50D "
            "--per_protein 1"
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to a FASTA file containing protein sequence(s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Directory for saving the embeddings as numbered bf16 safetensors shards.",
    )
    parser.add_argument(
        "--model",
        required=False,
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="ESM2 HuggingFace repo name or local path. "
        "Default: facebook/esm2_t33_650M_UR50D",
    )
    parser.add_argument(
        "--per_protein",
        type=int,
        default=0,
        help=(
            "0: per-residue embeddings (default). 1: mean-pooled per-protein embedding."
        ),
    )
    parser.add_argument(
        "--mean_pooled_output",
        type=str,
        default=None,
        help="Optional path to also save mean-pooled (D,) BF16 embeddings. "
        "When set, the main --output always stores per-residue (L, D).",
    )
    parser.add_argument(
        "--split_char",
        type=str,
        default="!",
        help=(
            "Character for splitting the FASTA header to get the identifier. Default: "
            "'!'"
        ),
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help=(
            "Index of the identifier field after splitting the FASTA header. Default: 0"
        ),
    )
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    seq_path = Path(args.input)
    emb_path = Path(args.output)
    per_protein = False if int(args.per_protein) == 0 else True
    mean_pooled_path = (
        Path(args.mean_pooled_output) if args.mean_pooled_output else None
    )

    get_embeddings(
        seq_path,
        emb_path,
        args.model,
        per_protein=per_protein,
        split_char=args.split_char,
        id_field=args.id,
        mean_pooled_path=mean_pooled_path,
    )


if __name__ == "__main__":
    main()
