#!/usr/bin/env python3
"""Generate per-sequence or pooled ESM-3 embeddings and store them in H5."""

import argparse
import time
from contextlib import ExitStack
from pathlib import Path

import h5py
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))


def read_fasta(fasta_path, split_char, id_field):
    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                sequences[uniprot_id] = ''
            else:
                sequences[uniprot_id] += ''.join(line.split()).replace("-", "")
    return sequences


def get_embeddings(seq_path, emb_path, model_name, per_protein, split_char, id_field,
                   mean_pooled_path=None):
    seq_dict = read_fasta(seq_path, split_char, id_field)
    print('Total number of sequences: {}'.format(len(seq_dict)))

    print("Loading ESM3 from: {}".format(model_name))
    client = ESM3.from_pretrained(model_name, device=device)
    client.eval()

    n_embeddings = 0
    n_skipped = 0
    start = time.time()

    with ExitStack() as stack:
        hf = stack.enter_context(h5py.File(str(emb_path), "w"))
        hf_pooled = stack.enter_context(h5py.File(str(mean_pooled_path), "w")) if mean_pooled_path is not None else None

        with torch.no_grad():
            for seq_idx, (identifier, seq) in enumerate(seq_dict.items(), 1):
                if identifier in hf:
                    n_embeddings += 1
                    continue
                print(f"Processing {seq_idx}/{len(seq_dict)}: {identifier} (L={len(seq)})")
                try:
                    protein = ESMProtein(sequence=seq)
                    protein_tensor = client.encode(protein)
                    output = client.forward_and_sample(
                        protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
                    )
                    emb = output.per_residue_embedding[1:-1]  # strip BOS and EOS tokens

                    if mean_pooled_path is not None:
                        emb_np = emb.cpu().to(torch.bfloat16).view(torch.int16).numpy()
                        ds = hf.create_dataset(identifier, data=emb_np, compression="lzf")
                        ds.attrs['dtype'] = 'bfloat16'
                        pooled_np = emb.mean(dim=0).cpu().to(torch.bfloat16).view(torch.int16).numpy()
                        ds_p = hf_pooled.create_dataset(identifier, data=pooled_np, compression="lzf")
                        ds_p.attrs['dtype'] = 'bfloat16'
                    else:
                        if per_protein:
                            emb = emb.mean(dim=0)
                        emb_np = emb.cpu().to(torch.bfloat16).view(torch.int16).numpy()
                        ds = hf.create_dataset(identifier, data=emb_np, compression="lzf")
                        ds.attrs['dtype'] = 'bfloat16'
                    n_embeddings += 1

                    if n_embeddings == 1:
                        print("Example: embedded {} with length {} to shape: {}".format(
                            identifier, len(seq), emb_np.shape))
                except Exception as e:
                    print(f"WARNING: skipping {identifier} (L={len(seq)}) — {type(e).__name__}: {e}")
                    n_skipped += 1
                    torch.cuda.empty_cache()

    end = time.time()

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(n_embeddings))
    print('Skipped sequences: {}'.format(n_skipped))
    if n_embeddings > 0:
        print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]'.format(
            end - start, (end - start) / n_embeddings))
    return True


def create_arg_parser():
    parser = argparse.ArgumentParser(description=(
        'esm3_embedding_script.py creates ESM3 embeddings for a given FASTA file '
        'and saves them in HDF5 format. '
        'Example: python esm3_embedding_script.py --input sequences.fasta '
        '--output embeddings.h5 --model esm3-open --per_protein 1'
    ))
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Path to a FASTA file containing protein sequence(s).')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Path for saving the embeddings as an HDF5 (.h5) file.')
    parser.add_argument('--model', required=False, type=str,
                        default='esm3-open',
                        help='ESM3 model name. Default: esm3-open')
    parser.add_argument('--per_protein', type=int, default=0,
                        help='0: per-residue embeddings (default). 1: mean-pooled per-protein embedding.')
    parser.add_argument('--mean_pooled_output', type=str, default=None,
                        help='Optional path to also save mean-pooled (D,) BF16 embeddings. '
                             'When set, the main --output always stores per-residue (L, D).')
    parser.add_argument('--split_char', type=str, default='!',
                        help="Character for splitting the FASTA header to get the identifier. Default: '!'")
    parser.add_argument('--id', type=int, default=0,
                        help='Index of the identifier field after splitting the FASTA header. Default: 0')
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    seq_path = Path(args.input)
    emb_path = Path(args.output)
    per_protein = False if int(args.per_protein) == 0 else True
    mean_pooled_path = Path(args.mean_pooled_output) if args.mean_pooled_output else None

    get_embeddings(seq_path, emb_path, args.model, per_protein, args.split_char, args.id,
                   mean_pooled_path=mean_pooled_path)


if __name__ == '__main__':
    main()
