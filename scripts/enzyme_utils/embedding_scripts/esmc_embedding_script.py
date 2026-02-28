#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import h5py
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

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


def get_embeddings(seq_path, emb_path, model_name, per_protein, split_char, id_field):
    seq_dict = read_fasta(seq_path, split_char, id_field)
    print('Total number of sequences: {}'.format(len(seq_dict)))

    print("Loading ESMC from: {}".format(model_name))
    client = ESMC.from_pretrained(model_name).to(device)
    client.eval()

    emb_dict = {}
    start = time.time()

    with torch.no_grad():
        for seq_idx, (identifier, seq) in enumerate(seq_dict.items(), 1):
            print(f"Processing {seq_idx}/{len(seq_dict)}: {identifier} (L={len(seq)})")
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            emb = logits_output.embeddings[0, 1:-1, :]  # strip BOS and EOS tokens

            if per_protein:
                emb = emb.mean(dim=0)
            emb_dict[identifier] = emb.cpu().numpy()

            if seq_idx == 1:
                print("Example: embedded {} with length {} to shape: {}".format(
                    identifier, len(seq), emb.shape))

    end = time.time()

    with h5py.File(str(emb_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(len(emb_dict)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]'.format(
        end - start, (end - start) / len(emb_dict)))
    return True


def create_arg_parser():
    parser = argparse.ArgumentParser(description=(
        'esmc_embedding_script.py creates ESMC embeddings for a given FASTA file '
        'and saves them in HDF5 format. '
        'Example: python esmc_embedding_script.py --input sequences.fasta '
        '--output embeddings.h5 --model esmc_600m --per_protein 1'
    ))
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Path to a FASTA file containing protein sequence(s).')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Path for saving the embeddings as an HDF5 (.h5) file.')
    parser.add_argument('--model', required=False, type=str,
                        default='esmc_600m',
                        help='ESMC model name. Default: esmc_600m')
    parser.add_argument('--per_protein', type=int, default=0,
                        help='0: per-residue embeddings (default). 1: mean-pooled per-protein embedding.')
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

    get_embeddings(seq_path, emb_path, args.model, per_protein, args.split_char, args.id)


if __name__ == '__main__':
    main()
