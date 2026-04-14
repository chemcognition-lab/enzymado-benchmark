#!/usr/bin/env python3
"""Generate per-sequence or pooled ProFluent E1 embeddings and store them in H5."""

import argparse
import time
from pathlib import Path

import h5py
import torch
from E1.batch_preparer import E1BatchPreparer
from E1.modeling import E1ForMaskedLM

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

    print("Loading Profluent E1 from: {}".format(model_name))
    model = E1ForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    batch_preparer = E1BatchPreparer()

    n_embeddings = 0
    n_skipped = 0
    start = time.time()

    with h5py.File(str(emb_path), "a") as hf:
        with torch.no_grad():
            for seq_idx, (identifier, seq) in enumerate(seq_dict.items(), 1):
                if identifier in hf:
                    n_embeddings += 1
                    continue
                print(f"Processing {seq_idx}/{len(seq_dict)}: {identifier} (L={len(seq)})")
                try:
                    batch = batch_preparer.get_batch_kwargs([seq], device=str(device))

                    use_autocast = device.type == "cuda"
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_autocast):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            within_seq_position_ids=batch["within_seq_position_ids"],
                            global_position_ids=batch["global_position_ids"],
                            sequence_ids=batch["sequence_ids"],
                            past_key_values=None,
                            use_cache=False,
                            output_attentions=False,
                            output_hidden_states=False,
                        )

                    embeddings = outputs.embeddings  # (1, L, E)
                    residue_selector = ~(batch_preparer.get_boundary_token_mask(batch["input_ids"]))
                    emb = embeddings[0, residue_selector[0]]  # (seq_len, E)

                    if per_protein:
                        emb = emb.mean(dim=0)
                    emb_np = emb.cpu().float().numpy()
                    hf.create_dataset(identifier, data=emb_np, compression="lzf")
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
        'profluent_embedding_script.py creates Profluent E1 embeddings for a given FASTA file '
        'and saves them in HDF5 format. '
        'Example: python profluent_embedding_script.py --input sequences.fasta '
        '--output embeddings.h5 --model Profluent-Bio/E1-300m --per_protein 1'
    ))
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Path to a FASTA file containing protein sequence(s).')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Path for saving the embeddings as an HDF5 (.h5) file.')
    parser.add_argument('--model', required=False, type=str,
                        default='Profluent-Bio/E1-300m',
                        help='Profluent E1 model name or local path. Default: Profluent-Bio/E1-300m')
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
