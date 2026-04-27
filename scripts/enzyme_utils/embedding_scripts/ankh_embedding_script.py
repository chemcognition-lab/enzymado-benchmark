#!/usr/bin/env python3
"""Generate per-sequence or pooled Ankh embeddings and store them in H5."""

import argparse
import time
from contextlib import ExitStack
from pathlib import Path

import ankh
import h5py
import torch
from transformers import T5EncoderModel, AutoTokenizer

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))

def _load_hf_model(repo_id):
    model = T5EncoderModel.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer

MODEL_LOADERS = {
    'ankh_base':    ankh.load_base_model,
    'ankh_large':   ankh.load_large_model,
    'ankh3_large':  lambda: _load_hf_model("ElnaggarLab/ankh3-large"),
    'ankh3_xl':     lambda: _load_hf_model("ElnaggarLab/ankh3-xl"),
}


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
                   mean_pooled_path=None, max_residues=4000, max_seq_len=1000, max_batch=100):
    seq_dict = read_fasta(seq_path, split_char, id_field)

    if model_name not in MODEL_LOADERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_LOADERS.keys())}"
        )

    print("Loading Ankh model: {}".format(model_name))
    model, tokenizer = MODEL_LOADERS[model_name]()
    model = model.to(device).eval()

    avg_length = sum(len(seq) for seq in seq_dict.values()) / len(seq_dict)
    n_long = sum(1 for seq in seq_dict.values() if len(seq) > max_seq_len)
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)

    print('Total number of sequences: {}'.format(len(seq_dict)))
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    start = time.time()
    batch = []

    with ExitStack() as stack:
        hf = stack.enter_context(h5py.File(str(emb_path), "w"))
        hf_pooled = stack.enter_context(h5py.File(str(mean_pooled_path), "w")) if mean_pooled_path is not None else None

        n_embeddings = sum(1 for pid, _ in seq_dict if pid in hf)
        seq_dict = [(pid, seq) for pid, seq in seq_dict if pid not in hf]
        if not seq_dict:
            print("All sequences already embedded.")
            return True
        print(f"Resuming: {n_embeddings} already done, {len(seq_dict)} remaining.")
        for seq_idx, (identifier, seq) in enumerate(seq_dict, 1):
            seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            seq_len = len(seq)
            batch.append((identifier, seq, seq_len))

            n_res_batch = sum(s_len for _, _, s_len in batch) + seq_len
            if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
                ids, seqs, seq_lens = zip(*batch)
                batch = []

                token_encoding = tokenizer(
                    [list(s) for s in seqs],
                    add_special_tokens=True,
                    padding=True,
                    is_split_into_words=True,
                    return_tensors="pt",
                )
                input_ids = token_encoding['input_ids'].to(device)
                attention_mask = token_encoding['attention_mask'].to(device)

                try:
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                except RuntimeError:
                    print("RuntimeError during embedding for {} (L={})".format(identifier, seq_len))
                    continue

                for batch_idx, bid in enumerate(ids):
                    s_len = seq_lens[batch_idx]
                    emb = outputs.last_hidden_state[batch_idx, 0:s_len]  # strip EOS token

                    if mean_pooled_path is not None:
                        emb_np = emb.detach().cpu().to(torch.bfloat16).view(torch.int16).numpy()
                        ds = hf.create_dataset(bid, data=emb_np, compression="lzf")
                        ds.attrs['dtype'] = 'bfloat16'
                        pooled_np = emb.mean(dim=0).detach().cpu().to(torch.bfloat16).view(torch.int16).numpy()
                        ds_p = hf_pooled.create_dataset(bid, data=pooled_np, compression="lzf")
                        ds_p.attrs['dtype'] = 'bfloat16'
                    else:
                        if per_protein:
                            emb = emb.mean(dim=0)
                        emb_np = emb.detach().cpu().to(torch.bfloat16).view(torch.int16).numpy()
                        ds = hf.create_dataset(bid, data=emb_np, compression="lzf")
                        ds.attrs['dtype'] = 'bfloat16'
                    n_embeddings += 1

                    if n_embeddings == 1:
                        print("Example: embedded {} with length {} to shape: {}".format(
                            bid, s_len, emb.shape))

    end = time.time()

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(n_embeddings))
    if n_embeddings == 0:
        print('No embeddings were generated. Check for GPU compatibility or RuntimeErrors above.')
        return False
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(
        end - start, (end - start) / n_embeddings, avg_length))
    return True


def create_arg_parser():
    parser = argparse.ArgumentParser(description=(
        'ankh_embedding_script.py creates Ankh embeddings for a given FASTA file '
        'and saves them in HDF5 format. '
        'Example: python ankh_embedding_script.py --input sequences.fasta '
        '--output embeddings.h5 --model ankh_large --per_protein 1'
    ))
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Path to a FASTA file containing protein sequence(s).')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Path for saving the embeddings as an HDF5 (.h5) file.')
    parser.add_argument('--model', required=False, type=str,
                        default='ankh_large',
                        help='Ankh model to use. Choices: ankh_base, ankh_large, ankh3_large, ankh3_xl. Default: ankh_large')
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
