"""Generate MLIP embeddings with MACE-OFF or MACE-MP-0.

Reads prepared .npz structure files produced by prepare_structures.py and
writes per-residue safetensors shards and an optional mean-pooled file.
Atom features from MACE descriptors are averaged to residue features using
mlip_common.to_residues.

Example:
    python embedding_scripts/mace_embedding_script.py \
        --family off --model medium \
        -i /scratch/.../structures/prepared \
        -o /scratch/.../embeddings/mace_off_medium_residue \
        --mean_pooled_output /scratch/.../embeddings/mace_off_medium_pooled.safetensors
"""

import argparse
from collections.abc import Callable

from ase import Atoms
from jaxtyping import Float
from mace.calculators.foundations_models import mace_mp, mace_off
import mace.modules.utils as mace_utils
import mlip_common
import numpy as np
import torch
from torch import Tensor

_DEFAULT_OFF_MODEL = "medium"
_DEFAULT_MP_MODEL = "medium-0b3"


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_calc(
    family: str,
    model: str | None,
    model_path: str | None,
    device: str,
):
    """Instantiate a MACECalculator for the requested family and model."""
    model_arg = model_path or model
    match family:
        case "off":
            model_arg = model_arg or _DEFAULT_OFF_MODEL
            return mace_off(
                model=model_arg,
                device=device,
                default_dtype="float32",
            )
        case "mp":
            model_arg = model_arg or _DEFAULT_MP_MODEL
            return mace_mp(
                model=model_arg,
                device=device,
                default_dtype="float32",
            )
        case _:
            raise ValueError(f"Unknown MACE family: {family!r}. Choose 'off' or 'mp'.")


def _descriptors_no_force(
    calc,
    atoms: Atoms,
) -> np.ndarray:
    """Extract invariant MACE descriptors without computing forces.

    Passes compute_force=False to the MACE forward to skip the force branch,
    which requires positions.requires_grad. Without this flag the TorchScript
    model raises 'element 0 of tensors does not require grad and does not have
    a grad_fn' under torch.no_grad(). Pilot measurement: max abs diff vs
    calc.get_descriptors 2.4e-6; GPU memory 40% lower.

    Uses model.products[0].linear.irreps_out directly (already an e3nn Irreps
    object) to avoid importing e3nn, which causes a weights-only load failure
    under torch 2.7 when imported after mace.
    """
    batch = calc._atoms_to_batch(atoms)
    model = calc.models[0]

    with torch.no_grad():
        node_feats = model(batch.to_dict(), compute_force=False)["node_feats"]

    irreps_out = model.products[0].linear.irreps_out  # Irreps object; no e3nn import
    l_max = irreps_out.lmax
    num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
    num_interactions = int(model.num_interactions)

    desc = mace_utils.extract_invariant(
        node_feats,
        num_layers=num_interactions,
        num_features=num_invariant_features,
        l_max=l_max,
    )
    to_keep = num_invariant_features * num_interactions
    return desc[:, :to_keep].detach().cpu().float().numpy()


def receptive_field(calc) -> float:
    """Distance in Angstrom that one atom's features can depend on.

    MACE passes messages num_interactions times over a radius-r_max graph,
    so the features of an atom are a function of the atoms within
    num_interactions * r_max. No global normalization, and MACE-OFF and
    MACE-MP take no total charge, so nothing else couples distant atoms.
    """
    model = calc.models[0]
    return float(model.num_interactions) * float(model.r_max)


def _is_oom(err: RuntimeError) -> bool:
    return "out of memory" in str(err).lower()


def _atom_features(calc, numbers: np.ndarray, positions: np.ndarray) -> np.ndarray:
    atoms = Atoms(numbers=numbers, positions=positions, pbc=False)
    return _descriptors_no_force(calc, atoms)  # [n x d]


def chunked_atom_features(
    calc, numbers: np.ndarray, positions: np.ndarray, max_atoms: int, halo: float
) -> np.ndarray:
    """Atom features computed on spatial chunks of at most max_atoms atoms.

    Each chunk runs on its core plus all atoms within halo, and only the core
    rows are kept, so the result equals a single pass when halo is at least
    receptive_field(calc). See mlip_common.spatial_chunks.
    """
    out: np.ndarray | None = None
    for core, context in mlip_common.spatial_chunks(positions, max_atoms, halo):
        feats = _atom_features(calc, numbers[context], positions[context])
        if out is None:
            out = np.zeros((len(numbers), feats.shape[1]), dtype=np.float32)
        out[core] = feats[np.searchsorted(context, core)]
    return out


def make_embed_fn(
    calc,
    chunk_atoms: int = 0,
    min_chunk_atoms: int = 1000,
) -> Callable[[mlip_common.Structure], Float[Tensor, "L d"]]:  # noqa: F821
    """Return an embed_fn compatible with mlip_common.embed_structures.

    Proteins with more than chunk_atoms atoms (0: never) are computed in
    spatial chunks. On a GPU out-of-memory error the chunk size is halved
    and the protein retried, down to min_chunk_atoms; below that the error
    is re-raised and mlip_common counts the protein as an OOM skip.
    """
    halo = receptive_field(calc)

    def embed_fn(
        structure: mlip_common.Structure,
    ) -> Float[Tensor, "L d"]:  # noqa: F821
        n_atoms = len(structure.numbers)
        limit = chunk_atoms if 0 < chunk_atoms < n_atoms else n_atoms
        while True:
            try:
                if limit >= n_atoms:
                    x_np = _atom_features(calc, structure.numbers, structure.positions)
                else:
                    x_np = chunked_atom_features(
                        calc, structure.numbers, structure.positions, limit, halo
                    )
                break
            except RuntimeError as err:
                if not _is_oom(err) or limit // 2 < min_chunk_atoms:
                    raise
                torch.cuda.empty_cache()
                limit = min(limit, n_atoms) // 2
                print(f"{structure.id}: OOM, retrying in chunks of {limit} atoms")
        x = torch.from_numpy(x_np)  # [N x d], float32

        ri = torch.from_numpy(structure.residue_index).long()
        n_res = len(structure.sequence)  # authoritative residue count from sequence
        return mlip_common.to_residues(x, ri, n_res)  # [L x d]

    return embed_fn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed proteins with MACE-OFF or MACE-MP-0."
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=["off", "mp"],
        help="MACE model family: 'off' for MACE-OFF, 'mp' for MACE-MP-0.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model size or name. "
            f"off default: {_DEFAULT_OFF_MODEL}. "
            f"mp default: {_DEFAULT_MP_MODEL}."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a local model checkpoint (overrides --model for downloading).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda' or 'cpu'. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--chunk_atoms",
        type=int,
        default=0,
        help=(
            "Compute proteins above this many atoms in exact spatial chunks "
            "(0: only after an out-of-memory error)."
        ),
    )
    mlip_common.add_common_args(parser)
    return parser


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    _device = _args.device or _resolve_device()
    print(f"Using device: {_device}")

    _calc = _build_calc(
        family=_args.family,
        model=_args.model,
        model_path=_args.model_path,
        device=_device,
    )
    print(f"Receptive field: {receptive_field(_calc):.2f} A")
    _embed_fn = make_embed_fn(_calc, chunk_atoms=_args.chunk_atoms)
    mlip_common.embed_structures(
        embed_fn=_embed_fn, **mlip_common.common_kwargs(_args)
    )
