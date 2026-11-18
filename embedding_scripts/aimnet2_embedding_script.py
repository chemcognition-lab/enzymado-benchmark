"""Generate MLIP embeddings with AIMNet2 (wb97m-d3, member 0).

Reads prepared .npz structure files produced by prepare_structures.py and
writes per-residue safetensors shards and an optional mean-pooled file.
Per-atom features (data["aim"], [N x 256]) from the final AIMNet2 MLP pass
are averaged to per-residue features using mlip_common.to_residues.

The default model is aimnet2-wb97m-d3_0 (registry alias "aimnet2"), which is
member 0 of the wb97m-d3 ensemble. Member 0 is pinned: each ensemble member
trains independently and their latent spaces are not comparable.

Coulomb and dispersion are disabled (needs_coulomb=False). The "aim" tensor
is computed entirely from the 5 A short-range AEV+MLP passes; the energy
head (which reads aim) appends Coulomb on top, but aim itself is not affected.
Disabling Coulomb prevents building an O(N^2) all-pairs neighbor list for
large proteins.

Example:
    python embedding_scripts/aimnet2_embedding_script.py \\
        --model_path /scratch/.../aimnet2/aimnet2_wb97m_d3_0.pt \\
        -i /scratch/.../structures/final/prepared \\
        -o /scratch/.../mlip_pilot/aimnet2/out_shards \\
        --mean_pooled_output /scratch/.../mlip_pilot/aimnet2/out_pooled.safetensors
"""

import argparse
from collections.abc import Callable

from aimnet.calculators import AIMNet2Calculator
from aimnet.calculators.neighbors import AdaptiveNeighborList
from jaxtyping import Float
import mlip_common
import numpy as np
import torch
from torch import Tensor

_DEFAULT_MODEL_NAME = "aimnet2-wb97m-d3_0"
_AIM_DIM = 256


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_calc(
    model: str,
    device: str,
) -> AIMNet2Calculator:
    """Load AIMNet2Calculator and expose the aim tensor via keys_out.

    The wb97m-d3 v2 model carries embedded D3TS in its module tree, which
    causes the calculator to set lr=True and build an AdaptiveNeighborList
    with cutoff=1e6 A.  That list starts with max_neighbors ~ 8e17 (from the
    density*volume formula with cutoff=1e6), which immediately OOMs.

    Fix: replace _nblist_lr with one at the SR cutoff (typically 5 A) after
    construction.  aim is set by the SR AEV+MLP pass in forward_prepared,
    before the D3TS energy head reads nbmat_dftd3.  Using the SR neighbor
    list for D3TS gives a wrong energy, but energy is discarded; aim is
    unaffected.
    """
    calc = AIMNet2Calculator(
        model=model,
        needs_coulomb=False,
        needs_dispersion=False,
        device=device,
        train=False,
    )
    if calc.is_nse:
        # NSE models read the multiplicity, which this script does not pass;
        # the closed-shell parity check below only fits non-NSE models.
        raise ValueError(f"{model} is an NSE model; this script expects closed-shell.")
    calc.keys_out = list(calc.keys_out) + ["aim"]
    if calc.lr and calc._nblist_lr is not None:  # embedded D3TS forced lr=True
        calc._nblist_lr = AdaptiveNeighborList(cutoff=calc.cutoff)
        calc.cutoff_lr = calc.cutoff
    return calc


def _check_electron_parity(numbers: np.ndarray, charge: int) -> None:
    """Raise ValueError if the system is open-shell (odd electron count).

    Closed-shell proteins must satisfy: sum(atomic_numbers) - total_charge is even.
    """
    n_electrons = int(numbers.sum()) - charge
    if n_electrons % 2 != 0:
        raise ValueError(
            f"Odd electron count {n_electrons} = sum(Z)={int(numbers.sum())}"
            f" - charge={charge}. Protein must be closed-shell."
            " Check that the charge was computed correctly by amber14."
        )


def _check_element_coverage(
    numbers: np.ndarray,
    implemented_species: list[int],
) -> None:
    """Raise ValueError if the structure contains elements the model cannot handle."""
    impl_set = set(implemented_species)
    unsupported = sorted({int(z) for z in numbers} - impl_set)
    if unsupported:
        raise ValueError(
            f"Atomic numbers {unsupported} are not in the model's"
            f" implemented_species {sorted(impl_set)}."
            " This protein contains unsupported elements."
        )


def make_embed_fn(
    calc: AIMNet2Calculator,
) -> Callable[[mlip_common.Structure], Float[Tensor, "L d"]]:  # noqa: F821
    """Return an embed_fn compatible with mlip_common.embed_structures."""
    impl_species: list[int] = list((calc.metadata or {}).get("implemented_species", []))
    device = calc.device

    def embed_fn(
        structure: mlip_common.Structure,
    ) -> Float[Tensor, "L d"]:  # noqa: F821
        if impl_species:
            _check_element_coverage(structure.numbers, impl_species)
        _check_electron_parity(structure.numbers, structure.charge)

        coord = torch.tensor(
            structure.positions, dtype=torch.float32, device=device
        )  # [N x 3]
        numbers = torch.tensor(
            structure.numbers, dtype=torch.int32, device=device
        )  # [N]
        # charge must be 1D per AIMNet2Base.prepare_input assertion
        charge = torch.tensor(
            [float(structure.charge)], dtype=torch.float32, device=device
        )  # [1]

        with torch.no_grad():
            result = calc.eval(
                {"coord": coord, "numbers": numbers, "charge": charge},
                forces=False,
            )

        n_atoms = len(structure.numbers)
        # process_output calls unpad_output which strips atom_feature_keys but
        # not "aim"; slice off the one padding atom added by pad_input
        aim = result["aim"].float()[:n_atoms]  # [N x 256]
        ri = torch.from_numpy(structure.residue_index).long().to(device)
        n_res = len(structure.sequence)
        return mlip_common.to_residues(aim, ri, n_res)  # [L x 256]

    return embed_fn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed proteins with AIMNet2 (wb97m-d3 member 0).",
        add_help=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            f"Path to a local AIMNet2 model file (.pt) or a registry name."
            f" Default: '{_DEFAULT_MODEL_NAME}' (auto-downloaded on first use)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda' or 'cpu'. Defaults to cuda if available.",
    )
    mlip_common.add_common_args(parser)
    return parser


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    _device = _args.device or _resolve_device()
    print(f"Using device: {_device}")

    _model = _args.model_path or _DEFAULT_MODEL_NAME
    print(f"Loading model: {_model}")
    _calc = _build_calc(model=_model, device=_device)
    print(f"Cutoff: {_calc.cutoff} A | LR enabled: {_calc.lr}")

    _embed_fn = make_embed_fn(_calc)
    mlip_common.embed_structures(
        embed_fn=_embed_fn, **mlip_common.common_kwargs(_args)
    )
