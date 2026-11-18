"""Generate MLIP embeddings with SO3LR Tauri (S, M, or L).

Reads prepared .npz structure files produced by prepare_structures.py and
writes per-residue safetensors shards and an optional mean-pooled file.
Atom features (atomic_representations, the grade-0 invariant x vector) are
averaged to residue features using mlip_common.to_residues.

Charge (from AMBER14 at pH 7) and spin multiplicity 1 (num_unpaired_electrons=0)
are passed to the model. Structures must be closed-shell: sum(Z) - charge must
be even. The model covers Z = 1..85 (H to At).

The model is built with return_representations_bool=True so only the backbone
runs and no long-range neighbor list is needed. Output x is what the energy
head reads: running the official ASE calculator (head 5, with electrostatics
and dispersion) and capturing the x passed to its EnergySparse module with
flax.linen.intercept_methods gave at most 4.6e-6 difference from this
script's x on CPU (Q96YL0, 1,526 atoms, 2026-11-11).

Shape bucketing: proteins are padded to geometric bucket sizes so JAX compiles
once per (node bucket, edge bucket) pair rather than once per protein. Node
buckets use ~1.25x steps covering 512 to 40,960 atoms; edge buckets are the
node buckets times 40 (S, M) or 50 (L). The jraph padding convention is used:
N_node_padded = N_bucket + 1 (1 extra node for the padding graph), padded
atoms get node_mask=False and batch_segments=1, padding edges are self-loops
on the padding node, whose cutoff function is 0 at d = 0. Slicing x[:N_real]
before to_residues removes all padding atoms. On CPU, padded and unpadded x
differ by at most 4.7e-6 on three pilot proteins (1,526 to 11,099 atoms).
Compiling one shape takes ~26 s on an L40S; a compiled call takes 0.02 s
(1,526 atoms) to 0.19 s (12,486 atoms).

Numerics: JAX runs float32 matrix products on the GPU at reduced (TF32)
precision by default, which moved x by up to 9e-3 from the CPU result, and
its scatter-adds are not deterministic (4e-3 between two identical calls).
This script therefore sets jax_default_matmul_precision='highest' and
--xla_gpu_deterministic_ops=true before JAX starts. With both, two identical
calls agree exactly, GPU and CPU agree to 1.1e-4, and two bucket sizes agree
to 6e-6 (2026-11-11). PyTorch models (MACE, AIMNet2) already compute matmuls
in full float32 by default.

JAX memory: XLA_PYTHON_CLIENT_PREALLOCATE is set to false before JAX loads so
the GPU is not pre-allocated to 75%. XLA OOM (RESOURCE_EXHAUSTED) is caught
and re-raised as RuntimeError containing "out of memory" so mlip_common counts
and skips the protein rather than aborting the run. After the run, peak JAX
GPU memory is printed via jax.devices()[0].memory_stats().

head_index / theory_mask: theory_mask is used only by the energy observable
(EnergySparse), which is never called with return_representations_bool=True.
Atomic representations x are therefore independent of theory_mask and the
--head_index argument has been removed.

Unused checkpoint parameters: the checkpoint contains params for observables
(energy_dense_final, partial_charges, Hirshfeld, C6 heads). These are present
in the params dict passed to model.apply but are never accessed during the
backbone-only forward pass. Flax silently ignores extra params; it only raises
on missing ones, which does not happen here.

Example:
    python embedding_scripts/so3lr_embedding_script.py \\
        --model M \\
        -i /scratch/.../structures/prepared \\
        -o /scratch/.../embeddings/so3lr_m_residue \\
        --mean_pooled_output /scratch/.../embeddings/so3lr_m_pooled.safetensors
"""

import os

# Must be set before JAX imports to prevent 75 % GPU preallocation.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Deterministic scatter-adds; read when the GPU backend starts (see docstring).
if "--xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
    ).strip()
# Full float32 matmuls instead of TF32 (see docstring).
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import argparse
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float
import mlip_common
import numpy as np
import torch
from torch import Tensor

# The env var only takes effect if JAX was not imported earlier; set it again.
jax.config.update(
    "jax_default_matmul_precision", os.environ["JAX_DEFAULT_MATMUL_PRECISION"]
)

# Exact versioned model names; aliases (so3lr-tauri-m) change with releases.
_VERSIONED_NAMES: dict[str, str] = {
    "S": "so3lr-tauri-s-2606",
    "M": "so3lr-tauri-m-2606",
    "L": "so3lr-tauri-l-2606",
}
_DEFAULT_SIZE = "M"

# Output feature dimension by size (num_features from the model config).
_EMBED_DIM: dict[str, int] = {"S": 128, "M": 128, "L": 256}

# Local cutoff in Angstrom by size (confirmed from hyperparameters.json).
_CUTOFF: dict[str, float] = {"S": 4.5, "M": 4.5, "L": 5.0}

# Geometric node buckets, ~1.25x steps, covering 512 to 40,960 atoms.
# jraph convention adds 1 extra node for the padding graph, so compiled shapes
# are (bucket + 1, edge_bucket).
_NODE_BUCKETS: list[int] = [
    512, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 4000,
    5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600, 32000, 40960,
]

# Edge buckets: 40x the corresponding node bucket. At 4.5 A cutoff the pilot
# measured a max of 33.2 edges/atom; factor 40 gives ~20 % safety margin.
# For L (5.0 A cutoff) the ratio is higher, so use 50x for L.
_EDGE_BUCKETS_DEFAULT: list[int] = [b * 40 for b in _NODE_BUCKETS]
_EDGE_BUCKETS_L: list[int] = [b * 50 for b in _NODE_BUCKETS]  # 5.0 A cutoff


def _next_bucket(n: int, buckets: list[int]) -> int:
    """Return the smallest bucket size >= n, or extend geometrically if too large."""
    for b in buckets:
        if b >= n:
            return b
    return buckets[-1] * 2  # extend if the protein exceeds the largest bucket


def _pad_to_buckets(
    numbers: np.ndarray,
    positions: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    charge: int | float,
    n_bucket: int,
    p_bucket: int,
) -> tuple:
    """Pad a single protein graph to fixed (n_bucket+1, p_bucket) shapes.

    Follows the jraph padding convention:
      - n_bucket+1 total nodes: N_real real atoms (batch_segments=0) plus
        1 padding-graph node (batch_segments=1, node_mask=False).
      - p_bucket total directed edges: real edges plus self-loops on the
        padding node to fill the fixed buffer.
      - 2 graphs: graph 0 is the real protein, graph 1 is the padding graph.
        graph_mask = [True, False].

    Returns (positions_p, numbers_p, idx_i_p, idx_j_p,
             node_mask, batch_segments, graph_mask,
             total_charge_2, num_unpaired_electrons_2).
    """
    n_real = len(numbers)
    p_real = len(idx_i)
    n_padded = n_bucket + 1  # extra node belongs to the padding graph
    pad_node = n_padded - 1  # index of the padding node

    numbers_p = np.ones(n_padded, dtype=np.int32)  # pad with H (Z=1)
    numbers_p[:n_real] = numbers

    positions_p = np.zeros((n_padded, 3), dtype=np.float32)
    positions_p[:n_real] = positions

    # Extra edges are self-loops on the padding node; they are masked and
    # contribute zero to any message passing aggregate.
    idx_i_p = np.full(p_bucket, pad_node, dtype=np.int32)
    idx_j_p = np.full(p_bucket, pad_node, dtype=np.int32)
    idx_i_p[:p_real] = idx_i
    idx_j_p[:p_real] = idx_j

    node_mask = np.zeros(n_padded, dtype=bool)
    node_mask[:n_real] = True

    batch_segments = np.ones(n_padded, dtype=np.int32)  # default = padding graph 1
    batch_segments[:n_real] = 0  # real atoms belong to graph 0

    graph_mask = np.array([True, False], dtype=bool)

    total_charge_2 = np.array([float(charge), 0.0], dtype=np.float32)
    num_unpaired_electrons_2 = np.zeros(2, dtype=np.float32)

    return (
        positions_p, numbers_p, idx_i_p, idx_j_p,
        node_mask, batch_segments, graph_mask,
        total_charge_2, num_unpaired_electrons_2,
    )


def _load_repr_model(versioned_name: str):
    """Load params and build a representation-only SO3krates model.

    Returns (net, params, cutoff_angstrom). The model is built with
    return_representations_bool=True so its forward pass returns only
    {'atomic_representations': x} without running any energy or property
    heads. The param tree is identical to the full model; unused observable
    params are silently present but never accessed during the forward pass.

    Proof that published weights load: the StackNetSparse architecture is
    identical for both the full model and the representation model.
    return_representations_bool is a forward-time flag that changes the
    control flow in __call__ only. Flax applies params to the declared module
    tree; since the same modules are declared, all parameter keys match.
    Observable params (observables_0 for energy, observables_2 for Hirshfeld,
    observables_3 for C6) are present in params but never accessed with
    return_representations_bool=True. Flax raises only on missing params, not
    extra ones, so apply succeeds.

    Proof that x is the energy-head input (stacknet_sparse.py lines):
        if self.return_representations_bool:
            return {'atomic_representations': quantities['x']}
        for o_fn in self.observables:
            o_dict = o_fn(quantities)   # EnergySparse reads quantities['x']
    The same quantities['x'] is returned by the repr model and consumed by
    the energy head. Numerical confirmation against the official calculator
    is in the module docstring.
    """
    from so3lr.mlff.calculators.potential import (
        cast_float_params,
        load_hyperparameters,
    )
    from so3lr.mlff.config.from_config import make_so3krates_sparse_from_config
    from so3lr.mlff.utils.checkpoint_utils import load_params_pkl
    from so3lr.mlff.utils.training_utils import expand_observables_to_theory_levels
    from so3lr.model_registry import resolve_model

    spec = resolve_model(versioned_name)
    if not spec.dir.exists():
        raise FileNotFoundError(f"Model directory not found: {spec.dir}")

    params = load_params_pkl(spec.dir / "params.pkl")
    params = expand_observables_to_theory_levels(params)
    params = cast_float_params(params, jnp.float32)

    cfg = load_hyperparameters(str(spec.dir))
    # Required by make_so3krates_sparse_from_config even though lr fields are
    # never used in the representation-only path.
    cfg.charge_embed_total_charge = None
    cfg.charge_equilibration_total_charge = None
    cfg.neighborlist_format_lr = "sparse"

    net = make_so3krates_sparse_from_config(cfg, return_representations_bool=True)
    cutoff = float(cfg.model.cutoff)
    return net, params, cutoff


def _local_pairs(
    positions: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute directed short-range pairs within cutoff using scipy KDTree.

    Returns (idx_i, idx_j) where messages flow from j to i. Both directions
    of each pair are included: for every (a, b) with |r_ab| < cutoff, both
    (a->b) and (b->a) appear. Only atom indices, no periodic images.
    """
    from scipy.spatial import KDTree

    tree = KDTree(positions)
    undirected = tree.query_pairs(r=cutoff, output_type="ndarray")  # [P/2 x 2]
    if len(undirected) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    # Add both directions.
    idx_i = np.concatenate([undirected[:, 0], undirected[:, 1]]).astype(np.int32)
    idx_j = np.concatenate([undirected[:, 1], undirected[:, 0]]).astype(np.int32)
    return idx_i, idx_j


def _jit_obs_fn(net) -> Callable:
    """Return a jit-compiled observable function for the given model.

    With bucketed padding, JAX sees a fixed (N_padded, P_padded) shape per
    bucket pair and compiles once per pair. Edges per atom stay within a
    narrow range for proteins (28 to 33 at 4.5 A in the pilot), so each node
    bucket meets only one or two edge buckets.
    """
    from so3lr.mlff.nn.stacknet.observable_function_sparse import (
        get_observable_fn_sparse,
    )

    return jax.jit(get_observable_fn_sparse(net))


def _check_elements(numbers: np.ndarray, context: str) -> None:
    """Raise ValueError if any element is outside SO3LR Tauri's coverage."""
    from so3lr.mlff.nn.observable.dispersion_ref_data import (
        check_atomic_numbers_supported,
        supported_atomic_numbers,
    )

    supported = supported_atomic_numbers(legacy=False, dispersion=True)
    check_atomic_numbers_supported(numbers, supported, context=context)


def make_embed_fn(
    net,
    params,
    cutoff: float,
    jit_fn: Callable,
    edge_multiplier: int = 40,
) -> Callable[[mlip_common.Structure], Float[Tensor, "L d"]]:  # noqa: F821
    """Return an embed_fn compatible with mlip_common.embed_structures.

    Uses geometric bucket padding so JAX compiles once per bucket size instead
    of once per protein. The padding follows the jraph convention: a single
    padding-graph node (node_mask=False) is appended, padding edges are self-
    loops on that node, and the real atoms are sliced out (x[:N_real]) before
    calling to_residues.

    theory_mask is not passed because return_representations_bool=True causes
    the model to return before the energy head reads theory_mask, the only
    module that uses it (stacknet_sparse.py, observable_sparse.py).

    Converts the JAX output to a CPU torch float32 tensor before returning
    so the safetensors writers (CPU torch build) can process it.

    XLA OOM is caught and re-raised as RuntimeError('CUDA out of memory: ...')
    so mlip_common counts and skips the protein without aborting the run.
    """
    edge_buckets = [b * edge_multiplier for b in _NODE_BUCKETS]

    def embed_fn(
        structure: mlip_common.Structure,
    ) -> Float[Tensor, "L d"]:  # noqa: F821
        n_electrons = int(np.sum(structure.numbers)) - structure.charge
        if n_electrons % 2 != 0:
            raise ValueError(
                f"{structure.id}: sum(Z) - charge = {n_electrons} is odd "
                "(open shell). Closed-shell structures required."
            )
        _check_elements(structure.numbers, context=structure.id)

        idx_i, idx_j = _local_pairs(structure.positions, cutoff)

        n_real = len(structure.numbers)
        p_real = len(idx_i)
        n_bucket = _next_bucket(n_real, _NODE_BUCKETS)
        p_bucket = _next_bucket(p_real, edge_buckets)

        (
            pos_p, num_p, ii_p, jj_p,
            node_mask, batch_seg, graph_mask,
            charge_2, spin_2,
        ) = _pad_to_buckets(
            structure.numbers, structure.positions,
            idx_i, idx_j, structure.charge,
            n_bucket, p_bucket,
        )

        try:
            result = jit_fn(
                params,
                jnp.array(pos_p, dtype=jnp.float32),
                jnp.array(num_p, dtype=jnp.int32),
                jnp.array(ii_p, dtype=jnp.int32),
                jnp.array(jj_p, dtype=jnp.int32),
                total_charge=jnp.array(charge_2),
                num_unpaired_electrons=jnp.array(spin_2),
                node_mask=jnp.array(node_mask),
                batch_segments=jnp.array(batch_seg, dtype=jnp.int32),
                graph_mask=jnp.array(graph_mask),
            )
        except Exception as err:
            msg = str(err).lower()
            if (
                "resource_exhausted" in msg
                or "out of memory" in msg
                or "resourceexhausted" in msg
            ):
                raise RuntimeError(f"CUDA out of memory: {err}") from err
            raise

        x_jax = result["atomic_representations"]  # [N_padded x d]
        x_real = x_jax[:n_real]  # [N_real x d], drop padding atoms

        # Convert to CPU torch float32 for the safetensors writers.
        x = torch.from_numpy(np.array(x_real, dtype=np.float32))

        ri = torch.from_numpy(structure.residue_index).long()
        n_res = len(structure.sequence)
        return mlip_common.to_residues(x, ri, n_res)  # [L x d]

    return embed_fn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed proteins with SO3LR Tauri (S, M, or L).",
        add_help=True,
    )
    parser.add_argument(
        "--model",
        choices=list(_VERSIONED_NAMES),
        default=_DEFAULT_SIZE,
        help=(
            f"Model size. S/M/L map to exact versioned names "
            f"({', '.join(f'{k}={v}' for k, v in _VERSIONED_NAMES.items())}). "
            f"Default: {_DEFAULT_SIZE}."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Path to a local SO3krates checkpoint directory (overrides --model). "
            "The directory must contain params.pkl and hyperparameters.json."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "JAX device to use: 'cuda' or 'cpu'. "
            "Defaults to the first available GPU."
        ),
    )
    mlip_common.add_common_args(parser)
    return parser


if __name__ == "__main__":
    _args = _build_parser().parse_args()

    if _args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")

    # Resolve the model name or path.
    if _args.model_path:
        _versioned = _args.model_path
        _size_key = _args.model  # used for logging only
    else:
        _size_key = _args.model
        _versioned = _VERSIONED_NAMES[_size_key]

    _edge_mult = 50 if _size_key == "L" else 40

    print(f"JAX devices: {jax.devices()}")
    print(
        f"matmul precision {jax.config.jax_default_matmul_precision}, "
        f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}"
    )
    print(f"Model: {_versioned}")
    print(f"Expected embedding dim: {_EMBED_DIM.get(_size_key, '?')}")

    _net, _params, _cutoff = _load_repr_model(_versioned)
    print(f"Model loaded. Local cutoff: {_cutoff} A")

    _jit_fn = _jit_obs_fn(_net)
    _embed_fn = make_embed_fn(
        _net, _params, _cutoff, _jit_fn, edge_multiplier=_edge_mult
    )

    mlip_common.embed_structures(
        embed_fn=_embed_fn, **mlip_common.common_kwargs(_args)
    )

    # Report JAX peak GPU memory after the run.
    try:
        _dev = jax.devices()[0]
        _stats = _dev.memory_stats()
        _peak_mb = _stats.get("peak_bytes_in_use", 0) / 1024 / 1024
        print(f"JAX peak GPU memory: {_peak_mb:.1f} MB")
    except (AttributeError, KeyError, RuntimeError):
        print("JAX peak GPU memory: unavailable")
