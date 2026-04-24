"""
Batch-size scan for the ConfigMLP head.

For every embedding dimension used in the benchmark the script runs two
micro-benchmarks over a grid of batch sizes:

  1. Per-step wall-clock latency  (including host→device transfer)
  2. Epoch-level total wall-clock time + final MAE after `num_epochs` epochs

Results are printed as tables and saved under
`results/mlp_hpo/batch_size_scan/`.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Unique embedding dimensions that appear in the benchmark
# (derived from notebooks/from_h5_to_json.py MODEL_DIMS)
EMBEDDING_DIMS = {
    "ankh_base":              768,
    "esmc_300m":              960,
    "prot5 / prost5 / profluent_e1_300m": 1024,
    "esmc_600m":             1152,
    "esm2_650M / profluent_e1_600m":      1280,
    "esm3_open / ankh_large / ankh3_large": 1536,
    "esm2_3B / ankh3_xl":    2560,
}

BATCH_SIZES   = [16, 32, 64, 128, 256]
NUM_SAMPLES   = 10_000   # synthetic dataset size
NUM_EPOCHS    = 20       # for the epoch-level benchmark
LATENCY_STEPS = 50       # steps measured in the per-step latency scan
WARMUP_STEPS  = 5        # warm-up steps (not measured)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConfigMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1,
                 expansion_ratio: float = 4.0, dropout_rate: float = 0.15):
        super().__init__()
        hidden_dim = int(input_dim * expansion_ratio)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.bfloat16),
            nn.LayerNorm(hidden_dim, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16),
            nn.LayerNorm(hidden_dim, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim, dtype=torch.bfloat16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(num_samples: int, input_dim: int, on_device: bool, device: torch.device):
    """Return a TensorDataset; tensors live on `device` if on_device=True."""
    torch.manual_seed(42)
    X = torch.randn(num_samples, input_dim, dtype=torch.bfloat16)
    w = torch.randn(input_dim, 1, dtype=torch.bfloat16)
    y = X @ w + 0.5 * torch.randn(num_samples, 1, dtype=torch.bfloat16)
    if on_device:
        X, y = X.to(device), y.to(device)
    return TensorDataset(X, y)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Benchmark 1: per-step wall-clock latency (data starts on CPU)
# ---------------------------------------------------------------------------

def run_latency_scan(input_dim: int, device: torch.device) -> list[dict]:
    dataset = make_dataset(NUM_SAMPLES, input_dim, on_device=False, device=device)
    results = []

    for bs in BATCH_SIZES:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        model  = ConfigMLP(input_dim).to(device)
        opt    = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.03)
        crit   = nn.L1Loss()

        model.train()
        for i, (xb, yb) in enumerate(loader):
            if i >= WARMUP_STEPS:
                break
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()

        latencies = []
        for i, (xb_cpu, yb_cpu) in enumerate(loader):
            if i >= LATENCY_STEPS:
                break
            sync(device)
            t0 = time.perf_counter()

            xb, yb = xb_cpu.to(device), yb_cpu.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

            sync(device)
            latencies.append(time.perf_counter() - t0)

        avg_ms = np.mean(latencies) * 1000
        results.append({
            "input_dim":       input_dim,
            "batch_size":      bs,
            "latency_ms":      round(avg_ms, 3),
            "throughput_s/it": round(bs / (avg_ms / 1000), 1),
        })
        print(f"  bs={bs:4d}  latency={avg_ms:8.2f} ms")

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: epoch-level wall-clock time + MAE (data pre-loaded on device)
# ---------------------------------------------------------------------------

def run_epoch_scan(input_dim: int, device: torch.device) -> list[dict]:
    # Data lives on the device to isolate compute cost; total wall-clock time
    # still captures the full pipeline (zero_grad, forward, backward, step).
    dataset = make_dataset(NUM_SAMPLES, input_dim, on_device=True, device=device)
    results = []

    for bs in BATCH_SIZES:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        model  = ConfigMLP(input_dim).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        opt  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.03)
        crit = nn.L1Loss()

        epoch_times = []
        final_mae   = 0.0

        for _epoch in range(NUM_EPOCHS):
            model.train()
            t_epoch_start = time.perf_counter()   # wall-clock start

            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            sync(device)                           # flush GPU before stopping clock
            epoch_times.append(time.perf_counter() - t_epoch_start)
            final_mae = epoch_loss / len(loader)

        results.append({
            "input_dim":        input_dim,
            "batch_size":       bs,
            "final_mae":        round(final_mae, 5),
            "avg_epoch_time_s": round(np.mean(epoch_times), 4),
            "total_time_s":     round(np.sum(epoch_times), 3),
        })
        print(f"  bs={bs:4d}  total={np.sum(epoch_times):.2f}s  mae={final_mae:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "mlp_hpo" / "batch_size_scan"


def save_figures(lat_df: pd.DataFrame, epoch_df: pd.DataFrame, out_dir: Path) -> None:
    dims = sorted(lat_df["input_dim"].unique())

    # --- latency figure ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for dim in dims:
        sub = lat_df[lat_df["input_dim"] == dim]
        ax.plot(sub["batch_size"], sub["latency_ms"], marker="o", label=f"dim={dim}")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency per Step (ms)")
    ax.set_title("Per-step Wall-Clock Latency (incl. H2D transfer)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_batch_size.png", dpi=150)
    plt.close(fig)

    # --- epoch time + MAE figures ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for dim in dims:
        sub = epoch_df[epoch_df["input_dim"] == dim]
        axes[0].plot(sub["batch_size"], sub["total_time_s"], marker="s", label=f"dim={dim}")
        axes[1].plot(sub["batch_size"], sub["final_mae"],    marker="o", label=f"dim={dim}")
    for ax, ylabel, title in zip(
        axes,
        ["Total Wall-Clock Time (s)", "Final MAE"],
        ["Total Runtime vs Batch Size", "Final MAE vs Batch Size"],
    ):
        ax.set_xlabel("Batch Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "epoch_metrics_vs_batch_size.png", dpi=150)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs: {gpu_count}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_latency = []
    all_epoch   = []

    for model_label, input_dim in EMBEDDING_DIMS.items():
        print(f"\n{'='*60}")
        print(f"Embedding dim {input_dim}  ({model_label})")
        print(f"{'='*60}")

        print("\n-- Per-step latency scan (data on CPU, incl. H2D transfer) --")
        all_latency.extend(run_latency_scan(input_dim, device))

        print(f"\n-- Epoch-level scan ({NUM_EPOCHS} epochs, total wall-clock) --")
        all_epoch.extend(run_epoch_scan(input_dim, device))

    lat_df   = pd.DataFrame(all_latency)
    epoch_df = pd.DataFrame(all_epoch)

    print("\n\n===== Per-step latency results =====")
    print(lat_df.to_string(index=False))

    print("\n\n===== Epoch-level results =====")
    print(epoch_df.to_string(index=False))

    combined = lat_df.merge(epoch_df, on=["input_dim", "batch_size"])
    csv_path = OUTPUT_DIR / "batch_size_scan_results.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    save_figures(lat_df, epoch_df, OUTPUT_DIR)
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
