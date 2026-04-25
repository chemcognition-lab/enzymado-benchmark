"""
Batch-size scan for the ConfigMLP head.

For every embedding dimension used in the benchmark the script runs a
unified benchmark over a grid of batch sizes. It measures both the 
per-step wall-clock latency and the total epoch-level time (including 
host→device transfer).

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
EMBEDDING_DIMS = {
    "ankh_base":              768,
    "esmc_300m":              960,
    "prot5 / prost5 / profluent_e1_300m": 1024,
    "esmc_600m":             1152,
    "esm2_650M / profluent_e1_600m":      1280,
    "esm3_open / ankh_large / ankh3_large": 1536,
    "esm2_3B / ankh3_xl":    2560,
    "sanity_check": 1500
}

BATCH_SIZES   = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
NUM_SAMPLES   = 10000    # synthetic dataset size
NUM_EPOCHS    = 20       # for the epoch-level benchmark
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
# Unified Benchmark (data starts on CPU to measure complete pipeline including transfer)
# ---------------------------------------------------------------------------

def run_benchmark_scan(input_dim: int, device: torch.device) -> list[dict]:
    dataset = make_dataset(NUM_SAMPLES, input_dim, on_device=False, device=device)
    results = []

    for bs in BATCH_SIZES:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        num_batches = len(loader)
        
        model  = ConfigMLP(input_dim).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        opt  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.03)
        crit = nn.L1Loss()

        # Warm-up 
        model.train()
        for i, (xb, yb) in enumerate(loader):
            if i >= WARMUP_STEPS:
                break
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        
        sync(device)

        epoch_times = []

        for _epoch in range(NUM_EPOCHS):
            model.train()
            t_epoch_start = time.perf_counter()   # wall-clock start

            for dict_xb, dict_yb in loader:
                xb, yb = dict_xb.to(device), dict_yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()

            sync(device)                           # flush GPU before stopping clock
            epoch_times.append(time.perf_counter() - t_epoch_start)

        avg_epoch_s = np.mean(epoch_times)
        total_time_s = np.sum(epoch_times)
        
        # Calculate step latency and throughput from the epoch times
        # (avg_epoch_s / num_batches) gives average time per batch in seconds
        avg_step_ms = (avg_epoch_s / num_batches) * 1000
        throughput = bs / (avg_step_ms / 1000)

        results.append({
            "input_dim":        input_dim,
            "batch_size":       bs,
            "latency_ms":       round(avg_step_ms, 3),
            "throughput_s/it":  round(throughput, 1),
            "avg_epoch_time_s": round(avg_epoch_s, 4),
            "total_time_s":     round(total_time_s, 3),
        })
        
        print(f"  bs={bs:4d} | step={avg_step_ms:6.2f}ms | epoch={avg_epoch_s:5.2f}s | total={total_time_s:5.2f}s")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "mlp_hpo" / "batch_size_scan"


def save_figures(df: pd.DataFrame, out_dir: Path) -> None:
    dims = sorted(df["input_dim"].unique())

    # --- latency figure ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for dim in dims:
        sub = df[df["input_dim"] == dim]
        ax.plot(sub["batch_size"], sub["latency_ms"], marker="o", label=f"dim={dim}")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency per Step (ms)")
    ax.set_title("Per-step Wall-Clock Latency (incl. H2D transfer)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_batch_size.png", dpi=150)
    plt.close(fig)

    # --- epoch time figure ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for dim in dims:
        sub = df[df["input_dim"] == dim]
        ax.plot(sub["batch_size"], sub["total_time_s"], marker="s", label=f"dim={dim}")
    
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Total Wall-Clock Time (s)")
    ax.set_title("Total Runtime vs Batch Size")
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

    all_results = []

    for model_label, input_dim in EMBEDDING_DIMS.items():
        print(f"\n{'='*60}")
        print(f"Embedding dim {input_dim}  ({model_label})")
        print(f"{'='*60}")

        all_results.extend(run_benchmark_scan(input_dim, device))

    df = pd.DataFrame(all_results)
    
    print("\n\n===== Benchmark Results =====")
    print(df.to_string(index=False))

    csv_path = OUTPUT_DIR / "batch_size_scan_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    save_figures(df, OUTPUT_DIR)
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
