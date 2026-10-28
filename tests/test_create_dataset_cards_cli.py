from __future__ import annotations

from pathlib import Path
import runpy
import sys

import enzymado.dataset_card.dataset_card_populator as dataset_card_populator

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data_prep"
    / "create_dataset_cards.py"
)


def run_cli(monkeypatch, arguments: list[str]) -> list[tuple[str, object]]:
    """Run the dataset-card CLI with a recorder in place of real processing."""
    calls: list[tuple[str, object]] = []

    class RecordingPopulator:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("init", kwargs))

        def run(self) -> None:
            calls.append(("run", None))

    monkeypatch.setattr(
        dataset_card_populator,
        "DatasetCardPopulator",
        RecordingPopulator,
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), *arguments])
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    return calls


def test_dataset_card_cli_defaults_are_stable(monkeypatch) -> None:
    """Freeze effective defaults before restructuring the script."""
    assert run_cli(monkeypatch, []) == [
        (
            "init",
            {
                "base_directory": "data/processed/properties",
                "emb_model": None,
                "embeddings_json": "embeddings_models_output.json",
                "seq_identity": 0.5,
                "allow_incomplete": False,
                "figures_dir": "results/figures",
            },
        ),
        ("run", None),
    ]


def test_dataset_card_cli_forwards_selected_options(monkeypatch) -> None:
    """Freeze option names, parsed types, and constructor argument mapping."""
    assert run_cli(
        monkeypatch,
        [
            "--base_dir",
            "custom/properties",
            "--emb_model",
            "ankh_large",
            "--embeddings_json",
            "custom_embeddings.json",
            "--seq_identity",
            "0.6",
        ],
    ) == [
        (
            "init",
            {
                "base_directory": "custom/properties",
                "emb_model": "ankh_large",
                "embeddings_json": "custom_embeddings.json",
                "seq_identity": 0.6,
                "allow_incomplete": False,
                "figures_dir": "results/figures",
            },
        ),
        ("run", None),
    ]
