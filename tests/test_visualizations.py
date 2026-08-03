"""Unit tests for the leaderboard data-extraction logic."""

from __future__ import annotations

import json
import unittest

from src.visualizations.leaderboard import build_payload, extract_records, render_html


def _entry(*, task_type: str, test_metrics: dict, invalidated: bool = False) -> dict:
    metrics = {"train": {}, "test": {"split": "test", "task_type": task_type, "n": 10, **test_metrics}}
    entry = {"metrics": metrics}
    if invalidated:
        entry["invalidated"] = True
    return entry


def _card() -> dict:
    return {
        "properties": ["phopt", "solubility"],
        "real_case": ["erickson_petase"],
        "predictions": {
            "no_attention": {
                "random": {
                    "linear": {
                        "phopt": {
                            "ankh_base": _entry(task_type="regression", test_metrics={"mae": 0.5, "rmse": 0.7}),
                            "esm2_650M": _entry(task_type="regression", test_metrics={"mae": 0.3, "rmse": 0.4}),
                        },
                        "solubility": {
                            "ankh_base": _entry(
                                task_type="classification", test_metrics={"f1": 0.8, "accuracy": 0.75}
                            ),
                        },
                    },
                    "xgboost": {
                        "phopt": {
                            # No metrics computed yet (mirrors real gaps in results_card.json,
                            # e.g. residue-level xgboost runs) -- must be skipped, not plotted as 0.
                            "ankh_base": {"metrics": {}},
                            "esm2_650M": _entry(
                                task_type="regression",
                                test_metrics={"mae": 0.2, "rmse": 0.3},
                                invalidated=True,
                            ),
                        },
                    },
                },
            },
        },
    }


class ExtractRecordsTests(unittest.TestCase):
    def test_skips_invalidated_and_empty_metrics_entries(self) -> None:
        records, task_meta = extract_records(_card())
        keys = {(r.task, r.family, r.embedding) for r in records}
        self.assertIn(("phopt", "linear", "ankh_base"), keys)
        self.assertIn(("phopt", "linear", "esm2_650M"), keys)
        self.assertNotIn(("phopt", "xgboost", "ankh_base"), keys)  # empty metrics
        self.assertNotIn(("phopt", "xgboost", "esm2_650M"), keys)  # invalidated

    def test_uses_primary_metric_per_task_type(self) -> None:
        records, task_meta = extract_records(_card())
        phopt_ankh = next(r for r in records if r.task == "phopt" and r.embedding == "ankh_base")
        self.assertAlmostEqual(phopt_ankh.value, 0.5)  # mae is the registered primary regression metric
        self.assertEqual(task_meta["phopt"].metric, "mae")
        self.assertFalse(task_meta["phopt"].higher_is_better)

        solubility = next(r for r in records if r.task == "solubility")
        self.assertAlmostEqual(solubility.value, 0.8)  # f1 is the registered primary classification metric
        self.assertEqual(task_meta["solubility"].metric, "f1")
        self.assertTrue(task_meta["solubility"].higher_is_better)

    def test_task_scope_resolved_from_card(self) -> None:
        card = _card()
        card["predictions"]["no_attention"]["random"]["linear"]["erickson_petase"] = {
            "ankh_base": _entry(task_type="regression", test_metrics={"mae": 0.1}),
        }
        _, task_meta = extract_records(card)
        self.assertEqual(task_meta["phopt"].scope, "properties")
        self.assertEqual(task_meta["erickson_petase"].scope, "real_case")


class BuildPayloadAndRenderTests(unittest.TestCase):
    def test_payload_orders_properties_before_real_case(self) -> None:
        card = _card()
        card["predictions"]["no_attention"]["random"]["linear"]["erickson_petase"] = {
            "ankh_base": _entry(task_type="regression", test_metrics={"mae": 0.1}),
        }
        records, task_meta = extract_records(card)
        payload = build_payload(records, task_meta)
        self.assertLess(
            payload["task_order"].index("phopt"),
            payload["task_order"].index("erickson_petase"),
        )

    def test_render_html_embeds_valid_json_payload(self) -> None:
        records, task_meta = extract_records(_card())
        payload = build_payload(records, task_meta)
        html = render_html(payload)
        self.assertIn('<script id="leaderboard-data" type="application/json">', html)

        start = html.index('type="application/json">') + len('type="application/json">')
        end = html.index("</script>", start)
        embedded = json.loads(html[start:end])
        self.assertEqual(embedded["records"], payload["records"])


if __name__ == "__main__":
    unittest.main()
