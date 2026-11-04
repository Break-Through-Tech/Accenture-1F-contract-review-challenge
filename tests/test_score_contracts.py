import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from risk.score_contracts import load_model, score_contract


class EvidenceRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rules_path = Path(__file__).parents[1] / "risk" / "risk_rules.json"
        cls.evidence_rules = json.loads(rules_path.read_text(encoding="utf-8"))["evidence_rules"]

    def load_predictions(self, categories, probabilities, texts):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        np.savez(
            root / "predictions_test.npz",
            categories=np.asarray(categories),
            probs=np.asarray(probabilities, dtype=float),
            contract_title=np.asarray(["sample-contract"] * len(texts)),
            chunk_index=np.arange(len(texts)),
        )
        pd.DataFrame({"chunk_text": texts}).to_csv(root / "chunks.csv", index=False)
        pd.DataFrame({"category": categories, "level": ["contract"] * len(categories),
                      "threshold": [0.5] * len(categories)}).to_csv(root / "thresholds.csv", index=False)
        return load_model(root, root / "thresholds.csv", root / "chunks.csv", "tuned", self.evidence_rules)

    def test_change_of_control_uses_matching_non_top_chunk_as_evidence(self):
        contracts = self.load_predictions(
            ["Change Of Control"], [[0.9], [0.4]],
            ["Ordinary termination language without a control event.",
             "Upon a change of control, the remaining advance shall be repaid."],
        )

        entry = contracts["sample-contract"]["Change Of Control"]
        self.assertEqual(entry["source"], "model, evidence rule matched")
        self.assertIn("change of control", entry["evidence"].lower())
        self.assertIn("contractual consequence", entry["rule_reason"])
        self.assertEqual(entry["confidence"], 0.9)

    def test_change_of_control_backstop_requires_trigger_and_consequence(self):
        contracts = self.load_predictions(
            ["Change Of Control"], [[0.1]],
            ["Following a merger, the counterparty may terminate this agreement."],
        )

        entry = contracts["sample-contract"]["Change Of Control"]
        self.assertEqual(entry["source"], "keyword backstop")
        self.assertEqual(entry["confidence"], 0.5)
        self.assertIn("Evidence backstop matched", entry["rule_reason"])

    def test_asset_assignment_exception_alone_does_not_trigger_change_of_control(self):
        contracts = self.load_predictions(
            ["Change Of Control"], [[0.1]],
            ["Assignment requires prior written consent, except to a person acquiring all or substantially all assets."],
        )

        self.assertNotIn("Change Of Control", contracts["sample-contract"])

    def test_category_specific_evidence_gates_downweight_unqualified_detections(self):
        examples = {
            "Termination For Convenience": "Either party may terminate for material breach.",
            "Notice Period To Terminate Renewal": "The agreement renews automatically after written notice.",
            "Post-Termination Services": "Services are provided until the termination date.",
            "Uncapped Liability": "Liability is subject to the cap set out in Section 8.",
        }
        for category, text in examples.items():
            with self.subTest(category=category):
                contracts = self.load_predictions([category], [[0.8]], [text])
                entry = contracts["sample-contract"][category]
                self.assertEqual(entry["source"], "model, evidence rule missing (downweighted)")
                self.assertEqual(entry["confidence"], 0.2)
                self.assertIn("Evidence rule not found", entry["rule_reason"])

    def test_category_specific_evidence_rules_retain_qualifying_chunks(self):
        examples = {
            "Termination For Convenience": "Either party may terminate this agreement at any time.",
            "Notice Period To Terminate Renewal": "Either party may terminate renewal by giving 60 days' notice.",
            "Post-Termination Services": "Following termination, supplier will continue to provide transition support.",
            "Uncapped Liability": "Liability for indemnification is without limitation.",
        }
        for category, text in examples.items():
            with self.subTest(category=category):
                contracts = self.load_predictions([category], [[0.8]], [text])
                entry = contracts["sample-contract"][category]
                self.assertEqual(entry["source"], "model, evidence rule matched")
                self.assertEqual(entry["confidence"], 0.8)
                self.assertIn("Evidence rule matched", entry["rule_reason"])
                self.assertEqual(entry["evidence"], text)

    def test_anti_assignment_remains_unmodified(self):
        contracts = self.load_predictions(
            ["Anti-Assignment"], [[0.8]], ["No party may assign this agreement without consent."],
        )

        entry = contracts["sample-contract"]["Anti-Assignment"]
        self.assertEqual(entry["source"], "model")
        self.assertNotIn("rule_reason", entry)
        self.assertEqual(entry["confidence"], 0.8)

    def test_gold_style_scoring_keeps_category_weight_and_confidence(self):
        found = {
            "Change Of Control": {
                "confidence": 1.0,
                "evidence": "Annotation evidence",
                "source": "annotation",
            }
        }
        levels = {"changeofcontrol": ("High", "Change Of Control")}
        rules = {
            "level_weights": {"High": 3, "Medium": 1, "Low": 0.5},
            "missing_protections": [],
            "combinations": [],
            "severity_floor": {"min_confidence": 0.8},
        }

        score, reasons, floor, register = score_contract(found, levels, rules)
        self.assertEqual(score, 3.0)
        self.assertEqual(reasons[0][1], 3.0)
        self.assertTrue(floor)
        self.assertEqual(register[0]["rule_reason"], "")


if __name__ == "__main__":
    unittest.main()