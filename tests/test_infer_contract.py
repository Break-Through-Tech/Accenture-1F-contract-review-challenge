import json
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from inference.infer_contract import chunk_contract, detect_clauses, get_or_compute_cutoffs, score_and_bucket
from risk.score_contracts import norm


class FakeTokenizer:
    """A minimal stand-in for a HF fast tokenizer: one token per whitespace-
    separated word, offsets mapped back to character spans. Just enough for
    paragraph_token_chunks()'s tok_offsets()/token_count() calls - no real
    model or downloaded tokenizer needed to test chunking."""

    is_fast = True
    _WORD_RE = re.compile(r"\S+")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, truncation=False, **kwargs):
        spans = [(m.start(), m.end()) for m in self._WORD_RE.finditer(text)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


class ChunkContractTests(unittest.TestCase):
    def test_chunks_cover_raw_text_with_no_gaps(self):
        text = (
            "Alpha bravo charlie delta echo foxtrot golf hotel.\n\n"
            "India juliet kilo lima mike november oscar papa."
        )
        chunks = chunk_contract(text, FakeTokenizer(), max_tokens=4, overlap=1)
        total_tokens = len(FakeTokenizer()(text)["input_ids"])

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["start_char"], 0)
        self.assertEqual(chunks[-1]["end_char"], len(text))
        # Coverage guarantee is in token space, like prepare_dataset.py's
        # check_chunks(): no gap between one chunk's last token and the next
        # chunk's first token (a 1-char whitespace gap in char space is
        # expected and fine - whitespace isn't itself a token).
        token_starts = [c["token_start"] for c in chunks]
        token_ends = [c["token_end"] for c in chunks]
        self.assertEqual(token_starts[0], 0)
        self.assertEqual(token_ends[-1], total_tokens)
        for i in range(1, len(chunks)):
            self.assertLessEqual(token_starts[i], token_ends[i - 1], "chunks must not leave a token gap")

    def test_single_short_paragraph_is_one_chunk(self):
        text = "Short agreement text that fits in one chunk."
        chunks = chunk_contract(text, FakeTokenizer(), max_tokens=100, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], text)


class DetectClausesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rules_path = Path(__file__).parents[1] / "risk" / "risk_rules.json"
        cls.evidence_rules = json.loads(rules_path.read_text(encoding="utf-8"))["evidence_rules"]

    def detect(self, categories, probs, texts):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        thresholds_csv = Path(temp_dir.name) / "thresholds.csv"
        pd.DataFrame({"category": categories, "level": ["contract"] * len(categories),
                      "threshold": [0.5] * len(categories)}).to_csv(thresholds_csv, index=False)
        return detect_clauses("sample-contract", categories, probs, texts, thresholds_csv, self.evidence_rules)

    def test_below_threshold_category_is_not_detected(self):
        found = self.detect(["Anti-Assignment"], [[0.2]], ["No party may assign this agreement."])
        self.assertNotIn("Anti-Assignment", found)

    def test_above_threshold_without_evidence_rule_is_detected_as_is(self):
        found = self.detect(["Anti-Assignment"], [[0.8]], ["No party may assign this agreement without consent."])
        entry = found["Anti-Assignment"]
        self.assertEqual(entry["source"], "model")
        self.assertEqual(entry["confidence"], 0.8)

    def test_evidence_rule_missing_downweights_confidence(self):
        found = self.detect(
            ["Termination For Convenience"], [[0.8]],
            ["Either party may terminate for material breach."],
        )
        entry = found["Termination For Convenience"]
        self.assertEqual(entry["source"], "model, evidence rule missing (downweighted)")
        self.assertEqual(entry["confidence"], 0.2)

    def test_evidence_rule_matched_keeps_confidence(self):
        found = self.detect(
            ["Termination For Convenience"], [[0.8]],
            ["Either party may terminate this agreement at any time."],
        )
        entry = found["Termination For Convenience"]
        self.assertEqual(entry["source"], "model, evidence rule matched")
        self.assertEqual(entry["confidence"], 0.8)


class ScoreAndBucketTests(unittest.TestCase):
    def setUp(self):
        self.levels = {
            norm("Cap On Liability"): ("Medium", "Cap On Liability"),
            norm("Uncapped Liability"): ("High", "Uncapped Liability"),
        }
        self.rules = {
            "level_weights": {"High": 3, "Medium": 1, "Low": 0.5},
            "missing_protections": [],
            "combinations": [],
            "severity_floor": {"min_confidence": 0.8, "floor_bucket": "Medium"},
        }
        self.cuts = (10.0, 16.0)

    def test_low_score_contract_is_low_bucket(self):
        found = {"Cap On Liability": {"confidence": 0.5, "evidence": "e", "source": "model"}}
        score, _, floor, _, bucket = score_and_bucket(found, self.levels, self.rules, self.cuts)
        self.assertLess(score, 10.0)
        self.assertFalse(floor)
        self.assertEqual(bucket, "Low")

    def test_high_confidence_high_risk_clause_floors_to_medium(self):
        found = {"Uncapped Liability": {"confidence": 0.9, "evidence": "e", "source": "model"}}
        score, _, floor, _, bucket = score_and_bucket(found, self.levels, self.rules, self.cuts)
        self.assertTrue(floor)
        self.assertEqual(bucket, "Medium")

    def test_score_above_high_cutoff_is_high_bucket(self):
        # 6 High-weighted clauses at full confidence: 6 * 3 = 18 >= the 16.0 high cutoff.
        found = {
            f"clause_{i}": {"confidence": 1.0, "evidence": "e", "source": "model"}
            for i in range(6)
        }
        levels = {f"clause_{i}": ("High", f"Clause {i}") for i in range(6)}
        score, _, _, _, bucket = score_and_bucket(found, levels, self.rules, self.cuts)
        self.assertGreaterEqual(score, 16.0)
        self.assertEqual(bucket, "High")


class CutoffCachingTests(unittest.TestCase):
    def test_cached_cutoffs_are_returned_without_recomputing(self):
        manifest = {
            "stages": {
                "cutoffs": {
                    "rules_path": "rules.json",
                    "rubric_path": "rubric.csv",
                    "train_json": "/does/not/exist.json",
                    "medium": 10.0,
                    "high": 16.0,
                }
            }
        }
        # train_json points nowhere real; if this weren't a cache hit,
        # load_gold() would raise trying to read it.
        medium, high, info = get_or_compute_cutoffs(manifest, "rules.json", "rubric.csv", "/does/not/exist.json")
        self.assertEqual((medium, high, info), (10.0, 16.0, None))

    def test_mismatched_cache_entry_is_not_reused(self):
        manifest = {
            "stages": {
                "cutoffs": {
                    "rules_path": "old_rules.json",
                    "rubric_path": "rubric.csv",
                    "train_json": "/does/not/exist.json",
                    "medium": 10.0,
                    "high": 16.0,
                }
            }
        }
        with self.assertRaises(FileNotFoundError):
            get_or_compute_cutoffs(manifest, "new_rules.json", "rubric.csv", "/does/not/exist.json")


if __name__ == "__main__":
    unittest.main()
