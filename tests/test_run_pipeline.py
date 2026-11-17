import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import run_pipeline


CONFIG = {
    "run_name": "config_run",
    "seed": 42,
    "paths": {
        "train_json": "data/cuad/train_separate_questions.json",
        "test_json": "data/cuad/test.json",
        "rubric": "evaluation/risk_categories.csv",
        "rules": "risk/risk_rules.json",
        "data_dir": "training/data",
        "train_out": "training/outputs",
        "eval_out": "training/outputs/evaluation",
        "risk_out": "risk/outputs",
    },
    "score": {"source": "model", "threshold_scheme": "tuned"},
}


class ScoreRunNameTests(unittest.TestCase):
    """'score --run-name' must read the named run's manifest, not the config's."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.runs = Path(temp_dir.name)

        # Keep the real runs/ directory out of the test entirely.
        patch_path = mock.patch.object(
            run_pipeline, "manifest_path",
            lambda run_name: self.runs / run_name / "manifest.json",
        )
        patch_path.start()
        self.addCleanup(patch_path.stop)

        # Capture the command instead of launching score_contracts.py.
        self.commands = []
        patch_run = mock.patch.object(
            run_pipeline, "run_command",
            lambda cmd, dry_run: self.commands.append([str(part) for part in cmd]),
        )
        patch_run.start()
        self.addCleanup(patch_run.stop)

    def write_manifest(self, run_name, run_dir, thresholds_csv):
        path = self.runs / run_name / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "run_name": run_name,
            "stages": {
                "train": {"run_name": run_name, "run_dir": run_dir},
                "evaluate": {"run_name": run_name, "thresholds_csv": thresholds_csv},
            },
        }), encoding="utf-8")

    def flag(self, cmd, name):
        return cmd[cmd.index(name) + 1]

    def score(self, run_name):
        run_pipeline.cmd_score(CONFIG, Namespace(run_name=run_name, source=None, dry_run=False))
        return self.commands[-1]

    def test_run_name_override_uses_that_runs_manifest(self):
        self.write_manifest("config_run", "/outputs/config_run", "/eval/thresholds_config_run.csv")
        self.write_manifest("registered", "/outputs/registered", "/eval/thresholds_registered.csv")

        cmd = self.score("registered")

        self.assertEqual(self.flag(cmd, "--run"), "/outputs/registered")
        self.assertEqual(self.flag(cmd, "--thresholds"), "/eval/thresholds_registered.csv")

    def test_run_name_override_changes_the_output_directory(self):
        self.write_manifest("registered", "/outputs/registered", "/eval/thresholds_registered.csv")

        cmd = self.score("registered")

        self.assertTrue(self.flag(cmd, "--out").endswith("risk/outputs/registered/model"))

    def test_run_name_override_records_the_score_stage_under_that_run(self):
        self.write_manifest("registered", "/outputs/registered", "/eval/thresholds_registered.csv")

        self.score("registered")

        written = json.loads((self.runs / "registered" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(written["stages"]["score"]["run_name"], "registered")
        # The config's own run must not pick up a score entry it didn't produce.
        self.assertFalse((self.runs / "config_run" / "manifest.json").exists())

    def test_no_override_falls_back_to_the_config_run_name(self):
        """This is the path 'all' takes: it never defines --run-name, so it is None."""
        self.write_manifest("config_run", "/outputs/config_run", "/eval/thresholds_config_run.csv")
        self.write_manifest("registered", "/outputs/registered", "/eval/thresholds_registered.csv")

        cmd = self.score(None)

        self.assertEqual(self.flag(cmd, "--run"), "/outputs/config_run")
        self.assertEqual(self.flag(cmd, "--thresholds"), "/eval/thresholds_config_run.csv")
        self.assertTrue(self.flag(cmd, "--out").endswith("risk/outputs/config_run/model"))


class ArgumentParserTests(unittest.TestCase):
    def parse(self, argv):
        return run_pipeline.build_arg_parser().parse_args(argv)

    def test_score_accepts_run_name(self):
        self.assertEqual(self.parse(["score", "--run-name", "registered"]).run_name, "registered")

    def test_score_run_name_defaults_to_none(self):
        self.assertIsNone(self.parse(["score"]).run_name)

    def test_all_does_not_define_run_name(self):
        """'all' calls cmd_score(), so main() has to default run_name for it."""
        self.assertFalse(hasattr(self.parse(["all"]), "run_name"))


class MainDefaultsTests(unittest.TestCase):
    """main() has to default run_name for the stages whose parser omits it,
    because 'all' reaches cmd_score()."""

    def run_main(self, argv):
        """Dispatch one CLI invocation with the stage body stubbed out, and
        return the args namespace it was handed.

        'all' is looked up by name inside main(), the other stages come from
        the STAGE_FUNCS dict, so the two need different patches.
        """
        seen = []
        stub = lambda cfg, args: seen.append(args)
        stage = argv[0]
        if stage in run_pipeline.STAGE_FUNCS:
            patch_stage = mock.patch.dict(run_pipeline.STAGE_FUNCS, {stage: stub})
        else:
            patch_stage = mock.patch.object(run_pipeline, f"cmd_{stage}", stub)
        with mock.patch.object(run_pipeline.sys, "argv", ["run_pipeline.py", *argv]), \
             mock.patch.object(run_pipeline, "load_config", lambda path: CONFIG), \
             patch_stage:
            run_pipeline.main()
        self.assertEqual(len(seen), 1)
        return seen[0]

    def test_all_gets_run_name_none(self):
        self.assertIsNone(self.run_main(["all"]).run_name)

    def test_single_stage_gets_run_name_none(self):
        self.assertIsNone(self.run_main(["prepare"]).run_name)

    def test_score_without_the_flag_gets_run_name_none(self):
        self.assertIsNone(self.run_main(["score"]).run_name)

    def test_score_keeps_an_explicit_run_name(self):
        self.assertEqual(self.run_main(["score", "--run-name", "registered"]).run_name, "registered")

if __name__ == "__main__":
    unittest.main()
