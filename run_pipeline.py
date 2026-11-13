#!/usr/bin/env python3
"""Pipeline orchestrator (Phase A of issue #23).

Connects chunking, clause classification, evaluation and risk scoring by
calling the existing scripts - training/prepare_dataset.py,
training/train_distilroberta.py, training/evaluate.py and
risk/score_contracts.py - as subprocesses, with consistent arguments taken
from pipeline.yaml. This script does not reimplement any stage; it only
builds argument lists and runs the existing scripts.

After each stage it writes/updates runs/<run_name>/manifest.json with the
config used, the seed, the output paths, and a timestamp. Later stages read
their inputs from that manifest instead of guessing directory names:
  - 'evaluate' reads the run directory that 'train' wrote.
  - 'score' (model source) reads the run directory from 'train' and the
    tuned-thresholds file from 'evaluate'.

Usage:
    python run_pipeline.py prepare
    python run_pipeline.py train
    python run_pipeline.py evaluate
    python run_pipeline.py score
    python run_pipeline.py all
    python run_pipeline.py all --dry-run      # print commands, write nothing
    python run_pipeline.py all --force         # rerun every stage even if done
    python run_pipeline.py --config my.yaml prepare
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Config and manifest helpers
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Read pipeline.yaml into a plain dict."""
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def resolve(path_str):
    """Turn a path from the config into an absolute path, relative to the repo root."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def manifest_path(run_name):
    return REPO_ROOT / "runs" / run_name / "manifest.json"


def load_manifest(run_name):
    """Read this run's manifest from disk, or start a fresh one if none exists yet."""
    path = manifest_path(run_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"run_name": run_name, "stages": {}}


def save_manifest(run_name, manifest, dry_run):
    """Write the manifest to disk, unless this is a dry run (then just show it)."""
    path = manifest_path(run_name)
    if dry_run:
        print(f"[dry-run] would write manifest to {path}:")
        print(json.dumps(manifest, indent=2))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {path}")


def record_stage(manifest, stage, info):
    """Add or replace one stage's entry in the manifest, stamped with the current time."""
    manifest.setdefault("stages", {})[stage] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **info,
    }


def run_command(cmd, dry_run):
    """Show the command; actually run it unless this is a dry run."""
    print("$ " + " ".join(str(part) for part in cmd))
    if dry_run:
        print("[dry-run] not executed")
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def require_stage(manifest, stage, dry_run, fallback, hint=""):
    """Fetch a prior stage's manifest entry, or fail with a clear message.

    In a dry run there may be no real manifest yet (nothing was actually
    written), so we fall back to the path that stage *would* write, purely
    so --dry-run can show the full chain of commands without requiring you
    to have already run anything for real.

    `hint` is an optional extra sentence appended to the failure message
    (e.g. pointing at 'register' as an alternative to retraining).
    """
    if stage in manifest.get("stages", {}):
        return manifest["stages"][stage]
    if dry_run:
        return fallback
    msg = (
        f"'{stage}' stage is missing from runs/{manifest['run_name']}/manifest.json. "
        f"Run 'python run_pipeline.py {stage}' first."
    )
    if hint:
        msg += " " + hint
    sys.exit(msg)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def cmd_prepare(cfg, args):
    """Chunk the CUAD contracts and build the tokenized dataset (prepare_dataset.py)."""
    paths, chunking = cfg["paths"], cfg["chunking"]
    data_dir = resolve(paths["data_dir"])

    print(
        "Note: prepare_dataset.py hardcodes the chunk strategy (paragraph, "
        f"{chunking['max_tokens']} tokens, {chunking['overlap']} overlap) internally; "
        "chunking.strategy/max_tokens/overlap in pipeline.yaml are recorded for "
        "documentation only and are not passed as arguments."
    )

    cmd = [
        sys.executable, str(REPO_ROOT / "training" / "prepare_dataset.py"),
        "--train", str(resolve(paths["train_json"])),
        "--test", str(resolve(paths["test_json"])),
        "--tokenizer", chunking["tokenizer"],
        "--out", str(data_dir),
        "--max-length", str(chunking["max_length"]),
        "--min-train-positives", str(chunking["min_train_positives"]),
    ]
    run_command(cmd, args.dry_run)

    manifest = load_manifest(cfg["run_name"])
    record_stage(manifest, "prepare", {
        "run_name": cfg["run_name"],
        "seed": cfg["seed"],
        "data_dir": str(data_dir),
        "chunking_config": chunking,
    })
    save_manifest(cfg["run_name"], manifest, args.dry_run)


def cmd_train(cfg, args):
    """Fine-tune DistilRoBERTa on the tokenized dataset (train_distilroberta.py)."""
    paths, model = cfg["paths"], cfg["model"]
    data_dir = resolve(paths["data_dir"])
    run_dir = resolve(paths["train_out"]) / cfg["run_name"]

    cmd = [
        sys.executable, str(REPO_ROOT / "training" / "train_distilroberta.py"),
        "--data-dir", str(data_dir),
        "--out", str(run_dir),
        "--model", model["name"],
        "--epochs", str(model["epochs"]),
        "--batch-size", str(model["batch_size"]),
        "--lr", str(model["lr"]),
        "--weight-decay", str(model["weight_decay"]),
        "--warmup-ratio", str(model["warmup_ratio"]),
        "--threshold", str(model["threshold"]),
        "--val-frac", str(model["val_frac"]),
        "--seed", str(cfg["seed"]),
        "--pos-weight-mode", model["pos_weight_mode"],
        "--pos-weight-cap", str(model["pos_weight_cap"]),
    ]
    run_command(cmd, args.dry_run)

    manifest = load_manifest(cfg["run_name"])
    record_stage(manifest, "train", {
        "run_name": cfg["run_name"],
        "seed": cfg["seed"],
        "run_dir": str(run_dir),
        "best_model_dir": str(run_dir / "best_model"),
        "predictions_test": str(run_dir / "predictions_test.npz"),
        "model_config": model,
    })
    save_manifest(cfg["run_name"], manifest, args.dry_run)


def cmd_evaluate(cfg, args):
    """Tune per-category thresholds and score the trained run (evaluate.py)."""
    paths, ev = cfg["paths"], cfg["evaluate"]
    manifest = load_manifest(cfg["run_name"])
    fallback_run_dir = resolve(paths["train_out"]) / cfg["run_name"]
    train_info = require_stage(manifest, "train", args.dry_run, {"run_dir": str(fallback_run_dir)})
    run_dir = Path(train_info["run_dir"])

    data_dir = resolve(paths["data_dir"])
    eval_out = resolve(paths["eval_out"])

    cmd = [
        sys.executable, str(REPO_ROOT / "training" / "evaluate.py"),
        "--runs", str(run_dir),
        "--data-dir", str(data_dir),
        "--out", str(eval_out),
        "--min-val-positives", str(ev["min_val_positives"]),
        "--top-n-errors", str(ev["top_n_errors"]),
    ]
    run_command(cmd, args.dry_run)

    # evaluate.py names its per-run files after the run directory's basename.
    thresholds_csv = eval_out / f"thresholds_{run_dir.name}.csv"
    record_stage(manifest, "evaluate", {
        "run_name": cfg["run_name"],
        "seed": cfg["seed"],
        "eval_out": str(eval_out),
        "thresholds_csv": str(thresholds_csv),
        "evaluate_config": ev,
    })
    save_manifest(cfg["run_name"], manifest, args.dry_run)


def cmd_score(cfg, args):
    """Score contracts for risk, gold annotations or the trained model (score_contracts.py)."""
    paths, score_cfg = cfg["paths"], cfg["score"]
    manifest = load_manifest(cfg["run_name"])
    source = args.source or score_cfg["source"]
    out_dir = resolve(paths["risk_out"]) / cfg["run_name"] / source

    cmd = [
        sys.executable, str(REPO_ROOT / "risk" / "score_contracts.py"),
        "--source", source,
        "--train", str(resolve(paths["train_json"])),
        "--test", str(resolve(paths["test_json"])),
        "--rubric", str(resolve(paths["rubric"])),
        "--rules", str(resolve(paths["rules"])),
        "--out", str(out_dir),
    ]

    if source == "model":
        fallback_run_dir = resolve(paths["train_out"]) / cfg["run_name"]
        train_info = require_stage(manifest, "train", args.dry_run, {"run_dir": str(fallback_run_dir)})
        fallback_thresholds = resolve(paths["eval_out"]) / f"thresholds_{cfg['run_name']}.csv"
        eval_info = require_stage(manifest, "evaluate", args.dry_run, {"thresholds_csv": str(fallback_thresholds)})

        run_dir = Path(train_info["run_dir"])
        thresholds_csv = Path(eval_info["thresholds_csv"])
        data_dir = resolve(paths["data_dir"])
        cmd += [
            "--run", str(run_dir),
            "--thresholds", str(thresholds_csv),
            "--threshold-scheme", score_cfg["threshold_scheme"],
            "--chunks", str(data_dir / "test_chunks.csv"),
        ]

    run_command(cmd, args.dry_run)

    record_stage(manifest, "score", {
        "run_name": cfg["run_name"],
        "seed": cfg["seed"],
        "source": source,
        "out": str(out_dir),
        "score_config": score_cfg,
    })
    save_manifest(cfg["run_name"], manifest, args.dry_run)


def cmd_infer(cfg, args):
    """Classify and risk-score one raw contract file (inference/infer_contract.py).

    Not part of 'all': this runs on demand against whatever model/thresholds
    are already recorded in the manifest, instead of running a pipeline stage.
    """
    paths = cfg["paths"]
    run_name = args.run_name or cfg["run_name"]
    manifest = load_manifest(run_name)
    hint = "Or run 'python run_pipeline.py register' to use an already-trained run."

    fallback_run_dir = resolve(paths["train_out"]) / run_name
    train_info = require_stage(
        manifest, "train", args.dry_run,
        {"best_model_dir": str(fallback_run_dir / "best_model")}, hint=hint,
    )
    fallback_thresholds = resolve(paths["eval_out"]) / f"thresholds_{run_name}.csv"
    eval_info = require_stage(
        manifest, "evaluate", args.dry_run,
        {"thresholds_csv": str(fallback_thresholds)}, hint=hint,
    )

    best_model_dir = Path(train_info["best_model_dir"])
    thresholds_csv = Path(eval_info["thresholds_csv"])

    # labels.json's location comes from the manifest's 'prepare' stage when
    # available (the data_dir that run actually wrote to); otherwise fall
    # back to the configured default.
    prepare_info = manifest.get("stages", {}).get("prepare")
    data_dir = Path(prepare_info["data_dir"]) if prepare_info else resolve(paths["data_dir"])
    labels_path = data_dir / "labels.json"

    rubric_path = resolve(paths["rubric"])
    rules_path = resolve(paths["rules"])
    train_json = resolve(paths["train_json"])

    if args.dry_run:
        print(f"[dry-run] would run inference on {args.contract} with:")
        for label, value in [
            ("best_model_dir", best_model_dir), ("thresholds_csv", thresholds_csv),
            ("labels_path", labels_path), ("rubric_path", rubric_path), ("rules_path", rules_path),
        ]:
            print(f"  {label} = {value}")
        print("[dry-run] not executed")
        return

    # Imported lazily so plain pipeline stages (prepare/train/evaluate/score,
    # and --dry-run above) don't need torch/transformers just to argparse.
    from inference import infer_contract as infer_mod

    medium, high, fresh = infer_mod.get_or_compute_cutoffs(manifest, rules_path, rubric_path, train_json)
    if fresh is not None:
        record_stage(manifest, "cutoffs", fresh)
        save_manifest(run_name, manifest, args.dry_run)
    print(f"Bucket cutoffs (runs/{run_name}/manifest.json): Medium >= {medium:.2f}, High >= {high:.2f}\n")

    result = infer_mod.infer_contract(
        args.contract, best_model_dir, labels_path, thresholds_csv, rubric_path, rules_path, (medium, high),
    )

    print(infer_mod.format_report(result))
    payload = json.dumps(result, indent=2)
    if args.out_json:
        out_path = resolve(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"\nWrote JSON to {out_path}")
    else:
        print("\nJSON:")
        print(payload)


def cmd_register(cfg, args):
    """Record an existing trained run's paths in runs/<run-name>/manifest.json,
    so a teammate's already-trained model can be used by 'infer' or 'score'
    without rerunning prepare/train/evaluate locally."""
    best_model_dir = resolve(args.best_model_dir)
    thresholds_csv = resolve(args.thresholds_csv)
    run_dir = resolve(args.run_dir) if args.run_dir else best_model_dir.parent
    predictions_test = resolve(args.predictions_test) if args.predictions_test else None

    to_check = {"--best-model-dir": best_model_dir, "--thresholds-csv": thresholds_csv}
    if args.run_dir:
        to_check["--run-dir"] = run_dir
    if predictions_test:
        to_check["--predictions-test"] = predictions_test
    missing = [f"{flag} ({p})" for flag, p in to_check.items() if not p.exists()]
    if missing:
        sys.exit("register: path(s) do not exist:\n  " + "\n  ".join(missing))

    manifest = load_manifest(args.run_name)
    train_entry = {
        "run_name": args.run_name,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "best_model_dir": str(best_model_dir),
        "registered": True,
    }
    if predictions_test:
        train_entry["predictions_test"] = str(predictions_test)
    record_stage(manifest, "train", train_entry)
    record_stage(manifest, "evaluate", {
        "run_name": args.run_name,
        "seed": args.seed,
        "thresholds_csv": str(thresholds_csv),
        "registered": True,
    })
    save_manifest(args.run_name, manifest, args.dry_run)


STAGE_ORDER = ["prepare", "train", "evaluate", "score"]
STAGE_FUNCS = {
    "prepare": cmd_prepare,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "score": cmd_score,
}


def stage_matches_config(entry, cfg, stage):
    """Check whether a manifest entry for `stage` was actually produced by
    the current config, not just a stage of the same name from a different
    run_name, seed, or (for score) source. A gold score entry does not count
    as "score already done" when the config now asks for a model score."""
    if entry.get("run_name") != cfg["run_name"]:
        return False
    if entry.get("seed") != cfg["seed"]:
        return False
    if stage == "score" and entry.get("source") != cfg["score"]["source"]:
        return False
    return True


def cmd_all(cfg, args):
    """Run prepare, train, evaluate and score in order.

    Skips any stage whose manifest entry matches the current config (same
    run_name/seed, and for score the same source), unless --force is
    passed, so a crash after training doesn't force a retrain.
    """
    manifest = load_manifest(cfg["run_name"])
    for stage in STAGE_ORDER:
        entry = manifest.get("stages", {}).get(stage)
        already_done = entry is not None and stage_matches_config(entry, cfg, stage)
        if already_done and not args.force:
            print(
                f"Skipping '{stage}': already recorded in "
                f"runs/{cfg['run_name']}/manifest.json (pass --force to rerun it)."
            )
            continue
        STAGE_FUNCS[stage](cfg, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "pipeline.yaml"),
        help="Path to the pipeline config file (default: pipeline.yaml)",
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    for name, help_text in [
        ("prepare", "Chunk the contracts and build the tokenized dataset."),
        ("train", "Fine-tune DistilRoBERTa."),
        ("evaluate", "Tune thresholds and evaluate the trained run."),
        ("score", "Score contracts for risk."),
        ("all", "Run prepare, train, evaluate and score in order."),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--dry-run", action="store_true",
            help="Print the commands and manifest that would run, without running anything.",
        )
        if name == "all":
            p.add_argument(
                "--force", action="store_true",
                help="Rerun every stage even if it already has a manifest entry.",
            )
        if name == "score":
            p.add_argument(
                "--source", choices=["gold", "model"], default=None,
                help="Override score.source from the config for this run.",
            )

    infer_p = sub.add_parser("infer", help="Classify and risk-score one raw contract file.")
    infer_p.add_argument("--contract", required=True, help="Path to a plain-text contract file.")
    infer_p.add_argument(
        "--run-name", default=None,
        help="Override run_name from the config for this call (e.g. a registered run).",
    )
    infer_p.add_argument(
        "--out-json", default=None,
        help="Also write the JSON result to this path (default: print to stdout only).",
    )
    infer_p.add_argument(
        "--dry-run", action="store_true",
        help="Print the paths that would be used, without running inference.",
    )

    register_p = sub.add_parser(
        "register", help="Record an existing trained run (model dir, thresholds) in the manifest.",
    )
    register_p.add_argument("--run-name", required=True, help="Run name to record this under.")
    register_p.add_argument("--best-model-dir", required=True, help="Directory from model.save_pretrained().")
    register_p.add_argument("--thresholds-csv", required=True, help="thresholds_<run>.csv from evaluate.py.")
    register_p.add_argument("--run-dir", default=None, help="Training run directory (default: best-model-dir's parent).")
    register_p.add_argument("--predictions-test", default=None, help="predictions_test.npz, if available.")
    register_p.add_argument("--seed", type=int, default=None, help="Seed the run was trained with, for the record.")
    register_p.add_argument(
        "--dry-run", action="store_true",
        help="Print the manifest entries that would be written, without writing them.",
    )

    return parser


def main():
    args = build_arg_parser().parse_args()
    if not hasattr(args, "force"):
        args.force = False  # only 'all' defines --force; it's unused by the single-stage commands
    if not hasattr(args, "source"):
        args.source = None  # only 'score' defines --source; 'all' also calls cmd_score(), so default it

    cfg = load_config(args.config)
    STAGE_FUNCS_WITH_ALL = {**STAGE_FUNCS, "all": cmd_all, "infer": cmd_infer, "register": cmd_register}
    STAGE_FUNCS_WITH_ALL[args.stage](cfg, args)


if __name__ == "__main__":
    main()
