#!/usr/bin/env python3
"""Driver for the headless PPO training pipeline.

Runs the JavaScriptCore trainer, streams its output so a long run stays
observable, and captures the two things a training run has to leave behind: a
metrics table you can plot, and the checkpoint that produced the best held-out
score.

JavaScriptCore has no file-write builtin, so the trainer emits structured lines
on stdout (METRIC/EVAL) and delimits its checkpoint with MODEL_BEGIN/MODEL_END.
This script is the piece that turns that stream into files on disk.

Usage:
    python3 training/run_training.py --stage 0 --iterations 300
    python3 training/run_training.py --stage 1 --iterations 400 --init models/policy_stage0.json
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

JSC = (
    "/System/Library/Frameworks/JavaScriptCore.framework/"
    "Versions/A/Helpers/jsc"
)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Arguments forwarded verbatim to train.js.
PASSTHROUGH = [
    "stage", "agents", "iterations", "rollout", "maxSteps", "seed", "lr",
    "criticLr", "entropy", "epochs", "minibatch", "hidden", "evalEvery",
    "evalEpisodes", "bc", "bcSteps", "bcEpochs", "bcLr", "bcAlpha",
    "aimShaping", "init", "fuelRescue", "world", "worldScale",
]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, default=0,
                        help="curriculum stage: 0 asteroids, 1 enemies, 2 boss, 3 full")
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--rollout", type=int, default=2048)
    parser.add_argument("--maxSteps", type=int, default=1000)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--criticLr", type=float, default=1e-3)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--hidden", default="64,64")
    parser.add_argument("--evalEvery", type=int, default=50)
    parser.add_argument("--evalEpisodes", type=int, default=30)
    parser.add_argument("--bc", type=int, default=0,
                        help="1 to behaviour-clone the rule-based controller first")
    parser.add_argument("--bcSteps", type=int, default=30000,
                        help="expert transitions to collect for BC (must match train.js default)")
    parser.add_argument("--bcEpochs", type=int, default=12)
    parser.add_argument("--bcLr", type=float, default=1e-3)
    parser.add_argument("--bcAlpha", type=float, default=0.75)
    # Must match the defaults in training/train.js and sim/env.js.
    parser.add_argument("--aimShaping", type=float, default=0.0,
                        help="potential-based aim shaping weight (0 disables)")
    parser.add_argument("--init", default=None,
                        help="policy JSON to warm start from (for curriculum transfer)")
    parser.add_argument("--fuelRescue", type=int, default=0,
                        help="1 to train with the alpha fuel-rescue mechanic (43-dim observation)")
    parser.add_argument("--world", choices=["torus", "sphere"], default="torus",
                        help="world geometry: flat wrap-around torus (default) or sphere")
    parser.add_argument("--worldScale", type=float, default=1.0,
                        help="sphere only: globe size relative to the 1200x600 chart (2.0 = R 382, 4x surface)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "training_runs"))
    parser.add_argument("--model-dir", default=os.path.join(REPO_ROOT, "models"))
    return parser


def main():
    args = build_parser().parse_args()

    if not os.path.exists(JSC):
        sys.exit(
            f"JavaScriptCore not found at {JSC}.\n"
            "This pipeline uses the JavaScriptCore that ships with macOS."
        )

    run_name = args.run_name or f"stage{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    cmd = [JSC, "training/train.js", "--"]
    values = vars(args)
    for key in PASSTHROUGH:
        value = values.get(key)
        if value is None:
            continue
        cmd += [f"--{key}", str(value)]

    print(f"Run:     {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output:  {run_dir}")
    if args.bc:
        print(f"\n[BC] Behaviour-cloning warmstart enabled  "
              f"(bcSteps={args.bcSteps}, bcEpochs={args.bcEpochs}, bcLr={args.bcLr})")
        print("[BC] Phase 1: collecting expert transitions → pre-training actor ...")
        print("[BC] Phase 2: PPO fine-tuning will follow automatically.\n")
    else:
        print()

    metrics_path = os.path.join(run_dir, "metrics.csv")
    evals_path = os.path.join(run_dir, "evaluations.jsonl")
    log_path = os.path.join(run_dir, "train.log")

    metrics_rows = []
    eval_rows = []
    model_lines = []
    capturing_model = False
    started = time.time()

    process = subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    with open(log_path, "w") as log_file:
        for line in process.stdout:
            line = line.rstrip("\n")

            if line == "MODEL_BEGIN":
                capturing_model = True
                continue
            if line == "MODEL_END":
                capturing_model = False
                continue
            if capturing_model:
                model_lines.append(line)
                continue

            if line.startswith("METRIC "):
                metrics_rows.append(json.loads(line[len("METRIC "):]))
                continue
            if line.startswith("EVAL "):
                eval_rows.append(json.loads(line[len("EVAL "):]))
                continue

            print(line, flush=True)
            log_file.write(line + "\n")

    process.wait()
    elapsed = time.time() - started

    if metrics_rows:
        with open(metrics_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"\nWrote {len(metrics_rows)} iterations to {metrics_path}")

    if eval_rows:
        with open(evals_path, "w") as handle:
            for row in eval_rows:
                handle.write(json.dumps(row) + "\n")
        print(f"Wrote {len(eval_rows)} evaluations to {evals_path}")

    if process.returncode != 0:
        sys.exit(f"Trainer exited with status {process.returncode}")

    if not model_lines:
        sys.exit("No checkpoint was emitted; refusing to report success.")

    payload = json.loads("\n".join(model_lines))
    run_model_path = os.path.join(run_dir, "policy.json")
    with open(run_model_path, "w") as handle:
        json.dump(payload, handle)

    meta = payload.get("meta", {})
    print(f"Saved checkpoint to {run_model_path}")

    # Only promote to models/ if the run actually beat the policy it replaces.
    # A pipeline that always publishes its output teaches you nothing.
    # Sphere runs publish under their own name so they never replace the torus
    # policies the browser loads by default; the world (and its scale) is part
    # of the task, so the checkpoints are not interchangeable.
    if args.world == "sphere":
        scale_tag = f"{args.worldScale:g}".replace(".", "p")
        published = os.path.join(args.model_dir, f"policy_stage{args.stage}_sphere_x{scale_tag}.json")
    else:
        published = os.path.join(args.model_dir, f"policy_stage{args.stage}.json")
    if meta.get("beatsHeuristic"):
        shutil.copyfile(run_model_path, published)
        print(f"Published to {published} (beat the rule-based controller it replaces)")
    else:
        print("NOT published: this run did not significantly beat the rule-based controller.")

    def fmt(value, spec=".2f"):
        return "n/a" if value is None else format(value, spec)

    print("\nSummary")
    print(f"  wall clock         {elapsed:.1f}s")
    print(f"  env steps          {meta.get('trainedSteps', 'n/a')}")
    print(f"  eval mean return   {fmt(meta.get('evalMeanReturn'))}")
    print(f"  eval mean score    {fmt(meta.get('evalMeanScore'), '.0f')}")
    print(f"  action selection   {meta.get('actionSelection', 'n/a')}")
    print("  vs rule-based      "
          f"{fmt(meta.get('evalMeanReturn', 0) - meta.get('heuristicMeanReturn', 0), '+.2f')}"
          f"  ({'significant' if meta.get('beatsHeuristic') else 'not significant'})")
    # Random button-mashing is the harder baseline in this game, because firing
    # costs nothing. Report it whether or not it flatters the run.
    print("  vs random          "
          f"{fmt(meta.get('vsRandomDiff'), '+.2f')}"
          f"  ({'significant' if meta.get('beatsRandom') else 'not significant'})")


if __name__ == "__main__":
    main()
