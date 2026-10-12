#!/usr/bin/env python3
"""Train a small matrix of PPO runs (stages x seeds) and summarise them.

RL_TRAINING.md notes that every published number came from a single training
seed, and that stages 1 and 2 were never trained. This driver closes both gaps:
it runs each requested stage with several seeds, in parallel, through
run_training.py, then writes one summary table with mean +/- spread across seeds
so a hyperparameter or reward change can be judged against seed noise rather
than against one lucky run.

Defaults reproduce the published stage-0 recipe (4096-step rollouts, entropy
0.004, 128x128) warm-started from models/policy_stage0.json.

Usage:
    python3 training/run_matrix.py                       # stages 1,2 x 3 seeds
    python3 training/run_matrix.py --stages 1 --seeds 2 --iterations 100
    python3 training/run_matrix.py --stages 0 1 2 3 --fuelRescue 1 --agents 4 --init none
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_TRAINING = os.path.join(REPO_ROOT, "training", "run_training.py")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, default=3, help="training seeds per stage")
    parser.add_argument("--seed-prefix", default="matrix")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--rollout", type=int, default=4096)
    parser.add_argument("--entropy", type=float, default=0.004)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--hidden", default="128,128")
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--evalEvery", type=int, default=50)
    parser.add_argument("--evalEpisodes", type=int, default=30)
    parser.add_argument("--aimShaping", type=float, default=0.0)
    parser.add_argument("--fuelRescue", type=int, default=0)
    parser.add_argument("--init", default=os.path.join("models", "policy_stage0.json"),
                        help="warm-start checkpoint, or 'none'. Must match the observation layout "
                             "(a 38-dim checkpoint cannot seed a --fuelRescue run).")
    parser.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="concurrent runs; each uses one core")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "training_runs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_one(args, stage, seed_index, matrix_dir):
    seed = f"{args.seed_prefix}-s{stage}-seed{seed_index}"
    run_name = f"matrix_stage{stage}_seed{seed_index}"
    cmd = [
        sys.executable, RUN_TRAINING,
        "--stage", str(stage),
        "--agents", str(args.agents),
        "--iterations", str(args.iterations),
        "--rollout", str(args.rollout),
        "--entropy", str(args.entropy),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--hidden", args.hidden,
        "--evalEvery", str(args.evalEvery),
        "--evalEpisodes", str(args.evalEpisodes),
        "--aimShaping", str(args.aimShaping),
        "--fuelRescue", str(args.fuelRescue),
        "--seed", seed,
        "--run-name", run_name,
        "--out-dir", matrix_dir,
        # Never publish from a matrix run automatically: promotion is a decision
        # to make after reading the whole table, not per run.
        "--model-dir", os.path.join(matrix_dir, "_promoted"),
    ]
    if args.init and args.init != "none":
        cmd += ["--init", args.init]

    started = time.time()
    if args.dry_run:
        print(" ".join(cmd))
        return {"stage": stage, "seed": seed, "run": run_name, "status": "dry-run"}

    log_path = os.path.join(matrix_dir, f"{run_name}.stdout.log")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started

    result = {"stage": stage, "seed": seed, "run": run_name,
              "status": "ok" if proc.returncode == 0 else f"exit {proc.returncode}",
              "minutes": round(elapsed / 60, 1)}
    policy_path = os.path.join(matrix_dir, run_name, "policy.json")
    if os.path.exists(policy_path):
        with open(policy_path) as handle:
            meta = json.load(handle).get("meta", {})
        result.update({
            "steps": meta.get("trainedSteps"),
            "evalReturn": meta.get("evalMeanReturn"),
            "evalScore": meta.get("evalMeanScore"),
            "heuristicReturn": meta.get("heuristicMeanReturn"),
            "beatsHeuristic": meta.get("beatsHeuristic"),
            "vsRandomDiff": meta.get("vsRandomDiff"),
            "vsRandomT": meta.get("vsRandomT"),
            "beatsRandom": meta.get("beatsRandom"),
            "mode": meta.get("actionSelection"),
        })
    return result


def fmt(value, spec=".2f"):
    return "n/a" if value is None else format(value, spec)


def summarise(results, matrix_dir, args):
    lines = ["# PPO training matrix", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')}  ",
             f"iterations={args.iterations} rollout={args.rollout} entropy={args.entropy} "
             f"lr={args.lr} hidden={args.hidden} agents={args.agents} "
             f"fuelRescue={args.fuelRescue} init={args.init}", ""]

    lines += ["## Per run", "",
              "| stage | seed | status | env steps | eval return | score | vs rule-based | vs random (t) | mode | min |",
              "|---:|---|---|---:|---:|---:|---|---|---|---:|"]
    for r in sorted(results, key=lambda x: (x["stage"], x["seed"])):
        vs_h = "n/a"
        if r.get("evalReturn") is not None and r.get("heuristicReturn") is not None:
            vs_h = f"{r['evalReturn'] - r['heuristicReturn']:+.2f}" + (" *" if r.get("beatsHeuristic") else "")
        vs_r = "n/a"
        if r.get("vsRandomDiff") is not None:
            vs_r = f"{r['vsRandomDiff']:+.2f} ({fmt(r.get('vsRandomT'))})" + (" *" if r.get("beatsRandom") else "")
        lines.append(
            f"| {r['stage']} | {r['seed']} | {r['status']} | {r.get('steps') or 'n/a'} | "
            f"{fmt(r.get('evalReturn'))} | {fmt(r.get('evalScore'), '.0f')} | {vs_h} | {vs_r} | "
            f"{r.get('mode') or 'n/a'} | {r.get('minutes', 'n/a')} |")
    lines += ["", "`*` = significant at the 5% level on paired held-out seeds.", ""]

    lines += ["## Across seeds", "",
              "| stage | seeds | eval return mean | std | min | max | beat rule-based | beat random |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stage in sorted({r["stage"] for r in results}):
        rows = [r for r in results if r["stage"] == stage and r.get("evalReturn") is not None]
        if not rows:
            lines.append(f"| {stage} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        vals = [r["evalReturn"] for r in rows]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lines.append(
            f"| {stage} | {len(rows)} | {statistics.mean(vals):.2f} | {std:.2f} | {min(vals):.2f} | "
            f"{max(vals):.2f} | {sum(1 for r in rows if r.get('beatsHeuristic'))}/{len(rows)} | "
            f"{sum(1 for r in rows if r.get('beatsRandom'))}/{len(rows)} |")

    lines += ["", "Checkpoints: `<matrix_dir>/matrix_stage<N>_seed<k>/policy.json`. Nothing is copied to "
              "`models/` automatically; promote the best seed by hand once the table has been read.", ""]

    text = "\n".join(lines)
    with open(os.path.join(matrix_dir, "summary.md"), "w") as handle:
        handle.write(text)
    with open(os.path.join(matrix_dir, "summary.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    print("\n" + text)


def main():
    args = build_parser().parse_args()
    if args.fuelRescue and args.init and args.init != "none":
        try:
            with open(os.path.join(REPO_ROOT, args.init)) as handle:
                if json.load(handle).get("obsDim") != 43:
                    sys.exit(f"--init {args.init} is not a fuel-feature (43-dim) checkpoint; "
                             "pass --init none for the first fuel run.")
        except OSError as exc:
            sys.exit(f"cannot read --init {args.init}: {exc}")

    matrix_dir = os.path.join(args.out_dir, f"matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(matrix_dir, exist_ok=True)
    jobs = [(stage, k) for stage in args.stages for k in range(args.seeds)]
    print(f"{len(jobs)} runs, {args.parallel} in parallel -> {matrix_dir}\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, args, stage, k, matrix_dir): (stage, k) for stage, k in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"done  stage {result['stage']}  {result['seed']}  {result['status']}  "
                  f"return {fmt(result.get('evalReturn'))}", flush=True)

    if not args.dry_run:
        summarise(results, matrix_dir, args)


if __name__ == "__main__":
    main()
