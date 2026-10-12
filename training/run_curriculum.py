#!/usr/bin/env python3
"""Train a full curriculum chain on the sphere world, several seeds in parallel.

Each seed runs stage 0 -> 1 -> 2 -> 3 (-> swarm), every stage warm-started from
the same seed's previous stage, through run_training.py. When every chain has
finished, the benchmark suite scores each seed's final checkpoint on the same
held-out seeds, so the published model is chosen on measured results and the
spread across seeds is reported rather than hidden.

Why this exists: the checkpoints trained before the aim-gate and action-safety
changes (see the comment above isAimedAtTarget in game.js) were evaluated under a
different effective action space than they were trained on. Retraining under the
current rules, on the world the app defaults to, is the only way to get numbers
that describe what players actually see.

Usage:
    python3 training/run_curriculum.py --tag r1 --seeds 4
    python3 training/run_curriculum.py --tag smoke --seeds 1 --iterations 3 --stages 0 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_TRAINING = os.path.join(REPO_ROOT, "training", "run_training.py")
RUN_BENCHMARK = os.path.join(REPO_ROOT, "training", "run_benchmark.py")

# 'swarm' is stage 3 with a fleet-sized opposition, matching the app defaults far
# better than the 2 enemies + 1 boss of the standard stage 3.
SWARM = {"stage": 3, "agents": 6, "enemies": 4, "bosses": 2}


def stage_args(stage):
    if stage == "swarm":
        return SWARM
    return {"stage": int(stage)}


def train_chain(seed_index, args):
    run_root = os.path.join(args.out_dir, f"curriculum_{args.tag}")
    init = args.init if args.init != "none" else None
    last_policy = None
    for stage in args.stages:
        spec = stage_args(stage)
        name = f"seed{seed_index}_stage{stage}"
        run_dir = os.path.join(run_root, name)
        cmd = [sys.executable, RUN_TRAINING,
               "--run-name", name, "--out-dir", run_root,
               "--model-dir", os.path.join(run_root, "published"),  # never touch models/
               "--seed", f"{args.tag}-s{seed_index}",
               "--iterations", str(args.iterations), "--rollout", str(args.rollout),
               "--entropy", str(args.entropy), "--lr", str(args.lr),
               "--epochs", str(args.epochs), "--hidden", args.hidden,
               "--evalEvery", str(args.evalEvery), "--evalEpisodes", str(args.evalEpisodes),
               "--world", "sphere", "--worldScale", str(args.worldScale)]
        for k, v in spec.items():
            cmd += [f"--{k}", str(v)]
        warm = last_policy or init
        if warm:
            cmd += ["--init", warm]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "driver.log"), "w") as f:
            f.write(proc.stdout + proc.stderr)
        policy = os.path.join(run_dir, "policy.json")
        if proc.returncode != 0 or not os.path.exists(policy):
            return seed_index, None, f"stage {stage} failed (see {run_dir}/driver.log)"
        last_policy = policy
        print(f"  seed {seed_index} stage {stage} done", flush=True)
    return seed_index, last_policy, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="names the output dir and seeds the runs")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stages", nargs="+", default=["0", "1", "2", "3", "swarm"])
    ap.add_argument("--iterations", type=int, default=250, help="per stage")
    ap.add_argument("--rollout", type=int, default=4096)
    ap.add_argument("--entropy", type=float, default=0.004)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--evalEvery", type=int, default=50)
    ap.add_argument("--evalEpisodes", type=int, default=30)
    ap.add_argument("--worldScale", type=float, default=2.0)
    ap.add_argument("--init", default="none", help="optional warm start for stage 0")
    ap.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "training_runs"))
    ap.add_argument("--bench-suite", default="standard", choices=["quick", "standard", "full"])
    args = ap.parse_args()

    started = time.time()
    print(f"curriculum '{args.tag}': {args.seeds} seeds x stages {args.stages}, "
          f"{args.iterations} iterations each, {args.parallel} in parallel", flush=True)

    finals = {}
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(train_chain, i, args) for i in range(args.seeds)]
        for fut in as_completed(futures):
            idx, policy, err = fut.result()
            if err:
                print(f"  seed {idx}: {err}", flush=True)
            else:
                finals[idx] = policy

    if not finals:
        sys.exit("every chain failed")

    run_root = os.path.join(args.out_dir, f"curriculum_{args.tag}")
    print(f"\nbenchmarking {len(finals)} final checkpoints "
          f"(suite={args.bench_suite}, took {time.time() - started:.0f}s so far)", flush=True)
    models = [os.path.relpath(p, REPO_ROOT) for _, p in sorted(finals.items())]
    out_json = os.path.join(run_root, "benchmark.json")
    out_md = os.path.join(run_root, "benchmark.md")
    subprocess.run([sys.executable, RUN_BENCHMARK, "--suite", args.bench_suite,
                    "--models", *models, "--out", out_json, "--markdown", out_md],
                   cwd=REPO_ROOT, check=True)
    print(f"\nreport: {out_md}\ntotal wall clock {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
