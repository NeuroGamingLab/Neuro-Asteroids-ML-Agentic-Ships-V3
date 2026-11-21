#!/usr/bin/env python3
"""Driver for Adversarial Self-Play League training (training/self_play.js).

Captures league round metrics and writes the resulting MAPPO/PPO checkpoint to
training_runs/selfplay_<tag>/policy.json (and promotes to models/ when --promote
is passed and the checkpoint beats the rule-based baseline).

Usage:
    python3 training/run_self_play.py --tag sp1 --rounds 4 --itersPerRound 20
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
from run_training import resolve_js_command  # noqa: E402
from update_registry import write_registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--itersPerRound", type=int, default=15)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--enemies", type=int, default=3)
    ap.add_argument("--bosses", type=int, default=1)
    ap.add_argument("--world", choices=["torus", "sphere"], default="sphere")
    ap.add_argument("--worldScale", type=float, default=2.0)
    ap.add_argument("--fuelRescue", type=int, default=0)
    ap.add_argument("--actionMode", choices=["flat", "factored"], default="factored")
    ap.add_argument("--centralizedCritic", type=int, default=1)
    ap.add_argument("--creditMode", choices=["ippo", "vdn", "coma"], default="coma")
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--init", default="none")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(REPO_ROOT, "training_runs", f"selfplay_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = resolve_js_command("training/self_play.js") + [
        "--rounds", str(args.rounds),
        "--itersPerRound", str(args.itersPerRound),
        "--rollout", str(args.rollout),
        "--agents", str(args.agents),
        "--enemies", str(args.enemies),
        "--bosses", str(args.bosses),
        "--world", args.world,
        "--worldScale", str(args.worldScale),
        "--fuelRescue", str(args.fuelRescue),
        "--actionMode", args.actionMode,
        "--centralizedCritic", str(args.centralizedCritic),
        "--creditMode", args.creditMode,
        "--hidden", args.hidden,
        "--init", args.init,
    ]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    model_lines, league_rows, capturing = [], [], False
    with open(os.path.join(out_dir, "train.log"), "w") as logf:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line == "MODEL_BEGIN":
                capturing = True
                continue
            if line == "MODEL_END":
                capturing = False
                continue
            if capturing:
                model_lines.append(line)
                continue
            if line.startswith("LEAGUE_ROUND "):
                league_rows.append(json.loads(line[len("LEAGUE_ROUND "):]))
            print(line, flush=True)
            logf.write(line + "\n")
    proc.wait()
    if proc.returncode != 0 or not model_lines:
        sys.exit(f"Self-play trainer failed (code={proc.returncode})")

    payload = json.loads("\n".join(model_lines))
    policy_path = os.path.join(out_dir, "policy.json")
    with open(policy_path, "w") as f:
        json.dump(payload, f)
    with open(os.path.join(out_dir, "league.json"), "w") as f:
        json.dump(league_rows, f, indent=2)
    print(f"Saved self-play checkpoint to {policy_path}")

    if args.promote and payload.get("meta", {}).get("beatsHeuristic"):
        dest = os.path.join(REPO_ROOT, "models", f"policy_selfplay_{args.tag}.json")
        shutil.copyfile(policy_path, dest)
        write_registry(os.path.join(REPO_ROOT, "models"))
        print(f"Promoted to {dest} and refreshed models/registry.json")


if __name__ == "__main__":
    main()
