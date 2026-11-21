#!/usr/bin/env python3
"""Scan models/policy_*.json and maintain models/registry.json.

Produces a structured model catalog so the Streamlit UI, arena, and CI tools can
inspect compatibility (world, worldScale, obsDim, actionMode, waves, bossPhases,
fuelFeatures) and pick the highest-scoring compatible checkpoint rather than
relying on alphabetical glob ordering.

Usage:
    python3 training/update_registry.py
    python3 training/update_registry.py --model-dir models
"""

import argparse
import glob
import json
import os
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def inspect_checkpoint(abs_path, repo_root=REPO_ROOT):
    with open(abs_path, "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("format") != "asteroids-ppo-policy":
        return None

    rel_path = os.path.relpath(abs_path, repo_root).replace(os.sep, "/")
    meta = payload.get("meta") or {}
    obs_dim = int(payload.get("obsDim", 0))
    feature_names = payload.get("featureNames") or []
    action_mode = payload.get("actionMode") or "flat"
    world = meta.get("world") or ("sphere" if "sphere" in os.path.basename(rel_path) else "torus")
    world_scale = float(meta.get("worldScale", 2.0 if "x2" in os.path.basename(rel_path) else 1.0))
    fuel_features = ("alphaFuelNorm" in feature_names) or bool(meta.get("fuelRescue")) or obs_dim >= 43
    entity_tokens = ("entAst0Dist" in feature_names) or bool(meta.get("entityTokens"))
    waves = bool(meta.get("waves", False))
    boss_phases = bool(meta.get("bossPhases", False))
    centralized_critic = bool(meta.get("centralizedCritic", False))

    eval_return = meta.get("evalMeanReturn")
    heur_return = meta.get("heuristicMeanReturn")
    beats_heuristic = bool(meta.get("beatsHeuristic", False))
    beats_random = bool(meta.get("beatsRandom", False))

    # Preference score used for default selection when multiple checkpoints match
    # the active world: favours sphere x2, full-feature checkpoints, beatsRandom,
    # and higher held-out return.
    rank_score = 0.0
    if beats_heuristic:
        rank_score += 20.0
    if beats_random:
        rank_score += 30.0
    if world == "sphere":
        rank_score += 10.0
    if fuel_features:
        rank_score += 8.0
    if waves:
        rank_score += 6.0
    if boss_phases:
        rank_score += 4.0
    if isinstance(eval_return, (int, float)):
        rank_score += min(50.0, max(-20.0, float(eval_return)))

    return {
        "name": os.path.splitext(os.path.basename(rel_path))[0],
        "path": rel_path,
        "sizeBytes": os.path.getsize(abs_path),
        "createdAt": payload.get("createdAt"),
        "stage": meta.get("stage", 0),
        "agents": meta.get("agents", 3),
        "world": world,
        "worldScale": world_scale,
        "obsDim": obs_dim,
        "actionSize": int(payload.get("actionSize", 48)),
        "actionMode": action_mode,
        "fuelFeatures": fuel_features,
        "entityTokens": entity_tokens,
        "waves": waves,
        "bossPhases": boss_phases,
        "centralizedCritic": centralized_critic,
        "trainedSteps": int(meta.get("trainedSteps", 0)),
        "evalMeanReturn": eval_return,
        "evalMeanScore": meta.get("evalMeanScore"),
        "heuristicMeanReturn": heur_return,
        "actionSelection": meta.get("actionSelection", "stochastic"),
        "beatsHeuristic": beats_heuristic,
        "beatsRandom": beats_random,
        "vsRandomDiff": meta.get("vsRandomDiff"),
        "vsRandomT": meta.get("vsRandomT"),
        "rankScore": round(rank_score, 3),
    }


def build_registry(model_dir, repo_root=REPO_ROOT):
    pattern = os.path.join(model_dir, "policy_*.json")
    entries = []
    for path in sorted(glob.glob(pattern)):
        try:
            entry = inspect_checkpoint(path, repo_root=repo_root)
            if entry:
                entries.append(entry)
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    entries.sort(key=lambda e: (-e["rankScore"], e["path"]))
    default_sphere = next((e["path"] for e in entries if e["world"] == "sphere"), None)
    default_torus = next((e["path"] for e in entries if e["world"] == "torus"), None)
    return {
        "format": "asteroids-model-registry",
        "version": 1,
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "defaults": {
            "sphere": default_sphere,
            "torus": default_torus,
            "overall": default_sphere or default_torus,
        },
        "models": entries,
    }


def write_registry(model_dir=None, repo_root=REPO_ROOT):
    target_dir = model_dir or os.path.join(repo_root, "models")
    os.makedirs(target_dir, exist_ok=True)
    registry = build_registry(target_dir, repo_root=repo_root)
    out_path = os.path.join(target_dir, "registry.json")
    with open(out_path, "w") as handle:
        json.dump(registry, handle, indent=2)
        handle.write("\n")
    return out_path, registry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=os.path.join(REPO_ROOT, "models"))
    args = ap.parse_args()
    out_path, registry = write_registry(args.model_dir)
    print(f"Wrote {out_path} ({len(registry['models'])} checkpoint(s))")
    for m in registry["models"]:
        flags = []
        if m["fuelFeatures"]:
            flags.append("fuel")
        if m["waves"]:
            flags.append("waves")
        if m["bossPhases"]:
            flags.append("boss+")
        if m["actionMode"] != "flat":
            flags.append(m["actionMode"])
        tag = f" [{'+'.join(flags)}]" if flags else ""
        ret = f"{m['evalMeanReturn']:.1f}" if isinstance(m["evalMeanReturn"], (int, float)) else "n/a"
        print(f"  - {m['path']}  ({m['world']} x{m['worldScale']:g}, {m['obsDim']}d{tag}, ret={ret}, rank={m['rankScore']})")


if __name__ == "__main__":
    main()
