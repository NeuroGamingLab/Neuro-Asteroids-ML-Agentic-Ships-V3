#!/usr/bin/env python3
"""Run the benchmark suite, store the report, and check it against a baseline.

The suite itself lives in training/benchmark.js. This driver does the parts
JavaScriptCore cannot: it writes the JSON report to disk, renders a markdown
table, and compares the run against a committed baseline.

The baseline gate is a regression detector for the *game*, not for policies. It
re-measures the two fixed reference policies (rule-based controller and seeded
random) and fails if either drifts far outside what the stored run implies. A
broken collision rule or a changed spawn table moves those numbers; ordinary
float noise between machines does not.

Usage:
    python3 training/run_benchmark.py --suite quick --check
    python3 training/run_benchmark.py --suite standard --models models/policy_stage3.json
    python3 training/run_benchmark.py --suite quick --update-baseline
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

JSC = ("/System/Library/Frameworks/JavaScriptCore.framework/"
       "Versions/A/Helpers/jsc")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(REPO_ROOT, "training", "benchmarks")
REFERENCE_POLICIES = ("heuristic", "random")


def run_suite(suite, models, scenarios, episodes):
    cmd = [JSC, "training/benchmark.js", "--", "--suite", suite]
    if models:
        cmd += ["--models", ",".join(models)]
    if scenarios:
        cmd += ["--scenarios", ",".join(scenarios)]
    if episodes:
        cmd += ["--episodes", str(episodes)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0 or "BENCH_DONE" not in proc.stdout:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        raise SystemExit("benchmark run failed")

    rows, paired, skipped = [], [], []
    for line in proc.stdout.splitlines():
        if line.startswith("BENCH "):
            rows.append(json.loads(line[len("BENCH "):]))
        elif line.startswith("PAIRED "):
            paired.append(json.loads(line[len("PAIRED "):]))
        elif line.startswith("BENCH_SKIP "):
            skipped.append(json.loads(line[len("BENCH_SKIP "):]))
    return rows, paired, skipped


def build_report(suite, rows, paired, skipped):
    return {
        "format": "asteroids-benchmark",
        "suite": suite,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": rows,
        "paired": paired,
        "skipped": skipped,
    }


def markdown(report):
    out = [f"### Benchmark `{report['suite']}`  ({report['generated']})", "",
           "| Scenario | Policy | Return | Score | Survivors | Kills / 1k steps (ast / enemy / boss) |",
           "| --- | --- | ---: | ---: | ---: | --- |"]
    for r in report["results"]:
        s = r["summary"]
        k = s.get("killsPer1k")
        kills = f"{k['asteroids']:.1f} / {k['enemies']:.1f} / {k['bosses']:.2f}" if k else "-"
        out.append(f"| {r['scenario']} | {r['policy']} | {s['meanReturn']:.2f} ± {s['ciReturn']:.2f} "
                   f"| {s['meanScore']:.0f} | {s['meanSurvivors']:.2f} | {kills} |")
    if report["paired"]:
        out += ["", "| Scenario | Policy | vs heuristic (t) | vs random (t) |", "| --- | --- | --- | --- |"]
        for p in report["paired"]:
            h, r = p["vsHeuristic"], p["vsRandom"]
            out.append(f"| {p['scenario']} | {p['policy']} | {h['diff']:+.2f} (t={h['t']:.2f}"
                       f"{', sig' if h['significant'] else ''}) | {r['diff']:+.2f} (t={r['t']:.2f}"
                       f"{', sig' if r['significant'] else ''}) |")
    for s in report["skipped"]:
        out.append(f"\n_skipped {s.get('path')}: {s['reason']}_")
    return "\n".join(out)


def check_against_baseline(report, baseline, tolerance):
    """Fail on reference-policy drift beyond max(tolerance x baseline, 2 x CI)."""
    base = {(r["scenario"], r["policy"]): r["summary"]
            for r in baseline["results"] if r["policy"] in REFERENCE_POLICIES}
    problems = []
    seen = 0
    for r in report["results"]:
        key = (r["scenario"], r["policy"])
        if key not in base:
            continue
        seen += 1
        old, new = base[key], r["summary"]
        for field, ci_field in (("meanScore", "ciScore"), ("meanLength", None)):
            allowed = max(tolerance * abs(old[field]), 2 * old.get(ci_field, 0) if ci_field else 0, 1.0)
            if abs(new[field] - old[field]) > allowed:
                problems.append(f"{key[0]}/{key[1]} {field}: {old[field]:.1f} -> {new[field]:.1f} "
                                f"(allowed ±{allowed:.1f})")
    if seen == 0:
        problems.append("no reference-policy rows overlap with the baseline")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="standard", choices=["quick", "standard", "full"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="checkpoints to evaluate (default: every models/policy_stage*.json)")
    ap.add_argument("--no-models", action="store_true", help="evaluate only the reference policies")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--markdown", default=None, help="write the markdown table here")
    ap.add_argument("--check", action="store_true", help="compare with the stored baseline, exit 1 on drift")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.25)
    args = ap.parse_args()

    if args.no_models:
        models = []
    elif args.models is None:
        models = sorted(os.path.relpath(p, REPO_ROOT)
                        for p in glob.glob(os.path.join(REPO_ROOT, "models", "policy_stage*.json")))
    else:
        models = args.models

    # Baselines are reference-policy only, so they do not depend on which
    # checkpoints happen to be in models/.
    if args.check or args.update_baseline:
        models = []

    rows, paired, skipped = run_suite(args.suite, models, args.scenarios, args.episodes)
    report = build_report(args.suite, rows, paired, skipped)
    table = markdown(report)
    print(table)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(table + "\n")

    baseline_path = os.path.join(BASELINE_DIR, f"baseline_{args.suite}.json")
    if args.update_baseline:
        os.makedirs(BASELINE_DIR, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nbaseline written: {os.path.relpath(baseline_path, REPO_ROOT)}")
    elif args.check:
        if not os.path.exists(baseline_path):
            raise SystemExit(f"no baseline at {baseline_path}; run with --update-baseline")
        with open(baseline_path) as f:
            baseline = json.load(f)
        problems = check_against_baseline(report, baseline, args.tolerance)
        if problems:
            print("\nBASELINE DRIFT:")
            for p in problems:
                print("  -", p)
            raise SystemExit(1)
        print("\nbaseline check: reference policies within tolerance")


if __name__ == "__main__":
    main()
