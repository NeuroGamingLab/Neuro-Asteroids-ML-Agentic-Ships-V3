# Training, benchmarks and checks

Everything here runs on the JavaScriptCore binary that ships with macOS. There is
nothing to install for the RL side.

## Which command do I want?

| I want to... | Run |
| --- | --- |
| run every unit test and syntax check | `./run_tests.sh` |
| measure policies on fixed scenarios | `python3 training/run_benchmark.py --suite standard` |
| check I did not break the game | `python3 training/run_benchmark.py --suite quick --check` |
| train a curriculum, several seeds | `python3 training/run_curriculum.py --tag r1 --seeds 4` |
| train one stage | `python3 training/run_training.py --stage 3 --world sphere --worldScale 2` |
| drive the real game in a browser | `python3 tests/run_browser_smoke.py` |
| check the Streamlit page template | `python3 tests/test_app_template.py` |

## Benchmark suite

`training/benchmark.js` (driven by `run_benchmark.py`) plays **fixed scenarios on
held-out seeds** under identical conditions for every policy, so comparisons are
paired and meaningful.

| Scenario | What it is |
| --- | --- |
| `s0_torus` | asteroids only, flat torus |
| `s3_torus` | asteroids + enemies + one boss, flat torus |
| `s3_sphere` | the same on the x2 globe (the app's default world) |
| `swarm_sphere` | 6 agents vs 4 enemies + 2 bosses on the x2 globe |

Suites: `quick` (2 scenarios, 6 episodes, 500 steps - about a second), `standard`
(3 scenarios, 20 episodes) and `full` (all four, 40 episodes).

Every scenario always includes two reference policies, the rule-based controller
(`heuristic`) and a seeded `random` policy. Random is the harder baseline in this
game because ammunition is free, so a learned policy is only credited when it beats
*both*. For every policy the report gives return, score, survivors and **kills per
1000 steps** (asteroids / enemies / bosses), and for checkpoints the paired
difference against each reference with its t-statistic.

```bash
# evaluate specific checkpoints (default: every models/policy_stage*.json)
python3 training/run_benchmark.py --suite standard --models models/policy_stage3.json \
    --out benchmark.json --markdown benchmark.md
```

A checkpoint whose observation layout or action space does not match the running
build is reported as skipped rather than silently dropped.

### The regression gate

`--check` re-measures only the two reference policies and compares them with
`training/benchmarks/baseline_<suite>.json`. It fails if a mean score or episode
length moves by more than `max(25%, 2 x CI)`. That is a detector for the **game**,
not for policies: a broken collision rule, a changed spawn table or a physics
regression moves those numbers, ordinary float noise does not. CI runs it on every
push.

When you change gameplay on purpose, regenerate and commit the baseline:

```bash
python3 training/run_benchmark.py --suite quick --update-baseline
```

Waves and the boss escort/enrage mechanics are **off** in the headless env by
default, so the baseline and all published training numbers keep describing the
same game. Pass `waves: true` / `bossPhases: true` to `AsteroidsEnv` to train or
evaluate against them.

## Curriculum training

`run_curriculum.py` trains stage 0 -> 1 -> 2 -> 3 -> `swarm` on the sphere world,
each stage warm-started from the same seed's previous stage, several seeds in
parallel, then benchmarks every seed's final checkpoint together. `swarm` is stage 3
with a fleet-sized opposition (6 agents, 4 enemies, 2 bosses), much closer to the
app defaults than the standard 2 + 1.

It never writes into `models/`; runs land in `training_runs/curriculum_<tag>/` with
a `benchmark.md` report. Promote a checkpoint by copying it yourself once the
numbers justify it.

> `models/` and `training_runs/` are git-ignored on purpose: weights are build
> outputs. The benchmark baselines under `training/benchmarks/` are the exception,
> because CI needs them.

## What the other checks cover

* `tests/run_browser_smoke.py` opens the real `index.html` in headless Chrome (no
  Playwright or Selenium), runs the actual update/draw loop on the globe, exercises
  the Streamlit settings hooks, the game-over flow, waves and the boss bar, and
  fails on any uncaught page error. It checks the namespacing too: internals must
  not be visible as bare identifiers, shared state must be.
* `tests/test_app_template.py` extracts `get_game_html()` from `app.py` with `ast`,
  calls it with a spread of settings (including hostile strings) and parse-checks
  every injected script with JavaScriptCore. A stray `{` in the f-string is a blank
  game in production; this catches it.
* `training/test_waves.js` covers the wave schedule, ceilings, determinism,
  restart, idempotent toggling and the boss escort/enrage mechanics.

## game.js is namespaced

`game.js` runs inside one function scope. Roughly 70 of its names are the public
surface other scripts use; they are declared once in the block at the top of the
file (accessor properties for shared mutable state, so `aiShips`, `score`,
`numAIShips` and friends still read and assign as bare names elsewhere). Everything
else is private. If a script needs a new name from `game.js`, add it to that block -
the browser smoke test will tell you if you forgot.

## Results: curriculum run `r1` (sphere x2, 4 seeds)

Recipe: `python3 training/run_curriculum.py --tag r1 --seeds 4` (stages 0, 1, 2, 3,
`swarm`; 250 iterations of 4096 steps each; 128x128 actor and critic; entropy 0.004;
lr 3e-4; about 90 minutes wall clock on 4 cores). Scored with the `full` suite on
fresh held-out seeds (`--seedBase 1500000`, 40 episodes), after the aim-gate and
action-safety rules, which is the game players see. Raw numbers:
`training/benchmarks/curriculum_r1_full.json`.

Mean return (± 95% CI); the three right-hand columns are the old shipped checkpoints
scored under today's rules.

| Scenario | heuristic | random | **seed 0** | seed 1 | seed 2 | seed 3 | old stage0 | old stage3 | old stage0 sphere |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s0_torus` | 15.8 | 42.3 | **73.4** | 54.1 | 73.1 | 73.9 | 44.4 | 43.7 | 42.5 |
| `s3_torus` | 7.1 | 20.7 | **47.4** | 49.5 | 46.1 | 45.6 | 16.9 | 23.0 | 24.6 |
| `s3_sphere` | 4.6 | 11.9 | **29.6** | 28.7 | 29.7 | 30.1 | 11.7 | 13.3 | 17.7 |
| `swarm_sphere` | 5.8 | 13.2 | **33.0** | 29.2 | 28.8 | 30.4 | 11.1 | 11.9 | 15.7 |

* Every retrained seed beats both baselines in every scenario (paired t = 5.0 to
  16.0 against random, so well past the 1.96 cutoff). In the fleet-sized
  `swarm_sphere` scenario seed 0 scores 2.5x random and 5.7x the rule-based
  controller, kills 1.6 bosses per 1000 steps against 0.8, and keeps 4.6 of 6 ships
  alive against 3.7.
* The old checkpoints are statistically indistinguishable from random in most
  scenarios now (t between -1.6 and 1.7; the one exception is the old sphere
  checkpoint on its own world, t = 4.5 on `s3_sphere` and 2.0 on `swarm_sphere`).
  That is what the changed action space did to them, and why a retrain was needed.
* The four seeds agree closely (28.7 to 30.1 on `s3_sphere`), with seed 1 the weak
  one on `s0_torus`. Seeds 0, 2 and 3 are within noise of each other; seed 0 was
  picked for the best mean return across scenarios and ships as
  `models/policy_swarm_sphere_x2.json`. Selection used one set of held-out seeds and
  these numbers come from another, so they are not inflated by the choice.

Caveats: the numbers are for the game with waves and boss escorts **off** (that is
what the trainer and benchmark play). The final stage trained 6 ships against 4
enemies and 2 bosses; behaviour with 10 AI ships against escalating waves has not
been measured.

