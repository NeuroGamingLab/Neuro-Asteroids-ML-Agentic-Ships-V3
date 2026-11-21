# Asteroids-AI — ML App Enhancement Review (v2.0.0)

| | |
|---|---|
| **Document version** | 2.0.0 |
| **Date** | 2026-11-20 |
| **Reviewer model** | Google Gemini (Antigravity) |
| **Repository state** | branch `next-steps` @ `793c247` (App v2.5.1) |
| **Supersedes / Builds on** | [2026-09-05-claude-fable-5.1-asteroids-ai-enhancement-v1.0.0.md](2026-09-05-claude-fable-5.1-asteroids-ai-enhancement-v1.0.0.md) |
| **Scope** | Correctness & train/serve parity, RL/MARL algorithms, 3D rendering & UX, Streamlit architecture, repository hygiene, CI & cloud deployment |
| **Baseline verification** | `./run_tests.sh` (`ALL TESTS PASSED`), `tests/test_app_template.py` (`ALL TEMPLATE CHECKS PASSED`) |

---

## Progress Since v1.0.0

Substantial engineering work has landed since the September 2026 review:

- **3D Spherical World & Visual Feel**: Exact great-circle transport (`world.js`), decoupled logical globe scale (`2×` default), Three.js 3D ship/boss hull meshes, glow halos, shield domes, hit flashes, SFX/ambient audio, boss-follow camera, and kill-cam (`sphere_view.js`, `fx.js`).
- **Gameplay Depth**: Escalating wave system and 3-phase boss encounters (escort summons at Phase 2, enrage at Phase 3) covered by `training/test_waves.js`.
- **RL Curriculum & Benchmarking**: Multi-seed curriculum training (`training/run_curriculum.py`, run `r1`) producing `models/policy_swarm_sphere_x2.json` (**2.5× random** and **5.7× heuristic** on `swarm_sphere`, paired $t \in [5.0, 16.0]$ across all 4 seeds), plus regression-gated CI benchmarks (`training/run_benchmark.py --check`).
- **Code & Test Hardening**: `game.js` namespaced behind an explicit public surface, plus automated headless Chrome smoke testing (`tests/run_browser_smoke.py`) and AST-based Streamlit template syntax validation (`tests/test_app_template.py`).

---

## Executive Summary

With the core 3D simulator, PPO pipeline, and CI regression gates now solid, the next major leaps come from fixing **silent train/serve mismatches** and upgrading **coordination, interactivity, and maintainability**:

1. **Geometry & Train/Serve Parity**: Relative features in `ml_features.js` and rewards in `sim/env.js` still compute flat Euclidean chart deltas instead of great-circle `WORLD.delta()`/`WORLD.distance()`, creating seam/polar blind spots on the default 3D globe. Meanwhile, default Streamlit settings run waves, boss phases, fuel rescue, and legacy non-learning MARL/TF.js code against a 38-dim non-fuel, non-wave checkpoint.
2. **RL / MARL Ceiling**: PPO remains flat-vector IPPO over single-nearest entities with a 48-way categorical head and post-hoc VDN logging. Moving to a **43-dim wave/fuel curriculum**, **factored multi-discrete actions**, **centralized critic (MAPPO)**, and **multi-entity attention** is the clearest path to genuine swarm tactics.
3. **UI, 3D UX & Architecture**: `app.py` drops browser telemetry packets (one-way `components.html`), tears down the iframe on every widget change, spawns six 2-second `setInterval` loops, and passes 50 arguments into a ~1.7 MB inline `srcdoc` payload. Replacing this with a bidirectional component + interactive 3D camera will unlock live inspection and zero-reset configuration.

---

## 1. High-Impact Quick Wins (Bugs & Mismatches in Current Code)

### 1.1 Route `ml_features.js` and `sim/env.js` Through Geodesic `WORLD.delta()` / `WORLD.distance()`
- **Impact**: High | **Effort**: Low
- **Where**: `ml_features.js` (`L53–L75`, `L77–L95`, `L119–L149`, `L255–L263`), `sim/env.js` (`L351–L374`), `world.js` (`L181–L208`)
- **Finding**: `world.js` already implements exact great-circle `WORLD.distance(x1, y1, x2, y2)` and local tangent-frame `WORLD.delta(fx, fy, tx, ty)`. However, `MLFeatureExtractor.findNearestEntity()`, `attentionThreat()`, ally proximity, formation error, and tanker bearing still compute flat `entity.x - agent.x` and `entity.y - agent.y`. `sim/env.js` does the same for `allyProximity` and `centerCampingPenalty`.
- **Consequence**: On the default 3D sphere, targets across the antimeridian ($x \approx 0$ vs. $x \approx W$) or near the poles appear artificially distant and at incorrect bearings to the neural network.
- **Recommendation**: Switch all relative distance/angle lookups in `ml_features.js` and `sim/env.js` to `WORLD.delta()` / `WORLD.distance()` (zero overhead on torus, where `WORLD.delta` returns plain `{ dx, dy }`). This eliminates spherical distortion and allows one policy to generalize cleanly across both worlds.

### 1.2 Clean Up Noisy & Off-Distribution Defaults in `app.py`
- **Impact**: High | **Effort**: Low
- **Where**: `app.py` (`L174–L194`, `L349–L415`, `L577–L580`)
- **Findings**:
  1. **Dead/scaffolding flags enabled by default**: `marl_enabled: True`, `marl_training: True`, `nn_models_enabled: True`, `ml_enabled: True`, and `ml_data_collection: True` all start ON. As documented in `marl_system.js` (`L1–L26`), `marl_training` fills an unused replay buffer, and `nn_models_enabled` unconditionally injects ~1.5 MB of TensorFlow.js/WASM CDN scripts and attempts to fetch a non-existent `/models/multitask_model/model.json`.
  2. **Suboptimal inference default**: `trained_policy_deterministic` defaults to `True`, even though the sidebar tooltip and `RL_TRAINING.md` demonstrate stochastic sampling scores higher than deterministic argmax.
  3. **Scenario vs. checkpoint mismatch**: Default `DEFAULTS` sets `10 AI, 10 enemies, 10 bosses`, `waves_enabled: True`, `boss_phases_enabled: True`, and `fuel_rescue_enabled: True`, while `models/policy_swarm_sphere_x2.json` is a 38-dim (non-fuel) checkpoint trained on `6 agents, 4 enemies, 2 bosses` with waves and boss phases disabled.
- **Recommendation**:
  - Set legacy TF.js / MARL scaffolding flags to `False` by default and conditionally include `TFJS_TAG` / `TFJS_WASM_TAG` only when `nn_models_enabled` is turned on.
  - Default `trained_policy_deterministic` to `False`.
  - Display a non-blocking compatibility badge in the sidebar whenever the active world mode, scale, or fuel-rescue toggle differs from the selected checkpoint's `meta`.

### 1.3 Support (or Properly Gate) the Critic Value Heatmap in 3D Sphere Mode
- **Impact**: Medium | **Effort**: Low–Medium
- **Where**: `game.js` (`L2997–L3009`, `L3037`), `policy_viz.js`, `app.py` (`L189`, `L465`), `index.html` (`L115`)
- **Finding**: `value_heatmap` defaults to `True` in both `app.py` and `index.html`, but `game.js` only invokes `drawValueHeatmap(ctx)` inside the 2D torus branch (`L3037`). In the default 3D sphere view, the toggle silently does nothing.
- **Recommendation**: Either render the critic value field onto the visible hemisphere in `sphere_view.js` (e.g., a low-poly vertex-colored spherical mesh patch updated incrementally) or disable/annotate the toggle in 3D mode so UI state matches what is drawn.

### 1.4 Replace the Six `setInterval(..., 2000)` Polling Loops in `app.py`
- **Impact**: Medium | **Effort**: Low
- **Where**: `app.py` (`L895–L1140`)
- **Finding**: Six independent 2-second `setInterval` loops repeatedly re-apply Streamlit widget settings inside the iframe. This pattern already caused the historical Alpha fuel-link freeze bug and adds unnecessary periodic state churn.
- **Recommendation**: Serialize all settings into a single `window.__ASTEROIDS_CONFIG__` JSON payload applied once at initialization (and via `postMessage` once a bidirectional component is added).

### 1.5 Fix IMDSv1 Bug & Dependency Drift in `terraform/aws/user_data.sh`
- **Impact**: Medium | **Effort**: Low
- **Where**: `terraform/aws/main.tf` (`L142–L145`), `terraform/aws/user_data.sh` (`L42`, `L168`), `Dockerfile`
- **Finding**: `main.tf` enforces IMDSv2 (`http_tokens = "required"`), so the unauthenticated IMDSv1 `curl http://169.254.169.254/latest/meta-data/public-ipv4` in `user_data.sh` (`L168`) fails silently. Additionally, `user_data.sh` installs unpinned `streamlit>=1.28.0` on bare EC2 rather than using the pinned `Dockerfile` (`streamlit==1.40.2`).
- **Recommendation**: Run the existing `Dockerfile` container on EC2 (or pin `requirements.txt` and switch the MOTD lookup to an IMDSv2 session token).

---

## 2. Reinforcement Learning & Multi-Agent AI Enhancements

### 2.1 Train a Full-Feature 43-Dim Curriculum Policy (Waves + Boss Phases + Fuel Rescue)
- **Impact**: High | **Effort**: Medium
- **Where**: `training/run_curriculum.py`, `sim/env.js`, `models/`
- **Finding**: All 6 published checkpoints in `models/` are 38-dim (non-fuel-aware) and trained with `waves: false` and `bossPhases: false`—whereas the live app defaults to all three mechanics enabled.
- **Recommendation**: Add a final curriculum stage in `run_curriculum.py` with `fuelRescue: true`, `waves: true`, and `bossPhases: true` on `sphere x2`, including a dedicated benchmark scenario in `training/benchmark.js` measuring refuel survival rate, dock-death rate, and wave survival alongside score.

### 2.2 Upgrade from Flat-Vector IPPO to Centralized Critic (MAPPO) + Multi-Entity Attention
- **Impact**: High | **Effort**: High
- **Where**: `training/ppo.js`, `training/nn.js`, `ml_features.js`
- **Findings**:
  - `PPOTrainer.decomposeTeamValue()` (`training/ppo.js`, `L92–L107`) computes VDN team value and counterfactual credits only as post-hoc metrics; training remains standard IPPO with fixed `teamShare = 0.3`.
  - `ml_features.js` exposes only the *single* nearest asteroid, enemy, boss, and ally, plus a single binary `bulletThreat` bit. Agents cannot reason about incoming bullet curtains, flanking enemies, or coordinated target allocation.
- **Recommendation**:
  - **Centralized Critic (CTDE / MAPPO)**: Provide the critic with global/team features (all allies' positions, health, targets, and Alpha fuel state) during training while keeping actors fully decentralized for browser inference.
  - **Entity-Set Encoder**: Replace single-nearest slots with top-$k$ entity tokens (e.g., 3 asteroids, 3 enemies/bosses, 2 incoming bullets with relative velocities, 2 allies) aggregated via a lightweight DeepSets or single-head attention pooling layer fast enough for Web Worker inference.
  - **Learned Credit Assignment**: Incorporate true VDN/QMIX or COMA counterfactual baselines directly into advantage estimation.

### 2.3 Factored Multi-Discrete Policy Head & PPO Stabilizers
- **Impact**: High | **Effort**: Medium
- **Where**: `training/ppo.js`, `training/nn.js`, `game.js` (`L777–L781`), `policy_viz.js`
- **Findings**:
  - The 48-action flat categorical head (`3 rotation × 2 thrust × 4 fire × 2 shield`) treats correlated sub-actions as completely independent classes.
  - `training/ppo.js` lacks linear LR/entropy annealing, value-function clipping, orthogonal initialization, and return/observation normalization. *(Minor doc cleanup: `training/ppo.js` header comment `L6–L8` still says `24` actions instead of `48`.)*
- **Recommendation**:
  - Replace the 48-logit output layer with 4 factored heads (`3 + 2 + 4 + 2 = 11` logits). Joint log-probability is the sum of factor log-probs, exploration converges faster, and `policy_viz.js` can read factor marginals directly.
  - Add cosine/linear LR and entropy decay schedules, orthogonal initialization (`gain = 0.01` on policy head, `1.0` on value head), and clipped value loss.

### 2.4 Cross-Platform & Vectorized Training Throughput
- **Impact**: High | **Effort**: Medium–High
- **Where**: `run_tests.sh`, `training/*.py`, `training/*.js`, `.github/workflows/ci.yml`
- **Finding**: Running pure-JS `Float64Array` loops inside macOS-only `jsc` takes ~90 minutes for 4 curriculum seeds and forces CI onto `macos-latest` runners.
- **Recommendation**:
  - Add a tiny runtime shim supporting **Node.js / Bun** in addition to `jsc` so tests and training run on Linux/Windows and cheaper Linux CI runners.
  - For larger architectures (MAPPO + attention), expose `sim/env.js` through a vectorized worker-thread server feeding a PyTorch/JAX CleanRL trainer, exporting checkpoints back to compact JSON or INT8 ONNX (`onnxruntime-web`).

### 2.5 Adversarial Enemy Self-Play & Tanker Obstacle Avoidance
- **Impact**: Medium–High | **Effort**: Medium–High
- **Where**: `ai_ship.js`, `enemy_ship.js`, `training/`
- **Recommendation**:
  - Add basic asteroid/bullet avoidance steering to `handleFuelShipBehavior()` in `ai_ship.js` so the scripted green tanker stops colliding head-on with asteroids during "Fuel Crisis".
  - Introduce a **Red Swarm / Boss PPO self-play league** trained against historical Blue checkpoints to prevent Blue policies from overfitting to deterministic enemy heuristics.

---

## 3. 3D Graphics, Interactive Inspection & Streamlit UX

### 3.1 Interactive 3D Globe Camera & Click-to-Inspect Agent Telemetry
- **Impact**: High | **Effort**: Medium
- **Where**: `sphere_view.js` (`L119`, `L826–L874`, `L1118`)
- **Finding**: Both `glCanvas` and `overlayCanvas` set `pointerEvents = 'none'`, limiting camera control to sidebar presets (`Follow Alpha`, `Follow fleet`, `Follow boss`, `Slow orbit`).
- **Recommendation**: Add mouse drag-to-orbit, scroll-wheel zoom, double-click reset, and **raycast click-on-ship** to lock the camera onto any agent and display a compact inspector card showing its live role, fuel, critic value $V(s)$, entropy, and top-3 action probabilities.

### 3.2 True Bidirectional Streamlit Bridge (Zero-Reload Settings + Live Python Charts)
- **Impact**: High | **Effort**: Medium–High
- **Where**: `app.py` (`L417–L445`, `L1166`), `hud.js` (`L386–L412`)
- **Finding**: `app.py` describes "Bidirectional telemetry streaming" and `hud.js` posts `ASTEROIDS_TELEMETRY` via `window.parent.postMessage(..., '*')` every 10 frames—but `st.components.v1.html` is strictly one-way. Packets are silently dropped, Mission Control controls can only live inside the iframe, and moving any sidebar slider destroys and restarts the entire game.
- **Recommendation**: Migrate to a lightweight custom Streamlit component (`declare_component` + `Streamlit.setComponentValue`) so:
  - Sidebar sliders/toggles push live updates via `postMessage` **without resetting the running simulation**.
  - Python receives throttled (~2 Hz) telemetry packets for live Plotly/Altair charts, VDN counterfactual credit breakdowns, and a training-monitor tab reading `training_runs/*/metrics.csv`.

### 3.3 WebGL Performance & Modern Three.js Upgrade
- **Impact**: Medium | **Effort**: Medium
- **Where**: `sphere_view.js` (`L129`), `app.py` (`L23–L27`), `index.html` (`L62–L64`)
- **Findings**:
  - `sphere_view.js` sets `preserveDrawingBuffer: true` unconditionally, forcing an extra GPU buffer copy every frame purely for on-demand `SphereView.snapshot()` calls.
  - Three.js is pinned at **r128** (2021), and asteroids/basic enemies still render as flat unlit line-segments while AI ships and bosses use lit 3D meshes.
- **Recommendation**:
  - Remove `preserveDrawingBuffer: true` and capture snapshots synchronously right after `renderer.render()` when requested.
  - Upgrade Three.js to a modern pinned release (with updated SRI hashes), add `UnrealBloomPass` post-processing, render displaced 3D meshes for asteroids and enemy hulls, and use `InstancedMesh` for high-volume projectiles/particles.

### 3.4 Split-Screen A/B Policy Arena & Flight Log Replay Scrubber
- **Impact**: Medium | **Effort**: Medium
- **Where**: `index.html`, `app.py`, `hud.js` (`L367–L384`)
- **Recommendation**:
  - **A/B Policy Arena**: Render two synchronized canvases initialized with the exact same `AsteroidsRNG` seed (e.g., Rule-Based vs. PPO, or Stage 0 vs. Swarm Sphere) to make policy improvements visually self-evident.
  - **Replay Scrubber**: Load exported `asteroids_telemetry_*.json` Flight Logs into a timeline viewer with play/pause/step scrubbing and critic-value overlays.

---

## 4. Architecture, Maintainability & Repository Hygiene

### 4.1 Decompose Monolithic Files & Replace the 50-Arg `get_game_html()`
- **Impact**: High | **Effort**: Medium
- **Where**: `ai_ship.js` (4,010 lines), `game.js` (3,370 lines), `enemy_ship.js` (1,450 lines), `app.py` (1,167 lines, `L481`), `sphere_view.js` (1,142 lines)
- **Finding**: Five files account for >52% of the ~21k-line codebase. `get_game_html()` in `app.py` accepts **50 parameters** and constructs a ~680-line Python f-string that re-inlines ~677 KB of JavaScript plus ~1 MB of PPO JSON into a `srcdoc` iframe on every Streamlit rerun.
- **Recommendation**:
  - Split `ai_ship.js` into focused modules (`ai_navigation.js`, `ai_formations.js`, `ai_combat.js`, `fuel_rescue.js`).
  - Replace the 50-parameter signature with a `GameConfig` dataclass serialized once with `json.dumps()`.
  - Serve static JS and checkpoint JSON assets via Streamlit static serving (or a lightweight static bundle) rather than inlining ~1.7 MB into `srcdoc` every rerun.

### 4.2 Introduce a Formal Model Registry (`models/registry.json`)
- **Impact**: Medium | **Effort**: Low
- **Where**: `app.py` (`L171`, `L187`), `training/run_training.py`, `training/run_benchmark.py`
- **Finding**: `app.py` picks the default checkpoint via `sorted(glob("models/policy_*.json"))[-1]` (purely alphabetical).
- **Recommendation**: Have `run_training.py` / `run_benchmark.py` maintain `models/registry.json` with structured metadata (`world`, `worldScale`, `obsDim`, `waves`, `bossPhases`, `fuelRescue`, paired benchmark scores, seed, promotion status) so the UI renders clean model cards and auto-selects the highest-scoring compatible checkpoint.

### 4.3 Consolidate Documentation & Archive Superseded Scaffolding
- **Impact**: Medium | **Effort**: Low
- **Where**: `README.md` (13 lines), 19 root-level `*.md` files, `marl_system.js`, `ml_model_loader.js`, `ml_model_adapter.js`
- **Finding**: The root directory holds 19 Markdown files (many historical/superseded like `MARL_IMPLEMENTATION.md`, `PHASE5_1_TEST_GUIDE.md`, `TENSORFLOW_TRAINING_README.md`), while the root `README.md` is only 13 lines and lacks architecture, quickstart commands, and benchmark tables.
- **Recommendation**:
  - Expand root `README.md` with project overview, architecture diagram, quickstart (`./start.sh`, `python3 -m http.server`, `streamlit run app.py`), test/benchmark commands, and latest `r1` curriculum results.
  - Move historical phase/design docs into `docs/archive/` and isolate or remove unused TF.js/MADDPG scaffolding once decoupled from `app.py`.

---

## 5. DevOps, CI & Cloud Security

### 5.1 Expand CI Coverage & Linting
- **Impact**: Medium | **Effort**: Low
- **Where**: `.github/workflows/ci.yml` (`L23`)
- **Finding**: `py_compile` currently checks only `app.py`, `training/run_training.py`, and `training/run_matrix.py`, skipping `training/run_benchmark.py`, `training/run_curriculum.py`, `tests/run_browser_smoke.py`, and `tests/test_app_template.py`.
- **Recommendation**: Compile-check all Python files, add `ruff` (Python) and `eslint` (JS) linting jobs, and migrate non-`jsc` or Node-shimmed jobs to `ubuntu-latest` for faster, lower-cost CI.

### 5.2 Harden AWS Terraform Deployment
- **Impact**: Medium | **Effort**: Medium
- **Where**: `terraform/aws/`
- **Recommendation**:
  - Add TLS/HTTPS termination (Caddy/Nginx with Let's Encrypt or AWS ALB + ACM) instead of serving Streamlit over plain HTTP on port 8501.
  - Configure a remote encrypted S3 Terraform state backend (with state locking) and remove local `terraform.tfstate*` / `tfplan` working artifacts from the workstation checkout.
  - Standardize EC2 provisioning around the repository's `Dockerfile` so local Docker builds and cloud deployments run identical bits.

---

## Recommended Phased Roadmap

| Priority | Horizon | Theme | Key Deliverables | Exit / Verification Criteria |
| --- | --- | --- | --- | --- |
| **P1 — Immediate** | 1–2 weeks | Correctness & Parity | Geodesic `WORLD.delta()` in `ml_features.js` & `sim/env.js`; clean `app.py` defaults & remove 6 `setInterval` loops; fix `user_data.sh` IMDSv2; expand root `README.md` & organize `docs/archive/` | `./run_tests.sh` green with new antimeridian/polar feature tests; zero failed TF.js fetches in Chrome console; `test_app_template.py` green |
| **P2 — Short-term** | 2–6 weeks | ML Parity & 3D UX | Train 43-dim `sphere x2` policy (waves + boss phases + fuel rescue); interactive 3D orbit/zoom/click-inspect in `sphere_view.js`; 3D critic heatmap; `models/registry.json`; tanker obstacle avoidance; expanded CI linting | 43-dim checkpoint beats both heuristic & random ($p < 0.05$) with `waves=true, fuelRescue=true`; interactive camera verified in `run_browser_smoke.py` |
| **P3 — Medium-term** | 1–3 months | Architecture & Visualization | Bidirectional custom Streamlit component (no iframe resets + live Python telemetry charts); factored 11-logit PPO head + LR/entropy annealing; `GameConfig` dataclass + static JS serving; split-screen A/B Policy Arena & replay scrubber; Node/Bun headless runner | Moving sidebar sliders updates sim without resetting score/episode; factored PPO matches or beats 48-way baseline in fewer steps; CI runs on `ubuntu-latest` |
| **P4 — Long-term** | 3–6 months | Next-Gen MARL & Visuals | Centralized Critic (MAPPO) + top-$k$ entity attention encoder; learned VDN/COMA credit assignment; adversarial Red Swarm self-play league; vectorized PyTorch/ONNX pipeline; Three.js modern bloom + 3D asteroid/enemy meshes; HTTPS Dockerized AWS deployment | Multi-agent cooperative metrics (focus fire, dock defense) measurably beat IPPO on held-out seeds; 60 FPS sustained with 20+ entities on 3D globe |

---

## Implementation & Verification Log (2026-11-20)

All four phases (P1–P4) were implemented sequentially and verified on `next-steps`:

### Phase 1 — Correctness, Geodesic Parity & Clean Defaults
- **Geodesic features & rewards**: `MLFeatureExtractor.relDelta()` in `ml_features.js` and `distFn` in `sim/env.js` now delegate to `WORLD.delta()` / `WORLD.distance()` on the 3D sphere while staying bit-identical on the flat torus. Verified with new antimeridian seam (`distNorm=0.100, aim=1.000`) and near-pole (`distNorm=0.050`) tests in `training/test_env.js`.
- **Streamlit `app.py` cleanup**: Legacy TF.js and non-learning MARL defaults disabled (`False`); `TFJS_TAG`/`TFJS_WASM_TAG` lazy-loaded only when enabled; `trained_policy_deterministic` defaults to `False`; off-distribution compatibility banner added; six `setInterval(..., 2000)` polling loops replaced with unified one-shot `applyAsteroidsConfig()`.
- **Deployment & Docs**: Fixed IMDSv1 token call in `terraform/aws/user_data.sh` to use IMDSv2 and pinned `streamlit==1.40.2`; expanded `.github/workflows/ci.yml` `py_compile` to all Python files; expanded root `README.md` and added `docs/INDEX.md`.

### Phase 2 — ML Parity, 43-Dim Full-Feature Training, Interactive 3D Globe & Registry
- **Tanker obstacle avoidance**: Added `computeTankerAvoidanceAngle()` in `ai_ship.js` to steer around asteroids and enemy bullets during transit/departure while preserving close-range docking alignment.
- **Interactive 3D globe & 3D critic heatmap**: Removed unconditional `preserveDrawingBuffer: true` in `sphere_view.js` (rendering on demand inside `snapshot()`); added mouse drag-orbit, scroll-zoom, double-click reset, click-to-inspect ship HUD card, and a vertex-colored 3D spherical critic value heatmap driven by `updateValueHeatmap()` / `getValueHeatmapData()` in `policy_viz.js`.
- **Model Registry**: Added `training/update_registry.py` generating `models/registry.json`, integrated with `app.py` checkpoint selector and auto-refreshed on promotion.
- **43-dim Full-Feature Policy (`models/policy_stage3_sphere_x2_fuel_waves.json`)**: Added `--waves`, `--bossPhases`, and automatic 38→43 input-layer widening on `--init` in `training/train.js`, plus `swarm_sphere_full` in `training/benchmark.js`. Fine-tuning `policy_swarm_sphere_x2.json` for 15 iterations (30,720 steps, 56.5 s) with fuel rescue + waves + boss phases on `sphere x2` lifted eval return from **25.88 → 28.16** (**11,367** score vs. heuristic **1,571** [$t = 8.90$] and random **3,821** [$t = 5.65$]).

### Phase 3 — Factored PPO, Stabilizers, Cross-Platform Runner, Bidirectional Bridge, Arena & Replay
- **Factored multi-discrete head (11 logits → 48 actions) & stabilizers**: Implemented in `training/nn.js`, `training/ppo.js`, `policy_runner.js`, and `policy_worker.js` along with orthogonal initialization, linear LR/entropy annealing, value clipping, and running return normalization. Finite-difference gradient check passed (`max rel err 1.05e-6`).
- **Cross-platform runtime**: Added `sim/runtime_shim.js` and updated `run_tests.sh`, `training/run_training.py`, and `training/run_benchmark.py` to auto-select macOS `jsc` or Node.js.
- **Bidirectional Streamlit bridge**: Added zero-npm `components/asteroids_bridge/index.html` wired into `app.py` so sidebar tweaks hot-update via `postMessage` without reloading the match, Python Mission Control buttons dispatch live interventions, and opt-in telemetry streams back into live Streamlit `st.metric` & `st.line_chart` dashboards.
- **Standalone analysis tools**: Added `arena.html` (synced-seed split-screen A/B policy arena) and `replay.html` (Flight Log JSON timeline scrubber), linked from `index.html`.

### Phase 4 — MAPPO Centralized Critic, Entity-Attention Encoder, COMA/VDN Credit, Self-Play & Hardened Deployment
- **Multi-entity tokens & `EntityAttentionNet`**: Added 10×6 entity slots (103-dim layout) and 12-dim global team state in `ml_features.js` / `sim/env.js`, plus permutation-invariant masked-attention `EntityAttentionNet` in `training/nn.js` (permutation diff `0.00e+0`, finite-difference gradient check `max rel err 4.47e-8`, exact JSON round-trip) with 3D geodesic attention sightlines in `sphere_view.js`.
- **MAPPO centralized critic & VDN/COMA credit**: Implemented in `training/ppo.js`, `policy_runner.js`, and `policy_worker.js` and smoke-verified end-to-end (`training/test_policy_runner.js` and CI job).
- **Adversarial Self-Play League**: Added `training/self_play.js` and `training/run_self_play.py` implementing PFSP opponent sampling across four escalating Red Swarm profiles.
- **Hardened Docker & Cloud Templates**: Added `docker-compose.yml`, `Caddyfile` (automatic HTTPS reverse proxy with WebSocket support), and `terraform/aws/backend.tf.example` (encrypted S3 + DynamoDB state locking).

