# Asteroids-AI — ML App Enhancement Review

| | |
|---|---|
| **Document version** | 1.0.0 |
| **Date** | 2026-09-05 |
| **Reviewer model** | Claude Fable 5.1 (Anthropic), running in Cursor |
| **Repository state** | branch `feature/advance-ml-6`, uncommitted fuel-rescue work in `game.js` / `app.py` / `theme.js` |
| **Scope** | Graphics, ML/RL technology, GUI. Phase 1 of the roadmap is implemented alongside this document; Phase 2 (spherical world + Three.js view) and 2b (world size decoupled from the screen, default 2×) followed the same day — see the last sections. |

---

## Assumptions

- Target remains the browser (Streamlit + Canvas) with the headless JavaScriptCore PPO pipeline (`sim/env.js`, `training/*.js`) as the real learning path. In-browser MARL (`marl_system.js`) stays scaffolding.
- Two-person team, no GPU cluster; training runs on a laptop or one cloud box.
- The Alpha fuel-rescue mechanic existed only in `game.js` and not in `sim/env.js`, so no trained policy had ever seen tankers or docking.

---

## Executive Summary

The three biggest gaps were: (1) graphics were Canvas 2D with `shadowBlur` glows, no particle system and no post-processing, while `game3d.js` sits on Three.js r128 without the agentic AI stack; (2) the Streamlit GUI was one-way (Python → JS) with no live metrics, so the ML story was invisible; (3) the RL is a 38-dim flat-vector IPPO on a plain-JS MLP, with curriculum stages 1–2 untrained, no centralized critic, and no entity encoder. Rendering *what the policy thinks* (critic heatmaps, action-probability wheels) is the cheapest way to make the app both graphically and technologically more advanced at once, and is where Phase 1 starts.

---

## Quick Wins (High Impact, Low Effort) — Phase 1, implemented

| Item | Where | Status |
|---|---|---|
| Cached glow sprites replacing `ctx.shadowBlur` | `fx.js` (`FX.glow`), call sites in `game.js` | done |
| Pooled particle system: thrusters, asteroid debris, explosions, dock plasma stream, HELP ripple | `fx.js`, hooks in `game.js` `update()` / `draw()` | done |
| Screen shake + hit flash on boss kill, ship loss, dock death | `fx.js` (`FX.shake`, `FX.flash`) | done |
| Critic value "danger heatmap" (grid of hypothetical positions through the loaded critic) | `policy_viz.js` | done |
| Action-probability wheel per ship (rotation / thrust / fire-mode / shield marginals) | `policy_viz.js`, `probs` added to `PolicyRunner` decisions | done |
| Web Worker policy inference (main thread extracts features, worker runs actor + critic) | `policy_worker.js`, `PolicyRunner.enableWorker()` | done |
| In-iframe HUD sparklines: score rate, alive ships, Alpha fuel, threat, policy value / entropy | `hud.js` | done |
| Sidebar reorganised into tabs with scenario presets (Training Demo, Boss Rush, Fuel Crisis, Policy vs Rules) | `app.py` | done |
| Fuel / tanker / dock in the headless env with reward terms and opt-in observation features | `sim/env.js`, `ml_features.js`, `game.js` | done |
| Training driver for stages 1–2 × 3 seeds; CI smoke test | `training/run_matrix.py`, `.github/workflows/ci.yml` | done |
| Stage 1 and 2 policies trained (3 seeds each) and best seeds promoted | `models/policy_stage1.json`, `models/policy_stage2.json` | done — results below |

### Stage 1–2 training results (2026-09-05, `python training/run_matrix.py --stages 1 2 --seeds 3`)

300 iterations × 4096-step rollouts, entropy 0.004, lr 3e-4, hidden 128×128, 3 agents, warm-started from `models/policy_stage0.json`. Evaluation on 30 paired held-out seeds; `*` = significant at 5%.

| stage | seed | env steps at best | eval return | score | vs rule-based | vs random (t) |
|---:|---|---:|---:|---:|---|---|
| 1 | seed0 | 614 400 | 25.15 | 5990 | +21.59 * | +2.02 (1.93) |
| 1 | **seed1** (promoted) | 1 228 800 | 26.18 | 6333 | +22.62 * | +3.05 (2.81) * |
| 1 | seed2 | 1 228 800 | 25.81 | 6177 | +22.25 * | +2.68 (2.22) * |
| 2 | **seed0** (promoted) | 819 200 | 20.04 | 6217 | +15.80 * | +1.96 (2.02) * |
| 2 | seed1 | 819 200 | 19.39 | 6040 | +15.15 * | +1.31 (1.11) |
| 2 | seed2 | 409 600 | 19.51 | 6243 | +15.27 * | +1.43 (1.30) |

Across seeds: stage 1 = 25.71 ± 0.52 (3/3 beat rule-based, 2/3 beat random); stage 2 = 19.65 ± 0.34 (3/3 beat rule-based, 1/3 beat random). Seed variance is small, so the stage-0 recipe transfers cleanly. The thin margin over *random* is the finding worth acting on: a uniformly random policy that thrusts and fires half the time collects a lot of asteroid score in this reward, so "beats random" is a weak bar and the Phase 2 reward work (survival-weighted return, boss-share credit) should be judged against the rule-based controller instead. Full logs, `metrics.csv` and per-seed checkpoints are under `training_runs/matrix_20260905_154204/` (git-ignored).

Numbers to expect: `shadowBlur` sites removed entirely; particles are pooled (max 2 000) with additive blending; heatmap is 24×16 evaluated one row per frame (full refresh every 16 frames, ~400 critic forwards spread across frames); worker inference adds one frame of action latency and removes ~0.5–3 ms/frame from the main thread with 5–10 ships.

---

## Strategic Improvements (High Impact, High Effort) — Phases 2–3

1. **Bidirectional Streamlit bridge → live ML dashboard.** Replace `components.html` with a custom component (`streamlit-component-lib`, `Streamlit.setComponentValue`) posting a 2 Hz telemetry frame. Python side: `st.plotly_chart` streams, `st.metric` deltas, a real "Export episode" download, plus a `training/` page (`st.navigation`) that tails `run_training.py` metrics and plots reward curves.
2. **Renderer migration to WebGL2/WebGPU via PixiJS v8.** Keep entity logic; swap the `draw()` layer for bloom, additive blending, thousands of particles, and a CRT filter. Alternative: upgrade `game3d.js` to Three.js r17x with `EffectComposer` + `UnrealBloomPass` and port formations/fuel/PPO — but that keeps two codebases in sync; PixiJS on the 2D path is the safer bet.
3. **PyTorch training with a vectorized Node env server.** Keep `sim/env.js` as ground truth; run N=64 envs in Node behind a WebSocket/ZeroMQ step API; train CleanRL-style MAPPO in PyTorch; export to ONNX and run with `onnxruntime-web` (WASM/WebGPU).
4. **MAPPO with centralized critic + entity-set encoder.** Replace the fixed vector with per-entity tokens (k-nearest asteroids/enemies/allies/tanker) through a 2-layer, 4-head, d=64 attention or DeepSets encoder; critic sees all allies' observations. Makes formation-holding and dock-guarding learnable rather than shaped.
5. **Adversarial enemy self-play (league / PFSP).** Train red swarm and boss policies against a pool of 8–16 blue checkpoints. Addresses the "modest margin over random" plateau in `RL_TRAINING.md`.
6. **Episode recorder + replay scrubber.** Log (state, action, probs, value) per frame to IndexedDB with a timeline UI to scrub, pause and step with overlays on.

---

## Detailed Analysis

### Graphics & rendering
- `[Impact: High | Effort: Low]` Offscreen-canvas glow sprites replacing `shadowBlur`. *(Phase 1)*
- `[Impact: High | Effort: Med]` Pooled particles with `globalCompositeOperation = 'lighter'`. *(Phase 1)*
- `[Impact: Med | Effort: Low]` Parallax starfield (3 layers), ship afterimage trails via a 12-position ring buffer. *(Phase 1: starfield)*
- `[Impact: Med | Effort: Med]` Offscreen post-processing: ¼-resolution bloom, vignette, optional CRT scanlines. *(Phase 2)*
- `[Impact: High | Effort: High]` PixiJS v8 renderer or Three.js upgrade (strategic 2). *(Phase 3)*
- `[Impact: Med | Effort: Med]` Procedural noise-displaced asteroids seeded from `rng.js` so replays stay deterministic. *(Phase 2)*
- `[Impact: Low | Effort: Low]` Web Audio synth graph replacing square-wave beeps; pan by x-position. *(Phase 2)*

### ML-driven visualization
- `[Impact: High | Effort: Low]` Critic value heatmap and action-probability wheels. *(Phase 1)*
- `[Impact: High | Effort: Med]` Input-gradient / occlusion saliency on the observation → arrows to attended entities; with the entity encoder, use attention weights directly. *(Phase 3)*
- `[Impact: Med | Effort: Med]` Policy rollout "ghost": clone env state, roll the policy 60 frames ahead every 15 frames, draw the predicted path. *(Phase 2)*
- `[Impact: Med | Effort: Low]` Entropy-coded hull saturation so hesitation is visible. *(Phase 2)*

### ML techniques
- `[Impact: High | Effort: Low]` Train stages 1–2, 3 seeds each; ablate aim shaping. *(Phase 1: `training/run_matrix.py`)*
- `[Impact: High | Effort: Med]` Fuel/tanker/dock in `sim/env.js`; rewards `refuelSuccess +3`, `dockDeath −8`, `lowFuel −0.01/frame`; opt-in observation features (`alphaFuelNorm`, `helpPingActive`, `tankerDx`, `tankerDy`, `fuelLinked`). The tanker is scripted, not a policy agent, so its off-policy trajectory does not enter PPO. *(Phase 1)*
- `[Impact: High | Effort: High]` MAPPO + entity encoder (strategic 4). *(Phase 3)*
- `[Impact: High | Effort: High]` Self-play league (strategic 5). *(Phase 3)*
- `[Impact: Med | Effort: Med]` Human-in-the-loop BC from the cyan player ship via `marl_data_collector.js`; the current BC clones a controller that is weaker than random. *(Phase 2)*
- `[Impact: Med | Effort: Med]` TarMAC-style 8-dim learned messages between allies once MAPPO exists. *(Phase 3)*
- `[Impact: Med | Effort: Low]` Optuna HPO over `DEFAULT_REWARD` and PPO clip/entropy using the existing 60-seed evaluation. *(Phase 2)*
- `[Impact: Low | Effort: Low]` INT8 ONNX quantization for the browser path once the encoder lands. *(Phase 3)*

### Agentic design

```
Agent: StrategistCommander
Role: Chooses formation, escort mode, and engage/escape posture at ~1 Hz
Tools: Summarised game state (threat, fuel, phases, enemy mix); formation API in game.js
Trigger: Every 60 frames, or threat-score crossing 0.6, or HELP ping
Benefit: Replaces the rule heuristic in ml_hierarchy.js with a learned (or LLM-driven) high-level policy
Coordinates with: Per-ship PPO policies (low level), FuelRescue logic
```
- `[Impact: Med | Effort: Med]` Learned version as a second PPO level with formation choice as the action. *(Phase 3)*
- `[Impact: Med | Effort: Low]` LLM version for demos (WebLLM in-browser or API at 0.5 Hz with a JSON command schema). 300–1500 ms latency and non-deterministic replays, so demo-only, every command logged. *(Phase 2)*

```
Agent: EvalGuardian
Role: Runs the 60-seed evaluation on every new checkpoint and blocks promotion on regression
Tools: training/evaluate.js, models registry, GitHub Actions
Trigger: New file in models/ or a training run finishing
Benefit: Champion/challenger gating; stops silent regressions
Coordinates with: Streamlit model selector (reads the registry it writes)
```

### Performance
- `[Impact: High | Effort: Low]` Glow sprites, Web Worker inference. *(Phase 1)*
- `[Impact: Med | Effort: Low]` Spatial hash (≈64 px cells) for collisions and k-nearest queries; current checks are O(n²). *(Phase 2)*
- `[Impact: Med | Effort: Med]` Batch all ships' observations into one forward pass. *(Phase 1: the worker already receives the whole batch per frame)*

### GUI / user experience
- `[Impact: High | Effort: Low]` Tabs, presets, collapsed "Advanced" sections. *(Phase 1)*
- `[Impact: High | Effort: Med]` In-iframe HUD sparklines (no round trip, 60 fps). *(Phase 1)*
- `[Impact: High | Effort: High]` Custom Streamlit component with JS→Python telemetry (strategic 1). *(Phase 2)*
- `[Impact: Med | Effort: Med]` Split-screen A/B: two canvases, same seed, rule-based vs PPO. *(Phase 2)*
- `[Impact: Med | Effort: Med]` Model registry UI from `models/*.json` metadata. *(Phase 2)*
- `[Impact: Med | Effort: High]` Episode recorder + scrubber (strategic 6). *(Phase 2)*
- `[Impact: Low | Effort: Low]` Colour-blind palette variant in `theme.js`, keyboard-shortcut overlay, touch controls. *(Phase 2)*

### Automation / MLOps
- `[Impact: High | Effort: Low]` GitHub Actions: `run_tests.sh` + short training smoke test on every PR. *(Phase 1)*
- `[Impact: Med | Effort: Low]` `models/registry.json` written by the trainer and read by the Streamlit selector and EvalGuardian. *(Phase 2)*
- `[Impact: Low | Effort: Low]` Dockerfile for the Streamlit app so the EC2 `user_data.sh` becomes `docker run`. *(Phase 2)*

### Innovation opportunities
- **Observer intervention score:** measure how often the human's shots change the critic's predicted outcome.
- **Auto-curriculum (PLR-style):** generate stage configs from where value estimates are worst.
- **Narrated after-action report:** feed the episode log to an LLM to narrate in the voice of `STORY.md`.

---

## Recommended Roadmap

- **Phase 1 (0–4 weeks) — implemented with this document:** glow sprites, particles, screen shake, parallax starfield; value heatmap and action wheels; Web Worker inference; sidebar tabs + presets; in-iframe HUD charts; fuel/tanker in `sim/env.js` with new rewards and opt-in features; stage 1–2 × 3-seed training driver; CI smoke test.
- **Phase 2 (1–3 months):** custom Streamlit component with telemetry + training dashboard page; model registry; split-screen A/B; episode recorder and scrubber; Node vectorized env server + PyTorch MAPPO; ONNX export to `onnxruntime-web`; human BC warm-start; post-processing bloom/CRT; procedural asteroids.
- **Phase 3 (3–6 months):** entity-attention encoder with attention-arrow visualization; enemy self-play league; learned StrategistCommander; PixiJS v8 renderer (or Three.js upgrade); auto-curriculum.

---

## Phase 1 implementation notes

- **Backward compatibility of checkpoints.** Fuel observation features are opt-in (`fuelRescue` env option → `RL_FEATURE_NAMES_FUEL`, 43 dims). `policy_runner.js` accepts either layout by matching the checkpoint's `featureNames` to a known layout, so `models/policy_stage0.json` and `models/policy_stage3.json` (38 dims) keep loading.
- **Scripted tanker.** Under policy control the `role === 'fuel'` ship runs `handleFuelShipBehavior()`; the env marks it not-alive so PPO never collects its transitions. Alpha selection under policy control is done by `ensureAlphaAssignment()` without re-centering the Alpha every frame.
- **Determinism.** `fx.js` uses its own RNG stream and never touches game state; the headless shim disables it entirely, so `training/test_env.js` determinism checks are unaffected.
- **Worker fallback.** If `Worker` or `Blob` is unavailable (or the worker fails), `PolicyRunner` falls back to synchronous inference on the main thread.
- **Fuel rescue defects found while verifying Phase 1 in the browser** (all fixed, regression-tested in `training/test_env.js` → "Fuel rescue under rule-based control"):
  - `findAlphaShip()` runs every frame and refilled the Alpha's tank each time, so under rule-based control the Alpha stayed at 100% and the rescue never fired. The refill now happens only on the frame a ship becomes Alpha.
  - Burn rates (`0.35`/thrust, `0.2`/shot with a 3-frame rapid-fire cooldown) drained a full tank in ~5 s while the tanker's round trip is 8–10 s, so the Alpha was pinging HELP ~70% of the time and could never be kept fuelled. Retuned to `0.02` passive, `0.08`/thrust, `0.03`/shot (≈20 s full → HELP).
  - An empty tank had no consequence. The Alpha now cannot thrust or fire while empty or while linked to the tanker (`AIShip.fuelStarved()`), which is what gives the `lowFuel`/`dockDeath` rewards teeth for RL.
  - `selectAlphaShip()` could promote the tanker to Alpha (an Alpha that is its own only fuel source), and an Alpha whose partner died off-path stayed "docked" forever and never pinged. The tanker is now excluded unless it is the last ship, and `processFuelTransfers()` clears orphaned links.
  - **Alpha frozen in "FUEL LINK" (reported from the browser).** `app.py` re-sends every sidebar setting on a 2 s `setInterval` "to make sure it sticks", and `setAutoAssignRoles()` reset every ship's role on each call — including the tanker's `'fuel'` role mid-refuel. The next frame re-converted the tanker with a full tank and cleared dock flags while the Alpha stayed `fuelDocked` to it, so both ships froze for good. `setAutoAssignRoles()` is now a no-op when the value is unchanged and never touches the tanker; `assignFuelShipRole()` restores rather than re-initialises a ship that is currently linked; the orphan guard requires the partner to be linked back; and an enemy-bullet kill releases the victim's fuel link. Regression test: "Fuel rescue survives the Streamlit settings re-send loop" (old code: 2421-frame link; fixed: ≤144).
- **Follow-up (not Phase 1).** The tanker flies straight at the Alpha with no asteroid avoidance and dies often; the dispatch-and-replace path covers it, but a small avoidance steer in `handleFuelShipBehavior()` would make "Fuel Crisis" read better.

## Phase 2 implementation notes — spherical world with a Three.js 3D view (2026-09-05)

Requested as a follow-on to Phase 1: replace the flat wrap-around rectangle with a sphere that all objects travel on, rendered in 3D. Delivered as a runtime world mode, not a fork, so the flat torus, the swarm AI, the fuel mechanic, the PPO policies and the headless trainer all keep working unchanged.

### Design

- **Chart, not coordinates, changed.** Every object still stores `x, y, angle, velocityX/Y` in canvas units. `world.js` decides what the chart means: `torus` (default, byte-identical to the original game) or `sphere`, where the canvas is an equirectangular map — `x` is longitude, `y` is latitude (`y=0` north pole), one unit of `y` is one unit of arc, so `R = H/π ≈ 191` for the 1200×600 canvas.
- **Motion is exact great-circle transport.** `WORLD.step()` lifts the chart velocity to a 3D tangent vector, rotates the position about the perpendicular axis, parallel-transports both velocity and heading, and re-expresses them in the new local frame. No small-angle approximation and no per-frame drift (a full great circle closes to within one step; test "a full great circle closes"). `x` wraps at the antimeridian; crossing a pole reflects `y`, shifts `x` by half a turn and flips heading/velocity.
- **Where `game.js` branches.** A handful of sites: the five `update()` movers (`Ship`, `Bullet`, `Asteroid`, `EnemyBullet`, `EnemyShip`), `checkCollision()` (arc distance instead of chart Euclid), the two edge-spawn constructors (`Asteroid`, `EnemyShip` — a sphere has no edge, so spawns are uniform and ≥150/200 units from the Alpha), and `draw()`. Everything else (flocking, avoidance, formations, targeting, `ml_features.js`) still reads chart `dx/dy`, which is faithful near the equator and stretched near the poles. This is the deliberate trade: the rule-based AI code is untouched, physics and collisions are correct everywhere, and AI judgement is mildly distorted at high latitude.
- **Renderer (`sphere_view.js`).** Three.js r128 (already pinned with SRI in `app.py`). A Lambert globe with a faint lat/lon grid and an additive rim, a fixed-seed starfield, and every game object rebuilt each frame into one dynamic `LineSegments` buffer (vector-display look, one draw call) plus two `Points` clouds for bullets and the `fx.js` particle pool (`FX.forEachParticle`). Formation spokes and fuel tethers are great-circle arcs. Far-hemisphere objects are culled before geometry is built. Camera: follow Alpha (default), follow fleet centroid, or slow orbit; distance 1.4–4.5 R. The game's own canvas has an opaque 2D context, so the WebGL canvas is stacked over it and `ctx` is repointed to a transparent overlay canvas — the HUD, policy labels, action wheels and a new minimap (whole map, visible-cap outline, "far side: N") draw there with unchanged code via `worldToScreen()`.
- **Fallback.** If Three.js fails to load, `setWorldMode('sphere')` still switches the physics; the sphere is then drawn on the flat canvas as its own map projection.
- **ML parity.** `sim/env.js` accepts `world: 'sphere'`, `train.js`/`run_training.py` take `--world sphere`, and the checkpoint `meta.world` records which geometry a policy was trained on. Observation layout is unchanged (38 / 43 dims), so existing torus checkpoints *load* on the sphere, but they were trained on a different world; retrain for the sphere with `--world sphere`. Smoke-verified: 2 PPO iterations on the sphere, `meta.world: "sphere"` written.

### Verification

- `training/test_env.js` gained two suites: "Sphere world geometry" (equatorial motion is straight, mid-latitude eastward motion bends toward the equator with the heading transported, speed conserved, pole crossing lands on the antimeridian heading south, great circle closes, antipodal distance = πR, near-pole points at opposite longitudes are close, projectiles follow great circles, torus path unchanged) and "Sphere world environment" (deterministic episodes, all objects stay inside the chart, finite observations, rule-based fleet completes a stage-3 fuel-rescue episode, a torus env restores the flat geometry). `./run_tests.sh`: all green.
- Browser (Streamlit, Fuel Crisis preset, World = Sphere): gold Alpha pinned at the equator under the camera, HELP ring pulsing, green tanker inbound along a great circle with its plasma stream, dashed formation spoke curving over the globe, minimap bottom-right, telemetry HUD top-right; orbit camera verified. Frame read-back is available as `SphereView.snapshot()` (PNG data URL) for screenshots and future tests.
- Rule-based fleet, 6 seeds × 1000 steps, fuel rescue on: stage 0 torus 450 vs sphere 1250 mean score; stage 3 torus 767 vs sphere 567 — same order of magnitude, so the chart-space AI is not broken by the geometry change. Random policies score higher on the sphere because uniform spawns put more asteroids in reach than edge spawns did.

### Known limitations / follow-ups

- Rule-based avoidance and targeting use chart deltas; give ships a `WORLD.delta()`-based local frame (already implemented in `world.js`) for the ~10 hot paths to remove the polar distortion.
- `ml_features.js` relative features likewise; a local-tangent-frame feature set would let one policy generalise across both worlds.
- The Alpha is pinned to the chart centre (equator, prime meridian) exactly as it is pinned to the screen centre on the torus; a "free Alpha" mode on the sphere would make the camera follow more interesting.
- No mouse orbit yet; camera is preset-driven from the sidebar (or `?cam=…&zoom=…` on `index.html`).
- Enemy boss shapes are stylised hexagons in 3D rather than the 2D sprites.

## Phase 2b — world size decoupled from the screen ("option B", 2026-09-05)

Motivation: at 1× the sphere is *smaller* than the flat map it replaced. Wrapping the 1200×600 chart onto a globe gives R = H/π ≈ 191 and 4πR² ≈ 458k units² of surface against 720k for the rectangle (−36%), and no two objects can ever be more than πR ≈ 600 units apart. The fleet, the Crimson Swarm and the asteroids were all within one screen of each other. Two ways to "make the sphere bigger" were considered: (A) scale the camera/mesh only — cosmetic, nothing gains room; (B) make the logical world larger than the screen — real room, touches physics, spawning, fuel and the renderer. B was implemented.

### Design

- **One number.** `WORLD.setScale(k)` (`world.js`). In sphere mode `WORLD.size()` returns `k × canvas`, so `R = k × 191`, surface grows as k², and the maximum separation grows as k. Ship size, speed, bullet range, formation offsets and collision radii are untouched — that is precisely what buys room. On the torus the scale is ignored (`effectiveScale() = 1`): there the canvas *is* the viewport and must equal the world.
- **`game.js` no longer reads `canvas.width/height` as world bounds.** All 73 world-logic sites (wrap, spawn, centre pin, edge thresholds, predictive-wrap, hyperspace, escape clamps, formation anchors) now go through `worldW()/worldH()`, which are the canvas on the torus and `WORLD.size()` on the sphere. Drawing code (`draw()`, overlay footer, HUD, heatmap) keeps the canvas. Torus behaviour is byte-identical (same values, `x / 1`).
- **Things that must grow with the world.** Spawn clearance from the Alpha: `150·k` / `200·k`. Fuel: the tanker's round trip is k× longer, so the three burn sites divide by `worldScale()` (`fuelPassiveDrain`, `fuelThrustCost`, `fuelFireCost`) — endurance scales with the trip and the Alpha never starves (measured min fuel 29–43% across 1.5×–3×, vs 31% at 1×). `ml_features.js` normalises positions by `WORLD.size()` on the sphere so features stay in range.
- **Renderer.** `sphere_view.js` watches `WORLD.radius()` every frame (`syncWorld()`) and rebuilds globe, grid, halo and starfield, camera far plane and key light when it changes — so the GUI slider takes effect without a reload. The lat/lon grid and the minimap are charts of the *world*, not the canvas. Camera distance semantics changed so ships keep their on-screen size: the "Distance" setting is in radii of the 1× globe and the effective distance is `1 + (d−1)/k` radii (2.6 → 1.8R at 2×, 1.53R at 3×). Visible cap shrinks from 67° to 56° of arc at 2× — the globe fills the frame instead of floating in it. Caption shows `SPHERE R=382 x2`.
- **GUI / URLs.** Streamlit: "Sphere world & camera" expander gains **World scale** 1.0–3.0 (step 0.25, default **2.0**); presets carry `world_scale`. `index.html`: `?scale=…` (default 2). `setWorldScale(k)` is exposed by `game.js` for both.
- **Training.** `AsteroidsEnv({ world: 'sphere', worldScale: k })`, `train.js --worldScale`, `run_training.py --worldScale`; the env resets the scale every episode so envs at different scales cannot leak into one another. `meta.worldScale` is written to checkpoints. Sphere runs publish as `models/policy_stage{N}_sphere_x{k}.json` so they never overwrite the torus policies the browser loads by default.

### Verification

- New `test_env.js` suite "Scaled sphere world" (17 checks): chart and radius double at 2×, surface ×4 and larger than the flat map (1.83M vs 720k), a 10-unit step is still 10 units of arc, `randomPoint` honours a clearance larger than the 1× radius, scale ignored on the torus, env applies the scale on reset, deterministic episodes, objects stay inside the 2× chart *and* use the space beyond the canvas, objects reach separations > πR₁ (1148 vs 600), finite observations, full stage-3 episode at 2×, thrust burns half the fuel per frame at 2×, a 1× env restores the scale. `./run_tests.sh`: all green.
- Rule-based fleet, 5 seeds × 1500 steps, stage 3, fuel rescue on (`/tmp/scale_compare.js`):

  | world | R | mean score | ships alive | Alpha min fuel |
  |---|---|---|---|---|
  | torus | – | 1180 | 4.8 | 27% |
  | sphere ×1 | 191 | 1480 | 4.4 | 31% |
  | sphere ×1.5 | 286 | 680 | 3.8 | 29% |
  | sphere ×2 | 382 | 880 | 3.0 | 34% |
  | sphere ×3 | 573 | 280 | 4.0 | 43% |

  Score falls with scale because encounters per unit time fall (that is the point — more distance, longer approaches); fuel never becomes the limiting factor.
- Browser (Streamlit, Fuel Crisis preset, default 2×): `WORLD.getScale() = 2`, `R = 382`, `size = 2400×1200`, effective camera 1.8R; Alpha renders at the same size as at 1×, globe fills the frame, minimap shows the whole 2400×1200 chart with 5 objects on the far side, HUD/overlays intact. Composited frame via `SphereView.snapshot()`.
- PPO: `run_training.py --stage 0 --iterations 150 --world sphere --worldScale 2` (`training_runs/stage0_sphere_x2`, 307k env steps, 110 s at ~17k sps): eval score 3037 ± 324 (n=30), +28.5 return over the rule-based controller (t=16.2) and +6.5 over random — published as `models/policy_stage0_sphere_x2.json` with `meta.world = "sphere"`, `meta.worldScale = 2`. Existing torus checkpoints load at any scale but are off-distribution; retrain per scale (stages 1–3 not yet run on the sphere).

### Follow-ups

- Rule-based sensing radii (asteroid 400, enemy 500, boss 600) are fixed world units; at 3× the fleet spends long stretches seeing nothing. Either scale the radii with `worldScale()` or (better) keep them and let the fleet patrol — a task for the strategic policy.
- Formation offsets could optionally grow with √k so wingmen spread with the room they have.
- Enemy spawn cadence is per-frame; a per-unit-area rate would keep density constant across scales.
