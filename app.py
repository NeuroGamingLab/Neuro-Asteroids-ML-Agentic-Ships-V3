import glob
import json
import os

import streamlit as st
import streamlit.components.v1 as components

# Third-party scripts are pinned by version *and* by content hash. Without
# Subresource Integrity a pinned version still means "whatever the CDN serves
# under that name today", which is a supply-chain risk we can remove for free.
# Regenerate a hash with:
#   curl -sSL <url> | openssl dgst -sha384 -binary | openssl base64 -A
TFJS_TAG = (
    "<script src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js' "
    "integrity='sha384-h/XNDFVVo8fubLoOyBCXZW8B+OPOs7cpygLoTtS6f8DDTPRPCoFNMaE4JXtqrSWj' "
    "crossorigin='anonymous'></script>"
)
TFJS_WASM_TAG = (
    "<script src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.15.0/dist/tf-backend-wasm.min.js' "
    "integrity='sha384-7msKZ1p5XOoSnbUEElD3mKhM37KLYWbIxIniRMT9TBnT8AajE7COuFypV+cUiirw' "
    "crossorigin='anonymous'></script>"
)
THREE_JS_TAG = (
    "<script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js' "
    "integrity='sha384-CI3ELBVUz9XQO+97x6nwMDPosPR5XvsxW2ua7N1Xeygeh1IxtgqtCkGfQY9WWdHu' "
    "crossorigin='anonymous'></script>"
)


def js_string(value):
    """Render a Python value as a safe JavaScript string literal.

    Several of these values come from st.text_input, so interpolating them
    directly into a <script> block lets a user close the string (or the script
    tag) and execute arbitrary JavaScript. json.dumps handles quotes, backslashes
    and control characters; escaping '<' additionally prevents '</script>' from
    terminating the surrounding block early.
    """
    return json.dumps("" if value is None else str(value)).replace("<", "\\u003c")


POLICY_DIR = "models"
# The whole checkpoint is inlined into the page, so an oversized file is a real
# problem before it is a security one: it would be sent to the browser on every
# rerun. A 128 MB actor would not fit the game loop's frame budget anyway.
MAX_POLICY_BYTES = 32 * 1024 * 1024


def load_policy_file(path):
    """Read and validate a PPO checkpoint that is about to be sent to the browser."""
    # glob('models/policy_*.json') would also match something like
    # 'models/policy_../../etc/passwd.json', so resolve and confirm containment
    # rather than trusting the pattern.
    resolved = os.path.realpath(path)
    root = os.path.realpath(POLICY_DIR)
    if os.path.commonpath([resolved, root]) != root:
        raise ValueError(f"checkpoint must live under {POLICY_DIR}/")

    size = os.path.getsize(resolved)
    if size > MAX_POLICY_BYTES:
        def mb(n):
            return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.0f} KB"
        raise ValueError(f"checkpoint is {mb(size)}, over the {mb(MAX_POLICY_BYTES)} limit")

    with open(resolved, 'r') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("format") != "asteroids-ppo-policy":
        raise ValueError("not an asteroids-ppo-policy checkpoint")
    return payload


# Page configuration
st.set_page_config(
    page_title="Asteroids Arcade Game",
    page_icon="",
    layout="wide"
)

# Custom CSS — Neuro-Engineering-Lab theme
st.markdown("""
    <style>
    :root {
        --nel-primary: #00D2FF;
        --nel-secondary: #0078FF;
        --nel-accent: #FF00FF;
        --nel-purple: #9D00FF;
        --nel-muted: #D1D1D1;
        --nel-glow: rgba(0, 210, 255, 0.35);
        --nel-accent-glow: rgba(255, 0, 255, 0.25);
    }
    .stApp {
        max-width: 100%;
        overflow-x: visible !important;
        background: linear-gradient(135deg, #000000 0%, #0a0a1a 45%, #1a0a2e 100%);
        color: var(--nel-muted);
    }
    .main > div {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    h1 {
        margin: 0 !important;
        text-align: left !important;
        background: linear-gradient(90deg, #00D2FF 0%, #0078FF 45%, #FF00FF 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 10px rgba(0, 210, 255, 0.35));
    }
    /* Logo and title sit in a two-column st.columns() row; center them on
       each other instead of each aligning to the top of its own column. */
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    .main {
        max-width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    .block-container {
        max-width: 100% !important;
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    body {
        overflow-x: visible !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050508 0%, #12081f 100%);
        border-right: 1px solid rgba(0, 210, 255, 0.25);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--nel-primary) !important;
    }
    .stSlider label, .stCheckbox label, .stSelectbox label, .stRadio label {
        color: var(--nel-muted) !important;
    }
    hr {
        border-color: rgba(0, 210, 255, 0.2) !important;
    }
    [data-testid="stImage"] img {
        border-radius: 50%;
        box-shadow: 0 0 20px var(--nel-glow), 0 0 36px var(--nel-accent-glow);
    }
    </style>
    """, unsafe_allow_html=True)

# Brand header + title: logo and title side by side.
logo_col, title_col = st.columns([1, 6])
with logo_col:
    st.image('assets/neuro-engineering-lab-logo.png', width=110)
with title_col:
    st.title("Observe: Agentic ML in Action")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar: scenario presets + four tabs. Every widget has a key so a preset can
# set it through st.session_state before the widgets render on the rerun.
# ---------------------------------------------------------------------------

# Published PPO checkpoints. Prefer models/registry.json (ranked by held-out
# return, world compatibility, and feature coverage) and fall back to globbing
# models/policy_*.json if no registry has been generated yet.
def load_model_registry():
    reg_path = os.path.join(POLICY_DIR, "registry.json")
    if os.path.exists(reg_path):
        try:
            with open(reg_path, "r") as handle:
                reg = json.load(handle)
            models = [m for m in reg.get("models", []) if os.path.exists(m.get("path", ""))]
            if models:
                return reg, models
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    paths = sorted(glob.glob(os.path.join(POLICY_DIR, "policy_*.json")))
    return None, [{"path": p, "name": os.path.splitext(os.path.basename(p))[0]} for p in paths]


model_registry, registry_entries = load_model_registry()
registry_by_path = {m["path"]: m for m in registry_entries}
policy_files = [m["path"] for m in registry_entries]
default_policy_path = (
    (model_registry.get("defaults", {}).get("sphere") if model_registry else None)
    or (policy_files[0] if policy_files else "none")
)
if default_policy_path not in policy_files:
    default_policy_path = policy_files[0] if policy_files else "none"

# Widget keys and their defaults. Presets override a subset of these.
# Legacy TensorFlow.js / scaffolding MARL options stay off by default so the
# browser does not fetch unused ~1.5 MB CDN bundles or fill an unconsumed replay
# buffer; the working headless-trained PPO policy drives the swarm directly.
DEFAULTS = {
    "num_ai_ships": 10, "num_enemy_ships": 10, "num_boss_ships": 10,
    "canvas_width": 1200, "canvas_height": 600,
    "player_ship_active": False,
    "alpha_attack_enabled": True, "formation_type": "arrowhead",
    "auto_assign_roles": True, "fuel_rescue_enabled": True,
    "adaptive_formation": True, "multi_target_mode": "focus", "escort_mode": "guard",
    "tactical_sequences": True, "formation_transitions": True, "advanced_flanking": True,
    "ml_data_collection": False, "ml_hierarchy_enabled": False, "ml_evaluation_enabled": False,
    "curriculum_stage": "3: Full Chaos",
    "nn_models_enabled": False, "ml_enabled": False, "ml_mode": "parameters",
    "marl_enabled": False, "marl_training": False,
    "trained_policy_enabled": True,
    "trained_policy_path": default_policy_path,
    "trained_policy_deterministic": False, "policy_overlay": True,
    "policy_worker": True, "value_heatmap": True, "action_wheels": True,
    "fx_enabled": True, "hud_enabled": True,
    "sfx_enabled": True, "music_enabled": True, "kill_cam_enabled": True,
    "waves_enabled": True, "boss_phases_enabled": True,
    "world_mode": "Sphere (3D)", "sphere_camera": "Follow Alpha", "sphere_zoom": 4.2, "world_scale": 2.0,
}

PRESETS = {
    "Custom": {},
    "Training Demo": {
        "num_ai_ships": 3, "num_enemy_ships": 2, "num_boss_ships": 1,
        "alpha_attack_enabled": False, "player_ship_active": False,
        "trained_policy_enabled": bool(policy_files), "policy_overlay": True,
        "value_heatmap": True, "action_wheels": True, "hud_enabled": True, "fx_enabled": True,
    },
    "Boss Rush": {
        "num_ai_ships": 5, "num_enemy_ships": 3, "num_boss_ships": 3,
        "alpha_attack_enabled": True, "formation_type": "circle", "fuel_rescue_enabled": True,
        "adaptive_formation": True, "trained_policy_enabled": False,
        "hud_enabled": True, "fx_enabled": True, "value_heatmap": False,
    },
    "Fuel Crisis": {
        "num_ai_ships": 4, "num_enemy_ships": 1, "num_boss_ships": 0,
        "alpha_attack_enabled": True, "formation_type": "arrowhead", "fuel_rescue_enabled": True,
        "trained_policy_enabled": False, "hud_enabled": True, "fx_enabled": True,
    },
    "Policy vs Rules": {
        "num_ai_ships": 3, "num_enemy_ships": 2, "num_boss_ships": 1,
        "alpha_attack_enabled": False,
        "trained_policy_enabled": bool(policy_files), "trained_policy_deterministic": False,
        "policy_overlay": True, "action_wheels": True, "value_heatmap": True, "hud_enabled": True,
    },
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def apply_preset():
    """Copy the chosen preset's values into the widget state before rerender."""
    name = st.session_state.get("scenario_preset", "Custom")
    for key, value in PRESETS.get(name, {}).items():
        st.session_state[key] = value


with st.sidebar:
    st.header("Scenario")
    st.selectbox(
        "Preset",
        list(PRESETS.keys()),
        key="scenario_preset",
        on_change=apply_preset,
        help=("Training Demo: trained policy with every overlay on. Boss Rush: big swarm in a "
              "defensive circle against three bosses. Fuel Crisis: watch the Alpha starve and the "
              "green tanker run the gauntlet. Policy vs Rules: PPO policy with decision overlays."),
    )

    tab_scene, tab_swarm, tab_ml, tab_telemetry, tab_visual = st.tabs(["Scenario", "Swarm AI", "ML / MARL", "Telemetry", "Visuals"])

    # ------------------------------------------------------------------ Scenario
    with tab_scene:
        num_ai_ships = st.slider("AI ships", 0, 10, step=1, key="num_ai_ships",
                                 help="Blue agentic ships. New ships spawn at random locations.")
        num_enemy_ships = st.slider("Enemy ships", 0, 10, step=1, key="num_enemy_ships",
                                    help="Red / purple enemies spawn at the screen edges.")
        num_boss_ships = st.slider("Boss ships", 0, 10, step=1, key="num_boss_ships",
                                   help="Bosses have phases, coordinated attack patterns and shields.")
        waves_enabled = st.toggle("Escalating waves", key="waves_enabled",
                                  help=("Turns the sliders above into ceilings. Wave 1 sends about a third of the enemies, "
                                        "the full count arrives by wave 5, a boss joins every third wave, and each wave "
                                        "fires faster and flies faster. Clearing a wave pays a bonus. Off: the whole fleet "
                                        "spawns at once and is never replaced."))
        boss_phases_enabled = st.toggle("Boss phase mechanics", key="boss_phases_enabled",
                                        help=("Health bar with phase marks, a banner on each phase change, phase 2 calls in "
                                              "two escort ships, phase 3 enrages the boss (fires and flies faster)."))
        player_ship_active = st.toggle("Player ship active", key="player_ship_active",
                                       help="The cyan Observer ship. Click the game area, then use the arrow keys.")
        world_mode = st.radio(
            "World", ["Flat torus (2D)", "Sphere (3D)"], key="world_mode", horizontal=True,
            help=("Flat: the classic rectangle whose edges wrap. Sphere: the same map wrapped onto a globe - "
                  "objects fly great circles, cross the poles and are rendered in 3D with Three.js. "
                  "The swarm AI, fuel rescue and PPO policies run unchanged; the minimap shows the far side."))
        sphere_on = world_mode.startswith("Sphere")
        sphere_camera = st.session_state.get("sphere_camera", "Follow Alpha")
        sphere_zoom = st.session_state.get("sphere_zoom", 4.2)
        world_scale = st.session_state.get("world_scale", 2.0)
        if sphere_on:
            with st.expander("Sphere world & camera", expanded=False):
                world_scale = st.slider("World scale", 1.0, 3.0, step=0.25, key="world_scale",
                                        help=("Size of the globe relative to the screen. 1x wraps the 1200×600 chart onto a "
                                              "sphere of R≈191 (36% less surface than the flat map). 2x doubles R and gives 4x "
                                              "the surface: ships, asteroids and bullets keep their size and speed, so there is "
                                              "more room and longer engagement distances. Fuel endurance and spawn clearances "
                                              "scale with it; PPO policies trained at 1x will be off-distribution."))
                sphere_camera = st.selectbox("Camera", ["Follow Alpha", "Follow fleet", "Follow boss", "Slow orbit"], key="sphere_camera",
                                             help="Follow keeps the target under the camera with local north up.")
                sphere_zoom = st.slider("Distance", 1.4, 4.5, step=0.1, key="sphere_zoom",
                                        help=("Camera height above the surface, in radii of the 1x globe. Ships keep the same "
                                              "on-screen size at any world scale; closer shows more detail, further shows more of the hemisphere."))
            st.caption("Sphere renders best at a 2:1 canvas (e.g. 1200×600).")
        with st.expander("Screen size"):
            canvas_width = st.slider("Canvas width", 400, 1920, step=100, key="canvas_width")
            canvas_height = st.slider("Canvas height", 600, 2000, step=100, key="canvas_height")

    # ------------------------------------------------------------------ Swarm AI
    with tab_swarm:
        alpha_attack_enabled = st.toggle(
            "Alpha attack pattern", key="alpha_attack_enabled",
            help="One gold Alpha holds the centre; the others fly formation around it."
        )
        formation_type = "arrowhead"
        auto_assign_roles = True
        fuel_rescue_enabled = False
        if alpha_attack_enabled:
            formation_type = st.selectbox(
                "Formation", ["arrowhead", "line", "circle", "diamond", "wedge"], key="formation_type"
            )
            auto_assign_roles = st.toggle("Auto-assign roles", key="auto_assign_roles",
                                          help="Scout, Tank, Support, DPS, Interceptor by formation slot.")
            fuel_rescue_enabled = st.toggle(
                "Alpha fuel rescue", key="fuel_rescue_enabled",
                help=("The Alpha burns fuel; below 35% it pings HELP and the green tanker docks. "
                      "An asteroid strike during the dock destroys both ships.")
            )

        with st.expander("Phase 3: adaptive coordination"):
            adaptive_formation_value = st.toggle("Adaptive formations", key="adaptive_formation",
                                                 help="Switch formation on threat, enemy count and health.")
            multi_target_mode_value = st.selectbox("Multi-target mode", ["focus", "split", "prioritize"],
                                                   key="multi_target_mode")
            escort_mode_value = st.selectbox("Escort & protection",
                                             ["none", "escort", "guard", "patrol", "intercept", "cover"],
                                             key="escort_mode")

        with st.expander("Advanced attack patterns"):
            tactical_sequences_enabled = st.toggle("Tactical attack sequences", key="tactical_sequences")
            formation_transitions_enabled = st.toggle("Dynamic formation transitions", key="formation_transitions")
            advanced_flanking_enabled = st.toggle("Advanced flanking", key="advanced_flanking")

    # ------------------------------------------------------------------ ML / RL
    with tab_ml:
        st.subheader("Trained PPO policy")
        trained_policy_path = "none"
        trained_policy_enabled = False
        trained_policy_deterministic = False
        policy_overlay = False
        policy_worker = False
        if not policy_files:
            st.caption(
                "No trained policy found. Train one with:\n\n"
                "`python3 training/run_training.py --stage 0 --iterations 300`"
            )
        else:
            trained_policy_enabled = st.toggle(
                "Drive AI ships with trained policy", key="trained_policy_enabled",
                help="Replace the hand-written controller with the PPO policy trained headlessly."
            )
            if trained_policy_enabled:
                if st.session_state.get("trained_policy_path") not in policy_files:
                    st.session_state["trained_policy_path"] = default_policy_path
                def _fmt_policy(path):
                    info = registry_by_path.get(path)
                    if not info or "world" not in info:
                        return os.path.basename(path)
                    tags = [f"{info['world']} x{info.get('worldScale', 1):g}", f"{info.get('obsDim', 38)}d"]
                    if info.get("fuelFeatures"):
                        tags.append("fuel")
                    if info.get("waves"):
                        tags.append("waves")
                    if info.get("actionMode") and info["actionMode"] != "flat":
                        tags.append(info["actionMode"])
                    return f"{info['name']} ({', '.join(tags)})"
                trained_policy_path = st.selectbox(
                    "Checkpoint", policy_files, key="trained_policy_path", format_func=_fmt_policy
                )
                trained_policy_deterministic = st.toggle(
                    "Deterministic (argmax)", key="trained_policy_deterministic",
                    help=("Off samples from the action distribution, which is how the policy was "
                          "trained and usually scores best. On is easier to read but often weaker.")
                )
                policy_worker = st.toggle(
                    "Run inference in a Web Worker", key="policy_worker",
                    help=("Moves the network off the render thread. Actions lag their observation by "
                          "one frame (16 ms). Falls back to the main thread if workers are unavailable.")
                )
                try:
                    loaded_cp = load_policy_file(trained_policy_path)
                    meta = loaded_cp.get("meta", {})
                    cp_world = meta.get("world", "torus")
                    cp_scale = meta.get("worldScale", 1)
                    cp_obs = loaded_cp.get("obsDim", 38)
                    cp_action_mode = loaded_cp.get("actionMode", "flat")
                    cp_fuel = cp_obs >= 43
                    st.caption(
                        f"stage {meta.get('stage', '?')} · {cp_world}"
                        + (f" x{cp_scale:g}" if cp_world == "sphere" else "")
                        + f" · {cp_obs}-dim ({cp_action_mode}) · {meta.get('trainedSteps', 0):,} steps · "
                        f"return {meta.get('evalMeanReturn', float('nan')):.1f} "
                        f"vs rules {meta.get('heuristicMeanReturn', float('nan')):.1f}"
                    )
                    if meta.get("beatsRandom") is False:
                        st.caption("Note: beat the rule-based controller but not random action selection.")
                    mismatches = []
                    active_world = "sphere" if sphere_on else "torus"
                    if cp_world != active_world:
                        mismatches.append(f"trained on {cp_world}, active world is {active_world}")
                    elif active_world == "sphere" and abs(float(cp_scale) - float(world_scale)) > 0.05:
                        mismatches.append(f"trained at x{cp_scale:g}, active scale is x{world_scale:g}")
                    if fuel_rescue_enabled and alpha_attack_enabled and not cp_fuel:
                        mismatches.append("38-dim policy has no fuel-rescue inputs")
                    if waves_enabled and not meta.get("waves"):
                        mismatches.append("trained without escalating waves")
                    if mismatches:
                        st.info("Off-distribution notice: " + "; ".join(mismatches) + ".")
                except (OSError, ValueError):
                    st.warning("Could not read metadata from this checkpoint.")

        with st.expander("Hybrid ML (Phase 5 / 6)"):
            ml_enabled = st.toggle("Enable ML mode", key="ml_enabled",
                                   help="TensorFlow.js parameter / priority tuning for the rule-based ships.")
            ml_mode = "parameters"
            if ml_enabled:
                ml_mode = st.selectbox("ML mode", ["parameters", "priorities", "both", "full"], key="ml_mode")
            ml_data_collection = st.toggle("Collect training data", key="ml_data_collection",
                                           help="Save samples to IndexedDB every 5 frames.")
            ml_hierarchy_enabled = st.toggle("Hierarchical ML (strategic + tactical)", key="ml_hierarchy_enabled")
            ml_evaluation_enabled = st.toggle("ML evaluation tracker", key="ml_evaluation_enabled")
            marl_curriculum_stage = st.selectbox(
                "Curriculum stage",
                ["0: Asteroids Only", "1: Enemies", "2: Boss Fight", "3: Full Chaos"],
                key="curriculum_stage"
            )
            curriculum_stage_value = int(marl_curriculum_stage.split(":")[0])

            nn_models_enabled = st.toggle("Neural network models (TF.js)", key="nn_models_enabled")
            model_version = "1.0.0"
            multitask_model_path = "none"
            parameter_model_path = "none"
            priority_model_path = "none"
            action_model_path = "none"
            prefer_wasm = True
            if nn_models_enabled:
                model_version = st.text_input("Model version", value="1.0.0")
                prefer_wasm = st.toggle("Use WASM backend", value=True)
                multitask_model_path = st.text_input("Multi-task model path",
                                                     value="/models/multitask_model/model.json")
                parameter_model_path = st.text_input("Parameter model path (5.1)", value="none")
                priority_model_path = st.text_input("Priority model path (5.2)", value="none")
                action_model_path = st.text_input("Action model path (5.3)", value="none")

            if ml_data_collection:
                st.caption("Export from the browser console: `exportMARLTrainingData()`")
            if ml_evaluation_enabled:
                st.caption("Browser console: `getMLSystemStatus()` / `getMLEvaluationSummary()`")

        with st.expander("MARL (experimental)"):
            marl_enabled = st.toggle("Enable MARL cooperative AI", key="marl_enabled")
            marl_training = False
            if marl_enabled:
                marl_training = st.toggle("Training mode", key="marl_training")

    # ------------------------------------------------------------------ Telemetry & Mission Control
    with tab_telemetry:
        st.subheader("🛰 Mission Control Telemetry")
        st.caption("Bidirectional live bridge: sidebar tweaks hot-update the simulation without restarting, and telemetry streams back to Python.")

        bridge_enabled = st.toggle(
            "Bidirectional live bridge (no reload on tweaks)",
            value=True,
            key="bridge_enabled",
            help="Uses components/asteroids_bridge to push config diffs via postMessage instead of restarting the game on every slider drag."
        )
        telemetry_sync = st.toggle(
            "Stream live telemetry to Python charts",
            value=False,
            key="telemetry_sync",
            help="Streams throttled (~0.8 Hz) game telemetry back to Streamlit for live metrics and sparkline charts below."
        )

        st.markdown("**Tactical Intervention Console (Live Dispatch):**")
        st.session_state.setdefault("mc_cmd_nonce", 0)
        st.session_state.setdefault("mc_cmd_name", None)
        st.session_state.setdefault("mc_reset_nonce", 0)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("⚡ Railgun Pod", use_container_width=True, help="Drops piercing ion powerup"):
                st.session_state["mc_cmd_nonce"] += 1
                st.session_state["mc_cmd_name"] = "AIRDROP_RAILGUN"
            if st.button("💥 Orbital EMP", use_container_width=True, help="Stuns all hostile craft 2.5s"):
                st.session_state["mc_cmd_nonce"] += 1
                st.session_state["mc_cmd_name"] = "DETONATE_EMP"
        with col_t2:
            if st.button("🛡 Swarm Barrier", use_container_width=True, help="Synchronizes forcefields"):
                st.session_state["mc_cmd_nonce"] += 1
                st.session_state["mc_cmd_name"] = "OVERCHARGE_SHIELDS"
            if st.button("⛽ Emergency Fuel", use_container_width=True, help="Refills fleet & alpha tanks"):
                st.session_state["mc_cmd_nonce"] += 1
                st.session_state["mc_cmd_name"] = "EMERGENCY_REFUEL"
        if st.button("↻ Restart Match (Keep Settings)", use_container_width=True):
            st.session_state["mc_reset_nonce"] += 1

        telemetry_rate = st.slider("Telemetry Broadcast Rate", 5, 60, step=5, value=10, key="telemetry_rate",
                                   help="Interval in game frames between broadcast telemetry packets (10 frames ≈ 6 Hz).")
        st.caption("Standalone tools (when served via HTTP): `arena.html` (A/B Policy Arena) · `replay.html` (Flight Log Scrubber).")

        with st.expander("MARL Value Decomposition (VDN)", expanded=False):
            st.markdown(r"""
            **Cooperative Swarm Objective:**
            $$\mathcal{V}_{team}(\mathbf{o}) = \sum_{i=1}^N \mathcal{V}_i(o_i)$$
            Counterfactual advantage measures an individual agent's marginal contribution against the swarm baseline:
            $$\mathcal{C}_i = \mathcal{V}_i(o_i) - \frac{1}{N}\sum_{j=1}^N \mathcal{V}_j(o_j)$$
            """)
            st.caption("Live credit decomposition and swarm entropy are rendered on the in-game HUD and included in telemetry logs.")

        with st.expander("Flight Recorder & Export"):
            st.caption("Export high-resolution telemetry JSON directly from the Mission Control bar, then inspect frame-by-frame in `replay.html`.")

    # ------------------------------------------------------------------ Visuals
    with tab_visual:
        fx_enabled = st.toggle("Particles, glow & screen shake", key="fx_enabled",
                               help="Thruster trails, debris, explosions, dock plasma stream, HELP ripple.")
        hud_enabled = st.toggle("Live telemetry HUD", key="hud_enabled",
                                help="Sparklines: score rate, ships alive, Alpha fuel, threat, policy value / entropy.")
        kill_cam_enabled = st.toggle("Kill-cam on boss kills", key="kill_cam_enabled",
                                     help=("Sphere world only: the camera swings to a boss kill and pulls in for a "
                                           "moment, then returns. At most once every ten seconds."))
        st.caption("Audio (browsers only allow sound after your first click or key press):")
        sfx_enabled = st.toggle("Sound effects", key="sfx_enabled",
                                help="Gunfire, explosions, power-up chime, boss-spawn siren, game-over tone.")
        music_enabled = st.toggle("Ambient music", key="music_enabled",
                                  help="Quiet generative drone. Press M in the game to mute or unmute it.")
        st.caption("Policy overlays (need a trained policy driving the ships):")
        policy_overlay = st.toggle("Decision text overlay", key="policy_overlay",
                                   help="Chosen action, confidence, entropy and value next to each ship.")
        action_wheels = st.toggle("Action-probability wheels", key="action_wheels",
                                  help="Rings around each ship: rotation (inner), thrust / shield, fire mode (outer).")
        value_heatmap = st.toggle("Critic value heatmap", key="value_heatmap",
                                  help=("Evaluates the critic with the Alpha (or first agent) moved to each cell of a "
                                        "24×16 grid. Magenta = the policy thinks that spot is dangerous."))
    st.markdown("---")
    st.markdown("**Designed by Dang Hoang, *AI/ML Eng.***")

# Read the game files and combine them
def read_source(path, fallback_comment=None):
    """Read a JS/CSS source file, returning a comment stub if it is missing."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"// {fallback_comment or path + ' not found'}\n"


def get_game_html(num_ai_ships, num_enemy_ships, num_boss_ships, canvas_width, canvas_height, player_ship_active, ml_enabled, ml_mode="parameters", marl_enabled=False, marl_training=False, alpha_attack_enabled=False, formation_type="arrowhead", auto_assign_roles=True, fuel_rescue_enabled=True, adaptive_formation=False, multi_target_mode="focus", escort_mode="none", tactical_sequences=True, formation_transitions=True, advanced_flanking=True, ml_data_collection=False, ml_hierarchy_enabled=False, ml_evaluation_enabled=False, curriculum_stage=3, nn_models_enabled=False, model_version="1.0.0", multitask_model_path="none", parameter_model_path="none", priority_model_path="none", action_model_path="none", prefer_wasm=True, trained_policy_path="none", trained_policy_enabled=False, trained_policy_deterministic=False, policy_overlay=False, policy_worker=False, value_heatmap=False, action_wheels=False, fx_enabled=True, hud_enabled=True, world_mode="torus", sphere_camera="alpha", sphere_zoom=2.6, world_scale=1.0, telemetry_rate=10, sfx_enabled=True, music_enabled=True, kill_cam_enabled=True, waves_enabled=True, boss_phases_enabled=True):
    # rng.js must be first: game.js calls gameRandom() and resolves it lexically,
    # so loading it later leaves the game with an undefined RNG.
    rng_js = read_source('rng.js', 'rng.js not found - game.js will fail to start')
    theme_js = read_source('theme.js')

    # Cosmetic and overlay layers. game.js checks for each of these at load
    # time and runs identically without them.
    fx_js = read_source('fx.js')
    viz_js = read_source('policy_viz.js')
    hud_js = read_source('hud.js')

    # World geometry + the Three.js sphere renderer. world.js must precede
    # game.js (it binds WORLD at load); sphere_view.js only registers an API.
    world_js = read_source('world.js')
    sphere_js = read_source('sphere_view.js')
    sphere_world = (world_mode == 'sphere')

    # Sources for the inference Web Worker. The page is a sandboxed srcdoc
    # iframe with no URL of its own, so the worker is assembled from a Blob of
    # nn.js + policy_worker.js at runtime; both are shipped as JS strings.
    nn_source = read_source('training/nn.js')
    worker_source = read_source('policy_worker.js')

    # Read CSS
    with open('style.css', 'r') as f:
        css = f.read()
    
    # Ship/Bullet/Asteroid/AIShip/EnemyBullet/EnemyShip/EnemyCoordinationSystem
    # live in these files, extracted out of game.js; they must load before
    # game.js (AIShip extends Ship, and game.js instantiates
    # EnemyCoordinationSystem at top level).
    entity_files = ['entities.js', 'ai_ship.js', 'enemy_ship.js', 'enemy_coordination.js']
    js = "".join(read_source(f) + "\n" for f in entity_files)
    with open('game.js', 'r') as f:
        js += f.read()
    
    # Read ML JavaScript modules (always loaded for data collection and NN support)
    #
    # training/nn.js and policy_runner.js are the browser half of the PPO path.
    # The browser deliberately runs the *same* nn.js forward pass the trainer ran,
    # over the same MLFeatureExtractor.extractRL() layout, so there is no model
    # conversion step that could silently reorder or rescale inputs.
    ml_js = ""
    ml_files = [
        'ml_features.js',
        'marl_data_collector.js',
        'ml_model_loader.js',
        'ml_model_adapter.js',
        'ml_hierarchy.js',
        'ml_evaluation.js',
        'training/nn.js',
        'policy_runner.js'
    ]
    for ml_file in ml_files:
        try:
            with open(ml_file, 'r') as f:
                ml_js += f.read() + "\n"
        except FileNotFoundError:
            ml_js += f"// {ml_file} not found\n"

    # Read MARL JavaScript files if MARL is enabled
    marl_js = ""
    if marl_enabled:
        try:
            with open('marl_environment.js', 'r') as f:
                marl_js += f.read() + "\n"
            with open('marl_system.js', 'r') as f:
                marl_js += f.read() + "\n"
            with open('marl_integration.js', 'r') as f:
                marl_js += f.read() + "\n"
        except FileNotFoundError as e:
            print(f"Warning: MARL file not found: {e}")
            marl_js = "// MARL files not found\n"
    
    # The component runs in a sandboxed srcdoc iframe, so it cannot fetch() a
    # policy file from the Streamlit server. Inline the checkpoint instead.
    #
    # It is embedded as a JS *string* that the page JSON.parses, rather than as a
    # bare object literal. That way correctness does not depend on JSON being a
    # subset of JavaScript, and js_string() handles the escaping.
    policy_json = "null"
    policy_error = ""
    if trained_policy_path and trained_policy_path != "none":
        try:
            policy_json = js_string(json.dumps(load_policy_file(trained_policy_path)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            policy_error = f"{type(exc).__name__}: {exc}"

    # Combine into single HTML
    tfjs_scripts = f"{TFJS_TAG}\n        {TFJS_WASM_TAG}" if (nn_models_enabled or ml_enabled) else ""
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {THREE_JS_TAG if sphere_world else ""}
        {tfjs_scripts}
        <style>
        {css}
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            width: {canvas_width}px;
            max-width: {canvas_width}px;
        }}
        .game-container {{
            padding: 10px;
            width: {canvas_width}px;
            max-width: {canvas_width}px;
            margin: 0 auto;
            box-sizing: border-box;
        }}
        #gameCanvas {{
            cursor: crosshair;
            width: {canvas_width}px !important;
            height: {canvas_height}px !important;
            display: block;
            margin: 0 auto;
            /* GPU Hardware Acceleration Hints */
            transform: translateZ(0);
            will-change: contents;
            -webkit-transform: translateZ(0);
            -moz-transform: translateZ(0);
            -ms-transform: translateZ(0);
            backface-visibility: hidden;
            -webkit-backface-visibility: hidden;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
        }}
        #gameCanvas:-webkit-full-screen {{
            width: 100%;
            height: 100%;
        }}
        #gameCanvas:-moz-full-screen {{
            width: 100%;
            height: 100%;
        }}
        #gameCanvas:-ms-fullscreen {{
            width: 100%;
            height: 100%;
        }}
        #gameCanvas:fullscreen {{
            width: 100%;
            height: 100%;
        }}
        .mission-control-bar {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin: 10px auto;
            max-width: 95%;
            padding: 8px 12px;
            background: rgba(10, 15, 30, 0.85);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 210, 255, 0.35);
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6), inset 0 0 15px rgba(0, 210, 255, 0.1);
        }}
        .mc-title {{
            font-family: monospace;
            font-weight: bold;
            font-size: 11px;
            color: #00D2FF;
            letter-spacing: 1px;
            text-shadow: 0 0 8px rgba(0, 210, 255, 0.6);
            margin-right: 4px;
        }}
        .mc-btn {{
            font-family: monospace;
            font-weight: bold;
            font-size: 10px;
            padding: 5px 11px;
            border-radius: 4px;
            cursor: pointer;
            border: 1px solid currentColor;
            background: rgba(0, 0, 0, 0.65);
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            outline: none;
            letter-spacing: 0.5px;
        }}
        .mc-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 0 12px currentColor;
        }}
        .mc-btn:active {{
            transform: translateY(1px);
        }}
        .mc-btn-green {{ color: #00FF66; border-color: #00FF66; }}
        .mc-btn-cyan {{ color: #00E5FF; border-color: #00E5FF; }}
        .mc-btn-gold {{ color: #FFD600; border-color: #FFD600; }}
        .mc-btn-pink {{ color: #FF0077; border-color: #FF0077; }}
        .mc-btn-blue {{ color: #0078FF; border-color: #0078FF; }}
        </style>
    </head>
    <body>
        <div class="game-container">
            <!-- game.js reads/writes these elements directly (score, best score,
                 ship counts); kept off-screen rather than removed so
                 updateScore()/updateAIShipCount()/etc. still have something to
                 write to instead of throwing on a null element. -->
            <div style="position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0);">
                <span id="score">0</span>
                <span id="bestScore">0</span>
                <span id="aiShipCount">0</span>
                <span id="enemyShipCount">0</span>
                <span id="bossShipCount">0</span>
            </div>
            <canvas id="gameCanvas"></canvas>
            <!-- Hidden by default; setupTouchControls() in game.js reveals it
                 only when it detects a touch-capable device. -->
            <div id="touchControls" class="touch-controls" hidden>
                <div class="touch-controls-row">
                    <button type="button" class="touch-btn" data-key="arrowleft" aria-label="Rotate left">◀</button>
                    <button type="button" class="touch-btn" data-key="arrowup" aria-label="Thrust">▲</button>
                    <button type="button" class="touch-btn" data-key="arrowright" aria-label="Rotate right">▶</button>
                </div>
                <div class="touch-controls-row">
                    <button type="button" class="touch-btn" data-key="f" aria-label="Shield">SHIELD</button>
                    <button type="button" class="touch-btn touch-btn-fire" data-key="space" aria-label="Fire">FIRE</button>
                    <button type="button" class="touch-btn" data-key="h" aria-label="Hyperspace">HYPER</button>
                </div>
            </div>
            <div class="mission-control-bar">
                <span class="mc-title">🛰 MISSION CONTROL:</span>
                <button type="button" class="mc-btn mc-btn-green" onclick="if (window.AsteroidsTelemetry) AsteroidsTelemetry.dispatchCommand('AIRDROP_RAILGUN')">⚡ AIRDROP RAILGUN</button>
                <button type="button" class="mc-btn mc-btn-cyan" onclick="if (window.AsteroidsTelemetry) AsteroidsTelemetry.dispatchCommand('OVERCHARGE_SHIELDS')">🛡 OVERCHARGE SHIELDS</button>
                <button type="button" class="mc-btn mc-btn-gold" onclick="if (window.AsteroidsTelemetry) AsteroidsTelemetry.dispatchCommand('DETONATE_EMP')">💥 DETONATE EMP</button>
                <button type="button" class="mc-btn mc-btn-pink" onclick="if (window.AsteroidsTelemetry) AsteroidsTelemetry.dispatchCommand('EMERGENCY_REFUEL')">⛽ EMERGENCY REFUEL</button>
                <button type="button" class="mc-btn mc-btn-blue" onclick="if (window.AsteroidsTelemetry) AsteroidsTelemetry.exportJSON()">💾 EXPORT FLIGHT LOG</button>
            </div>
            <div class="controls">
                <p><strong>Controls:</strong></p>
                <p>Arrow Keys: Move ship (↑ forward, ↓ back, ← left, → right)</p>
                <p>Space: Fire / Speed Boost</p>
                <p>F: Shield Force</p>
                <p>H: Hyperspace</p>
                <p>P: Pause</p>
                <p>M: Toggle music</p>
                <p>Waves: clear every hostile to call the next wave; a boss joins every third one</p>
                <p>ESC: Toggle Fullscreen</p>
                <br>
                <p class="credit-line">
                    Designed by Dang Hoang, <em>AI/ML Eng.</em>
                </p>
            </div>
        </div>
        <script>
        {theme_js}
        {rng_js}
        {world_js}
        {fx_js}
        {ml_js}
        {viz_js}
        {hud_js}
        {sphere_js}
        {marl_js}
        {js}
        // Gameplay layers: waves and boss phase mechanics.
        (function() {{
            if (typeof setWavesEnabled === 'function') setWavesEnabled({str(waves_enabled).lower()});
            if (typeof setBossPhasePlusEnabled === 'function') setBossPhasePlusEnabled({str(boss_phases_enabled).lower()});
        }})();
        // Visual layers (cosmetic; never touch game state).
        (function() {{
            if (typeof setFxEnabled === 'function') setFxEnabled({str(fx_enabled).lower()});
            if (typeof setHudEnabled === 'function') setHudEnabled({str(hud_enabled).lower()});
            if (typeof setSoundEnabled === 'function') setSoundEnabled({str(sfx_enabled).lower()});
            if (typeof setMusicEnabled === 'function') setMusicEnabled({str(music_enabled).lower()});
            if (typeof setValueHeatmap === 'function') setValueHeatmap({str(value_heatmap).lower()});
            if (typeof setActionWheels === 'function') setActionWheels({str(action_wheels).lower()});
            if (typeof AsteroidsTelemetry !== 'undefined' && AsteroidsTelemetry.setBroadcastInterval) {{
                AsteroidsTelemetry.setBroadcastInterval({int(telemetry_rate)});
            }}
        }})();

        // World geometry. Switching to the sphere re-routes motion, wrapping and
        // collisions through world.js and hands rendering to sphere_view.js.
        // If Three.js failed to load, the sphere still runs (great circles, pole
        // crossings) and is drawn on the flat canvas as its equirectangular map.
        (function() {{
            if (typeof setWorldMode !== 'function') return;
            const sphere = {str(sphere_world).lower()};
            if (typeof setWorldScale === 'function') setWorldScale({float(world_scale)});
            const ok = setWorldMode(sphere ? 'sphere' : 'torus');
            if (sphere && typeof SphereView !== 'undefined') {{
                SphereView.setCamera({js_string(sphere_camera)});
                SphereView.setZoom({float(sphere_zoom)});
                if (SphereView.setKillCamEnabled) SphereView.setKillCamEnabled({str(kill_cam_enabled).lower()});
                if (!SphereView.ready()) console.warn('[Sphere] Three.js renderer unavailable; showing the sphere as a flat map');
            }}
            console.log('[World] mode=' + (ok ? 'sphere' : 'torus') + (ok ? ' scale=' + WORLD.getScale() + ' R=' + WORLD.radius().toFixed(1) : ''));
        }})();

        // Hand the trained PPO policy to the in-browser runner.
        (function() {{
            const serialised = {policy_json};
            const payload = serialised ? JSON.parse(serialised) : null;
            const loadError = {js_string(policy_error)};
            if (loadError) {{
                console.error('[Policy] could not read checkpoint:', loadError);
                return;
            }}
            if (!payload) return;
            try {{
                const runner = initializePolicyRunner();
                runner.loadFromObject(payload);
                runner.setDeterministic({str(trained_policy_deterministic).lower()});
                runner.setEnabled({str(trained_policy_enabled).lower()});
                setPolicyOverlay({str(policy_overlay).lower()});
                console.log('[Policy] active:', runner.enabled,
                            'deterministic:', runner.deterministic,
                            'fuel features:', runner.featureOptions.fuelFeatures === true);
                if ({str(policy_worker).lower()} && runner.enabled) {{
                    const started = runner.enableWorker({js_string(nn_source)}, {js_string(worker_source)});
                    if (!started) console.log('[Policy] Web Worker unavailable; inference stays on the main thread');
                }}
            }} catch (err) {{
                // loadFromObject throws on an observation/action shape mismatch.
                // Running anyway would produce confident-looking nonsense.
                console.error('[Policy] refused to load:', err.message);
            }}
        }})();

        // Unified one-shot configuration (replaces legacy 2-second polling loops)
        // and live hot-update listener for bidirectional bridge messages.
        (function() {{
            const initialConfig = {{
                numAIShips: {int(num_ai_ships)},
                numEnemyShips: {int(num_enemy_ships)},
                numBossShips: {int(num_boss_ships)},
                canvasWidth: {int(canvas_width)},
                canvasHeight: {int(canvas_height)},
                playerShipActive: {str(bool(player_ship_active)).lower()},
                mlEnabled: {str(bool(ml_enabled)).lower()},
                mlMode: {js_string(ml_mode)},
                marlEnabled: {str(bool(marl_enabled)).lower()},
                marlTraining: {str(bool(marl_training)).lower()},
                alphaAttackEnabled: {str(bool(alpha_attack_enabled)).lower()},
                fuelRescueEnabled: {str(bool(fuel_rescue_enabled)).lower()},
                formationType: {js_string(formation_type)},
                autoAssignRoles: {str(bool(auto_assign_roles)).lower()},
                adaptiveFormation: {str(bool(adaptive_formation)).lower()},
                multiTargetMode: {js_string(multi_target_mode)},
                escortMode: {js_string(escort_mode)},
                tacticalSequences: {str(bool(tactical_sequences)).lower()},
                formationTransitions: {str(bool(formation_transitions)).lower()},
                advancedFlanking: {str(bool(advanced_flanking)).lower()},
                mlDataCollection: {str(bool(ml_data_collection)).lower()},
                mlHierarchyEnabled: {str(bool(ml_hierarchy_enabled)).lower()},
                mlEvaluationEnabled: {str(bool(ml_evaluation_enabled)).lower()},
                curriculumStage: {int(curriculum_stage)},
                nnModelsEnabled: {str(bool(nn_models_enabled)).lower()},
                modelVersion: {js_string(model_version)},
                multitaskModelPath: {js_string(multitask_model_path)},
                parameterModelPath: {js_string(parameter_model_path)},
                priorityModelPath: {js_string(priority_model_path)},
                actionModelPath: {js_string(action_model_path)},
                preferWasm: {str(bool(prefer_wasm)).lower()},
                fxEnabled: {str(bool(fx_enabled)).lower()},
                hudEnabled: {str(bool(hud_enabled)).lower()},
                sfxEnabled: {str(bool(sfx_enabled)).lower()},
                musicEnabled: {str(bool(music_enabled)).lower()},
                killCamEnabled: {str(bool(kill_cam_enabled)).lower()},
                wavesEnabled: {str(bool(waves_enabled)).lower()},
                bossPhasesEnabled: {str(bool(boss_phases_enabled)).lower()},
                valueHeatmap: {str(bool(value_heatmap)).lower()},
                actionWheels: {str(bool(action_wheels)).lower()},
                policyOverlay: {str(bool(policy_overlay)).lower()},
                trainedPolicyEnabled: {str(bool(trained_policy_enabled)).lower()},
                trainedPolicyDeterministic: {str(bool(trained_policy_deterministic)).lower()},
                worldMode: {js_string(world_mode)},
                sphereCamera: {js_string(sphere_camera)},
                sphereZoom: {float(sphere_zoom)},
                worldScale: {float(world_scale)},
                telemetryRate: {int(telemetry_rate)}
            }};
            window.__ASTEROIDS_CONFIG__ = initialConfig;

            function applyAsteroidsConfig(cfg, isLiveUpdate) {{
                if (!cfg || typeof cfg !== 'object') return;
                if (cfg.canvasWidth && cfg.canvasHeight && typeof setCanvasSize === 'function' && !isLiveUpdate) {{
                    setCanvasSize(cfg.canvasWidth, cfg.canvasHeight);
                }}
                if (typeof setWavesEnabled === 'function' && cfg.wavesEnabled !== undefined) setWavesEnabled(!!cfg.wavesEnabled);
                if (typeof setBossPhasePlusEnabled === 'function' && cfg.bossPhasesEnabled !== undefined) setBossPhasePlusEnabled(!!cfg.bossPhasesEnabled);
                if (typeof setNumAIShips === 'function' && cfg.numAIShips !== undefined) setNumAIShips(cfg.numAIShips);
                if (typeof setNumEnemyShips === 'function' && cfg.numEnemyShips !== undefined) setNumEnemyShips(cfg.numEnemyShips);
                if (typeof setNumBossShips === 'function' && cfg.numBossShips !== undefined) setNumBossShips(cfg.numBossShips);
                if (typeof setPlayerShipActive === 'function' && cfg.playerShipActive !== undefined) setPlayerShipActive(!!cfg.playerShipActive);
                if (typeof setAlphaAttackEnabled === 'function' && cfg.alphaAttackEnabled !== undefined) setAlphaAttackEnabled(!!cfg.alphaAttackEnabled);
                if (typeof setFuelRescueEnabled === 'function' && cfg.fuelRescueEnabled !== undefined) setFuelRescueEnabled(!!cfg.fuelRescueEnabled);
                if (typeof setFormationType === 'function' && cfg.formationType !== undefined) setFormationType(cfg.formationType);
                if (typeof setAutoAssignRoles === 'function' && cfg.autoAssignRoles !== undefined) setAutoAssignRoles(!!cfg.autoAssignRoles);
                if (typeof setAdaptiveFormation === 'function' && cfg.adaptiveFormation !== undefined) setAdaptiveFormation(!!cfg.adaptiveFormation);
                if (typeof setMultiTargetMode === 'function' && cfg.multiTargetMode !== undefined) setMultiTargetMode(cfg.multiTargetMode);
                if (typeof setEscortMode === 'function' && cfg.escortMode !== undefined) setEscortMode(cfg.escortMode);
                if (typeof setTacticalSequences === 'function' && cfg.tacticalSequences !== undefined) setTacticalSequences(!!cfg.tacticalSequences);
                if (typeof setFormationTransitions === 'function' && cfg.formationTransitions !== undefined) setFormationTransitions(!!cfg.formationTransitions);
                if (typeof setAdvancedFlanking === 'function' && cfg.advancedFlanking !== undefined) setAdvancedFlanking(!!cfg.advancedFlanking);
                if (typeof setMLMode === 'function' && cfg.mlEnabled !== undefined) setMLMode(!!cfg.mlEnabled, cfg.mlMode);
                if (typeof setMARLMode === 'function' && cfg.marlEnabled !== undefined) setMARLMode(!!cfg.marlEnabled, !!cfg.marlTraining);
                if (typeof setMLDataCollection === 'function' && cfg.mlDataCollection !== undefined) setMLDataCollection(!!cfg.mlDataCollection);
                if (typeof setMLHierarchy === 'function' && cfg.mlHierarchyEnabled !== undefined) setMLHierarchy(!!cfg.mlHierarchyEnabled);
                if (typeof setMLEvaluation === 'function' && cfg.mlEvaluationEnabled !== undefined) setMLEvaluation(!!cfg.mlEvaluationEnabled, !!cfg.mlEnabled);
                if (typeof setMARLCurriculumStage === 'function' && cfg.curriculumStage !== undefined) setMARLCurriculumStage(cfg.curriculumStage);
                if (cfg.mlDataCollection && typeof initializeMARLDataCollector === 'function') initializeMARLDataCollector();
                if (cfg.nnModelsEnabled && typeof setNeuralNetworkModels === 'function') {{
                    setNeuralNetworkModels(true, {{
                        version: cfg.modelVersion,
                        multiTaskModelPath: cfg.multitaskModelPath,
                        parameterModelPath: cfg.parameterModelPath,
                        priorityModelPath: cfg.priorityModelPath,
                        actionModelPath: cfg.actionModelPath,
                        preferWasm: !!cfg.preferWasm
                    }});
                }}
                if (isLiveUpdate) {{
                    if (typeof setFxEnabled === 'function' && cfg.fxEnabled !== undefined) setFxEnabled(!!cfg.fxEnabled);
                    if (typeof setHudEnabled === 'function' && cfg.hudEnabled !== undefined) setHudEnabled(!!cfg.hudEnabled);
                    if (typeof setSoundEnabled === 'function' && cfg.sfxEnabled !== undefined) setSoundEnabled(!!cfg.sfxEnabled);
                    if (typeof setMusicEnabled === 'function' && cfg.musicEnabled !== undefined) setMusicEnabled(!!cfg.musicEnabled);
                    if (typeof setValueHeatmap === 'function' && cfg.valueHeatmap !== undefined) setValueHeatmap(!!cfg.valueHeatmap);
                    if (typeof setActionWheels === 'function' && cfg.actionWheels !== undefined) setActionWheels(!!cfg.actionWheels);
                    if (typeof setPolicyOverlay === 'function' && cfg.policyOverlay !== undefined) setPolicyOverlay(!!cfg.policyOverlay);
                    if (typeof setTrainedPolicyEnabled === 'function' && cfg.trainedPolicyEnabled !== undefined) {{
                        setTrainedPolicyEnabled(!!cfg.trainedPolicyEnabled, !!cfg.trainedPolicyDeterministic);
                    }}
                    if (typeof setWorldScale === 'function' && cfg.worldScale !== undefined) setWorldScale(cfg.worldScale);
                    if (typeof setWorldMode === 'function' && cfg.worldMode !== undefined) {{
                        const sphere = cfg.worldMode === 'sphere';
                        setWorldMode(sphere ? 'sphere' : 'torus');
                        if (sphere && typeof SphereView !== 'undefined') {{
                            if (cfg.sphereCamera) SphereView.setCamera(cfg.sphereCamera);
                            if (cfg.sphereZoom !== undefined) SphereView.setZoom(cfg.sphereZoom);
                            if (SphereView.setKillCamEnabled && cfg.killCamEnabled !== undefined) SphereView.setKillCamEnabled(!!cfg.killCamEnabled);
                        }}
                    }}
                    if (typeof AsteroidsTelemetry !== 'undefined' && AsteroidsTelemetry.setBroadcastInterval && cfg.telemetryRate !== undefined) {{
                        AsteroidsTelemetry.setBroadcastInterval(cfg.telemetryRate);
                    }}
                }}
            }}

            window.applyAsteroidsConfig = applyAsteroidsConfig;
            applyAsteroidsConfig(initialConfig, false);

            window.addEventListener('message', function(event) {{
                if (!event.data || typeof event.data !== 'object') return;
                if (event.data.type === 'ASTEROIDS_UPDATE_CONFIG') {{
                    applyAsteroidsConfig(event.data.config, true);
                }} else if (event.data.type === 'ASTEROIDS_RESET_GAME' && typeof initGame === 'function') {{
                    initGame(event.data.seed);
                }}
            }});
        }})();
        </script>
    </body>
    </html>
    """
    return html

# Embed the game
formation_type_value = formation_type if alpha_attack_enabled else "arrowhead"
auto_assign_roles_value = auto_assign_roles if alpha_attack_enabled else True
fuel_rescue_value = fuel_rescue_enabled if alpha_attack_enabled else False
world_mode_value = "sphere" if world_mode.startswith("Sphere") else "torus"
sphere_camera_value = {"Follow Alpha": "alpha", "Follow fleet": "fleet", "Follow boss": "boss", "Slow orbit": "orbit"}.get(sphere_camera, "alpha")

game_html = get_game_html(num_ai_ships, num_enemy_ships, num_boss_ships, canvas_width, canvas_height, player_ship_active, ml_enabled, ml_mode, marl_enabled, marl_training, alpha_attack_enabled, formation_type_value, auto_assign_roles_value, fuel_rescue_value, adaptive_formation_value, multi_target_mode_value, escort_mode_value, tactical_sequences_enabled, formation_transitions_enabled, advanced_flanking_enabled, ml_data_collection, ml_hierarchy_enabled, ml_evaluation_enabled, curriculum_stage_value, nn_models_enabled, model_version, multitask_model_path, parameter_model_path, priority_model_path, action_model_path, prefer_wasm, trained_policy_path, trained_policy_enabled, trained_policy_deterministic, policy_overlay,
                          policy_worker=policy_worker, value_heatmap=value_heatmap, action_wheels=action_wheels,
                          fx_enabled=fx_enabled, hud_enabled=hud_enabled,
                          world_mode=world_mode_value,
                          sphere_camera=sphere_camera_value,
                          sphere_zoom=sphere_zoom, world_scale=world_scale,
                          telemetry_rate=telemetry_rate,
                          sfx_enabled=sfx_enabled, music_enabled=music_enabled,
                          kill_cam_enabled=kill_cam_enabled,
                          waves_enabled=waves_enabled, boss_phases_enabled=boss_phases_enabled)
component_height = canvas_height + 270

BRIDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "asteroids_bridge")
live_telemetry = None
if bridge_enabled and os.path.isdir(BRIDGE_DIR):
    _bridge_component = components.declare_component("asteroids_bridge", path=BRIDGE_DIR)
    # Only structural changes (canvas size, checkpoint, TF.js/MARL script bundle,
    # or toggling Three.js on/off) rebuild the inner iframe; all other sidebar
    # tweaks flow live via postMessage without resetting the match.
    boot_key = "|".join([
        str(canvas_width), str(canvas_height), world_mode_value,
        str(trained_policy_path), str(bool(policy_worker)),
        str(bool(marl_enabled)), str(bool(nn_models_enabled or ml_enabled)),
    ])
    live_config = {
        "numAIShips": int(num_ai_ships),
        "numEnemyShips": int(num_enemy_ships),
        "numBossShips": int(num_boss_ships),
        "playerShipActive": bool(player_ship_active),
        "alphaAttackEnabled": bool(alpha_attack_enabled),
        "fuelRescueEnabled": bool(fuel_rescue_value),
        "formationType": formation_type_value,
        "autoAssignRoles": bool(auto_assign_roles_value),
        "adaptiveFormation": bool(adaptive_formation_value),
        "multiTargetMode": multi_target_mode_value,
        "escortMode": escort_mode_value,
        "tacticalSequences": bool(tactical_sequences_enabled),
        "formationTransitions": bool(formation_transitions_enabled),
        "advancedFlanking": bool(advanced_flanking_enabled),
        "fxEnabled": bool(fx_enabled),
        "hudEnabled": bool(hud_enabled),
        "sfxEnabled": bool(sfx_enabled),
        "musicEnabled": bool(music_enabled),
        "killCamEnabled": bool(kill_cam_enabled),
        "wavesEnabled": bool(waves_enabled),
        "bossPhasesEnabled": bool(boss_phases_enabled),
        "valueHeatmap": bool(value_heatmap),
        "actionWheels": bool(action_wheels),
        "policyOverlay": bool(policy_overlay),
        "trainedPolicyEnabled": bool(trained_policy_enabled),
        "trainedPolicyDeterministic": bool(trained_policy_deterministic),
        "worldMode": world_mode_value,
        "sphereCamera": sphere_camera_value,
        "sphereZoom": float(sphere_zoom),
        "worldScale": float(world_scale),
        "telemetryRate": int(telemetry_rate),
    }
    live_telemetry = _bridge_component(
        bootKey=boot_key,
        html=game_html,
        config=live_config,
        height=component_height,
        telemetrySync=bool(telemetry_sync),
        command=st.session_state.get("mc_cmd_name"),
        commandNonce=int(st.session_state.get("mc_cmd_nonce", 0)),
        resetNonce=int(st.session_state.get("mc_reset_nonce", 0)),
        key="asteroids_live_bridge",
        default=None,
    )
else:
    components.html(game_html, height=component_height, scrolling=False)

# Live Python Telemetry Dashboard (renders when telemetry_sync streams packets back)
if isinstance(live_telemetry, dict) and telemetry_sync:
    hist = st.session_state.setdefault("telemetry_history", [])
    if not hist or hist[-1].get("frame") != live_telemetry.get("frame"):
        hist.append({
            "frame": live_telemetry.get("frame", 0),
            "score": live_telemetry.get("score", 0),
            "scoreRate": live_telemetry.get("scoreRate", 0),
            "alphaFuel": live_telemetry.get("alphaFuel", 100),
            "fleetCount": live_telemetry.get("fleetCount", 0),
        })
        if len(hist) > 120:
            del hist[:-120]
    with st.expander("📡 Live Python Telemetry Dashboard", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score", f"{live_telemetry.get('score', 0):,}")
        m2.metric("Score / s", f"{live_telemetry.get('scoreRate', 0)}")
        m3.metric("Alpha Fuel", f"{live_telemetry.get('alphaFuel', 100)}%")
        m4.metric("Active Fleet", f"{live_telemetry.get('fleetCount', 0)}")
        if len(hist) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                st.line_chart({"Score Rate (/s)": [h["scoreRate"] for h in hist]}, height=150)
            with c2:
                st.line_chart({"Alpha Fuel (%)": [h["alphaFuel"] for h in hist]}, height=150)


