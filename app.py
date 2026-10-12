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

# Published PPO checkpoints. A run is only published if it significantly beat
# the rule-based controller on held-out seeds, so anything listed here has
# already cleared that bar.
policy_files = sorted(glob.glob(os.path.join("models", "policy_*.json")))

# Widget keys and their defaults. Presets override a subset of these.
DEFAULTS = {
    "num_ai_ships": 10, "num_enemy_ships": 10, "num_boss_ships": 10,
    "canvas_width": 1200, "canvas_height": 600,
    "player_ship_active": False,
    "alpha_attack_enabled": True, "formation_type": "arrowhead",
    "auto_assign_roles": True, "fuel_rescue_enabled": True,
    "adaptive_formation": True, "multi_target_mode": "focus", "escort_mode": "guard",
    "tactical_sequences": True, "formation_transitions": True, "advanced_flanking": True,
    "ml_data_collection": True, "ml_hierarchy_enabled": True, "ml_evaluation_enabled": True,
    "curriculum_stage": "3: Full Chaos",
    "nn_models_enabled": True, "ml_enabled": True, "ml_mode": "full",
    "marl_enabled": True, "marl_training": True,
    "trained_policy_enabled": True,
    "trained_policy_path": policy_files[-1] if policy_files else "none",
    "trained_policy_deterministic": True, "policy_overlay": True,
    "policy_worker": True, "value_heatmap": True, "action_wheels": True,
    "fx_enabled": True, "hud_enabled": True,
    "sfx_enabled": True, "music_enabled": True, "kill_cam_enabled": True,
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
                    st.session_state["trained_policy_path"] = policy_files[-1]
                trained_policy_path = st.selectbox("Checkpoint", policy_files, key="trained_policy_path")
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
                    meta = load_policy_file(trained_policy_path).get("meta", {})
                    st.caption(
                        f"stage {meta.get('stage', '?')} · {meta.get('trainedSteps', 0):,} env steps · "
                        f"eval return {meta.get('evalMeanReturn', float('nan')):.1f} "
                        f"vs rule-based {meta.get('heuristicMeanReturn', float('nan')):.1f}"
                    )
                    if meta.get("beatsRandom") is False:
                        st.caption("Note: beat the rule-based controller but not random action selection.")
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
        st.caption("Bidirectional telemetry streaming between Streamlit and the Asteroids simulation.")

        st.markdown("**Tactical Intervention Console:**")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.info("⚡ **Railgun Pod**\nDrops piercing ion powerup")
            st.info("💥 **Orbital EMP**\nStuns all hostile craft 2.5s")
        with col_t2:
            st.info("🛡 **Swarm Barrier**\nSynchronizes forcefields")
            st.info("⛽ **Emergency Fuel**\nRefills fleet & alpha tanks")

        telemetry_rate = st.slider("Telemetry Broadcast Rate", 5, 60, step=5, value=10, key="telemetry_rate",
                                   help="Interval in game frames between broadcast telemetry packets (10 frames ≈ 6 Hz).")
        st.caption("Use the interactive Mission Control bar below the canvas or press `T` to toggle the live in-canvas flight telemetry HUD.")

        with st.expander("MARL Value Decomposition (VDN)", expanded=True):
            st.markdown(r"""
            **Cooperative Swarm Objective:**
            $$\mathcal{V}_{team}(\mathbf{o}) = \sum_{i=1}^N \mathcal{V}_i(o_i)$$
            Counterfactual advantage measures an individual agent's marginal contribution against the swarm baseline:
            $$\mathcal{C}_i = \mathcal{V}_i(o_i) - \frac{1}{N}\sum_{j=1}^N \mathcal{V}_j(o_j)$$
            """)
            st.caption("Live credit decomposition and swarm entropy are rendered on the in-game HUD and included in telemetry logs.")

        with st.expander("Flight Recorder & Export"):
            st.caption("Export high-resolution telemetry log containing past agent trajectories, fuel levels, subsystem integrity, and MARL metrics directly from the browser Mission Control bar.")

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


def get_game_html(num_ai_ships, num_enemy_ships, num_boss_ships, canvas_width, canvas_height, player_ship_active, ml_enabled, ml_mode="parameters", marl_enabled=False, marl_training=False, alpha_attack_enabled=False, formation_type="arrowhead", auto_assign_roles=True, fuel_rescue_enabled=True, adaptive_formation=False, multi_target_mode="focus", escort_mode="none", tactical_sequences=True, formation_transitions=True, advanced_flanking=True, ml_data_collection=False, ml_hierarchy_enabled=False, ml_evaluation_enabled=False, curriculum_stage=3, nn_models_enabled=False, model_version="1.0.0", multitask_model_path="none", parameter_model_path="none", priority_model_path="none", action_model_path="none", prefer_wasm=True, trained_policy_path="none", trained_policy_enabled=False, trained_policy_deterministic=False, policy_overlay=False, policy_worker=False, value_heatmap=False, action_wheels=False, fx_enabled=True, hud_enabled=True, world_mode="torus", sphere_camera="alpha", sphere_zoom=2.6, world_scale=1.0, telemetry_rate=10, sfx_enabled=True, music_enabled=True, kill_cam_enabled=True):
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
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {THREE_JS_TAG if sphere_world else ""}
        {TFJS_TAG}
        {TFJS_WASM_TAG}
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

        // Set number of AI ships from Streamlit
        (function() {{
            const targetCount = {num_ai_ships};
            if (typeof setNumAIShips === 'function') {{
                setNumAIShips(targetCount);
            }} else {{
                // Wait for function to be available
                setTimeout(function() {{
                    if (typeof setNumAIShips === 'function') {{
                        setNumAIShips(targetCount);
                    }}
                }}, 100);
            }}
        }})();
        
        // Set number of enemy ships from Streamlit
        (function() {{
            const targetCount = {num_enemy_ships};
            if (typeof setNumEnemyShips === 'function') {{
                setNumEnemyShips(targetCount);
            }} else {{
                // Wait for function to be available
                setTimeout(function() {{
                    if (typeof setNumEnemyShips === 'function') {{
                        setNumEnemyShips(targetCount);
                    }}
                }}, 100);
            }}
        }})();
        
        // Set number of boss ships from Streamlit
        (function() {{
            const targetCount = {num_boss_ships};
            if (typeof setNumBossShips === 'function') {{
                setNumBossShips(targetCount);
            }} else {{
                // Wait for function to be available
                setTimeout(function() {{
                    if (typeof setNumBossShips === 'function') {{
                        setNumBossShips(targetCount);
                    }}
                }}, 100);
            }}
        }})();
        
        // Set canvas size from Streamlit
        (function() {{
            const width = {canvas_width};
            const height = {canvas_height};
            if (typeof setCanvasSize === 'function') {{
                setCanvasSize(width, height);
            }} else {{
                // Wait for function to be available
                setTimeout(function() {{
                    if (typeof setCanvasSize === 'function') {{
                        setCanvasSize(width, height);
                    }}
                }}, 100);
            }}
        }})();
        
        // Set player ship active state from Streamlit
        (function() {{
            // Default to true if not explicitly set
            const activeState = {str(player_ship_active).lower()};
            const isActive = activeState === 'true' || activeState === true;
            
            // Set state immediately if function is available
            if (typeof setPlayerShipActive === 'function') {{
                setPlayerShipActive(isActive);
            }}
            
            // Ensure function is called after game initialization
            function setActiveState() {{
                if (typeof setPlayerShipActive === 'function') {{
                    setPlayerShipActive(isActive);
                }} else {{
                    setTimeout(setActiveState, 100);
                }}
            }}
            
            // Wait for game to initialize, then set state again to ensure it sticks
            setTimeout(setActiveState, 300);
            
            // Also set it periodically to override any resets (but less frequently)
            setInterval(function() {{
                if (typeof setPlayerShipActive === 'function') {{
                    setPlayerShipActive(isActive);
                }}
            }}, 2000);
        }})();
        
        // Set ML mode from Streamlit
        (function() {{
            const mlState = {str(ml_enabled).lower()};
            const mlEnabled = mlState === 'true' || mlState === true;
            const mlMode = {js_string(ml_mode)};
            
            // Set state immediately if function is available
            if (typeof setMLMode === 'function') {{
                setMLMode(mlEnabled, mlMode);
            }}
            
            // Ensure function is called after game initialization
            function setMLState() {{
                if (typeof setMLMode === 'function') {{
                    setMLMode(mlEnabled, mlMode);
                }} else {{
                    setTimeout(setMLState, 100);
                }}
            }}
            
            // Wait for game to initialize, then set state again to ensure it sticks
            setTimeout(setMLState, 300);
            
            // Also set it periodically to override any resets (but less frequently)
            setInterval(function() {{
                if (typeof setMLMode === 'function') {{
                    setMLMode(mlEnabled);
                }}
            }}, 2000);
        }})();
        
        // Set MARL mode from Streamlit
        (function() {{
            const marlState = {str(marl_enabled).lower()};
            const marlTrainingState = {str(marl_training).lower()};
            const marlEnabled = marlState === 'true' || marlState === true;
            const marlTraining = marlTrainingState === 'true' || marlTrainingState === true;
            
            // Set state immediately if function is available
            if (typeof setMARLMode === 'function') {{
                setMARLMode(marlEnabled, marlTraining);
            }}
            
            // Ensure function is called after game initialization
            function setMARLState() {{
                if (typeof setMARLMode === 'function') {{
                    setMARLMode(marlEnabled, marlTraining);
                }} else {{
                    setTimeout(setMARLState, 100);
                }}
            }}
            
            // Wait for game to initialize, then set state again to ensure it sticks
            setTimeout(setMARLState, 500);
            
            // Also set it periodically to override any resets (but less frequently)
            setInterval(function() {{
                if (typeof setMARLMode === 'function') {{
                    setMARLMode(marlEnabled, marlTraining);
                }}
            }}, 2000);
        }})();
        
        // Advanced ML Phase 6 configuration
        (function() {{
            const dataCollection = {str(ml_data_collection).lower()};
            const hierarchy = {str(ml_hierarchy_enabled).lower()};
            const evaluation = {str(ml_evaluation_enabled).lower()};
            const curriculum = {curriculum_stage};
            const nnEnabled = {str(nn_models_enabled).lower()};
            const mlEnabledState = {str(ml_enabled).lower()};
            
            function initAdvancedML() {{
                if (typeof setMLDataCollection === 'function') {{
                    setMLDataCollection(dataCollection === 'true' || dataCollection === true);
                }}
                if (typeof setMLHierarchy === 'function') {{
                    setMLHierarchy(hierarchy === 'true' || hierarchy === true);
                }}
                if (typeof setMLEvaluation === 'function') {{
                    setMLEvaluation(evaluation === 'true' || evaluation === true, mlEnabledState === 'true');
                }}
                if (typeof setMARLCurriculumStage === 'function') {{
                    setMARLCurriculumStage(curriculum);
                }}
                if (typeof initializeMARLDataCollector === 'function') {{
                    initializeMARLDataCollector();
                }}
                if (nnEnabled === 'true' || nnEnabled === true) {{
                    if (typeof setNeuralNetworkModels === 'function') {{
                        setNeuralNetworkModels(true, {{
                            version: {js_string(model_version)},
                            multiTaskModelPath: {js_string(multitask_model_path)},
                            parameterModelPath: {js_string(parameter_model_path)},
                            priorityModelPath: {js_string(priority_model_path)},
                            actionModelPath: {js_string(action_model_path)},
                            preferWasm: {str(prefer_wasm).lower()}
                        }});
                    }}
                }}
            }}
            
            setTimeout(initAdvancedML, 400);
            setTimeout(initAdvancedML, 1000);
        }})();
        
        // Set Alpha Attack mode from Streamlit
        (function() {{
            const alphaState = {str(alpha_attack_enabled).lower()};
            const alphaEnabled = alphaState === 'true' || alphaState === true;
            
            // Set state immediately if function is available
            if (typeof setAlphaAttackEnabled === 'function') {{
                setAlphaAttackEnabled(alphaEnabled);
            }}
            
            // Ensure function is called after game initialization
            function setAlphaState() {{
                if (typeof setAlphaAttackEnabled === 'function') {{
                    setAlphaAttackEnabled(alphaEnabled);
                }} else {{
                    setTimeout(setAlphaState, 100);
                }}
            }}
            
            setTimeout(setAlphaState, 300);
            
            // Also set it periodically to ensure it sticks
            setInterval(function() {{
                if (typeof setAlphaAttackEnabled === 'function') {{
                    setAlphaAttackEnabled(alphaEnabled);
                }}
            }}, 2000);
        }})();

        // Alpha fuel rescue (HELP ping + green fuel ship refuel)
        (function() {{
            const fuelRescueState = {str(fuel_rescue_enabled).lower()};
            const fuelRescueOn = fuelRescueState === 'true' || fuelRescueState === true;

            function setFuelRescueState() {{
                if (typeof setFuelRescueEnabled === 'function') {{
                    setFuelRescueEnabled(fuelRescueOn);
                }} else {{
                    setTimeout(setFuelRescueState, 100);
                }}
            }}

            setTimeout(setFuelRescueState, 350);
            setInterval(function() {{
                if (typeof setFuelRescueEnabled === 'function') {{
                    setFuelRescueEnabled(fuelRescueOn);
                }}
            }}, 2000);
        }})();
        
        // PHASE 1 & 2: Set Formation Type and Role Assignment
        (function() {{
            const formationType = {js_string(formation_type)};
            const autoAssignRoles = {str(auto_assign_roles).lower()};
            const adaptiveFormation = {str(adaptive_formation).lower()};
            const multiTargetMode = {js_string(multi_target_mode)};
            const escortMode = {js_string(escort_mode)};
            const tacticalSequences = {str(tactical_sequences).lower()};
            const formationTransitions = {str(formation_transitions).lower()};
            const advancedFlanking = {str(advanced_flanking).lower()};
            
            // Set formation type
            if (typeof setFormationType === 'function') {{
                setFormationType(formationType);
            }}
            
            // Set auto-assign roles
            if (typeof setAutoAssignRoles === 'function') {{
                setAutoAssignRoles(autoAssignRoles === 'true' || autoAssignRoles === true);
            }}
            
            // PHASE 3: Set adaptive formation
            if (typeof setAdaptiveFormation === 'function') {{
                setAdaptiveFormation(adaptiveFormation === 'true' || adaptiveFormation === true);
            }}
            
            // PHASE 3: Set multi-target mode
            if (typeof setMultiTargetMode === 'function') {{
                setMultiTargetMode(multiTargetMode);
            }}
            
            // PHASE 3: Set escort mode
            if (typeof setEscortMode === 'function') {{
                setEscortMode(escortMode);
            }}
            
            // Ensure functions are called after game initialization
            function setFormationState() {{
                if (typeof setFormationType === 'function') {{
                    setFormationType(formationType);
                }} else {{
                    setTimeout(setFormationState, 100);
                }}
            }}
            
            function setRoleState() {{
                if (typeof setAutoAssignRoles === 'function') {{
                    setAutoAssignRoles(autoAssignRoles === 'true' || autoAssignRoles === true);
                }} else {{
                    setTimeout(setRoleState, 100);
                }}
            }}
            
            setTimeout(setFormationState, 300);
            setTimeout(setRoleState, 300);
            
            // Also set periodically to ensure they stick
            setInterval(function() {{
                if (typeof setFormationType === 'function') {{
                    setFormationType(formationType);
                }}
                if (typeof setAutoAssignRoles === 'function') {{
                    setAutoAssignRoles(autoAssignRoles === 'true' || autoAssignRoles === true);
                }}
                if (typeof setAdaptiveFormation === 'function') {{
                    setAdaptiveFormation(adaptiveFormation === 'true' || adaptiveFormation === true);
                }}
                if (typeof setMultiTargetMode === 'function') {{
                    setMultiTargetMode(multiTargetMode);
                }}
                if (typeof setEscortMode === 'function') {{
                    setEscortMode(escortMode);
                }}
                if (typeof setTacticalSequences === 'function') {{
                    setTacticalSequences(tacticalSequences === 'true' || tacticalSequences === true);
                }}
                if (typeof setFormationTransitions === 'function') {{
                    setFormationTransitions(formationTransitions === 'true' || formationTransitions === true);
                }}
                if (typeof setAdvancedFlanking === 'function') {{
                    setAdvancedFlanking(advancedFlanking === 'true' || advancedFlanking === true);
                }}
            }}, 2000);
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
game_html = get_game_html(num_ai_ships, num_enemy_ships, num_boss_ships, canvas_width, canvas_height, player_ship_active, ml_enabled, ml_mode, marl_enabled, marl_training, alpha_attack_enabled, formation_type_value, auto_assign_roles_value, fuel_rescue_value, adaptive_formation_value, multi_target_mode_value, escort_mode_value, tactical_sequences_enabled, formation_transitions_enabled, advanced_flanking_enabled, ml_data_collection, ml_hierarchy_enabled, ml_evaluation_enabled, curriculum_stage_value, nn_models_enabled, model_version, multitask_model_path, parameter_model_path, priority_model_path, action_model_path, prefer_wasm, trained_policy_path, trained_policy_enabled, trained_policy_deterministic, policy_overlay,
                          policy_worker=policy_worker, value_heatmap=value_heatmap, action_wheels=action_wheels,
                          fx_enabled=fx_enabled, hud_enabled=hud_enabled,
                          world_mode="sphere" if world_mode.startswith("Sphere") else "torus",
                          sphere_camera={"Follow Alpha": "alpha", "Follow fleet": "fleet", "Follow boss": "boss", "Slow orbit": "orbit"}.get(sphere_camera, "alpha"),
                          sphere_zoom=sphere_zoom, world_scale=world_scale,
                          telemetry_rate=telemetry_rate,
                          sfx_enabled=sfx_enabled, music_enabled=music_enabled,
                          kill_cam_enabled=kill_cam_enabled)
# Adjust component dimensions to match canvas + UI elements
# Height: canvas height + status bar (~40px) + mission control bar (~50px) + controls section (~160px) + padding (~20px)
component_height = canvas_height + 270
# Note: components.html() doesn't support width parameter, width is controlled via CSS
components.html(game_html, height=component_height, scrolling=False)

