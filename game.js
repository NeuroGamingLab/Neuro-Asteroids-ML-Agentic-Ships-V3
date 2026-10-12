// game.js - game loop, input, rules and the headless training hooks.
//
// Everything below runs inside one function scope, so its ~150 top-level names no
// longer leak into the global scope where any script could read or overwrite them
// by accident. The deliberate public surface is declared once, here, and is derived
// from what the other scripts (entities, ai_ship, enemy_ship, the HUD, the sphere
// view, the RL environment, the tests) actually use.
//
// Shared mutable state is exposed through accessor properties rather than copied
// values: other files keep reading and assigning bare `aiShips`, `score`,
// `numAIShips` and so on, and the accessors forward to the one real variable. They
// are installed before any other code in this file runs because initGame() at the
// bottom calls back into scripts that read these names.
(function (root) {
    // ---- public surface ------------------------------------------------------
    const publicState = {
        canvas: [() => canvas],
        ctx: [() => ctx, (value) => { ctx = value; }],
        WORLD_GEO: [() => WORLD_GEO],
        bestScore: [() => bestScore, (value) => { bestScore = value; }],
        score: [() => score, (value) => { score = value; }],
        gameRunning: [() => gameRunning, (value) => { gameRunning = value; }],
        playerShipActive: [() => playerShipActive, (value) => { playerShipActive = value; }],
        episodeStepCount: [() => episodeStepCount, (value) => { episodeStepCount = value; }],
        episodeId: [() => episodeId, (value) => { episodeId = value; }],
        episodeDone: [() => episodeDone, (value) => { episodeDone = value; }],
        fuelEvents: [() => fuelEvents, (value) => { fuelEvents = value; }],
        policyControlEnabled: [() => policyControlEnabled, (value) => { policyControlEnabled = value; }],
        FIRE_FORWARD: [() => FIRE_FORWARD],
        FIRE_AT_ASTEROID: [() => FIRE_AT_ASTEROID],
        FIRE_AT_ENEMY: [() => FIRE_AT_ENEMY],
        ACTION_SPACE_SIZE: [() => ACTION_SPACE_SIZE],
        expertActionRecording: [() => expertActionRecording, (value) => { expertActionRecording = value; }],
        FXL: [() => FXL],
        mlGlobalFrame: [() => mlGlobalFrame, (value) => { mlGlobalFrame = value; }],
        mlUseNeuralNetwork: [() => mlUseNeuralNetwork, (value) => { mlUseNeuralNetwork = value; }],
        mlDataCollectionEnabled: [() => mlDataCollectionEnabled, (value) => { mlDataCollectionEnabled = value; }],
        ship: [() => ship],
        aiShips: [() => aiShips, (value) => { aiShips = value; }],
        bullets: [() => bullets, (value) => { bullets = value; }],
        enemyBullets: [() => enemyBullets, (value) => { enemyBullets = value; }],
        enemyShips: [() => enemyShips, (value) => { enemyShips = value; }],
        powerUps: [() => powerUps, (value) => { powerUps = value; }],
        asteroids: [() => asteroids, (value) => { asteroids = value; }],
        numAIShips: [() => numAIShips, (value) => { numAIShips = value; }],
        numEnemyShips: [() => numEnemyShips, (value) => { numEnemyShips = value; }],
        numBossShips: [() => numBossShips, (value) => { numBossShips = value; }],
        alphaAttackEnabled: [() => alphaAttackEnabled, (value) => { alphaAttackEnabled = value; }],
        formationType: [() => formationType, (value) => { formationType = value; }],
        autoAssignRoles: [() => autoAssignRoles, (value) => { autoAssignRoles = value; }],
        adaptiveFormationEnabled: [() => adaptiveFormationEnabled, (value) => { adaptiveFormationEnabled = value; }],
        multiTargetMode: [() => multiTargetMode, (value) => { multiTargetMode = value; }],
        escortMode: [() => escortMode, (value) => { escortMode = value; }],
        tacticalAttackSequencesEnabled: [() => tacticalAttackSequencesEnabled, (value) => { tacticalAttackSequencesEnabled = value; }],
        dynamicFormationTransitionsEnabled: [() => dynamicFormationTransitionsEnabled, (value) => { dynamicFormationTransitionsEnabled = value; }],
        advancedFlankingEnabled: [() => advancedFlankingEnabled, (value) => { advancedFlankingEnabled = value; }],
        fuelRescueEnabled: [() => fuelRescueEnabled, (value) => { fuelRescueEnabled = value; }],
        fuelRescueDispatchCooldown: [() => fuelRescueDispatchCooldown, (value) => { fuelRescueDispatchCooldown = value; }],
    };
    for (const name of Object.keys(publicState)) {
        const [get, set] = publicState[name];
        Object.defineProperty(root, name, { get, set, configurable: true, enumerable: false });
    }

    // Functions are hoisted, so these are valid before their declarations run.
    root.sphereWorld = sphereWorld;
    root.worldW = worldW;
    root.worldH = worldH;
    root.worldScale = worldScale;
    root.setWorldScale = setWorldScale;
    root.setWorldMode = setWorldMode;
    root.worldToScreen = worldToScreen;
    root.playFireSound = playFireSound;
    root.exitFullscreen = exitFullscreen;
    root.consumeAgentCredit = consumeAgentCredit;
    root.consumeFuelEvents = consumeFuelEvents;
    root.decodeAction = decodeAction;
    root.encodeAction = encodeAction;
    root.setPolicyControl = setPolicyControl;
    root.setPolicyOverlay = setPolicyOverlay;
    root.setExpertActionRecording = setExpertActionRecording;
    root.readExpertActions = readExpertActions;
    root.updateAlphaHelpPing = updateAlphaHelpPing;
    root.ensureAlphaAssignment = ensureAlphaAssignment;
    root.getAlphaCombatWingmen = getAlphaCombatWingmen;
    root.assignFuelShipRole = assignFuelShipRole;
    root.initGame = initGame;
    root.setupTouchControls = setupTouchControls;
    root.update = update;
    root.triggerEMPBlast = triggerEMPBlast;
    root.draw = draw;
    root.updateScore = updateScore;
    root.updateAIShipCount = updateAIShipCount;

    // ---- game ----------------------------------------------------------------
// Game Canvas Setup
const canvas = document.getElementById('gameCanvas');
// Try to get hardware-accelerated 2D context, fallback to regular 2D
let ctx = null;
try {
    // Request hardware acceleration hints
    ctx = canvas.getContext('2d', {
        alpha: false, // Opaque background for better performance
        desynchronized: false, // Keep synchronized for better compatibility
        willReadFrequently: false // Optimize for rendering, not reading
    });
    
    // Enable hardware acceleration if available
    if (ctx && ctx.imageSmoothingEnabled !== undefined) {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
    }
    
    console.log('[GPU] 2D Canvas context initialized with hardware acceleration hints');
} catch (e) {
    // Fallback to regular 2D context
    ctx = canvas.getContext('2d');
    console.log('[GPU] Using standard 2D context (hardware acceleration may not be available)');
}
// The 2D context above is opaque (alpha: false), so when the sphere view is on,
// the WebGL canvas is stacked over it and `ctx` is repointed at a transparent
// overlay canvas for HUD/labels. Kept so setWorldMode('torus') can restore it.
const baseCtx = ctx;

// World geometry (world.js). Optional: without it the game is the flat torus it
// always was. game.js only branches into WORLD when the sphere is active, so the
// headless sim and the browser share one geometry implementation.
const WORLD_GEO = (typeof WORLD !== 'undefined') ? WORLD : null;
if (WORLD_GEO) WORLD_GEO.configure(() => ({ w: canvas.width, h: canvas.height }));
function sphereWorld() { return WORLD_GEO !== null && WORLD_GEO.isSphere(); }

// Logical world bounds. Equal to the canvas on the torus (the canvas *is* the
// viewport there); on the sphere the chart can be WORLD.getScale() times larger
// than the screen, so all wrap/spawn/centre logic below reads these instead of
// canvas.width/height. Drawing code keeps using the canvas.
function worldW() { return sphereWorld() ? WORLD_GEO.size().w : canvas.width; }
function worldH() { return sphereWorld() ? WORLD_GEO.size().h : canvas.height; }
// Multiplier for distances that should grow with the world (spawn clearances,
// fuel endurance): 1 on the torus.
function worldScale() { return sphereWorld() ? WORLD_GEO.effectiveScale() : 1; }
function setWorldScale(k) {
    if (!WORLD_GEO) return 1;
    WORLD_GEO.setScale(k);
    return WORLD_GEO.getScale();
}
function sphereViewReady() {
    return sphereWorld() && typeof SphereView !== 'undefined' && SphereView.ready();
}

// Switch between the flat torus and the sphere. Objects keep their chart
// coordinates, so the switch is instant; only motion, wrapping, distances and
// the renderer change.
function setWorldMode(mode) {
    if (!WORLD_GEO) return false;
    const sphere = mode === 'sphere';
    WORLD_GEO.setMode(sphere ? 'sphere' : 'torus');
    if (sphere && typeof SphereView !== 'undefined') {
        if (!SphereView.ready()) SphereView.init(canvas);
        if (SphereView.ready()) ctx = SphereView.overlayContext();
        SphereView.setVisible(true);
    } else {
        ctx = baseCtx;
        if (typeof SphereView !== 'undefined' && SphereView.ready()) SphereView.setVisible(false);
    }
    return WORLD_GEO.isSphere();
}

// Chart -> screen mapping used by the policy overlays. Identity on the flat map;
// on the sphere it is the camera projection and returns null for the far side.
function worldToScreen(x, y) {
    if (sphereViewReady()) return SphereView.project(x, y);
    return { x: x, y: y, scale: 1 };
}

const scoreElement = document.getElementById('score');
const bestScoreElement = document.getElementById('bestScore');
const aiShipCountElement = document.getElementById('aiShipCount');
const enemyShipCountElement = document.getElementById('enemyShipCount');
const bossShipCountElement = document.getElementById('bossShipCount');

// Persisted across reloads via localStorage, keyed so it never collides with
// another app's storage. Wrapped in try/catch because the Streamlit build
// embeds this page in a sandboxed srcdoc iframe (see app.py), where
// localStorage has no origin to attach to and throws rather than just
// returning null.
function loadBestScore() {
    try {
        return parseInt(localStorage.getItem('asteroidsAI.bestScore'), 10) || 0;
    } catch (e) {
        return 0;
    }
}
function saveBestScore(value) {
    try {
        localStorage.setItem('asteroidsAI.bestScore', String(value));
    } catch (e) {
        // No persistent storage available here (sandboxed iframe, privacy
        // mode, etc.) -- best score just won't survive a reload.
    }
}
let bestScore = loadBestScore();
if (bestScoreElement) bestScoreElement.textContent = bestScore;

// Audio Context for sound effects
let audioContext = null;
let audioContextResumed = false;

// Audio switches: the Streamlit sidebar drives these, and M toggles music
// in-game. Sound effects and music are separate so either can be muted.
let sfxEnabled = true;
let musicEnabled = true;
window.setSoundEnabled = function(on) { sfxEnabled = !!on; };
window.setMusicEnabled = function(on) { musicEnabled = !!on; };

// Initialize audio context
function initAudioContext() {
    if (!audioContext) {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.log('Web Audio API not supported');
            return;
        }
    }
    
    // Resume audio context if suspended (required by browsers)
    if (audioContext.state === 'suspended') {
        audioContext.resume().then(() => {
            audioContextResumed = true;
            console.log('Audio context resumed');
        }).catch(err => {
            console.log('Could not resume audio context:', err);
        });
    } else {
        audioContextResumed = true;
    }
}

// Resume audio on user interaction
canvas.addEventListener('click', () => {
    if (!audioContextResumed) {
        initAudioContext();
    }
}, { once: false });

// Also try to resume on key press
document.addEventListener('keydown', () => {
    if (!audioContextResumed) {
        initAudioContext();
    }
}, { once: false });

// Initialize audio context on page load
initAudioContext();

// Sound effects functions
function playFireSound(isAIBullet = false) {
    if (!sfxEnabled) return;
    if (!audioContext || !audioContextResumed) {
        // Try to resume if not already resumed
        if (audioContext && audioContext.state === 'suspended') {
            initAudioContext();
        }
        return;
    }
    
    try {
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        // Different frequencies for player vs AI
        if (isAIBullet) {
            // AI ship sound - slightly lower pitch
            oscillator.frequency.setValueAtTime(300, audioContext.currentTime);
            oscillator.frequency.exponentialRampToValueAtTime(200, audioContext.currentTime + 0.05);
        } else {
            // Player ship sound - higher pitch
            oscillator.frequency.setValueAtTime(400, audioContext.currentTime);
            oscillator.frequency.exponentialRampToValueAtTime(300, audioContext.currentTime + 0.05);
        }
        
        oscillator.type = 'square'; // Square wave for sharper sound
        
        // Envelope for quick attack and decay (increased volume)
        gainNode.gain.setValueAtTime(0, audioContext.currentTime);
        gainNode.gain.linearRampToValueAtTime(0.2, audioContext.currentTime + 0.01); // Increased from 0.1 to 0.2
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.05);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.05);
        // Release the nodes once playback finishes. Without this the graph keeps
        // every oscillator/gain pair alive for the whole session.
        oscillator.onended = () => {
            oscillator.disconnect();
            gainNode.disconnect();
        };
    } catch (e) {
        console.log('Error playing sound:', e);
    }
}

// Shared noise for every explosion: generated once, reused. A tiny LCG rather
// than Math.random(), so the audio path never touches the seeded game RNG and
// stays reproducible.
let sfxNoiseBuffer = null;
let lastExplosionAt = -1;
let lastBossAlertAt = -100;

function playExplosionSound(scale = 1) {
    if (!sfxEnabled || !audioContext || !audioContextResumed) return;
    const now = audioContext.currentTime;
    // A busy swarm can pop several things in one frame; thin them out so the
    // mix stays readable instead of turning into a wall of noise.
    if (now - lastExplosionAt < 0.07) return;
    lastExplosionAt = now;
    try {
        if (!sfxNoiseBuffer) {
            const len = Math.floor(audioContext.sampleRate * 1.2);
            sfxNoiseBuffer = audioContext.createBuffer(1, len, audioContext.sampleRate);
            const data = sfxNoiseBuffer.getChannelData(0);
            let seed = 12345;
            for (let i = 0; i < len; i++) {
                seed = (seed * 1664525 + 1013904223) >>> 0;
                data[i] = seed / 2147483648 - 1;
            }
        }
        const dur = Math.min(1.1, 0.22 + 0.22 * scale);
        const peak = Math.min(0.3, 0.07 + 0.05 * scale);
        const src = audioContext.createBufferSource();
        src.buffer = sfxNoiseBuffer;
        const filter = audioContext.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(Math.min(3200, 1400 + 600 * scale), now);
        filter.frequency.exponentialRampToValueAtTime(90, now + dur);
        const gain = audioContext.createGain();
        gain.gain.setValueAtTime(peak, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + dur);
        src.connect(filter); filter.connect(gain); gain.connect(audioContext.destination);
        src.start(now); src.stop(now + dur + 0.05);
        src.onended = () => { src.disconnect(); filter.disconnect(); gain.disconnect(); };
        if (scale >= 2) {
            // Bosses and big blasts get a low thump under the noise.
            const osc = audioContext.createOscillator();
            const og = audioContext.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(110, now);
            osc.frequency.exponentialRampToValueAtTime(38, now + dur);
            og.gain.setValueAtTime(peak * 1.2, now);
            og.gain.exponentialRampToValueAtTime(0.001, now + dur);
            osc.connect(og); og.connect(audioContext.destination);
            osc.start(now); osc.stop(now + dur + 0.05);
            osc.onended = () => { osc.disconnect(); og.disconnect(); };
        }
    } catch (e) {
        console.log('Error playing explosion sound:', e);
    }
}

// Two-tone siren when a boss spawns. Rate-limited: bosses spawn in batches.
function playBossAlert() {
    if (!sfxEnabled || !audioContext || !audioContextResumed) return;
    const now = audioContext.currentTime;
    if (now - lastBossAlertAt < 6) return;
    lastBossAlertAt = now;
    try {
        const osc = audioContext.createOscillator();
        const filter = audioContext.createBiquadFilter();
        const gain = audioContext.createGain();
        osc.type = 'sawtooth';
        filter.type = 'lowpass';
        filter.frequency.value = 900;
        for (let i = 0; i < 4; i++) osc.frequency.setValueAtTime(i % 2 ? 330 : 440, now + i * 0.22);
        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(0.1, now + 0.03);
        gain.gain.setValueAtTime(0.1, now + 0.8);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.95);
        osc.connect(filter); filter.connect(gain); gain.connect(audioContext.destination);
        osc.start(now); osc.stop(now + 1.0);
        osc.onended = () => { osc.disconnect(); filter.disconnect(); gain.disconnect(); };
    } catch (e) {
        console.log('Error playing boss alert:', e);
    }
}

// Falling tone for the game-over screen.
function playGameOverSound() {
    if (!sfxEnabled || !audioContext || !audioContextResumed) return;
    const now = audioContext.currentTime;
    try {
        const osc = audioContext.createOscillator();
        const gain = audioContext.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(330, now);
        osc.frequency.exponentialRampToValueAtTime(82, now + 1.1);
        gain.gain.setValueAtTime(0.16, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 1.2);
        osc.connect(gain); gain.connect(audioContext.destination);
        osc.start(now); osc.stop(now + 1.25);
        osc.onended = () => { osc.disconnect(); gain.disconnect(); };
    } catch (e) {
        console.log('Error playing game-over sound:', e);
    }
}

// Short rising two-note chime for collecting a power-up.
function playPickupSound() {
    if (!sfxEnabled || !audioContext || !audioContextResumed) return;
    const now = audioContext.currentTime;
    try {
        [660, 880].forEach((freq, i) => {
            const t = now + i * 0.07;
            const osc = audioContext.createOscillator();
            const gain = audioContext.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, t);
            gain.gain.setValueAtTime(0, t);
            gain.gain.linearRampToValueAtTime(0.12, t + 0.01);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.14);
            osc.connect(gain); gain.connect(audioContext.destination);
            osc.start(t); osc.stop(t + 0.16);
            osc.onended = () => { osc.disconnect(); gain.disconnect(); };
        });
    } catch (e) {
        console.log('Error playing pickup sound:', e);
    }
}

// Ambient music: a slow low drone (two sine/triangle pairs under a gently
// sweeping low-pass) plus a sparse, quiet pentatonic note every couple of
// seconds. Entirely synthesized, so there is no audio asset to ship or load.
let musicNodes = null;

function startMusic() {
    try {
        const now = audioContext.currentTime;
        const master = audioContext.createGain();
        master.gain.setValueAtTime(0, now);
        master.gain.linearRampToValueAtTime(0.055, now + 3);
        const filter = audioContext.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.value = 900;
        filter.connect(master);
        master.connect(audioContext.destination);

        const oscs = [55, 82.41, 110, 164.81].map((freq, i) => {
            const o = audioContext.createOscillator();
            const g = audioContext.createGain();
            o.type = i % 2 ? 'triangle' : 'sine';
            o.frequency.value = freq;
            o.detune.value = (i - 1.5) * 6;
            g.gain.value = 0.25;
            o.connect(g); g.connect(filter);
            o.start();
            return o;
        });

        const lfo = audioContext.createOscillator();
        const lfoGain = audioContext.createGain();
        lfo.frequency.value = 0.07;
        lfoGain.gain.value = 350;
        lfo.connect(lfoGain); lfoGain.connect(filter.frequency);
        lfo.start();

        const scale = [220, 246.94, 277.18, 329.63, 369.99, 440];
        const timer = setInterval(() => {
            if (!musicNodes || !audioContext) return;
            const t = audioContext.currentTime;
            const o = audioContext.createOscillator();
            const g = audioContext.createGain();
            o.type = 'sine';
            o.frequency.value = scale[Math.floor(Math.random() * scale.length)] * (Math.random() < 0.3 ? 2 : 1);
            g.gain.setValueAtTime(0, t);
            g.gain.linearRampToValueAtTime(0.5, t + 0.4);
            g.gain.exponentialRampToValueAtTime(0.001, t + 3);
            o.connect(g); g.connect(filter);
            o.start(t); o.stop(t + 3.1);
            o.onended = () => { o.disconnect(); g.disconnect(); };
        }, 2200);

        musicNodes = { master, filter, oscs: oscs.concat([lfo]), timer };
    } catch (e) {
        console.log('Error starting music:', e);
        musicNodes = null;
    }
}

function stopMusic() {
    if (!musicNodes) return;
    const nodes = musicNodes;
    musicNodes = null;
    clearInterval(nodes.timer);
    try {
        const now = audioContext.currentTime;
        nodes.master.gain.cancelScheduledValues(now);
        nodes.master.gain.setValueAtTime(nodes.master.gain.value, now);
        nodes.master.gain.linearRampToValueAtTime(0, now + 0.5);
    } catch (e) { /* context may already be closed */ }
    setTimeout(() => {
        nodes.oscs.forEach(o => { try { o.stop(); o.disconnect(); } catch (e) {} });
        try { nodes.filter.disconnect(); nodes.master.disconnect(); } catch (e) {}
    }, 600);
}

// Called every frame: starts or stops the drone to match the toggle and the
// browser's audio-unlock state (nothing may play before a user gesture).
function syncMusic() {
    const want = musicEnabled && audioContext && audioContextResumed;
    if (want && !musicNodes) startMusic();
    else if (!want && musicNodes) stopMusic();
}

// Set canvas size
let defaultWidth = 1200;
let defaultHeight = 600;
canvas.width = defaultWidth;
canvas.height = defaultHeight;

// Function to set canvas size (called from Streamlit)
window.setCanvasSize = function(width, height) {
    defaultWidth = width;
    defaultHeight = height;
    canvas.width = width;
    canvas.height = height;
    
    // Reposition player ship to center if needed
    if (ship) {
        ship.x = worldW() / 2;
        ship.y = worldH() / 2;
    }
    
    // Update AI ships positions if needed
    if (aiShips && aiShips.length > 0) {
        aiShips.forEach(aiShip => {
            if (aiShip) {
                // Keep ships within bounds
                if (aiShip.x > worldW()) aiShip.x = worldW() * 0.25;
                if (aiShip.y > worldH()) aiShip.y = worldH() * 0.25;
            }
        });
    }
};

// Fullscreen state
let isFullscreen = false;

// Fullscreen functions
function enterFullscreen() {
    const elem = canvas;
    if (elem.requestFullscreen) {
        elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) {
        elem.webkitRequestFullscreen();
    } else if (elem.mozRequestFullScreen) {
        elem.mozRequestFullScreen();
    } else if (elem.msRequestFullscreen) {
        elem.msRequestFullscreen();
    }
}

function exitFullscreen() {
    if (document.exitFullscreen) {
        document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
    } else if (document.mozCancelFullScreen) {
        document.mozCancelFullScreen();
    } else if (document.msExitFullscreen) {
        document.msExitFullscreen();
    }
}

function toggleFullscreen() {
    if (!isFullscreen) {
        enterFullscreen();
    } else {
        exitFullscreen();
    }
}

// Handle fullscreen change events
function handleFullscreenChange() {
    isFullscreen = !!(document.fullscreenElement || document.webkitFullscreenElement || 
                     document.mozFullScreenElement || document.msFullscreenElement);
    
    if (isFullscreen) {
        // Resize canvas to fullscreen
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    } else {
        // Restore default size
        canvas.width = defaultWidth;
        canvas.height = defaultHeight;
    }
}

// Listen for fullscreen changes
document.addEventListener('fullscreenchange', handleFullscreenChange);
document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
document.addEventListener('mozfullscreenchange', handleFullscreenChange);
document.addEventListener('MSFullscreenChange', handleFullscreenChange);

// Game state
let score = 0;
let gameRunning = true;
let playerShipActive = true; // Player ship active state

// Manual pause (P key / touch button). Distinct from gameRunning, which
// tracks episode-over vs episode-in-progress for the RL loop -- this is a
// purely presentational stop that headless training never touches (nothing
// sets keys['p'] or calls togglePause() there).
let gamePaused = false;
function togglePause() {
    gamePaused = !gamePaused;
}

// Game over: a non-blocking overlay instead of a blocking alert(). The old
// alert() froze the whole tab (and the animation loop with it) until it was
// dismissed, and left the game running behind a dialog the user could not see
// past. Now the simulation just stops, the frozen scene stays visible behind
// a score card, and the next run starts on R/Enter or after a few seconds.
// Headless training/test harnesses (which stub alert() as a no-op and expect
// the world to reset immediately) keep that behaviour.
// Swings the sphere camera over to a boss kill (SphereView rate-limits this and
// ignores it unless the kill-cam is enabled and the sphere view is active).
function onBossKilled(boss) {
    if (typeof SphereView !== 'undefined' && sphereViewReady()) SphereView.killCam(boss.x, boss.y, 90);
}

let gameOverState = null; // { score, best, newBest, frames } while the card is up
const GAME_OVER_AUTO_RESTART_FRAMES = 300;

function triggerGameOver() {
    if (typeof globalThis !== 'undefined' && globalThis.__HEADLESS__) {
        initGame();
        gameRunning = true;
        return;
    }
    gameOverState = { score, best: bestScore, newBest: score > 0 && score >= bestScore, frames: 0 };
    playGameOverSound();
}

// Episode bookkeeping. Collected training samples are useless without these:
// without an episode id and a terminal flag you cannot reconstruct trajectories,
// compute returns, or avoid bootstrapping across an episode boundary.
let episodeStepCount = 0;
let episodeSequence = 0;
let episodeId = 'ep_uninitialised';
let episodeDone = false;

// Per-agent credit ledger. The reward function reads and clears this each step
// so that a kill is paid to the agent whose bullet landed it, instead of being
// split uniformly across the team (which pays free-riders the same as shooters).
let agentCredit = [];

function resetAgentCredit(agentCount) {
    agentCredit = [];
    for (let i = 0; i < agentCount; i++) {
        agentCredit.push({ asteroidKills: 0, enemyKills: 0, bossKills: 0, scoreEarned: 0 });
    }
}

function creditAgent(agentId, field, amount) {
    if (agentId === null || agentId === undefined) return; // player or unowned bullet
    if (!agentCredit[agentId]) {
        agentCredit[agentId] = { asteroidKills: 0, enemyKills: 0, bossKills: 0, scoreEarned: 0 };
    }
    agentCredit[agentId][field] += amount;
}

// Drain the ledger: returns the credit accrued since the last call and zeroes it.
function consumeAgentCredit() {
    const snapshot = agentCredit.map(c => ({ ...c }));
    agentCredit.forEach(c => {
        c.asteroidKills = 0;
        c.enemyKills = 0;
        c.bossKills = 0;
        c.scoreEarned = 0;
    });
    return snapshot;
}

// Fuel-rescue event ledger, drained by the reward function the same way as
// agentCredit. Records which agents were on each end of a completed refuel or
// a dock death so the env can pay (or charge) the right ships.
let fuelEvents = { refuels: [], dockDeaths: [], lowFuelFrames: 0 };

function recordFuelEvent(kind, shipA, shipB) {
    if (!fuelEvents[kind]) return;
    fuelEvents[kind].push({
        a: shipA ? shipA.agentId : undefined,
        b: shipB ? shipB.agentId : undefined
    });
}

function consumeFuelEvents() {
    const snapshot = fuelEvents;
    fuelEvents = { refuels: [], dockDeaths: [], lowFuelFrames: 0 };
    return snapshot;
}

// External policy control. When enabled, `policyActions[i]` drives AI ship i
// instead of its rule-based makeDecision(). The same path is used by the headless
// trainer and by the browser when running a trained policy, so there is exactly
// one action-application code path to keep consistent.
let policyControlEnabled = false;
let policyActions = null;

// Factored discrete action space:
//   rotation {-1,0,+1} x thrust {0,1} x fireMode {0,1,2,3} x shield {0,1}
// = 48 joint actions, small enough for a flat categorical policy head.
//
// fireMode exists because AIShip.fire(target) computes a predictive intercept
// angle and launches the bullet along it, independent of the ship's heading -
// the AI ships have a turret, not a fixed forward gun. The rule-based controller
// uses that, so an agent restricted to firing along its heading is playing with
// a strictly weaker weapon and cannot be compared against it fairly. Exposing
// the target choice also puts the interesting decision (what to shoot, and when)
// in the policy instead of asking it to rediscover trigonometry the game
// already implements.
const FIRE_NONE = 0;
const FIRE_FORWARD = 1;
const FIRE_AT_ASTEROID = 2;
const FIRE_AT_ENEMY = 3;
const ACTION_SPACE_SIZE = 48;

// Publish the action-space width so a loaded policy can be checked against it.
// A top-level `const` lives in the global lexical environment rather than on
// globalThis, so without this assignment policy_runner.js sees `undefined` and
// cannot tell a stale 24-action checkpoint from a current 48-action one.
// ACTION_SPACE_SIZE is part of the public surface declared at the top of this file.

function decodeAction(index) {
    const i = Math.max(0, Math.min(ACTION_SPACE_SIZE - 1, index | 0));
    const rotationIndex = Math.floor(i / 16);  // 0,1,2
    const remainder = i % 16;
    return {
        rotation: rotationIndex - 1,           // -1, 0, +1
        thrust: (remainder >> 3) & 1,
        fire: (remainder >> 1) & 3,            // 0..3
        shield: remainder & 1
    };
}

function encodeAction(action) {
    const rotationIndex = (action.rotation | 0) + 1;
    const fireMode = Math.max(0, Math.min(3, action.fire | 0));
    return rotationIndex * 16 + ((action.thrust ? 1 : 0) << 3) + (fireMode << 1) + (action.shield ? 1 : 0);
}

// Nearest asteroid / enemy to a ship, used to resolve targeted fire modes.
function nearestTargetFor(aiShip, kind) {
    const pool = kind === 'asteroid'
        ? asteroids
        : enemyShips;
    let best = null;
    let bestDistance = Infinity;
    for (let i = 0; i < pool.length; i++) {
        const candidate = pool[i];
        if (!candidate) continue;
        const dx = candidate.x - aiShip.x;
        const dy = candidate.y - aiShip.y;
        const distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
            bestDistance = distance;
            best = candidate;
        }
    }
    return best;
}

// Blocks a specific sub-action (targeted fire, thrust) that the docs
// describe as always safety-validated ("All ML actions are validated before
// execution"), but which previously only AIShip.isSafeMLAction() enforced --
// and that function is only reachable from the legacy per-ship ML fallback
// inside makeDecision(), which is skipped entirely once a trained PPO policy
// is driving the ship (see the policyControlEnabled branch below). This
// mirrors isSafeMLAction()'s own two checks (same 40-unit friendly-fire
// radius, same thrust-into-asteroid projection) against applyAgentAction's
// {rotation,thrust,fire,shield} action shape instead of isSafeMLAction's
// {action,target} shape, since the two don't share a representation.
//
// applyAgentAction() is the same function the headless trainer calls via
// sim/env.js's setPolicyControl(), so this also shields training rollouts,
// not just live inference -- consistent with the shared-code architecture's
// goal of training and inference never diverging, but it does mean a
// checkpoint trained before this change is now evaluated under a slightly
// different effective action space than it was trained on. In practice this
// should rarely fire (firing point-blank at an ally or ramming an asteroid
// head-on is already bad for score, so a competent policy rarely picks it),
// but the RL_TRAINING.md benchmark numbers were measured without it.
function isSafeAgentSubAction(aiShip, kind, target) {
    if (kind === 'fire' && target) {
        for (const ally of aiShips) {
            if (!ally || ally === aiShip) continue;
            const dx = target.x - ally.x;
            const dy = target.y - ally.y;
            if (Math.sqrt(dx * dx + dy * dy) < 40) return false;
        }
    }
    if (kind === 'thrust') {
        const nextX = aiShip.x + Math.cos(aiShip.angle) * 20;
        const nextY = aiShip.y + Math.sin(aiShip.angle) * 20;
        for (const asteroid of asteroids) {
            const dx = nextX - asteroid.x;
            const dy = nextY - asteroid.y;
            if (Math.sqrt(dx * dx + dy * dy) < (asteroid.radius || 20) + aiShip.radius) return false;
        }
    }
    return true;
}

// Requires the hull to be roughly pointed at a target before a targeted shot
// fires, instead of firing the instant cooldown allows regardless of facing.
// AIShip.fire(target) always computes its own predictive intercept angle,
// independent of the hull's facing -- it is a turret, not a fixed gun -- so
// this changes nothing about accuracy; a spinning ship already lands every
// shot it takes. What it changes is the *look* of firing: a ship still
// turning toward a target now holds fire until it is roughly facing it,
// instead of spraying in every direction while it spins. 0.4 rad (~23
// degrees) matches the cone the rule-based AIShip.makeDecision() already
// uses before its own fire(target) calls.
const AIM_TOLERANCE = 0.4;
function isAimedAtTarget(aiShip, target) {
    const bearing = Math.atan2(target.y - aiShip.y, target.x - aiShip.x);
    return Math.abs(aiShip.normalizeAngle(bearing - aiShip.angle)) < AIM_TOLERANCE;
}

// Single place where an agent action becomes game input.
function applyAgentAction(aiShip, action) {
    if (!aiShip || !action) return;
    const a = typeof action === 'number' ? decodeAction(action) : action;

    if (a.rotation === -1) aiShip.rotate(-1);
    else if (a.rotation === 1) aiShip.rotate(1);

    // A docked pair cannot manoeuvre: the rule-based alpha stops for the
    // transfer, and the policy-driven alpha is held to the same rule.
    const fuelLocked = aiShip.fuelDocked || aiShip.fuelTransferActive;
    if (a.thrust === 1 && !fuelLocked && isSafeAgentSubAction(aiShip, 'thrust')) aiShip.thrust();

    if (a.fire && aiShip.shootCooldown <= 0) {
        if (a.fire === FIRE_FORWARD) {
            aiShip.fire();
        } else if (a.fire === FIRE_AT_ASTEROID) {
            const target = nearestTargetFor(aiShip, 'asteroid');
            if (!target) aiShip.fire();
            else if (isAimedAtTarget(aiShip, target) && isSafeAgentSubAction(aiShip, 'fire', target)) aiShip.fire(target);
        } else if (a.fire === FIRE_AT_ENEMY) {
            const target = nearestTargetFor(aiShip, 'enemy');
            if (!target) aiShip.fire();
            else if (isAimedAtTarget(aiShip, target) && isSafeAgentSubAction(aiShip, 'fire', target)) aiShip.fire(target);
        }
    }

    if (a.shield === 1 && aiShip.shieldCooldown <= 0 && aiShip.health < aiShip.maxHealth) {
        aiShip.shieldActive = true;
        aiShip.shieldDuration = 60;
        aiShip.shieldCooldown = 300;
    }
}

function setPolicyControl(enabled, actions) {
    policyControlEnabled = !!enabled;
    policyActions = actions || null;
}

// Explainability overlay. A learned policy is otherwise a black box on screen:
// you can see that a ship moved, but not what it chose, how confident it was, or
// what it thought the position was worth. Showing the decision next to the ship
// makes it possible to tell competent play from lucky play.
let policyOverlayEnabled = false;

function setPolicyOverlay(enabled) {
    policyOverlayEnabled = !!enabled;
}

function drawPolicyOverlay() {
    if (!policyOverlayEnabled || typeof getPolicyRunner !== 'function') return;
    const runner = getPolicyRunner();
    if (!runner || !runner.enabled || !runner.lastDecisions.length) return;

    ctx.save();
    ctx.font = '11px monospace';
    ctx.textAlign = 'left';

    aiShips.forEach(aiShip => {
        if (!aiShip) return;
        const decision = runner.lastDecisions[aiShip.agentId];
        if (!decision) return;

        const sp = worldToScreen(aiShip.x, aiShip.y);
        if (!sp) return; // far side of the sphere
        const x = sp.x + 14;
        const y = sp.y - 14;

        // Confidence drives the colour: uncertain decisions are the interesting
        // ones, so make them visually distinct rather than uniform.
        const confidence = decision.confidence;
        ctx.fillStyle = confidence > 0.6 ? GAME_THEME.confidenceHigh : (confidence > 0.3 ? GAME_THEME.confidenceMid : GAME_THEME.confidenceLow);
        ctx.fillText(describePolicyDecision(decision), x, y);

        if (decision.value !== null && decision.value !== undefined) {
            ctx.fillStyle = GAME_THEME.valueText;
            ctx.fillText(`V=${decision.value.toFixed(1)}  H=${decision.entropy.toFixed(2)}`, x, y + 12);
        }

        // Small confidence bar for the top-3 actions.
        const bars = decision.topActions || [];
        for (let i = 0; i < bars.length; i++) {
            ctx.fillStyle = i === 0 ? GAME_THEME.barPrimary : GAME_THEME.barSecondary;
            ctx.fillRect(x, y + 18 + i * 4, Math.max(1, bars[i].p * 40), 3);
        }
    });

    ctx.fillStyle = GAME_THEME.overlayMuted;
    const inWorker = runner.worker && runner.workerReady;
    ctx.fillText(inWorker
        ? 'policy inference: Web Worker (off main thread, 1-frame lag)'
        : `policy inference ${runner.inferenceMs.toFixed(2)}ms/frame`, 10, canvas.height - 10);
    ctx.restore();
}

// Expert action recording. The rule-based controller does not emit an action
// index - it calls rotate()/thrust()/fire() directly - so to imitate it we record
// which primitives it invoked each frame and encode them into the same 24-action
// space the policy uses. This is what makes behaviour cloning possible.
let expertActionRecording = false;

function setExpertActionRecording(enabled) {
    expertActionRecording = !!enabled;
    if (!enabled && typeof aiShips !== 'undefined') {
        aiShips.forEach(s => { if (s) s._actionRecord = null; });
    }
}

function armExpertActionRecords() {
    aiShips.forEach(s => {
        if (s) s._actionRecord = { rotation: 0, thrust: 0, fire: 0, shield: 0 };
    });
}

function readExpertActions() {
    const out = [];
    aiShips.forEach(s => {
        if (s && s._actionRecord && s.agentId !== undefined) {
            out[s.agentId] = encodeAction(s._actionRecord);
        }
    });
    return out;
}

// Visual effects layer (fx.js). Optional: the headless trainer never loads it,
// and every call site checks FXL so the game logic is identical either way.
const FXL = (typeof FX !== 'undefined') ? FX : null;

let mlGlobalFrame = 0;
let mlUseNeuralNetwork = false;
let mlHierarchyEnabled = false;
let mlEvaluationEnabled = false;
let mlDataCollectionEnabled = false;

// Function to set player ship active state (called from Streamlit)
window.setPlayerShipActive = function(isActive) {
    playerShipActive = isActive;
    console.log('Player ship active state changed to:', isActive);
};

// Game objects
const ship = new Ship();
let aiShips = [];
let bullets = [];
let enemyBullets = []; // Separate array for enemy bullets
let enemyShips = []; // Array for all enemy ships (basic, advanced, and boss)
let powerUps = []; // Tactical power-up pickups (railgun, overshield, emp, fuel)

const isHeadlessEnv = () => typeof globalThis !== 'undefined' && !!globalThis.__HEADLESS__;
const tacticalDropRng = (typeof AsteroidsRNG !== 'undefined' && AsteroidsRNG.createStream)
    ? AsteroidsRNG.createStream('tactical-drops')
    : Math.random;


// Global coordination system instance
let enemyCoordinationSystem = new EnemyCoordinationSystem();
let asteroids = [];
let numAIShips = 1; // Default number of AI ships
let numEnemyShips = 1; // Default number of enemy ships (basic/advanced)
let numBossShips = 1; // Default number of boss ships
let alphaAttackEnabled = false; // Toggle for alpha attack pattern
let formationType = 'arrowhead'; // Default formation type: 'arrowhead', 'line', 'circle', 'diamond', 'wedge'
let autoAssignRoles = true; // Auto-assign roles or use manual assignment
let adaptiveFormationEnabled = false; // Phase 3: Enable adaptive formation switching
let multiTargetMode = 'focus'; // Phase 3: 'focus', 'split', 'prioritize'
let escortMode = 'none'; // Phase 3: 'none', 'escort', 'guard', 'patrol', 'intercept', 'cover'
let tacticalAttackSequencesEnabled = true; // Advanced: Enable tactical attack sequences
let dynamicFormationTransitionsEnabled = true; // Advanced: Enable dynamic formation transitions
let advancedFlankingEnabled = true; // Advanced: Enable advanced flanking patterns
let fuelRescueEnabled = true; // Alpha HELP ping + fuel ship refuel rescue
let fuelRescueDispatchCooldown = 0;

function updateAlphaHelpPing(ship) {
    if (!ship) return;
    if (ship.fuelDocked || ship.fuelTransferActive) {
        ship.helpPingActive = false;
        return;
    }
    ship.helpPingActive = ship.fuel / ship.maxFuel <= ship.fuelHelpThreshold;
}

function isFuelLinked(ship) {
    return !!(ship && (ship.fuelTransferActive || ship.fuelDocked) && ship.fuelTransferPartner);
}

function clearFuelLink(shipA, shipB) {
    [shipA, shipB].forEach(ship => {
        if (!ship) return;
        ship.fuelTransferActive = false;
        ship.fuelDocked = false;
        ship.fuelTransferPartner = null;
        ship.fuelDockHoldTimer = 0;
        if (ship.isAlpha) updateAlphaHelpPing(ship);
    });
}

function snapFuelShipDocked(fuelShip, alpha) {
    const dockAngle = alpha.angle - Math.PI / 2;
    const dist = alpha.size + fuelShip.size + 8;
    fuelShip.x = alpha.x + Math.cos(dockAngle) * dist;
    fuelShip.y = alpha.y + Math.sin(dockAngle) * dist;
    fuelShip.velocityX = 0;
    fuelShip.velocityY = 0;
    fuelShip.angle = alpha.angle;
    alpha.velocityX = 0;
    alpha.velocityY = 0;
}

function startFuelDockHold(fuelShip, alpha) {
    fuelShip.fuelTransferActive = false;
    alpha.fuelTransferActive = false;   // was left set, which suppressed every later HELP ping
    fuelShip.fuelTransferTimer = 0;
    alpha.fuelTransferTimer = 0;
    fuelShip.fuelDocked = true;
    alpha.fuelDocked = true;
    fuelShip.fuelTransferPartner = alpha;
    alpha.fuelTransferPartner = fuelShip;
    fuelShip.fuelDockHoldTimer = fuelShip.fuelDockHoldFrames;
    alpha.fuelDockHoldTimer = fuelShip.fuelDockHoldFrames;
    alpha.helpPingActive = false;
    snapFuelShipDocked(fuelShip, alpha);
    recordFuelEvent('refuels', alpha, fuelShip);
    if (FXL) { FXL.emitRipple(alpha.x, alpha.y, GAME_THEME.fuelTransfer); FXL.flash(0.12, GAME_THEME.fuelTransfer); }
}

// Alpha selection for externally-driven ships. The rule-based controller picks
// its alpha inside makeDecision(), which a policy-controlled ship never calls,
// so without this there is no alpha (and therefore no fuel mechanic) whenever a
// trained policy is flying. Unlike findAlphaShip() it does not re-centre the
// alpha every frame: a policy has to be allowed to move its leader.
function ensureAlphaAssignment(aiShips) {
    if (!alphaAttackEnabled) return null;
    let alpha = null;
    for (let i = 0; i < aiShips.length; i++) {
        if (aiShips[i] && aiShips[i].isAlpha) { alpha = aiShips[i]; break; }
    }
    if (!alpha) {
        alpha = AIShip.selectAlphaShip(aiShips);
        if (!alpha) return null;
        if (!alpha.originalMaxHealth) alpha.originalMaxHealth = alpha.maxHealth;
        alpha.maxHealth = Math.ceil(alpha.originalMaxHealth * 1.25);
        alpha.health = Math.min(alpha.health, alpha.maxHealth);
        alpha.isAlpha = true;
        alpha.alphaShip = null;
        alpha.maxFuel = 100;
        alpha.fuel = 100;
        updateAlphaHelpPing(alpha);
    }
    for (let i = 0; i < aiShips.length; i++) {
        const s = aiShips[i];
        if (!s || s === alpha) continue;
        s.isAlpha = false;
        s.alphaShip = alpha;
    }
    assignFuelShipRole(aiShips);
    return alpha;
}

function releaseFuelDock(fuelShip, alpha) {
    fuelShip.fuelDocked = false;
    alpha.fuelDocked = false;
    fuelShip.fuelTransferActive = false;
    alpha.fuelTransferActive = false;
    fuelShip.fuelTransferPartner = null;
    alpha.fuelTransferPartner = null;
    fuelShip.fuelDockHoldTimer = 0;
    alpha.fuelDockHoldTimer = 0;
    fuelShip.fuelDeparting = true;
    fuelShip.fuelDepartTimer = 0;
    fuelShip.fuelStandby = true;
    const dx = fuelShip.x - alpha.x;
    const dy = fuelShip.y - alpha.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    fuelShip.velocityX = (dx / dist) * 3;
    fuelShip.velocityY = (dy / dist) * 3;
    updateAlphaHelpPing(alpha);
}

// Backward-compatible alias used by collision cleanup.
function endFuelTransfer(shipA, shipB) {
    clearFuelLink(shipA, shipB);
}

function getAlphaCombatWingmen(aiShips, alpha) {
    if (!alpha) return [];
    return aiShips.filter(ship =>
        ship &&
        ship !== alpha &&
        ship.alphaShip === alpha &&
        ship.role !== 'fuel'
    );
}

function assignFuelShipRole(aiShips) {
    if (!fuelRescueEnabled || !alphaAttackEnabled) return;
    if (aiShips.some(s => s && s.role === 'fuel')) return;
    const alpha = aiShips.find(s => s && s.isAlpha);
    if (!alpha) return;
    const wingmen = aiShips.filter(s => s && s.alphaShip === alpha && !s.isAlpha);
    // Prefer converting a wingman when 2+ exist; otherwise emergency dispatch handles it.
    if (wingmen.length < 2) return;
    // Never re-initialise a ship that is currently linked to the alpha: if its
    // role was wiped externally mid-refuel, restore the role and keep the link.
    const linked = wingmen.find(s => (s.fuelDocked || s.fuelTransferActive) && s.fuelTransferPartner === alpha);
    if (linked) {
        linked.role = 'fuel';
        linked.roleAssigned = true;
        return;
    }
    const wingman = wingmen.find(s => s.role !== 'fuel') || wingmen[0];
    if (wingman) {
        wingman.role = 'fuel';
        wingman.roleAssigned = true;
        wingman.maxFuel = 100;
        wingman.fuel = 100;
        wingman.fuelStandby = true;
        wingman.fuelDeparting = false;
        wingman.fuelDocked = false;
        wingman.standbyTarget = {
            x: Math.max(60, worldW() * 0.12),
            y: Math.max(60, worldH() * 0.88)
        };
    }
}

// Spawn a dedicated green tanker when Alpha pings HELP and no fuel ship is alive.
function ensureFuelRescueShip(aiShips) {
    if (!fuelRescueEnabled || !alphaAttackEnabled) return;
    if (fuelRescueDispatchCooldown > 0) {
        fuelRescueDispatchCooldown--;
        return;
    }
    const alpha = aiShips.find(s => s && s.isAlpha);
    if (!alpha) return;
    updateAlphaHelpPing(alpha);
    if (!alpha.helpPingActive) return;
    if (aiShips.some(s => s && s.role === 'fuel')) return;

    const spawnX = Math.max(80, worldW() * 0.1);
    const spawnY = Math.max(80, worldH() * 0.9);
    const fuelShip = new AIShip(spawnX, spawnY);
    fuelShip.role = 'fuel';
    fuelShip.roleAssigned = true;
    fuelShip.alphaShip = alpha;
    fuelShip.isAlpha = false;
    fuelShip.fuelStandby = false;
    fuelShip.fuelDeparting = false;
    fuelShip.fuelDocked = false;
    fuelShip.maxFuel = 100;
    fuelShip.fuel = 100;
    // A dispatched tanker is scripted, never policy-driven, and never fires, so
    // it needs no agent identity. Giving it aiShips.length as an id collided
    // with live agents once earlier ships had died, and resetting the credit
    // ledger here discarded kills earned in the same frame.
    fuelShip.agentId = undefined;
    fuelShip.formationType = formationType || 'arrowhead';
    fuelShip.standbyTarget = { x: spawnX, y: spawnY };
    aiShips.push(fuelShip);
    fuelRescueDispatchCooldown = 180;
}

function processFuelTransfers() {
    if (!fuelRescueEnabled || !alphaAttackEnabled) return;

    ensureFuelRescueShip(aiShips);

    // An alpha whose link partner is gone (killed by something other than an
    // asteroid, or re-rolled) would otherwise stay "docked" forever, and a
    // docked alpha never pings.
    aiShips.forEach(s => {
        if (!s || !s.isAlpha || !(s.fuelDocked || s.fuelTransferActive)) return;
        const p = s.fuelTransferPartner;
        const partnerLinked = p && (p.fuelDocked || p.fuelTransferActive) && p.fuelTransferPartner === s;
        if (!p || !aiShips.includes(p) || p.role !== 'fuel' || !partnerLinked) {
            clearFuelLink(s, null);
        }
    });

    aiShips.forEach(fuelShip => {
        if (!fuelShip || fuelShip.role !== 'fuel') return;
        const alpha = aiShips.find(s => s && s.isAlpha);
        if (!alpha) return;

        if (fuelShip.fuelDocked) {
            snapFuelShipDocked(fuelShip, alpha);
            fuelShip.fuelDockHoldTimer--;
            alpha.fuelDockHoldTimer = fuelShip.fuelDockHoldTimer;
            if (fuelShip.fuelDockHoldTimer <= 0) {
                releaseFuelDock(fuelShip, alpha);
            }
            return;
        }

        if (fuelShip.fuelDeparting || fuelShip.fuelStandby) return;

        // Once a transfer has started it runs to completion. The first transfer
        // frame clears the alpha's HELP ping, so gating every frame on the ping
        // (as an earlier version did) stalled the link after 2.2 units of fuel
        // and left both ships frozen in the "FUEL LINK" state indefinitely.
        const transferring = fuelShip.fuelTransferActive && fuelShip.fuelTransferPartner === alpha;
        if (!transferring && (!alpha.helpPingActive || fuelShip.fuel <= 0)) return;

        const dx = alpha.x - fuelShip.x;
        const dy = alpha.y - fuelShip.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (transferring || distance < fuelShip.fuelTransferRange) {
            fuelShip.fuelTransferActive = true;
            alpha.fuelTransferActive = true;
            fuelShip.fuelTransferPartner = alpha;
            alpha.fuelTransferPartner = fuelShip;
            snapFuelShipDocked(fuelShip, alpha);
            alpha.helpPingActive = false;

            const amount = Math.min(
                fuelShip.fuelTransferRate,
                fuelShip.fuel,
                alpha.maxFuel - alpha.fuel
            );
            if (amount > 0) {
                fuelShip.fuel -= amount;
                alpha.fuel += amount;
            }

            fuelShip.fuelTransferTimer++;
            alpha.fuelTransferTimer = fuelShip.fuelTransferTimer;

            const transferComplete =
                alpha.fuel >= alpha.maxFuel * 0.85 ||
                fuelShip.fuel <= 0 ||
                fuelShip.fuelTransferTimer >= fuelShip.fuelTransferDurationMax;

            if (transferComplete) {
                startFuelDockHold(fuelShip, alpha);
            }
        }
    });
}

// Initialize game with 5 asteroids.
// `seed` makes the episode replayable: the same seed always produces the same
// asteroid layout, spawn positions and every subsequent stochastic decision.
//
// This is a *full* reset. Anything left over from a previous episode - surviving
// AI ships, the player's velocity, coordination-system state, the frame counter
// that gates ML inference - silently leaks into the next episode and makes two
// runs of the same seed diverge.
function initGame(seed) {
    if (seed !== undefined && seed !== null) {
        setGameSeed(seed);
    }
    episodeStepCount = 0;
    episodeDone = false;
    episodeId = `ep_${getGameSeed()}_${episodeSequence++}`;

    gameRunning = true;
    score = 0;
    mlGlobalFrame = 0;
    shootCooldown = 0;
    enemySpawnTimer = 0;

    bullets = [];
    enemyBullets = [];
    enemyShips = [];
    aiShips = [];
    powerUps = [];

    // Reset the player ship in place: `ship` is a const binding, but copying a
    // freshly constructed Ship over it restores every field, including any added
    // later, without needing to enumerate them here.
    if (typeof ship !== 'undefined' && ship) {
        Object.assign(ship, new Ship());
    }
    if (typeof keys !== 'undefined' && keys) {
        Object.keys(keys).forEach(k => { keys[k] = false; });
    }
    enemyCoordinationSystem = new EnemyCoordinationSystem();

    asteroids = [];
    for (let i = 0; i < 5; i++) {
        asteroids.push(new Asteroid());
    }

    updateScore();
    // Initialize AI ships at random locations
    updateAIShips();
    // Initialize enemy ships (basic/advanced)
    updateEnemyShips();
    // Initialize boss ships
    updateBossShips();
    // Don't reset playerShipActive here - preserve current state
    // playerShipActive state is managed by Streamlit toggle
}

// Spawn enemy ship
function spawnEnemyShip(type = 'basic') {
    enemyShips.push(new EnemyShip(null, null, type));
    if (type === 'boss') playBossAlert();
}

// Spawn enemy ships periodically
let enemySpawnTimer = 0;
const enemySpawnInterval = 600; // Spawn every 10 seconds at 60fps

// Update AI ships based on desired count
function updateAIShips() {
    // Remove excess ships if count decreased
    if (aiShips.length > numAIShips) {
        aiShips = aiShips.slice(0, numAIShips);
    }
    // Add new ships if count increased
    while (aiShips.length < numAIShips) {
        const x = gameRandom() * worldW();
        const y = gameRandom() * worldH();
        const newShip = new AIShip(x, y);
        newShip.formationType = formationType || 'arrowhead'; // Set formation type
        aiShips.push(newShip);
    }
    // Update existing ships' formation type and keep agentId aligned with index,
    // which is the key the credit ledger and the policy batch are indexed by.
    aiShips.forEach((ship, index) => {
        if (ship) {
            ship.formationType = formationType || 'arrowhead';
            ship.agentId = index;
        }
    });
    resetAgentCredit(aiShips.length);
    // Update display
    updateAIShipCount();
}

// Function to set number of AI ships (called from Streamlit)
window.setNumAIShips = function(count) {
    numAIShips = Math.max(0, Math.min(10, count)); // Limit between 0 and 10
    updateAIShips();
};

// Update enemy ships based on desired count (basic and advanced only, not boss)
function updateEnemyShips() {
    // Count non-boss ships
    const nonBossShips = enemyShips.filter(ship => ship && ship.type !== 'boss');
    
    // Remove excess non-boss ships if count decreased
    if (nonBossShips.length > numEnemyShips) {
        let removed = 0;
        const toRemove = nonBossShips.length - numEnemyShips;
        for (let i = enemyShips.length - 1; i >= 0 && removed < toRemove; i--) {
            if (enemyShips[i] && enemyShips[i].type !== 'boss') {
                enemyShips.splice(i, 1);
                removed++;
            }
        }
    }
    
    // Add new non-boss ships if count increased
    while (enemyShips.filter(ship => ship && ship.type !== 'boss').length < numEnemyShips) {
        // Spawn random enemy type based on score (but not boss)
        let enemyType = 'basic';
        if (score > 1000) {
            enemyType = gameRandom() < 0.5 ? 'advanced' : 'basic';
        }
        spawnEnemyShip(enemyType);
    }
    updateEnemyShipCount();
}

// Update boss ships based on desired count
function updateBossShips() {
    // Count boss ships
    const currentBossShips = enemyShips.filter(ship => ship && ship.type === 'boss');
    
    // Remove excess boss ships if count decreased
    if (currentBossShips.length > numBossShips) {
        let removed = 0;
        const toRemove = currentBossShips.length - numBossShips;
        for (let i = enemyShips.length - 1; i >= 0 && removed < toRemove; i--) {
            if (enemyShips[i] && enemyShips[i].type === 'boss') {
                enemyShips.splice(i, 1);
                removed++;
            }
        }
    }
    
    // Add new boss ships if count increased
    while (enemyShips.filter(ship => ship && ship.type === 'boss').length < numBossShips) {
        spawnEnemyShip('boss');
    }
    updateBossShipCount();
}

// Function to set number of enemy ships (called from Streamlit)
window.setNumEnemyShips = function(count) {
    numEnemyShips = Math.max(0, Math.min(10, count)); // Limit between 0 and 10
    updateEnemyShips();
};

// Function to set number of boss ships (called from Streamlit)
window.setNumBossShips = function(count) {
    numBossShips = Math.max(0, Math.min(10, count)); // Limit between 0 and 10
    updateBossShips();
};

// Phase 5.1 & 5.2: Function to set ML mode (called from Streamlit)
window.setMLMode = function(enabled, mode = 'parameters') {
    const mlEnabled = enabled === true || enabled === 'true';
    // mode can be: 'parameters', 'priorities', 'both', or 'full'
    
    let updatedCount = 0;
    aiShips.forEach(aiShip => {
        if (aiShip) {
            aiShip.mlEnabled = mlEnabled;
            
            // Phase 5.1: ML Parameter Tuning
            if (mode === 'parameters' || mode === 'both' || mode === 'full') {
                aiShip.useMLParameters = mlEnabled;
            } else {
                aiShip.useMLParameters = false;
            }
            
            // Phase 5.2: ML Priority Weights
            if (mode === 'priorities' || mode === 'both' || mode === 'full') {
                aiShip.useMLPriorities = mlEnabled;
            } else {
                aiShip.useMLPriorities = false;
            }
            
            // Phase 5.3: Full ML Decision Making
            if (mode === 'full') {
                aiShip.useMLDecisions = mlEnabled;
            } else {
                aiShip.useMLDecisions = false;
            }
            
            // Phase 5.2: ML Priority Weights
            if (mode === 'priorities' || mode === 'both' || mode === 'full') {
                aiShip.useMLPriorities = mlEnabled;
            } else {
                aiShip.useMLPriorities = false;
            }
            
            updatedCount++;
        }
    });
    
    const modeText = mode === 'parameters' ? 'Parameter Tuning' : 
                     mode === 'priorities' ? 'Priority Weights' : 
                     mode === 'both' ? 'Parameters & Priorities' : 
                     mode === 'full' ? 'Full ML' : 'Parameter Tuning';
    
    console.log(`[ML] ${modeText} mode: ${mlEnabled ? 'ENABLED' : 'DISABLED'} (${updatedCount} AI ships updated)`);
    if (mlEnabled) {
        if (mode === 'parameters' || mode === 'both' || mode === 'full') {
            console.log('[ML] Using ML-suggested parameters for detection radius, firing range, flock weight, etc.');
        }
        if (mode === 'priorities' || mode === 'both' || mode === 'full') {
            console.log('[ML] Using ML-suggested priority weights for decision making.');
        }
    }
};

// Phase 5.1: Function to export training data (for future ML training)
window.exportMLTrainingData = function() {
    if (typeof exportMARLTrainingData === 'function') {
        return exportMARLTrainingData();
    }
    let allTrainingData = [];
    aiShips.forEach(aiShip => {
        if (aiShip && aiShip.trainingDataBuffer && aiShip.trainingDataBuffer.length > 0) {
            allTrainingData = allTrainingData.concat(aiShip.trainingDataBuffer);
        }
    });
    if (allTrainingData.length === 0) {
        console.log('No training data collected yet. Enable data collection first.');
        return null;
    }
    const dataStr = JSON.stringify(allTrainingData, null, 2);
    console.log(`Exported ${allTrainingData.length} training samples`);
    return dataStr;
};

window.setMLDataCollection = function(enabled) {
    mlDataCollectionEnabled = enabled === true || enabled === 'true';
    aiShips.forEach(ship => { if (ship) ship.collectTrainingData = mlDataCollectionEnabled; });
    if (typeof setMARLDataCollection === 'function') setMARLDataCollection(mlDataCollectionEnabled);
    console.log(`[ML] Data collection: ${mlDataCollectionEnabled ? 'ENABLED' : 'DISABLED'}`);
};

window.setMLHierarchy = function(enabled) {
    mlHierarchyEnabled = enabled === true || enabled === 'true';
    if (typeof setMLHierarchyEnabled === 'function') setMLHierarchyEnabled(mlHierarchyEnabled);
};

window.setMLEvaluation = function(enabled, mlMode = false) {
    mlEvaluationEnabled = enabled === true || enabled === 'true';
    if (typeof setMLEvaluationEnabled === 'function') setMLEvaluationEnabled(mlEvaluationEnabled, mlMode);
};

window.setNeuralNetworkModels = async function(enabled, config = {}) {
    mlUseNeuralNetwork = enabled === true || enabled === 'true';
    if (!mlUseNeuralNetwork) return;

    if (typeof initializeMLModelLoader === 'function' && typeof initializeMLModelAdapter === 'function') {
        const loader = initializeMLModelLoader();
        await loader.loadModels(config);
        window.mlModelAdapter = initializeMLModelAdapter(loader);
        aiShips.forEach(ship => {
            if (ship) ship.mlModel = window.mlModelAdapter;
        });
        console.log('[ML] Neural network models loaded:', loader.getStatus());
    }
};

window.getMLSystemStatus = function() {
    return {
        mlEnabled: aiShips.some(s => s && s.mlEnabled),
        neuralNetwork: mlUseNeuralNetwork,
        hierarchy: mlHierarchyEnabled,
        evaluation: mlEvaluationEnabled,
        dataCollection: mlDataCollectionEnabled,
        modelStatus: (typeof mlModelLoader !== 'undefined' && mlModelLoader) ? mlModelLoader.getStatus() : null,
        evalSummary: (typeof getMLEvaluationSummary === 'function') ? getMLEvaluationSummary() : null,
        dataStats: (typeof getMARLDataCollectorStats === 'function') ? getMARLDataCollectorStats() : null,
        marlStats: (typeof getMARLStats === 'function') ? getMARLStats() : null
    };
};

// Keyboard input handling
const keys = {};

// Ensure canvas gets focus for keyboard events
canvas.setAttribute('tabindex', '0');
canvas.style.outline = 'none';
canvas.focus();

// Click to focus
canvas.addEventListener('click', () => {
    canvas.focus();
});

// Handle both key and keyCode for better compatibility
function getKeyName(e) {
    // Handle arrow keys by keyCode for better compatibility
    if (e.keyCode) {
        switch(e.keyCode) {
            case 37: return 'arrowleft';
            case 38: return 'arrowup';
            case 39: return 'arrowright';
            case 40: return 'arrowdown';
            case 32: return ' ';
            case 72: return 'h';
            case 65: return 'a';
            case 68: return 'd';
            case 27: return 'escape';
        }
    }
    return e.key ? e.key.toLowerCase() : '';
}

window.addEventListener('keydown', (e) => {
    const keyName = getKeyName(e);
    if (keyName) {
        keys[keyName] = true;
        
        // Prevent default for game keys
        if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(keyName)) {
            e.preventDefault();
            e.stopPropagation();
        }
    }
}, true);

window.addEventListener('keyup', (e) => {
    const keyName = getKeyName(e);
    if (keyName) {
        keys[keyName] = false;
        if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(keyName)) {
            e.preventDefault();
            e.stopPropagation();
        }
    }
}, true);

// Also listen on document for better capture
document.addEventListener('keydown', (e) => {
    const keyName = getKeyName(e);
    if (keyName) {
        keys[keyName] = true;
        if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(keyName)) {
            e.preventDefault();
            e.stopPropagation();
        }
    }
}, true);

document.addEventListener('keyup', (e) => {
    const keyName = getKeyName(e);
    if (keyName) {
        keys[keyName] = false;
        if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(keyName)) {
            e.preventDefault();
            e.stopPropagation();
        }
    }
}, true);

// Touch controls: on-screen buttons for touch devices, driving the same
// `keys` map the keyboard listeners above use. handleInput()/handleShooting()
// read from `keys` either way, so they need no changes to support touch.
function setupTouchControls() {
    // The headless JSC training/test harness (sim/browser_shim.js) stubs out
    // `document`/`window` but defines neither `navigator` nor
    // querySelectorAll() on its element stubs -- game.js loads there too, so
    // both must be feature-detected rather than assumed to exist.
    if (typeof navigator === 'undefined' || typeof document === 'undefined') return;
    const container = document.getElementById('touchControls');
    if (!container || typeof container.querySelectorAll !== 'function') return;
    const isTouchDevice = (typeof window !== 'undefined' && 'ontouchstart' in window) || navigator.maxTouchPoints > 0;
    if (!isTouchDevice) return;
    container.hidden = false;

    container.querySelectorAll('.touch-btn').forEach(btn => {
        const keyName = btn.dataset.key === 'space' ? ' ' : btn.dataset.key;
        const press = (e) => { e.preventDefault(); keys[keyName] = true; };
        const release = (e) => { e.preventDefault(); keys[keyName] = false; };
        btn.addEventListener('touchstart', press, { passive: false });
        btn.addEventListener('touchend', release, { passive: false });
        btn.addEventListener('touchcancel', release, { passive: false });
    });
}
setupTouchControls();

// Handle input
function handleInput() {
    // Rotation
    if (keys['arrowleft']) {
        ship.rotate(-1);
    }
    if (keys['arrowright']) {
        ship.rotate(1);
    }

    // Movement
    if (keys['arrowup']) {
        ship.thrust();
    }
    if (keys['arrowdown']) {
        ship.moveBackward();
    }
    if (keys['a'] || (keys['arrowleft'] && !keys['arrowright'])) {
        // Strafe left (when only left arrow, or 'a' key)
        if (!keys['arrowup'] && !keys['arrowdown']) {
            ship.moveLeft();
        }
    }
    if (keys['d'] || (keys['arrowright'] && !keys['arrowleft'])) {
        // Strafe right (when only right arrow, or 'd' key)
        if (!keys['arrowup'] && !keys['arrowdown']) {
            ship.moveRight();
        }
    }

    // Space bar for shooting and speed boost
    if (keys[' ']) {
        ship.thrust(1.5); // Speed boost (reduced from 2 to prevent excessive speed)
    }

    // Hyperspace
    if (keys['h']) {
        ship.hyperspace();
        keys['h'] = false; // Prevent continuous hyperspace
    }

    // Shield force field (F key)
    ship.shieldActive = keys['f'] || false;

    // Fullscreen toggle (ESC key)
    if (keys['escape']) {
        toggleFullscreen();
        keys['escape'] = false; // Prevent continuous toggling
    }
}

// Shooting
let shootCooldown = 0;
function handleShooting() {
    if (keys[' '] && shootCooldown <= 0) {
        // Fire from center gun (primary) - straight ahead
        const centerX = ship.x + Math.cos(ship.angle) * ship.size;
        const centerY = ship.y + Math.sin(ship.angle) * ship.size;
        bullets.push(new Bullet(centerX, centerY, ship.angle));
        playFireSound(false); // Play player firing sound
        
        // Fire from left gun barrel (15 degrees to the left)
        const leftAngle = ship.angle - Math.PI / 12; // 15 degrees
        // Calculate left gun position relative to ship center
        const leftOffsetX = Math.cos(ship.angle) * (ship.size - 3) + Math.cos(ship.angle - Math.PI / 2) * (-ship.size / 4);
        const leftOffsetY = Math.sin(ship.angle) * (ship.size - 3) + Math.sin(ship.angle - Math.PI / 2) * (-ship.size / 4);
        bullets.push(new Bullet(ship.x + leftOffsetX, ship.y + leftOffsetY, leftAngle));
        
        // Fire from right gun barrel (15 degrees to the right)
        const rightAngle = ship.angle + Math.PI / 12; // 15 degrees
        // Calculate right gun position relative to ship center
        const rightOffsetX = Math.cos(ship.angle) * (ship.size - 3) + Math.cos(ship.angle + Math.PI / 2) * (ship.size / 4);
        const rightOffsetY = Math.sin(ship.angle) * (ship.size - 3) + Math.sin(ship.angle + Math.PI / 2) * (ship.size / 4);
        bullets.push(new Bullet(ship.x + rightOffsetX, ship.y + rightOffsetY, rightAngle));
        
        shootCooldown = 10; // Cooldown between shots
    }
    if (shootCooldown > 0) {
        shootCooldown--;
    }
}

// Collision detection
function checkCollision(obj1, obj2) {
    const pos1 = obj1.getPosition();
    const pos2 = obj2.getPosition();
    if (sphereWorld()) {
        // Arc distance: chart dx is meaningless near the poles and across the
        // antimeridian.
        return WORLD_GEO.distance(pos1.x, pos1.y, pos2.x, pos2.y) < (pos1.radius + pos2.radius);
    }
    const dx = pos1.x - pos2.x;
    const dy = pos1.y - pos2.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    return distance < (pos1.radius + pos2.radius);
}

// Update game state
// Satellite / pickup EMP: clears nearby enemy fire and stuns the enemy fleet.
// Declared at top level so Mission Control's DETONATE EMP button (hud.js) can reach
// it; it used to be nested inside update(), where that button's typeof check
// always failed and the click silently did nothing.
function triggerEMPBlast(x, y) {
    if (FXL) {
        FXL.shake(8, 25);
        FXL.emitExplosion(x, y, '#FFD600', 4.0);
    }
    playExplosionSound(4);
    // Clear all enemy bullets within 400px
    enemyBullets = enemyBullets.filter(b => Math.hypot(b.x - x, b.y - y) > 400);
    // Stun all enemy ships for 4 seconds
    enemyShips.forEach(e => {
        if (!e) return;
        e.stunTimer = 240;
        if (FXL) FXL.emitDebris(e.x, e.y, e.size);
    });
    if (typeof killFeedPush === 'function') {
        killFeedPush('💥 EMP SHOCKWAVE DISCHARGED!', '#FFD600');
    }
}

function update() {
    if (!gameRunning) return;

    syncMusic();
    // Edge-triggered so holding the key doesn't toggle every frame.
    if (keys['p']) {
        togglePause();
        keys['p'] = false;
    }
    if (keys['m']) {
        musicEnabled = !musicEnabled;
        keys['m'] = false;
    }
    if (gamePaused) return;

    if (gameOverState) {
        gameOverState.frames++;
        const wantsRestart = keys['r'] || keys['enter'];
        if (gameOverState.frames > 30 && (wantsRestart || gameOverState.frames > GAME_OVER_AUTO_RESTART_FRAMES)) {
            gameOverState = null;
            initGame();
        }
        return;
    }

    episodeStepCount++;

    // Only handle input and update player ship if active
    if (playerShipActive) {
        handleInput();
        handleShooting();
        ship.update();
    } else {
        // Ship inactive - apply friction to slow it down
        ship.velocityX *= ship.friction;
        ship.velocityY *= ship.friction;
        ship.update();
    }

    // Update AI ships
    mlGlobalFrame++;

    // Hierarchical ML strategic layer
    if (mlHierarchyEnabled && typeof initializeMLHierarchy === 'function') {
        const hierarchy = initializeMLHierarchy();
        const shipsFeatures = aiShips.filter(s => s).map(ship => ({
            ship,
            features: ship.extractCanonicalFeatures(asteroids, aiShips, ship, enemyShips, enemyBullets),
            aiShips
        }));
        hierarchy.update(mlGlobalFrame, shipsFeatures);
    }

    // Batched neural network inference (every 5 frames)
    if (mlUseNeuralNetwork && typeof mlModelAdapter !== 'undefined' && mlModelAdapter && mlGlobalFrame % 5 === 0) {
        mlModelAdapter.tickCache(mlGlobalFrame);
        const batchData = aiShips.filter(s => s).map((ship, i) => ({
            features: ship.extractCanonicalFeatures(asteroids, aiShips, ship, enemyShips, enemyBullets),
            shipId: `ship_${i}`
        }));
        mlModelAdapter.batchPredict(batchData).then(results => {
            aiShips.forEach((ship, i) => {
                if (ship && results[i]) {
                    ship._cachedMLParams = results[i].params;
                    ship._cachedMLPriorities = results[i].priorities;
                }
            });
        }).catch(err => console.warn('[ML] Batch inference failed:', err));
    }

    let marlActions = null;
    if (typeof updateMARL === 'function' && typeof marlEnabled !== 'undefined' && marlEnabled) {
        marlActions = updateMARL();
    }

    // A loaded PPO policy decides for all agents here, before the per-ship loop.
    if (typeof policyRunnerTick === 'function') {
        policyRunnerTick();
    }

    const previousScoreForMARL = typeof marlPreviousScore !== 'undefined' ? marlPreviousScore : score;
    let usedMLCount = 0;

    if (expertActionRecording) armExpertActionRecords();

    // The rule-based controller selects its alpha inside makeDecision(). A
    // policy-driven swarm never gets there, so do it here.
    if (alphaAttackEnabled && policyControlEnabled) ensureAlphaAssignment(aiShips);

    aiShips.forEach((aiShip, index) => {
        if (aiShip) {
            // Must run before the decision, and for every controller.
            aiShip.updateCooldowns();

            const scriptedTanker = fuelRescueEnabled && alphaAttackEnabled && aiShip.role === 'fuel';
            const useMARL = marlActions && marlActions[index] && typeof executeMARLAction === 'function' && marlEnabled;
            if (scriptedTanker && (useMARL || policyControlEnabled)) {
                // The tanker is part of the environment, not of the policy: its
                // flight is scripted whoever is flying the combat ships.
                aiShip.handleFuelShipBehavior(aiShips);
            } else if (useMARL) {
                executeMARLAction(aiShip, marlActions[index]);
            } else if (policyControlEnabled) {
                // Dead ships are spliced out of aiShips, so array position is not a
                // stable agent identity. Always look the action up by agentId.
                const agentId = aiShip.agentId !== undefined ? aiShip.agentId : index;
                const action = policyActions ? policyActions[agentId] : null;
                if (action !== undefined && action !== null) {
                    applyAgentAction(aiShip, action);
                }
            } else {
                aiShip.makeDecision(asteroids, aiShips, ship, enemyShips, enemyBullets);
                if (aiShip.mlEnabled) usedMLCount++;
            }

            aiShip.updateAIPhase();
            aiShip.updateAIShield();
            aiShip.tickFuel();
            if (aiShip.isAlpha && aiShip.helpPingActive) fuelEvents.lowFuelFrames++;
            aiShip.update();
        }
    });

    processFuelTransfers();
    if (FXL) FXL.update();

    if (typeof storeMARLExperience === 'function' && typeof marlTrainingMode !== 'undefined' && marlTrainingMode) {
        storeMARLExperience(previousScoreForMARL, score);
        if (typeof trainMARL === 'function') trainMARL();
    }

    if (mlEvaluationEnabled && typeof initializeMLEvaluation === 'function') {
        const tracker = initializeMLEvaluation();
        const formationErrors = aiShips.filter(s => s).map(s => {
            const f = s.extractCanonicalFeatures(asteroids, aiShips, s, enemyShips, enemyBullets);
            return f.formationError || 0;
        });
        tracker.recordFrame({
            aiShips,
            score,
            inferenceMs: (typeof mlModelAdapter !== 'undefined' && mlModelAdapter) ? mlModelAdapter.getInferenceMs() : 0,
            usedML: usedMLCount > 0,
            formationErrors
        });
    }
    
    // Update AI ship count display
    updateAIShipCount();
    updateEnemyShipCount();
    updateBossShipCount();

    // Update bullets
    bullets = bullets.filter(bullet => {
        bullet.update();
        return bullet.isAlive();
    });

    // Update enemy bullets
    enemyBullets = enemyBullets.filter(bullet => {
        bullet.update();
        return bullet.isAlive();
    });

    // Update tactical power-ups (interactive sessions only)
    if (!isHeadlessEnv() && powerUps.length > 0) {
        powerUps.forEach(p => p.update());
        powerUps = powerUps.filter(p => p.lifetime > 0);

        // Check power-up collection by player or AI ships
        const activeCollectors = [];
        if (playerShipActive) activeCollectors.push(ship);
        aiShips.forEach(s => { if (s && s.health > 0) activeCollectors.push(s); });

        const collectedPowerUps = new Set();
        powerUps.forEach(p => {
            for (const col of activeCollectors) {
                const dist = Math.hypot(p.x - col.x, p.y - col.y);
                if (dist < p.radius + col.radius + 6) {
                    collectedPowerUps.add(p);
                    if (p.type === 'railgun') {
                        col.buffRailgun = 600; // 10s piercing railgun
                        if (typeof killFeedPush === 'function') killFeedPush('⚡ RAILGUN OVERCHARGE ACTIVATED!', '#00FF66');
                    } else if (p.type === 'overshield') {
                        col.buffShield = 480; // 8s
                        col.shieldActive = true;
                        col.shieldRadius = 45;
                        if (typeof killFeedPush === 'function') killFeedPush('🛡 OVERSHIELD BARRIER ONLINE!', '#00E5FF');
                    } else if (p.type === 'emp') {
                        triggerEMPBlast(p.x, p.y);
                    } else if (p.type === 'fuel') {
                        if (col.fuel !== undefined) col.fuel = Math.min(col.maxFuel, col.fuel + 50);
                        const alpha = aiShips.find(s => s && s.isAlpha);
                        if (alpha && alpha.fuel !== undefined) {
                            alpha.fuel = Math.min(alpha.maxFuel, alpha.fuel + 50);
                            if (alpha.fuel > alpha.maxFuel * alpha.fuelHelpThreshold) alpha.helpPingActive = false;
                        }
                        if (typeof killFeedPush === 'function') killFeedPush('⛽ FLEET EMERGENCY FUEL RESTORED!', '#FF0077');
                    }
                    if (FXL) FXL.emitExplosion(p.x, p.y, p.color, 1.2);
                    playPickupSound();
                    score += 150;
                    updateScore();
                    break;
                }
            }
        });
        if (collectedPowerUps.size) powerUps = powerUps.filter(p => !collectedPowerUps.has(p));
    }

    // Update enemy ships
    // COORDINATION: Update coordination system
    enemyCoordinationSystem.update(enemyShips, aiShips);
    
    enemyShips.forEach(enemyShip => {
        if (enemyShip) {
            enemyShip.makeDecision(asteroids, enemyShips, aiShips);
            enemyShip.update();
        }
    });

    // Update asteroids
    asteroids.forEach(asteroid => asteroid.update());

    // Check asteroid-to-asteroid collisions
    const newFragments = [];
    const asteroidsToRemove = [];
    
    for (let i = 0; i < asteroids.length; i++) {
        for (let j = i + 1; j < asteroids.length; j++) {
            const asteroid1 = asteroids[i];
            const asteroid2 = asteroids[j];
            
            if (checkCollision(asteroid1, asteroid2)) {
                // Calculate collision normal (direction from asteroid1 to asteroid2)
                const dx = asteroid2.x - asteroid1.x;
                const dy = asteroid2.y - asteroid1.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance > 0) {
                    const normalX = dx / distance;
                    const normalY = dy / distance;
                    
                    // Calculate relative velocity
                    const relativeVelX = asteroid2.velocityX - asteroid1.velocityX;
                    const relativeVelY = asteroid2.velocityY - asteroid1.velocityY;
                    
                    // Calculate relative speed along collision normal
                    const relativeSpeed = relativeVelX * normalX + relativeVelY * normalY;
                    
                    // Only break apart if asteroids are moving towards each other
                    // (negative relative speed means approaching)
                    if (relativeSpeed < -0.5) { // Threshold to prevent breaking on slow collisions
                        // Mark both asteroids for removal
                        if (!asteroidsToRemove.includes(i)) asteroidsToRemove.push(i);
                        if (!asteroidsToRemove.includes(j)) asteroidsToRemove.push(j);
                        if (FXL) {
                            FXL.emitDebris((asteroid1.x + asteroid2.x) / 2, (asteroid1.y + asteroid2.y) / 2,
                                (asteroid1.radius + asteroid2.radius) / 2);
                        }
                        
                        // Create fragments from both asteroids
                        const fragments1 = asteroid1.breakApart(normalX, normalY, relativeVelX, relativeVelY);
                        const fragments2 = asteroid2.breakApart(-normalX, -normalY, -relativeVelX, -relativeVelY);
                        
                        // Add fragments to array
                        newFragments.push(...fragments1);
                        newFragments.push(...fragments2);
                    } else {
                        // Elastic collision - bounce asteroids apart
                        // Calculate masses (proportional to area)
                        const mass1 = asteroid1.size * asteroid1.size;
                        const mass2 = asteroid2.size * asteroid2.size;
                        const totalMass = mass1 + mass2;
                        
                        // Calculate velocity change using conservation of momentum
                        const velocityChange = 2 * relativeSpeed / totalMass;
                        
                        // Update velocities (elastic collision)
                        asteroid1.velocityX += velocityChange * normalX * mass2;
                        asteroid1.velocityY += velocityChange * normalY * mass2;
                        asteroid2.velocityX -= velocityChange * normalX * mass1;
                        asteroid2.velocityY -= velocityChange * normalY * mass1;
                        
                        // Separate asteroids to prevent overlap
                        const overlap = (asteroid1.radius + asteroid2.radius) - distance;
                        if (overlap > 0) {
                            const separationX = normalX * overlap * 0.5;
                            const separationY = normalY * overlap * 0.5;
                            asteroid1.x -= separationX;
                            asteroid1.y -= separationY;
                            asteroid2.x += separationX;
                            asteroid2.y += separationY;
                        }
                    }
                }
            }
        }
    }
    
    // Remove broken asteroids (in reverse order to maintain indices)
    asteroidsToRemove.sort((a, b) => b - a);
    asteroidsToRemove.forEach(index => {
        asteroids.splice(index, 1);
    });
    
    // Add new fragments
    asteroids.push(...newFragments);

    // SHIELD SYSTEM: Apply shield force for enemy ships (basic and advanced)
    enemyShips.forEach(enemyShip => {
        if ((enemyShip.type === 'basic' || enemyShip.type === 'advanced') && enemyShip.shieldActive) {
            asteroids.forEach(asteroid => {
                const dx = asteroid.x - enemyShip.x;
                const dy = asteroid.y - enemyShip.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const shieldRadius = enemyShip.shieldRadius || 40;
                
                // If asteroid is within shield radius, apply repulsion (same as blue ships)
                if (distance < shieldRadius + asteroid.radius && distance > 0) {
                    const angle = Math.atan2(dy, dx);
                    const force = enemyShip.shieldForce || 0.5; // Same shield force as blue ships
                    asteroid.velocityX += Math.cos(angle) * force;
                    asteroid.velocityY += Math.sin(angle) * force;
                }
            });
        }
    });
    
    // Apply shield force with size-based physics
    if (ship.shieldActive) {
        asteroids.forEach(asteroid => {
            const dx = asteroid.x - ship.x;
            const dy = asteroid.y - ship.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // If asteroid is within shield radius, apply size-based repulsion
            if (distance < ship.shieldRadius + asteroid.radius && distance > 0) {
                const angle = Math.atan2(dy, dx);
                const force = ship.shieldForce * (1 - distance / (ship.shieldRadius + asteroid.radius));
                
                // Compare asteroid size to ship size
                const asteroidSize = asteroid.size || asteroid.radius * 2;
                const shipSize = ship.size;
                
                if (asteroidSize > shipSize) {
                    // Large asteroid: bounce the ship away (opposite direction)
                    const shipBounceForce = force * (asteroidSize / shipSize) * 0.3; // Scale force by size ratio
                    ship.velocityX -= Math.cos(angle) * shipBounceForce;
                    ship.velocityY -= Math.sin(angle) * shipBounceForce;
                    
                    // Also push asteroid slightly (but ship gets pushed more)
                    asteroid.velocityX += Math.cos(angle) * force * 0.2;
                    asteroid.velocityY += Math.sin(angle) * force * 0.2;
                } else {
                    // Small/equal asteroid: bounce asteroid away (normal behavior)
                    asteroid.velocityX += Math.cos(angle) * force;
                    asteroid.velocityY += Math.sin(angle) * force;
                }
            }
        });
    }


    // Check bullets (player and AI) vs enemy ships
    //
    // Marks hits into Sets instead of splicing bullets/enemyShips while
    // forEach is iterating them (splicing mid-forEach shifts the next live
    // element into the removed slot, which forEach then skips for this
    // pass). The `break` after a bullet's first hit also restores the
    // intended one-bullet-one-target semantics: without it, a bullet
    // overlapping two enemies in the same frame could damage both.
    const bulletsHittingEnemies = new Set();
    const enemiesDestroyed = new Set();
    bullets.forEach(bullet => {
        for (const enemyShip of enemyShips) {
            if (!enemyShip || enemiesDestroyed.has(enemyShip)) continue;

            // Boss subsystem targeting: check if bullet hits any active subsystem first (interactive mode)
            let subsystemHit = false;
            if (!isHeadlessEnv() && enemyShip.type === 'boss' && enemyShip.subsystems) {
                for (const sub of enemyShip.subsystems) {
                    if (sub.destroyed) continue;
                    const subPos = enemyShip.getSubsystemWorldPos(sub);
                    const dist = Math.hypot(bullet.x - subPos.x, bullet.y - subPos.y);
                    if (dist < bullet.radius + sub.radius) {
                        subsystemHit = true;
                        sub.health--;
                        if (FXL) FXL.emitExplosion(subPos.x, subPos.y, sub.color, 0.7);
                        playExplosionSound(0.5);
                        if (sub.health <= 0) {
                            sub.destroyed = true;
                            playExplosionSound(2.4);
                            if (FXL) {
                                FXL.emitExplosion(subPos.x, subPos.y, sub.color, 2.5);
                                FXL.shake(4, 14);
                            }
                            score += 300;
                            creditAgent(bullet.ownerAgentId, 'scoreEarned', 300);
                            updateScore();
                            // Guaranteed powerup drop from destroyed subsystem
                            if (!isHeadlessEnv()) {
                                powerUps.push(new PowerUp(subPos.x, subPos.y));
                            }
                            if (typeof killFeedPush === 'function') {
                                killFeedPush(`💥 ${sub.name.toUpperCase()} DESTROYED! +300`, sub.color);
                            }
                            enemyShip.onSubsystemDestroyed(sub.id);
                        }
                        if (!bullet.piercing || bullet.pierceHits >= 2) {
                            bulletsHittingEnemies.add(bullet);
                        } else {
                            bullet.pierceHits++;
                        }
                        break;
                    }
                }
            }
            if (subsystemHit) {
                if (bulletsHittingEnemies.has(bullet)) break;
                continue;
            }

            if (!checkCollision(bullet, enemyShip)) continue;

            // SHIELD SYSTEM: Shield blocks bullets for basic/advanced enemy ships
            if ((enemyShip.type === 'basic' || enemyShip.type === 'advanced') && enemyShip.shieldActive) {
                bulletsHittingEnemies.add(bullet);
                break; // Shield blocks the bullet - remove it but don't damage ship
            }

            // Boss shield blocks damage
            if (enemyShip.type === 'boss' && enemyShip.shieldActive) {
                bulletsHittingEnemies.add(bullet);
                break; // Shield blocks the bullet but doesn't destroy it
            }

            if (!bullet.piercing || bullet.pierceHits >= 2) {
                bulletsHittingEnemies.add(bullet);
            } else {
                bullet.pierceHits++;
            }
            enemyShip.health--;
            if (enemyShip.health <= 0) {
                enemiesDestroyed.add(enemyShip);
                let killValue = 0;
                if (enemyShip.type === 'basic') killValue = 100;
                else if (enemyShip.type === 'advanced') killValue = 250;
                else if (enemyShip.type === 'boss') killValue = 1000;
                if (FXL) {
                    const isBoss = enemyShip.type === 'boss';
                    FXL.emitExplosion(enemyShip.x, enemyShip.y, GAME_THEME.enemy, isBoss ? 3 : 1);
                    if (isBoss) FXL.shake(7, 22); else FXL.shake(2, 6);
                }
                playExplosionSound(enemyShip.type === 'boss' ? 3 : 1);
                if (enemyShip.type === 'boss') onBossKilled(enemyShip);
                score += killValue;
                creditAgent(bullet.ownerAgentId, enemyShip.type === 'boss' ? 'bossKills' : 'enemyKills', 1);
                creditAgent(bullet.ownerAgentId, 'scoreEarned', killValue);
                updateScore();
                updateEnemyShipCount();
                updateBossShipCount();
                // Kill-feed
                if (typeof killFeedPush === 'function') {
                    const icon = enemyShip.type === 'boss' ? '💀 BOSS' : enemyShip.type === 'advanced' ? '⚡ ADV' : '✕ SHIP';
                    killFeedPush(`${icon} +${killValue}`, enemyShip.type === 'boss' ? '#FF6B00' : '#FF4444');
                }
                // Boss or advanced ship drops tactical power-up
                if (!isHeadlessEnv() && (enemyShip.type === 'boss' || (enemyShip.type === 'advanced' && tacticalDropRng() < 0.45))) {
                    powerUps.push(new PowerUp(enemyShip.x, enemyShip.y));
                }
            }
            if (bulletsHittingEnemies.has(bullet)) break; // bullet is consumed; stop checking further enemies
        }
    });
    if (bulletsHittingEnemies.size) bullets = bullets.filter(b => !bulletsHittingEnemies.has(b));
    if (enemiesDestroyed.size) enemyShips = enemyShips.filter(e => !enemiesDestroyed.has(e));

    // Check enemy bullets vs player ship
    //
    // A hit ends the episode (initGame() rebuilds every entity array,
    // enemyBullets included), so this stops at the first hit instead of
    // continuing a forEach over the pre-reset array -- letting it continue
    // could check a second stale bullet against the just-reset ship and
    // trigger a second, spurious reset in the same frame.
    if (playerShipActive) {
        const hitByEnemyBullet = enemyBullets.some(bullet => checkCollision(bullet, ship));
        if (hitByEnemyBullet) {
            triggerGameOver();
            return;
        }
    }

    // Check enemy bullets vs AI ships
    //
    // aiShips.splice() below is safe as-is: bulletHit stops further checks
    // for this bullet after the first hit, so at most one splice happens per
    // enemyBullets.forEach callback, and each callback re-fetches aiShips
    // fresh. The outer enemyBullets.splice() is the actual bug -- it mutates
    // enemyBullets while forEach is iterating it -- so that one is collected
    // into a Set and filtered afterward instead.
    const enemyBulletsToRemove = new Set();
    enemyBullets.forEach(bullet => {
        let bulletHit = false;
        aiShips.forEach((aiShip, aiIndex) => {
            if (aiShip && !bulletHit && checkCollision(bullet, aiShip)) {
                bulletHit = true;
                // AI ship shield blocks damage
                if (aiShip.shieldActive) {
                    // Shield blocks the bullet - remove it but no damage
                    // Visual effect handled in draw
                    // Bullet will be removed below
                } else {
                    // Phase-Based Defense: Damage reduction in Phase 2/3
                    let damage = 1; // Base damage
                    if (aiShip.aiPhase === 2) {
                        damage = 0.75; // 25% damage reduction in Phase 2
                    } else if (aiShip.aiPhase === 3) {
                        damage = 0.5; // 50% damage reduction in Phase 3
                    }

                    // Reduce AI ship health instead of instant destruction
                    aiShip.health = Math.max(0, aiShip.health - damage);
                    if (FXL) FXL.emitMuzzle(bullet.x, bullet.y, Math.atan2(-bullet.velocityY || 0, -bullet.velocityX || 1), GAME_THEME.enemy);
                    if (aiShip.health <= 0) {
                        // AI ship destroyed by enemy bullet. Release the fuel
                        // link so the surviving partner is not left "docked"
                        // to a ship that no longer exists.
                        if (isFuelLinked(aiShip)) clearFuelLink(aiShip, aiShip.fuelTransferPartner);
                        aiShips.splice(aiIndex, 1);
                        if (FXL) { FXL.emitExplosion(aiShip.x, aiShip.y, GAME_THEME.aiBlue, 1.2); FXL.shake(3, 10); }
                        playExplosionSound(1.2);
                        if (typeof killFeedPush === 'function') {
                            const name = aiShip.isAlpha ? '★ Alpha' : (aiShip.role ? aiShip.role : 'Unit');
                            killFeedPush(`✦ ${name} destroyed`, '#66AAFF');
                        }
                        updateAIShipCount();
                    }
                }
            }
        });
        if (bulletHit) {
            enemyBulletsToRemove.add(bullet);
        }
    });
    if (enemyBulletsToRemove.size) {
        enemyBullets = enemyBullets.filter(b => !enemyBulletsToRemove.has(b));
    }

    // Check bullet-asteroid collisions
    //
    // Same fix as the bullets-vs-enemyShips pass above: mark hits into Sets
    // instead of splicing bullets/asteroids mid-forEach, and stop checking
    // further asteroids once a bullet has hit one (one bullet, one asteroid).
    // Fragments and the population top-up are applied after the scan so
    // `new Asteroid()`/`breakApart()` keep firing in the same bullet-major,
    // ascending order the original (buggy) forward loop intended -- both
    // consume the seeded RNG, and episodes must replay identically for a
    // given seed.
    const bulletsHittingAsteroids = new Set();
    const asteroidsDestroyed = new Set();
    const newAsteroidFragments = [];
    bullets.forEach(bullet => {
        for (const asteroid of asteroids) {
            if (asteroidsDestroyed.has(asteroid)) continue;
            if (!checkCollision(bullet, asteroid)) continue;

            if (!bullet.piercing || bullet.pierceHits >= 2) {
                bulletsHittingAsteroids.add(bullet);
            } else {
                bullet.pierceHits++;
            }
            asteroidsDestroyed.add(asteroid);

            // ── 3-Tier scoring ────────────────────────────────────────
            // Large (≥45): splits, 50 pts  – hard to finish off
            // Medium (≥25): splits, 100 pts
            // Small (<25):  no split, 200 pts bonus – precision reward
            const isLarge  = asteroid.size >= 45;
            const isMedium = asteroid.size >= 25 && asteroid.size < 45;
            const isSmall  = asteroid.size < 25;

            let tileScore = 100; // default / medium
            if (isLarge)  tileScore = 50;
            if (isSmall)  tileScore = 200;

            if (FXL) FXL.emitDebris(asteroid.x, asteroid.y, asteroid.radius);
            playExplosionSound(isSmall ? 0.3 : isMedium ? 0.5 : 0.7);
            score += tileScore;
            creditAgent(bullet.ownerAgentId, 'asteroidKills', 1);
            creditAgent(bullet.ownerAgentId, 'scoreEarned', tileScore);
            updateScore();
            // Post kill-feed event
            if (typeof killFeedPush === 'function') {
                const tier = isLarge ? 'L' : isMedium ? 'M' : 'S';
                killFeedPush(`☄ [${tier}] +${tileScore}`, GAME_THEME.primary);
            }

            // Mineral asteroids drop guaranteed tactical power-up
            if (!isHeadlessEnv() && asteroid.isMineral) {
                powerUps.push(new PowerUp(asteroid.x, asteroid.y));
                if (typeof killFeedPush === 'function') {
                    killFeedPush('💎 MINERAL CRITICAL MASS EXTRACTED!', '#FFD700');
                }
            }

            // Large/medium asteroids break into fragments; small ones vanish
            if (!isSmall) {
                const bvx = bullet.speed ? Math.cos(bullet.angle) * bullet.speed : 0;
                const bvy = bullet.speed ? Math.sin(bullet.angle) * bullet.speed : 0;
                const nx  = asteroid.x > 0 ? (bullet.x - asteroid.x) / (asteroid.radius || 1) : 1;
                const ny  = asteroid.y > 0 ? (bullet.y - asteroid.y) / (asteroid.radius || 1) : 0;
                newAsteroidFragments.push(...asteroid.breakApart(nx, ny, bvx, bvy));
            }
            if (bulletsHittingAsteroids.has(bullet)) break; // bullet is consumed; stop checking further asteroids
        }
    });
    if (bulletsHittingAsteroids.size) bullets = bullets.filter(b => !bulletsHittingAsteroids.has(b));
    if (asteroidsDestroyed.size) asteroids = asteroids.filter(a => !asteroidsDestroyed.has(a));
    asteroids.push(...newAsteroidFragments);
    // Keep total population at 5 if everything has been cleared
    while (asteroids.length < 5) {
        asteroids.push(new Asteroid());
    }

    // Check player ship-asteroid collisions (only if shield is not active and ship is active)
    if (playerShipActive && !ship.shieldActive) {
        if (asteroids.some(asteroid => checkCollision(ship, asteroid))) {
            triggerGameOver();
            return;
        }
    }

    // Check enemy ships vs asteroids
    //
    // Marks destroyed ships into a Set instead of splicing enemyShips while
    // forEach is iterating it (see the bullets-vs-enemyShips fix above for
    // why). The shield-bounce branch doesn't remove anything, so it's safe
    // to keep applying it per overlapping asteroid; once a ship is
    // destroyed it stops checking further asteroids against it.
    const enemyShipsDestroyedByAsteroids = new Set();
    enemyShips.forEach(enemyShip => {
        if (!enemyShip) return;
        for (const asteroid of asteroids) {
            if (!checkCollision(enemyShip, asteroid)) continue;

            // SHIELD SYSTEM: Shield blocks asteroid collision for basic and advanced enemy ships
            if ((enemyShip.type === 'basic' || enemyShip.type === 'advanced') && enemyShip.shieldActive) {
                // Shield blocks the collision - bounce asteroid away (same as blue ships)
                const dx = asteroid.x - enemyShip.x;
                const dy = asteroid.y - enemyShip.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance > 0) {
                    const angle = Math.atan2(dy, dx);
                    const force = enemyShip.shieldForce || 0.5; // Same shield force as blue ships
                    asteroid.velocityX += Math.cos(angle) * force;
                    asteroid.velocityY += Math.sin(angle) * force;
                }
                continue; // Skip damage, asteroid bounces off
            }

            // Boss shield already handled separately
            if (enemyShip.type === 'boss' && enemyShip.shieldActive) {
                continue; // Shield blocks damage
            }

            // No shield or shield inactive - enemy ship destroyed
            enemyShipsDestroyedByAsteroids.add(enemyShip);
            if (FXL) FXL.emitExplosion(enemyShip.x, enemyShip.y, GAME_THEME.enemy, enemyShip.type === 'boss' ? 3 : 1);
            playExplosionSound(enemyShip.type === 'boss' ? 3 : 1);
            if (enemyShip.type === 'boss') onBossKilled(enemyShip);
            break;
        }
    });
    if (enemyShipsDestroyedByAsteroids.size) {
        enemyShips = enemyShips.filter(e => !enemyShipsDestroyedByAsteroids.has(e));
    }

    // Check AI ship-asteroid collisions
    const linkedFuelDestruction = new Set();
    aiShips = aiShips.filter(aiShip => {
        if (!aiShip || linkedFuelDestruction.has(aiShip)) return false;
        for (let i = 0; i < asteroids.length; i++) {
            if (checkCollision(aiShip, asteroids[i])) {
                // Linked vulnerability: asteroid hit during fuel transfer destroys both ships
                if (
                    fuelRescueEnabled &&
                    isFuelLinked(aiShip) &&
                    aiShips.includes(aiShip.fuelTransferPartner)
                ) {
                    const partner = aiShip.fuelTransferPartner;
                    linkedFuelDestruction.add(partner);
                    recordFuelEvent('dockDeaths', aiShip, partner);
                    if (FXL) {
                        FXL.emitExplosion(aiShip.x, aiShip.y, GAME_THEME.fuelTransfer, 2.2);
                        FXL.emitExplosion(partner.x, partner.y, GAME_THEME.gold, 2.2);
                        playExplosionSound(2.4);
                        FXL.shake(9, 28);
                    }
                    clearFuelLink(aiShip, partner);
                    return false;
                }

                // AI ship shield blocks asteroid collision
                if (aiShip.shieldActive) {
                    // Shield blocks the collision - bounce asteroid away
                    const dx = asteroids[i].x - aiShip.x;
                    const dy = asteroids[i].y - aiShip.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance > 0) {
                        const angle = Math.atan2(dy, dx);
                        const force = 0.3;
                        asteroids[i].velocityX += Math.cos(angle) * force;
                        asteroids[i].velocityY += Math.sin(angle) * force;
                    }
                    continue; // Skip damage, asteroid bounces off
                }
                
                // Phase-Based Defense: Damage reduction in Phase 2/3
                let damage = 1; // Base damage
                if (aiShip.aiPhase === 2) {
                    damage = 0.75; // 25% damage reduction in Phase 2
                } else if (aiShip.aiPhase === 3) {
                    damage = 0.5; // 50% damage reduction in Phase 3
                }
                
                // Reduce AI ship health on asteroid collision
                aiShip.health = Math.max(0, aiShip.health - damage);
                if (aiShip.health <= 0) {
                    // AI ship destroyed - remove it permanently
                    if (FXL) { FXL.emitExplosion(aiShip.x, aiShip.y, GAME_THEME.aiBlue, 1.2); FXL.shake(3, 10); }
                    playExplosionSound(1.2);
                    if (typeof killFeedPush === 'function') {
                        const name = aiShip.isAlpha ? '★ Alpha' : (aiShip.role ? aiShip.role : 'Unit');
                        killFeedPush(`☄ ${name} hit by asteroid`, '#FF9944');
                    }
                    return false;
                }
                if (FXL) FXL.shake(1, 4);
            }
        }
        return true; // Keep ship if no collision
    });
    aiShips = aiShips.filter(ship => !linkedFuelDestruction.has(ship));

    // HEALING SYSTEM: Check AI ship-AI ship collisions for healing
    aiShips.forEach((phase3Ship, phase3Index) => {
        if (!phase3Ship || phase3Ship.aiPhase !== 3) return; // Only Phase 3 ships can be healed
        
        // Check if healing cooldown is active
        if (phase3Ship.healingCooldown > 0) return;
        
        aiShips.forEach((phase1Ship, phase1Index) => {
            if (!phase1Ship || phase1Ship.aiPhase !== 1) return; // Only Phase 1 ships can heal
            if (phase3Index === phase1Index) return; // Don't check self
            
            // Check if ships are in contact (within healing range)
            const dx = phase1Ship.x - phase3Ship.x;
            const dy = phase1Ship.y - phase3Ship.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < phase3Ship.healingRange) {
                // Healing contact! Phase 3 ship gains health
                const oldHealth = phase3Ship.health;
                phase3Ship.health = Math.min(phase3Ship.maxHealth, phase3Ship.health + 1);
                
                // Set healing cooldown to prevent spam
                phase3Ship.healingCooldown = phase3Ship.healingCooldownMax;
                
                // Visual feedback: Create healing particles/effect
                // (We'll add visual feedback in the draw function)
                phase3Ship.healingEffect = true;
                phase3Ship.healingEffectTimer = 30; // 0.5 seconds at 60fps
                
                // Log healing (optional, for debugging)
                if (oldHealth < phase3Ship.health) {
                    console.log(`[Healing] Phase 3 ship healed from ${oldHealth.toFixed(1)} to ${phase3Ship.health.toFixed(1)} HP`);
                }
            }
        });
    });
    
    // Update AI ship count after collision check
    updateAIShipCount();
}

// Render game
// COORDINATION: Draw visual indicators for enemy coordination
function drawEnemyCoordinationVisuals(enemyShips) {
    if (!enemyShips || enemyShips.length === 0) return;
    
    // Draw formation lines between coordinating enemies
    const coordinatingEnemies = enemyShips.filter(e => e && e.isCoordinating);
    
    if (coordinatingEnemies.length >= 2) {
        ctx.strokeStyle = GAME_THEME.enemy;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.3;
        ctx.setLineDash([3, 3]);
        
        // Draw lines between coordinating enemies
        for (let i = 0; i < coordinatingEnemies.length; i++) {
            for (let j = i + 1; j < coordinatingEnemies.length; j++) {
                const e1 = coordinatingEnemies[i];
                const e2 = coordinatingEnemies[j];
                if (e1 && e2) {
                    ctx.beginPath();
                    ctx.moveTo(e1.x, e1.y);
                    ctx.lineTo(e2.x, e2.y);
                    ctx.stroke();
                }
            }
        }
        
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
    }
    
    // Draw target lines (from enemies to their targets)
    enemyShips.forEach(enemy => {
        if (!enemy || !enemy.target) return;
        
        const target = enemy.target.ship || enemy.target;
        if (!target) return;
        
        ctx.strokeStyle = GAME_THEME.orange;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.4;
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        ctx.moveTo(enemy.x, enemy.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
    });
    
    // Draw pattern indicator text (optional, for debugging)
    if (enemyCoordinationSystem.currentPattern) {
        const leader = enemyCoordinationSystem.leader;
        if (leader) {
            ctx.fillStyle = GAME_THEME.text;
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.globalAlpha = 0.7;
            ctx.fillText(
                `Pattern: ${enemyCoordinationSystem.currentPattern.toUpperCase()}`,
                leader.x,
                leader.y - leader.size - 20
            );
            ctx.globalAlpha = 1.0;
        }
    }
}

function drawGameOverOverlay() {
    if (!gameOverState) return;
    const g = gameOverState;
    const w = canvas.width, h = canvas.height;
    const fade = Math.min(1, g.frames / 20);
    ctx.save();
    ctx.globalAlpha = fade;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.68)';
    ctx.fillRect(0, 0, w, h);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = GAME_THEME.enemy;
    ctx.font = 'bold 54px Arial';
    ctx.fillText('GAME OVER', w / 2, h / 2 - 64);
    ctx.fillStyle = GAME_THEME.primary;
    ctx.font = 'bold 26px Arial';
    ctx.fillText('Score  ' + g.score, w / 2, h / 2 - 12);
    ctx.font = '18px Arial';
    ctx.fillStyle = g.newBest ? '#FFD700' : (GAME_THEME.muted || '#D1D1D1');
    ctx.fillText(g.newBest ? 'NEW BEST!' : 'Best  ' + g.best, w / 2, h / 2 + 24);
    const secs = Math.max(0, Math.ceil((GAME_OVER_AUTO_RESTART_FRAMES - g.frames) / 60));
    ctx.font = '15px Arial';
    ctx.fillStyle = GAME_THEME.primary;
    ctx.fillText('Press R or Enter to restart  -  auto-restart in ' + secs + 's', w / 2, h / 2 + 62);
    ctx.restore();
}

function drawPauseOverlay() {
    if (!gamePaused) return;
    ctx.save();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = GAME_THEME.primary;
    ctx.font = 'bold 42px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('PAUSED', canvas.width / 2, canvas.height / 2 - 16);
    ctx.font = '16px Arial';
    ctx.fillText('Press P to resume', canvas.width / 2, canvas.height / 2 + 24);
    ctx.restore();
}

// Sphere mode: the world is rendered by Three.js (sphere_view.js) on a WebGL
// canvas; `ctx` is the transparent overlay above it and only carries
// screen-space layers (HUD, policy labels, minimap).
function drawSphereFrame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    SphereView.render({
        aiShips: aiShips,
        asteroids: asteroids,
        bullets: bullets,
        enemyBullets: enemyBullets,
        enemyShips: enemyShips,
        powerUps: powerUps,
        ship: ship,
        playerShipActive: playerShipActive,
        alphaAttackEnabled: alphaAttackEnabled,
        fuelRescueEnabled: fuelRescueEnabled,
        fx: FXL
    });
    SphereView.drawOverlay(ctx);
    drawPolicyOverlay();
    if (typeof drawActionWheels === 'function') drawActionWheels(ctx);
    if (typeof drawHUD === 'function') drawHUD(ctx);
    drawActiveBuffBadges(ctx);
    drawPauseOverlay();
    drawGameOverOverlay();
}

function draw() {
    if (sphereViewReady()) {
        drawSphereFrame();
        return;
    }

    // Clear canvas
    ctx.fillStyle = GAME_THEME.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Screen-shake transform for everything in world space.
    if (FXL) FXL.beginFrame(ctx);

    // Draw stars background
    if (FXL) {
        FXL.drawStarfield(ctx, canvas.width, canvas.height);
    } else {
        ctx.fillStyle = GAME_THEME.text;
        for (let i = 0; i < 50; i++) {
            const x = (i * 37) % canvas.width;
            const y = (i * 53) % canvas.height;
            ctx.fillRect(x, y, 1, 1);
        }
    }

    // Critic value heatmap sits under everything else.
    if (typeof drawValueHeatmap === 'function') drawValueHeatmap(ctx);

    // Draw game objects
    asteroids.forEach(asteroid => asteroid.draw());
    if (FXL) FXL.drawParticles(ctx);
    powerUps.forEach(powerUp => powerUp.draw());
    bullets.forEach(bullet => bullet.draw());
    enemyBullets.forEach(bullet => bullet.draw());
    // Draw enemy ships
    enemyShips.forEach(enemyShip => {
        if (enemyShip) {
            enemyShip.draw();
        }
    });
    
    // COORDINATION: Draw formation lines and target lines
    drawEnemyCoordinationVisuals(enemyShips);
    
    // Draw player ship (with visual indication if inactive)
    if (playerShipActive) {
        ship.draw();
    } else {
        // Draw inactive ship with reduced opacity
        ctx.save();
        ctx.globalAlpha = 0.3;
        ship.draw();
        ctx.globalAlpha = 1.0;
        ctx.restore();
    }
    // ALPHA ATTACK: Draw formation lines (before drawing ships)
    if (alphaAttackEnabled) {
        aiShips.forEach(aiShip => {
            if (aiShip && aiShip.alphaShip && aiShip.alphaShip.isAlpha) {
                const alpha = aiShip.alphaShip;
                ctx.strokeStyle = GAME_THEME.gold; // Gold color for formation lines
                ctx.lineWidth = 1;
                ctx.globalAlpha = 0.4;
                ctx.setLineDash([5, 5]); // Dashed lines
                ctx.beginPath();
                ctx.moveTo(alpha.x, alpha.y);
                ctx.lineTo(aiShip.x, aiShip.y);
                ctx.stroke();
                ctx.setLineDash([]); // Reset line dash
                ctx.globalAlpha = 1.0;
            }
        });
    }
    
    // Draw fuel-transfer tether lines before ships (world space)
    if (fuelRescueEnabled && alphaAttackEnabled) {
        aiShips.forEach(ship => {
            if (!ship || ship.role !== 'fuel' || (!ship.fuelTransferActive && !ship.fuelDocked) || !ship.fuelTransferPartner) return;
            const partner = ship.fuelTransferPartner;
            if (!aiShips.includes(partner)) return;
            if (FXL) FXL.emitDockStream(ship.x, ship.y, partner.x, partner.y, GAME_THEME.fuelTransfer);
            ctx.strokeStyle = GAME_THEME.fuelTransfer;
            ctx.lineWidth = ship.fuelDocked ? 3 : 2;
            ctx.globalAlpha = ship.fuelDocked ? 0.95 : 0.75;
            ctx.setLineDash(ship.fuelDocked ? [] : [4, 4]);
            ctx.beginPath();
            ctx.moveTo(ship.x, ship.y);
            ctx.lineTo(partner.x, partner.y);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1.0;
        });
    }

    // Draw AI ships
    aiShips.forEach((aiShip) => {
        if (aiShip) {
            aiShip.draw();
        }
    });

    drawPolicyOverlay();
    if (typeof drawActionWheels === 'function') drawActionWheels(ctx);

    if (FXL) FXL.endFrame(ctx, canvas.width, canvas.height);

    // Screen-space HUD, drawn after the shake transform is removed.
    if (typeof drawHUD === 'function') drawHUD(ctx);
    drawActiveBuffBadges(ctx);
    drawPauseOverlay();
    drawGameOverOverlay();
}

function drawActiveBuffBadges(ctx) {
    const activeBuffs = [];
    if (playerShipActive && ship.buffRailgun > 0) {
        activeBuffs.push({ label: '⚡ RAILGUN OVERCHARGE', time: (ship.buffRailgun / 60).toFixed(1) + 's', color: '#00FF66' });
    }
    if (playerShipActive && ship.buffShield > 0) {
        activeBuffs.push({ label: '🛡 OVERSHIELD BARRIER', time: (ship.buffShield / 60).toFixed(1) + 's', color: '#00E5FF' });
    }
    let fleetRailgun = 0, fleetShield = 0;
    aiShips.forEach(s => {
        if (!s) return;
        if (s.buffRailgun > fleetRailgun) fleetRailgun = s.buffRailgun;
        if (s.buffShield > fleetShield) fleetShield = s.buffShield;
    });
    if (!playerShipActive && fleetRailgun > 0) {
        activeBuffs.push({ label: '⚡ FLEET RAILGUN', time: (fleetRailgun / 60).toFixed(1) + 's', color: '#00FF66' });
    }
    if (!playerShipActive && fleetShield > 0) {
        activeBuffs.push({ label: '🛡 FLEET OVERSHIELD', time: (fleetShield / 60).toFixed(1) + 's', color: '#00E5FF' });
    }

    if (!activeBuffs.length) return;

    ctx.save();
    let bx = 20, by = canvas.height - 35;
    activeBuffs.forEach(buff => {
        ctx.fillStyle = 'rgba(10, 20, 30, 0.75)';
        ctx.strokeStyle = buff.color;
        ctx.lineWidth = 1.5;
        const w = 185, h = 24;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(bx, by - h, w, h, 4);
        else ctx.rect(bx, by - h, w, h);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = buff.color;
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${buff.label} [${buff.time}]`, bx + 8, by - h / 2);
        by -= (h + 6);
    });
    ctx.restore();
}

// Update score display
function updateScore() {
    scoreElement.textContent = score;
    if (score > bestScore) {
        bestScore = score;
        saveBestScore(bestScore);
        if (bestScoreElement) bestScoreElement.textContent = bestScore;
    }
}

// Update AI ship count display
function updateAIShipCount() {
    if (aiShipCountElement) {
        const activeCount = aiShips.filter(ship => ship !== null).length;
        aiShipCountElement.textContent = activeCount;
    }
}

// Update enemy ship count display (basic and advanced only)
function updateEnemyShipCount() {
    if (enemyShipCountElement) {
        const activeCount = enemyShips.filter(ship => ship && ship.type !== 'boss').length;
        enemyShipCountElement.textContent = activeCount;
    }
}

// Update boss ship count display
function updateBossShipCount() {
    if (bossShipCountElement) {
        const activeCount = enemyShips.filter(ship => ship && ship.type === 'boss').length;
        bossShipCountElement.textContent = activeCount;
    }
}

// ALPHA ATTACK: Set alpha attack enabled state
window.setAlphaAttackEnabled = function(enabled) {
    alphaAttackEnabled = enabled;
    // Reset alpha assignments when toggled
    if (!enabled) {
        aiShips.forEach(ship => {
            if (ship) {
                ship.isAlpha = false;
                ship.alphaShip = null;
                ship.alphaAttackTarget = null;
                ship.role = null;
                ship.roleAssigned = false;
                ship.helpPingActive = false;
                clearFuelLink(ship, ship.fuelTransferPartner);
            }
        });
    }
};

// Visual effects toggle (particles, glow, shake). Cosmetic only.
window.setFxEnabled = function(enabled) {
    if (FXL) FXL.setEnabled(enabled === true || enabled === 'true');
};

window.setFuelRescueEnabled = function(enabled) {
    fuelRescueEnabled = enabled === true || enabled === 'true';
    if (!fuelRescueEnabled) {
        aiShips.forEach(ship => {
            if (ship) {
                ship.helpPingActive = false;
                clearFuelLink(ship, ship.fuelTransferPartner);
            }
        });
    } else if (alphaAttackEnabled) {
        assignFuelShipRole(aiShips);
    }
};

// PHASE 1: Set formation type
window.setFormationType = function(type) {
    if (typeof type === 'string' && ['arrowhead', 'line', 'circle', 'diamond', 'wedge'].includes(type)) {
        formationType = type;
        // Update all ships' formation type
        if (typeof aiShips !== 'undefined' && Array.isArray(aiShips)) {
            aiShips.forEach(ship => {
                if (ship) {
                    ship.formationType = type;
                    ship.formationTypeChanged = true;
                }
            });
        }
    }
};

// PHASE 2: Set auto-assign roles
window.setAutoAssignRoles = function(enabled) {
    const next = enabled === true || enabled === 'true';
    // The Streamlit page re-sends every setting on a 2 s timer. Resetting
    // roles on each call wiped the tanker's 'fuel' role mid-refuel; the next
    // frame re-converted it with a full tank and cleared dock flags while the
    // alpha stayed docked to it, freezing both ships in FUEL LINK forever.
    if (next === autoAssignRoles) return;
    autoAssignRoles = next;
    // Reset formation roles. The tanker is not a formation role, so keep it.
    if (typeof aiShips !== 'undefined' && Array.isArray(aiShips)) {
        aiShips.forEach(ship => {
            if (ship && ship.role !== 'fuel') {
                ship.role = null;
                ship.roleAssigned = false;
            }
        });
    }
};

// PHASE 3: Set adaptive formation
window.setAdaptiveFormation = function(enabled) {
    adaptiveFormationEnabled = enabled === true || enabled === 'true';
    if (typeof aiShips !== 'undefined' && Array.isArray(aiShips)) {
        aiShips.forEach(ship => {
            if (ship && ship.isAlpha) {
                ship.adaptiveFormationEnabled = adaptiveFormationEnabled;
            }
        });
    }
};

// PHASE 3: Set multi-target mode
window.setMultiTargetMode = function(mode) {
    if (typeof mode === 'string' && ['focus', 'split', 'prioritize'].includes(mode)) {
        multiTargetMode = mode;
        if (typeof aiShips !== 'undefined' && Array.isArray(aiShips)) {
            aiShips.forEach(ship => {
                if (ship) {
                    ship.multiTargetMode = mode;
                }
            });
        }
    }
};

// PHASE 3: Set escort mode
window.setEscortMode = function(mode) {
    if (typeof mode === 'string' && ['none', 'escort', 'guard', 'patrol', 'intercept', 'cover'].includes(mode)) {
        escortMode = mode;
        if (typeof aiShips !== 'undefined' && Array.isArray(aiShips)) {
            aiShips.forEach(ship => {
                if (ship) {
                    ship.escortMode = mode;
                    if (mode === 'patrol' && !ship.patrolCenter) {
                        ship.patrolCenter = { x: ship.x, y: ship.y };
                    }
                    if (mode === 'none') {
                        ship.escortTarget = null;
                        ship.interceptTarget = null;
                    }
                }
            });
        }
    }
};

// ADVANCED PATTERNS: Set tactical attack sequences
window.setTacticalSequences = function(enabled) {
    tacticalAttackSequencesEnabled = enabled === true || enabled === 'true';
};

// ADVANCED PATTERNS: Set dynamic formation transitions
window.setFormationTransitions = function(enabled) {
    dynamicFormationTransitionsEnabled = enabled === true || enabled === 'true';
};

// ADVANCED PATTERNS: Set advanced flanking patterns
window.setAdvancedFlanking = function(enabled) {
    advancedFlankingEnabled = enabled === true || enabled === 'true';
    if (typeof aiShips !== 'undefined' && Array.isArray(aiShips)) {
        aiShips.forEach(ship => {
            if (ship && !enabled) {
                // Reset flanking when disabled
                ship.flankPattern = 'none';
                ship.flankPosition = null;
            }
        });
    }
};

// Game loop
function gameLoop() {
    update();
    draw();
    requestAnimationFrame(gameLoop);
}

// Start game
initGame();

// Set initial player ship state - default to active
playerShipActive = true;

// Ensure state is set before game loop starts
if (typeof setPlayerShipActive === 'function') {
    setPlayerShipActive(true);
}

gameLoop();
})(typeof globalThis !== 'undefined' ? globalThis : this);
