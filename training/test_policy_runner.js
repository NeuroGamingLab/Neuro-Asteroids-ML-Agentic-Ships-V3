// End-to-end test of the browser inference path.
//
// This loads the same files, in the same order, that app.py injects into the
// Streamlit component, then drives the real update() loop with a real
// checkpoint. It exists because the browser path is where train/serve
// divergence would appear, and a mismatch there is silent: the ships still fly,
// they just fly on meaningless activations.

load('sim/browser_shim.js');
load('rng.js');
load('theme.js');
load('ml_features.js');
load('entities.js');
load('ai_ship.js');
load('enemy_ship.js');
load('enemy_coordination.js');
load('game.js');
load('training/nn.js');
load('policy_runner.js');

let failures = 0;
function check(name, condition, detail) {
    if (condition) {
        print(`  ok    ${name}`);
    } else {
        failures++;
        print(`  FAIL  ${name}${detail ? '  -> ' + detail : ''}`);
    }
}

const CHECKPOINT = 'models/policy_stage0.json';

print('\npolicy_runner end-to-end\n');

// --- loading -------------------------------------------------------------
let payload;
try {
    payload = JSON.parse(read(CHECKPOINT));
} catch (e) {
    print(`  SKIP  no checkpoint at ${CHECKPOINT}; train one first`);
    quit(0);
}

const runner = initializePolicyRunner();
runner.loadFromObject(payload);
check('checkpoint loads', runner.actor !== null);
check('critic loads', runner.critic !== null);
check('observation width matches this build', payload.obsDim === RL_FEATURE_DIM,
    `checkpoint ${payload.obsDim} vs build ${RL_FEATURE_DIM}`);
check('action count matches this build', payload.actionSize === ACTION_SPACE_SIZE,
    `checkpoint ${payload.actionSize} vs build ${ACTION_SPACE_SIZE}`);

// --- shape mismatches must be refused, not tolerated ---------------------
function refuses(name, mutate) {
    const broken = JSON.parse(JSON.stringify(payload));
    mutate(broken);
    let threw = false;
    try {
        new PolicyRunner().loadFromObject(broken);
    } catch (e) {
        threw = true;
    }
    check(name, threw);
}
refuses('refuses wrong observation width', p => { p.obsDim = 30; });
refuses('refuses wrong action count', p => { p.actionSize = 24; });
refuses('refuses reordered features', p => { p.featureNames = p.featureNames.slice().reverse(); });
refuses('refuses a foreign file', p => { p.format = 'something-else'; });

// --- the browser must see exactly what the trainer saw --------------------
// This is the whole justification for reusing game.js and nn.js instead of
// exporting to another format, so it is worth asserting rather than assuming.
// The bug this catches: the runner previously read the browser's
// marlCurriculumStage, so a stage-0 policy was served an observation whose
// curriculumStageNorm said stage 3, with no visible symptom.
(function testObservationsMatchTraining() {
    load('sim/env.js');

    const trainedStage = payload.meta && payload.meta.stage !== undefined ? payload.meta.stage : 0;
    const env = new AsteroidsEnv({ numAgents: 3, maxSteps: 100, curriculumStage: trainedStage });
    env.reset('obs-parity');
    env.step([0, 0, 0]);

    const runner2 = new PolicyRunner();
    runner2.loadFromObject(payload);
    check('runner adopts the stage the checkpoint was trained on',
        runner2.curriculumStage === trainedStage,
        `runner ${runner2.curriculumStage} vs checkpoint ${trainedStage}`);

    // Same live game state, observation built the trainer's way and the
    // browser's way.
    let mismatches = 0;
    let worst = 0;
    let worstFeature = '';
    for (let i = 0; i < aiShips.length; i++) {
        const shipRef = aiShips[i];
        if (!shipRef) continue;
        const fromEnv = MLFeatureExtractor.extractRL(
            shipRef, asteroids, aiShips, ship, enemyShips, enemyBullets,
            { curriculumStage: env.curriculumStage })._vector;
        const fromRunner = MLFeatureExtractor.extractRL(
            shipRef, asteroids, aiShips, ship, enemyShips, enemyBullets,
            { curriculumStage: runner2.curriculumStage })._vector;
        for (let f = 0; f < fromEnv.length; f++) {
            const delta = Math.abs(fromEnv[f] - fromRunner[f]);
            if (delta > 1e-12) {
                mismatches++;
                if (delta > worst) { worst = delta; worstFeature = RL_FEATURE_NAMES[f]; }
            }
        }
    }
    check('browser observation is identical to the training observation',
        mismatches === 0,
        `${mismatches} differing value(s), worst ${worstFeature} by ${worst}`);

    // And confirm the check has teeth: a deliberately wrong stage must differ.
    if (trainedStage !== 3) {
        const wrong = MLFeatureExtractor.extractRL(
            aiShips[0], asteroids, aiShips, ship, enemyShips, enemyBullets,
            { curriculumStage: 3 })._vector;
        const right = MLFeatureExtractor.extractRL(
            aiShips[0], asteroids, aiShips, ship, enemyShips, enemyBullets,
            { curriculumStage: trainedStage })._vector;
        let differs = false;
        for (let f = 0; f < right.length; f++) if (right[f] !== wrong[f]) differs = true;
        check('a wrong curriculum stage would have been detected', differs);
    }
})();

// --- driving the real game loop ------------------------------------------
setNumAIShips(3);
initGame(4242);
runner.setEnabled(true);
runner.setDeterministic(false);
check('runner reports enabled', runner.enabled);

const before = aiShips.map(s => ({ x: s.x, y: s.y, angle: s.angle }));
let framesWithDecisions = 0;
const actionsSeen = {};

for (let frame = 0; frame < 600; frame++) {
    policyRunnerTick();
    update();
    if (runner.lastDecisions.length) {
        framesWithDecisions++;
        runner.lastDecisions.forEach(d => {
            if (d) actionsSeen[d.action] = (actionsSeen[d.action] || 0) + 1;
        });
    }
}

check('produced a decision every frame', framesWithDecisions === 600,
    `${framesWithDecisions}/600`);
check('policy control was handed to the game', policyControlEnabled);

const moved = aiShips.some((s, i) =>
    before[i] && (s.x !== before[i].x || s.y !== before[i].y || s.angle !== before[i].angle));
check('ships actually moved', moved);

const distinct = Object.keys(actionsSeen).length;
check('explores more than one action', distinct > 1, `${distinct} distinct actions`);
check('scored points', score > 0, `score ${score}`);

// --- the overlay needs every field it draws ------------------------------
const decision = runner.lastDecisions.find(d => d);
check('decision carries a decoded action', decision && decision.decoded &&
    typeof decision.decoded.rotation === 'number');
check('decision carries a confidence', decision && decision.confidence > 0 &&
    decision.confidence <= 1, decision && String(decision.confidence));
check('decision carries entropy', decision && isFinite(decision.entropy));
check('decision carries a value estimate', decision && decision.value !== null &&
    isFinite(decision.value));
check('decision carries top-k actions', decision && decision.topActions.length === 3);
check('top-k is sorted by probability', decision &&
    decision.topActions[0].p >= decision.topActions[1].p &&
    decision.topActions[1].p >= decision.topActions[2].p);
check('describes itself without throwing',
    typeof describePolicyDecision(decision) === 'string' &&
    describePolicyDecision(decision).length > 0);
check('decision carries the full action distribution', decision && decision.probs &&
    decision.probs.length === ACTION_SPACE_SIZE &&
    Math.abs(decision.probs.reduce((a, b) => a + b, 0) - 1) < 1e-6);

// --- overlays: heatmap + action wheels ------------------------------------
load('policy_viz.js');
load('hud.js');
setValueHeatmap(true);
setActionWheels(true);
setHudEnabled(true);
let overlaysClean = true;
try {
    for (let f = 0; f < 20; f++) { policyRunnerTick(); update(); draw(); }
} catch (e) {
    overlaysClean = false;
    print(`        overlays threw: ${e}`);
}
check('heatmap, wheels and HUD render through the stub without throwing', overlaysClean);
check('heatmap completed a full refresh', getPolicyVizState().refreshed);
check('heatmap restores the focus ship position', (function () {
    const s = aiShips.find(x => x);
    const x0 = s.x, y0 = s.y;
    drawValueHeatmap(canvas.getContext('2d'));
    return s.x === x0 && s.y === y0;
})());
setValueHeatmap(false);
setActionWheels(false);
setHudEnabled(false);

// --- worker protocol (driven synchronously, no real Worker in jsc) --------
(function testWorkerHandler() {
    load('policy_worker.js');
    const sent = [];
    const savedPost = globalThis.postMessage;
    globalThis.postMessage = m => sent.push(m);
    try {
        __policyWorkerHandle({ type: 'load', actor: payload.actor, critic: payload.critic, seed: 'test' });
        const { agentIds, obs } = runner.observe();
        __policyWorkerHandle({ type: 'infer', frame: 1, deterministic: true, agentIds, obs });
    } finally {
        if (savedPost === undefined) delete globalThis.postMessage; else globalThis.postMessage = savedPost;
    }
    check('worker acknowledges the load', sent[0] && sent[0].type === 'ready', JSON.stringify(sent[0]));
    const result = sent[1];
    check('worker returns one decision per agent', result && result.decisions.length === runner.observe().obs.length);
    if (result) {
        // Same weights, same observation, greedy: worker and main thread must agree.
        const main = runner.observe();
        let agree = true;
        for (let k = 0; k < result.decisions.length; k++) {
            const logits = runner.actor.forward(main.obs[k]);
            const probs = NN.logSoftmax(logits).probs;
            let best = 0;
            for (let a = 1; a < probs.length; a++) if (probs[a] > probs[best]) best = a;
            if (best !== result.decisions[k].action) agree = false;
        }
        check('worker greedy actions match main-thread inference', agree);
        // Feed the result back through the runner as the browser would.
        runner.onWorkerMessage(result);
        check('runner adopts worker decisions', runner.lastDecisions.some(d => d && d.probs));
    }
})();

// --- fuel-feature checkpoints are accepted, other layouts refused --------
(function testFuelLayout() {
    const fuelPayload = JSON.parse(JSON.stringify(payload));
    fuelPayload.obsDim = RL_FEATURE_DIM_FUEL;
    fuelPayload.featureNames = RL_FEATURE_NAMES_FUEL.slice();
    // Widen the first layer so the network accepts the extra inputs. W is
    // row-major [inDim x outDim], so extra inputs are extra zero rows at the end.
    const extra = RL_FEATURE_DIM_FUEL - RL_FEATURE_DIM;
    function widen(net) {
        if (!net) return;
        const first = net.layers[0];
        for (let k = 0; k < extra * first.outDim; k++) first.W.push(0);
        first.inDim = RL_FEATURE_DIM_FUEL;
        net.spec.inDim = RL_FEATURE_DIM_FUEL;
    }
    widen(fuelPayload.actor);
    widen(fuelPayload.critic);
    let loaded = null;
    try {
        loaded = new PolicyRunner().loadFromObject(fuelPayload);
    } catch (e) {
        print(`        fuel layout refused: ${e.message}`);
    }
    check('a fuel-feature checkpoint loads', loaded !== null);
    check('and selects the fuel extractor', loaded && loaded.featureOptions.fuelFeatures === true);
})();

// The overlay draws through the canvas stub; this catches a missing ctx method.
setPolicyOverlay(true);
let drewCleanly = true;
try {
    draw();
} catch (e) {
    drewCleanly = false;
    print(`        overlay threw: ${e}`);
}
check('overlay renders without throwing', drewCleanly);

// --- determinism of greedy selection -------------------------------------
runner.setDeterministic(true);
initGame(99);
policyRunnerTick();
const firstPass = runner.lastDecisions.map(d => d && d.action);
initGame(99);
policyRunnerTick();
const secondPass = runner.lastDecisions.map(d => d && d.action);
check('greedy selection is reproducible',
    JSON.stringify(firstPass) === JSON.stringify(secondPass),
    `${JSON.stringify(firstPass)} vs ${JSON.stringify(secondPass)}`);

// --- disabling gives the ships back --------------------------------------
runner.setEnabled(false);
check('disabling releases policy control', !policyControlEnabled);
check('disabling clears stale decisions', runner.lastDecisions.length === 0);

print(`\n${failures === 0 ? 'PASS' : 'FAIL'}  ${failures} failure(s)\n`);
quit(failures === 0 ? 0 : 1);
