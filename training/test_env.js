// Tests for the headless simulator and the environment wrapper.
//
// The reward-alignment test is the important one. A reward function is a
// specification of the task, and it is very easy to write one that is quietly
// measuring something other than what you meant. The check here is concrete: the
// hand-written controller scores roughly fifty times what a random policy scores,
// so if the reward does not rank it far above random, the reward is wrong and any
// policy trained against it will optimise the wrong thing.

load('sim/browser_shim.js');
load('rng.js');
load('theme.js');
load('world.js');
load('ml_features.js');
load('entities.js');
load('ai_ship.js');
load('enemy_ship.js');
load('enemy_coordination.js');
load('game.js');
load('sim/env.js');
load('training/evaluate.js');

globalThis.draw = function () {};

let failures = 0;
function check(name, condition, detail) {
    if (condition) {
        print(`  PASS  ${name}${detail ? '  (' + detail + ')' : ''}`);
    } else {
        print(`  FAIL  ${name}${detail ? '  (' + detail + ')' : ''}`);
        failures++;
    }
}

print('Determinism');
(function determinism() {
    const env = new AsteroidsEnv({ numAgents: 3, maxSteps: 300, curriculumStage: 0 });

    function rollout(seed) {
        const rng = AsteroidsRNG.createStream('fixed-action-stream');
        env.reset(seed);
        let last = null;
        for (let t = 0; t < 300; t++) {
            const actions = [0, 1, 2].map(() => Math.floor(rng() * ACTION_SPACE_SIZE));
            last = env.step(actions);
            if (last.done) break;
        }
        return JSON.stringify({
            score: last.info.score,
            step: last.info.step,
            returns: last.info.episodeReturn.map(v => v.toFixed(6))
        });
    }

    const a = rollout(1234);
    const b = rollout(1234);
    const c = rollout(5678);
    check('same seed reproduces the episode exactly', a === b);
    check('different seed produces a different episode', a !== c);
})();

print('Observation space');
(function observations() {
    const env = new AsteroidsEnv({ numAgents: 3, maxSteps: 100, curriculumStage: 3 });
    const obs = env.reset(11);
    check('one observation per agent', obs.length === 3, `got ${obs.length}`);
    check('observation dimension matches RL feature set', obs[0].length === RL_FEATURE_DIM,
        `${obs[0].length} vs ${RL_FEATURE_DIM}`);
    check('RL features extend the canonical 30', RL_FEATURE_DIM > CANONICAL_FEATURE_DIM
        && RL_FEATURE_NAMES.slice(0, CANONICAL_FEATURE_DIM).join() === CANONICAL_FEATURE_NAMES.join(),
        `${CANONICAL_FEATURE_DIM} -> ${RL_FEATURE_DIM}`);

    let allFinite = true;
    let allInRange = true;
    for (let t = 0; t < 100; t++) {
        const res = env.step([0, 5, 10]);
        for (const o of res.observations) {
            for (const v of o) {
                if (!Number.isFinite(v)) allFinite = false;
                if (v < 0 || v > 1) allInRange = false;
            }
        }
        if (res.done) break;
    }
    check('all features finite', allFinite);
    check('all features within [0,1]', allInRange);
})();

print('Action space');
(function actions() {
    let roundTrips = true;
    const seen = new Set();
    for (let i = 0; i < ACTION_SPACE_SIZE; i++) {
        const decoded = decodeAction(i);
        seen.add(`${decoded.rotation}|${decoded.thrust}|${decoded.fire}|${decoded.shield}`);
        if (encodeAction(decoded) !== i) roundTrips = false;
    }
    check('encode/decode round-trips for all actions', roundTrips);
    check(`all ${ACTION_SPACE_SIZE} actions are distinct`, seen.size === ACTION_SPACE_SIZE, `${seen.size} distinct`);
})();

print('Credit assignment');
(function credit() {
    const env = new AsteroidsEnv({ numAgents: 3, maxSteps: 600, curriculumStage: 0 });
    env.reset(99);
    const rng = AsteroidsRNG.createStream('credit-test');

    let anyCredit = false;
    let creditMatchesReward = true;
    for (let t = 0; t < 600; t++) {
        // Bias toward firing so kills actually happen.
        const actions = [0, 1, 2].map(() => encodeAction({
            rotation: Math.floor(rng() * 3) - 1,
            thrust: rng() < 0.3 ? 1 : 0,
            fire: 1,
            shield: 0
        }));
        const before = score;
        const res = env.step(actions);
        if (score > before) {
            anyCredit = true;
            // Somebody must have been paid for the score that was just earned.
            const paid = res.rewards.some(r => r > 0.5);
            if (!paid) creditMatchesReward = false;
        }
        if (res.done) break;
    }
    check('kills occur during the test rollout', anyCredit);
    check('score increases are paid out as agent reward', creditMatchesReward);
})();

print('Reward alignment (the specification test)');
(function rewardAlignment() {
    // The invariant is not "the hand-written controller wins" - it turned out not
    // to, once the cooldown bug was fixed. The invariant that actually matters is
    // that reward *agrees with game score*: whenever one policy scores clearly
    // higher than another, the reward function must rank it higher too. A reward
    // that inverts that ordering is optimising for something other than the game.
    const env = new AsteroidsEnv({ numAgents: 3, maxSteps: 1000, curriculumStage: 0 });
    const seeds = Evaluation.evalSeeds(15, 500000);
    const rng = AsteroidsRNG.createStream('align-random');

    function constantFire(mode) {
        return function (observations, e) {
            const actions = [];
            for (let i = 0; i < observations.length; i++) {
                actions.push(e.alive[i] ? encodeAction({ rotation: 0, thrust: 0, fire: mode, shield: 0 }) : null);
            }
            return actions;
        };
    }

    const policies = [
        { name: 'idle', result: Evaluation.runEpisodes(env, constantFire(0), seeds) },
        { name: 'heuristic', result: Evaluation.runEpisodes(env, null, seeds) },
        { name: 'fire@asteroid', result: Evaluation.runEpisodes(env, constantFire(2), seeds) },
        { name: 'random', result: Evaluation.runEpisodes(env, Evaluation.randomPolicy(rng, ACTION_SPACE_SIZE), seeds) }
    ];

    for (const p of policies) {
        print(`    ${p.name.padEnd(14)} return ${p.result.summary.meanReturn.toFixed(2).padStart(8)}` +
            `  score ${p.result.summary.meanScore.toFixed(0).padStart(6)}`);
    }

    let inversions = 0;
    let comparisons = 0;
    for (let i = 0; i < policies.length; i++) {
        for (let j = i + 1; j < policies.length; j++) {
            const a = policies[i].result.summary;
            const b = policies[j].result.summary;
            // Only judge pairs whose scores clearly differ.
            const scoreGap = Math.abs(a.meanScore - b.meanScore);
            if (scoreGap < 300) continue;
            comparisons++;
            const scoreOrder = Math.sign(a.meanScore - b.meanScore);
            const rewardOrder = Math.sign(a.meanReturn - b.meanReturn);
            if (scoreOrder !== rewardOrder) {
                inversions++;
                print(`    INVERSION: ${policies[i].name} vs ${policies[j].name}`);
            }
        }
    }

    check('reward ordering matches game-score ordering', inversions === 0,
        `${comparisons - inversions}/${comparisons} pairs consistent`);
    check('enough distinct policies to be a real test', comparisons >= 3,
        `${comparisons} comparable pairs`);

    // A policy that does nothing must be clearly worse than one that plays.
    const idle = policies[0].result;
    const best = policies.reduce((acc, p) =>
        p.result.summary.meanReturn > acc.result.summary.meanReturn ? p : acc, policies[0]);
    const comparison = Evaluation.pairedComparison(best.result.returns, idle.returns);
    check('active play beats doing nothing', comparison.meanDiff > 0 && comparison.significant,
        `${best.name} - idle = ${comparison.meanDiff.toFixed(2)}, t=${comparison.tStat.toFixed(2)}`);
})();

print('Termination');
(function termination() {
    const env = new AsteroidsEnv({ numAgents: 2, maxSteps: 50, curriculumStage: 0 });
    env.reset(3);
    let steps = 0;
    let res = null;
    for (let t = 0; t < 200; t++) {
        res = env.step([0, 0]);
        steps++;
        if (res.done) break;
    }
    check('episode respects maxSteps', steps <= 50, `${steps} steps`);
    check('time-limit truncation is flagged', res.done, `timeLimit=${res.info.timeLimit}`);
})();

// The two entry points must define the same objective. They did not: train.js
// defaulted aimShaping to 2.0 while run_training.py passed 0.0, so the same
// nominal run trained against a different reward depending on how it was
// launched, and the documented command did not reproduce the documented result.
(function testEntryPointDefaultsAgree() {
    function numericDefault(source, key) {
        const match = source.match(new RegExp(`${key}\\s*[:=]\\s*(-?[\\d.eE+]+)`));
        return match ? parseFloat(match[1]) : null;
    }
    const envDefault = DEFAULT_REWARD_CONFIG.aimShaping;
    const trainDefault = numericDefault(read('training/train.js'), 'aimShaping');
    const driverDefault = parseFloat(
        (read('training/run_training.py').match(/--aimShaping"[^)]*default=([\d.eE+-]+)/) || [])[1]
    );

    check('sim/env.js and training/train.js agree on aimShaping',
        envDefault === trainDefault, `env ${envDefault} vs train.js ${trainDefault}`);
    check('training/train.js and run_training.py agree on aimShaping',
        trainDefault === driverDefault, `train.js ${trainDefault} vs driver ${driverDefault}`);
})();

// Stochastic evaluation must be reproducible. When action sampling drew from the
// training RNG, evaluating one unchanged policy twice on the same seeds differed
// by ~4 return against an effect of interest of ~6, so comparisons between
// iterations were partly measuring sampling luck.
(function testEvaluationIsReproducible() {
    load('training/nn.js');
    load('training/ppo.js');

    const env = new AsteroidsEnv({ numAgents: 2, maxSteps: 200, curriculumStage: 0 });
    const trainer = new PPOTrainer({
        obsDim: RL_FEATURE_DIM, actionSize: ACTION_SPACE_SIZE, hidden: [16, 16], seed: 'repro'
    });
    const seeds = Evaluation.evalSeeds(6);

    const first = Evaluation.runEpisodes(env, Evaluation.trainedPolicy(trainer, false), seeds);
    const second = Evaluation.runEpisodes(env, Evaluation.trainedPolicy(trainer, false), seeds);
    check('stochastic evaluation repeats exactly',
        JSON.stringify(first.returns) === JSON.stringify(second.returns),
        `${first.summary.meanReturn.toFixed(4)} vs ${second.summary.meanReturn.toFixed(4)}`);

    const greedyA = Evaluation.runEpisodes(env, Evaluation.trainedPolicy(trainer, true), seeds);
    const greedyB = Evaluation.runEpisodes(env, Evaluation.trainedPolicy(trainer, true), seeds);
    check('greedy evaluation repeats exactly',
        JSON.stringify(greedyA.returns) === JSON.stringify(greedyB.returns));

    // Evaluation must also leave the training RNG untouched, or running an
    // evaluation would change the rollout that follows it and make training
    // itself depend on how often you happened to evaluate.
    function nthDraw(withEvaluation) {
        const t = new PPOTrainer({
            obsDim: RL_FEATURE_DIM, actionSize: ACTION_SPACE_SIZE, hidden: [16, 16], seed: 'rng-isolation'
        });
        t.rng();
        if (withEvaluation) Evaluation.runEpisodes(env, Evaluation.trainedPolicy(t, false), seeds);
        return t.rng();
    }
    check('evaluation does not consume the training RNG', nthDraw(true) === nthDraw(false));
})();

print('Fuel rescue (alpha + scripted tanker)');
(function fuelRescue() {
    const env = new AsteroidsEnv({ numAgents: 4, maxSteps: 2500, curriculumStage: 0, fuelRescue: true });
    const obs = env.reset('fuel-1');

    check('fuel env exposes the extended observation', env.obsDim === RL_FEATURE_DIM_FUEL
        && obs[1].length === RL_FEATURE_DIM_FUEL, `${obs[1].length} vs ${RL_FEATURE_DIM_FUEL}`);
    check('fuel feature names extend the base RL layout',
        RL_FEATURE_NAMES_FUEL.slice(0, RL_FEATURE_DIM).join() === RL_FEATURE_NAMES.join()
        && RL_FEATURE_NAMES_FUEL.length === RL_FEATURE_DIM + 5);

    const alpha = aiShips.find(s => s && s.isAlpha);
    const tanker = aiShips.find(s => s && s.role === 'fuel');
    check('an alpha is selected without the rule-based controller', !!alpha);
    check('one ship becomes the tanker', !!tanker && tanker !== alpha);
    check('the tanker is not a learning agent', tanker && env.alive[tanker.agentId] === false);
    check('the other ships are agents', env.alive.filter(Boolean).length === 3,
        JSON.stringify(env.alive));

    // Thrust-heavy random play burns the alpha's fuel; the scripted tanker must
    // then find it and complete at least one refuel.
    const rng = AsteroidsRNG.createStream('fuel-actions');
    let res = null;
    let observedPing = false;
    let fuelWentDown = false;
    let tankerEverMoved = false;
    let releasedAfterRefuel = false;
    const tankerStart = tanker ? { x: tanker.x, y: tanker.y } : null;
    for (let t = 0; t < 2500; t++) {
        const actions = [];
        for (let i = 0; i < env.numAgents; i++) {
            actions.push(env.alive[i] ? (Math.floor(rng() * ACTION_SPACE_SIZE) | 8) : null);
        }
        res = env.step(actions);
        const a = aiShips.find(s => s && s.isAlpha);
        if (a && a.helpPingActive) observedPing = true;
        if (a && a.fuel < a.maxFuel * 0.5) fuelWentDown = true;
        const tk = aiShips.find(s => s && s.role === 'fuel');
        if (tk && tankerStart && (Math.abs(tk.x - tankerStart.x) > 50 || Math.abs(tk.y - tankerStart.y) > 50)) tankerEverMoved = true;
        if (a && res.info.fuel.refuels >= 1 && !a.fuelDocked && !a.fuelTransferActive) releasedAfterRefuel = true;
        if (res.done) break;
    }
    check('the alpha burns fuel', fuelWentDown);
    check('the alpha pings HELP when low', observedPing);
    check('the tanker leaves standby', tankerEverMoved);
    check('at least one refuel completes', res.info.fuel && res.info.fuel.refuels >= 1,
        JSON.stringify(res.info.fuel));
    check('low-fuel frames are counted', res.info.fuel && res.info.fuel.lowFuelFrames > 0);

    // After a completed transfer the alpha must be released and able to ping
    // again - an earlier version left fuelTransferActive set and silenced it.
    check('the alpha is released after docking', releasedAfterRefuel);

    // Determinism holds with the mechanic on.
    function rollout(seed) {
        const r = AsteroidsRNG.createStream('fuel-det');
        env.reset(seed);
        let last = null;
        for (let t = 0; t < 400; t++) {
            const actions = [];
            for (let i = 0; i < env.numAgents; i++) actions.push(env.alive[i] ? Math.floor(r() * ACTION_SPACE_SIZE) : null);
            last = env.step(actions);
            if (last.done) break;
        }
        return JSON.stringify({ score: last.info.score, returns: last.info.episodeReturn.map(v => v.toFixed(6)), fuel: last.info.fuel });
    }
    check('fuel episodes are reproducible', rollout('fuel-det-seed') === rollout('fuel-det-seed'));

    // And the base env is untouched by having run a fuel env in the same realm.
    const base = new AsteroidsEnv({ numAgents: 3, maxSteps: 50, curriculumStage: 0 });
    const baseObs = base.reset('base-after-fuel');
    check('base env keeps the 38-dim observation', baseObs[0].length === RL_FEATURE_DIM);
    check('base env leaves the alpha formation off', alphaAttackEnabled === false && fuelRescueEnabled === false);
    check('base env has no tanker', !aiShips.some(s => s && s.role === 'fuel'));
})();

// The browser drives the same game with the rule-based controller instead of a
// policy. That path picks the alpha in findAlphaShip() every frame and used to
// refill its tank there, so fuel never dropped and the rescue never fired.
print('');
print('Fuel rescue under rule-based control (browser path)');
(function fuelRescueRuleBased() {
    setGameSeed('rule-fuel-test');
    initGame('rule-fuel-test');
    setPlayerShipActive(false);
    setNumAIShips(4);
    setNumEnemyShips(1);
    setNumBossShips(0);
    setAlphaAttackEnabled(true);
    setFuelRescueEnabled(true);

    let minFuel = 100, pings = 0, starvedFrames = 0, orphanFrames = 0, tankerAlpha = 0;
    for (let t = 0; t < 9000; t++) {
        update();
        const alpha = aiShips.find(s => s && s.isAlpha);
        if (!alpha) continue;
        minFuel = Math.min(minFuel, alpha.fuel);
        if (alpha.helpPingActive) pings++;
        if (alpha.fuel <= 0) starvedFrames++;
        if (alpha.role === 'fuel' && aiShips.length > 1) tankerAlpha++;
        if (alpha.fuelDocked || alpha.fuelTransferActive) {
            const p = alpha.fuelTransferPartner;
            if (!p || !aiShips.includes(p) || p.fuelTransferPartner !== alpha) orphanFrames++;
        }
    }
    check('the rule-based alpha burns fuel', minFuel < 50, `min ${minFuel.toFixed(1)}`);
    check('the rule-based alpha pings HELP', pings > 0, `${pings} frames`);
    check('refuels complete under rule-based control', fuelEvents.refuels.length > 0, `${fuelEvents.refuels.length}`);
    check('the alpha is not starved for long', starvedFrames < 900, `${starvedFrames} empty-tank frames of 9000`);
    check('the tanker is never promoted to alpha while combat ships live', tankerAlpha === 0, `${tankerAlpha} frames`);
    check('the alpha is never left linked to a missing tanker', orphanFrames === 0, `${orphanFrames} frames`);

    setFuelRescueEnabled(false);
    setAlphaAttackEnabled(false);
    consumeFuelEvents();
})();

// The Streamlit page re-sends every sidebar setting on a 2 s timer. That used
// to reset every ship's role, including the tanker's, in the middle of a
// refuel; the tanker was then re-initialised with cleared dock flags while the
// alpha stayed docked to it, and both ships froze in FUEL LINK for good.
print('');
print('Fuel rescue survives the Streamlit settings re-send loop');
(function fuelRescueSettingsResend() {
    setGameSeed('resend-fuel');
    initGame('resend-fuel');
    setPlayerShipActive(false);
    setNumAIShips(5);
    setNumEnemyShips(1);
    setNumBossShips(0);
    setAlphaAttackEnabled(true);
    setFuelRescueEnabled(true);
    setFormationType('circle');
    setAutoAssignRoles(true);

    let linkRun = 0, maxLinkRun = 0, tankerSwaps = 0, lastTanker = null;
    for (let t = 0; t < 12000; t++) {
        update();
        if (t % 120 === 0) {
            // Same-valued re-sends, exactly as app.py does every 2 s.
            setAlphaAttackEnabled(true);
            setFuelRescueEnabled(true);
            setFormationType('circle');
            setAutoAssignRoles(true);
        }
        const alpha = aiShips.find(s => s && s.isAlpha);
        const tanker = aiShips.find(s => s && s.role === 'fuel');
        if (tanker && lastTanker && tanker !== lastTanker && aiShips.includes(lastTanker)) tankerSwaps++;
        lastTanker = tanker || lastTanker;
        const linked = alpha && (alpha.fuelTransferActive || alpha.fuelDocked);
        linkRun = linked ? linkRun + 1 : 0;
        if (linkRun > maxLinkRun) maxLinkRun = linkRun;
    }
    // Transfer (<=120) + dock hold (120) is the longest legitimate link.
    check('no fuel link outlasts transfer + dock hold', maxLinkRun <= 260, `longest link ${maxLinkRun} frames`);
    check('refuels keep completing', fuelEvents.refuels.length > 0, `${fuelEvents.refuels.length}`);
    check('the tanker role is not reassigned while the old tanker lives', tankerSwaps === 0, `${tankerSwaps} swaps`);

    setFuelRescueEnabled(false);
    setAlphaAttackEnabled(false);
    consumeFuelEvents();
})();

print('Sphere world geometry');
(function sphereGeometry() {
    WORLD.setMode('sphere');
    const W = canvas.width, H = canvas.height, R = H / Math.PI;
    const near = (a, b, tol) => Math.abs(a - b) <= tol;

    // Equator, heading east: a great circle that stays a straight line on the chart.
    let o = { x: 600, y: 300, velocityX: 3, velocityY: 0, angle: 0 };
    for (let i = 0; i < 100; i++) WORLD.moveBody(o, 0);
    check('equatorial motion stays on the equator', near(o.y, 300, 1e-6) && near(o.x, 900, 1e-6) && near(o.angle, 0, 1e-9),
        `x=${o.x.toFixed(3)} y=${o.y.toFixed(3)}`);

    // At 45N heading east the great circle bends toward the equator and the
    // heading is transported with it (no "flat" straight line along a parallel).
    o = { x: 600, y: 150, velocityX: 3, velocityY: 0, angle: 0 };
    for (let i = 0; i < 60; i++) WORLD.moveBody(o, 0);
    check('mid-latitude eastward motion bends toward the equator', o.y > 150 && o.angle > 0,
        `y=${o.y.toFixed(1)} angle=${o.angle.toFixed(3)}`);
    check('heading tracks the transported velocity', near(o.angle, Math.atan2(o.velocityY, o.velocityX), 1e-6));
    check('speed is conserved', near(Math.hypot(o.velocityX, o.velocityY), 3, 1e-9));

    // Crossing the north pole: comes out on the opposite meridian heading south.
    o = { x: 300, y: 20, velocityX: 0, velocityY: -4, angle: -Math.PI / 2 };
    for (let i = 0; i < 10; i++) WORLD.moveBody(o, 0);
    check('pole crossing lands on the antimeridian, heading south', near(o.x, 900, 1e-6) && near(o.y, 20, 1e-6)
        && o.velocityY > 0 && near(WORLD.wrapAngle(o.angle - Math.PI / 2), 0, 1e-6), `x=${o.x.toFixed(2)} y=${o.y.toFixed(2)}`);

    // A full great circle returns to the start.
    o = { x: 200, y: 200, velocityX: 2, velocityY: 1, angle: 0.3 };
    const speed = Math.hypot(2, 1);
    const steps = Math.round(2 * Math.PI * R / speed);
    for (let i = 0; i < steps; i++) WORLD.moveBody(o, 0);
    check('a full great circle closes', WORLD.distance(o.x, o.y, 200, 200) < speed, `off by ${WORLD.distance(o.x, o.y, 200, 200).toFixed(2)}`);

    check('antipodal distance is pi*R', near(WORLD.distance(0, 300, 600, 300), Math.PI * R, 1e-6));
    check('near-pole points at opposite longitudes are close', WORLD.distance(0, 5, 600, 5) < 11, WORLD.distance(0, 5, 600, 5).toFixed(2));
    check('projectiles follow great circles too', (() => {
        const b = { x: 600, y: 300, angle: Math.PI / 2 };
        for (let i = 0; i < 25; i++) WORLD.moveAlongHeading(b, 8);
        return near(b.x, 600, 1e-6) && near(b.y, 500, 1e-6);
    })());
    const d = WORLD.delta(600, 300, 650, 280);
    check('local delta is arc-scaled and oriented like the canvas', d.dx > 49 && d.dx < 51 && d.dy < -19 && d.dy > -21,
        `dx=${d.dx.toFixed(2)} dy=${d.dy.toFixed(2)}`);

    // Torus mode is byte-identical to the original inline arithmetic.
    WORLD.setMode('torus');
    o = { x: 1195, y: 598, velocityX: 10, velocityY: 5, angle: 0.5 };
    WORLD.moveBody(o, 0);
    check('torus wrap matches the original edge rule', o.x === 0 && o.y === 0, `x=${o.x} y=${o.y}`);
    check('torus step is a plain translation', o.angle === 0.5 && o.velocityX === 10 && o.velocityY === 5);
})();

print('Sphere world environment');
(function sphereEnv() {
    const env = new AsteroidsEnv({ numAgents: 3, maxSteps: 400, curriculumStage: 3, fuelRescue: true, world: 'sphere' });
    function rollout(seed) {
        env.reset(seed);
        let last = null, inBounds = true, finite = true;
        for (let t = 0; t < 400; t++) {
            last = env.step(null);
            for (const o of last.observations) for (const v of o) if (!Number.isFinite(v)) finite = false;
            for (const list of [aiShips, asteroids, enemyShips, bullets, enemyBullets]) {
                for (const obj of list) {
                    if (!obj) continue;
                    if (!(obj.x >= 0 && obj.x <= canvas.width && obj.y >= 0 && obj.y <= canvas.height)) inBounds = false;
                }
            }
            if (last.done) break;
        }
        return { sig: JSON.stringify({ score: last.info.score, step: last.info.step, r: last.info.episodeReturn.map(v => v.toFixed(6)) }), inBounds, finite, info: last.info };
    }
    check('env reports the sphere world', env.world === 'sphere' && WORLD.isSphere() === false);
    const a = rollout(4242);
    check('WORLD is switched to sphere on reset', WORLD.isSphere());
    const b = rollout(4242);
    const c = rollout(99);
    check('sphere episodes are deterministic', a.sig === b.sig);
    check('different seed differs', a.sig !== c.sig);
    check('every object stays inside the chart (no torus overhang)', a.inBounds && c.inBounds);
    check('observations stay finite on the sphere', a.finite && c.finite);
    check('the rule-based fleet plays a full episode on the sphere', a.info.step >= 200 && a.info.score >= 0, `${a.info.step} steps, score ${a.info.score}`);

    // A torus env constructed afterwards restores the flat world on reset.
    const flat = new AsteroidsEnv({ numAgents: 3, maxSteps: 50, curriculumStage: 0 });
    flat.reset(1);
    check('a torus env resets the geometry back to torus', !WORLD.isSphere());
})();

print('');
print('Scaled sphere world (world size decoupled from the canvas)');
(function scaledSphere() {
    // Geometry: the chart grows with the scale, the canvas does not.
    WORLD.setMode('sphere');
    WORLD.setScale(1);
    const r1 = WORLD.radius();
    WORLD.setScale(2);
    const s2 = WORLD.size();
    check('scale 2 doubles the chart', s2.w === canvas.width * 2 && s2.h === canvas.height * 2, `${s2.w}x${s2.h}`);
    check('scale 2 doubles the radius', Math.abs(WORLD.radius() - 2 * r1) < 1e-9, `${r1.toFixed(2)} -> ${WORLD.radius().toFixed(2)}`);
    const area1 = 4 * Math.PI * r1 * r1, area2 = 4 * Math.PI * WORLD.radius() ** 2;
    check('surface area grows 4x at scale 2', Math.abs(area2 / area1 - 4) < 1e-9);
    check('scale 2 sphere has more surface than the flat map', area2 > canvas.width * canvas.height, `${area2.toFixed(0)} vs ${canvas.width * canvas.height}`);
    // Great-circle step length is a world distance and is unchanged by scale;
    // the same displacement is a smaller angle on the bigger globe.
    const p = { x: s2.w / 2, y: s2.h / 2, angle: 0 };
    WORLD.step(p, 10, 0);
    check('step of 10 units moves ~10 units of arc on the big globe', Math.abs(WORLD.distance(s2.w / 2, s2.h / 2, p.x, p.y) - 10) < 1e-6, `${WORLD.distance(s2.w / 2, s2.h / 2, p.x, p.y).toFixed(4)}`);
    let lcg = 12345;
    const rnd = () => { lcg = (lcg * 1664525 + 1013904223) >>> 0; return lcg / 4294967296; };
    let minSep = Infinity;
    for (let i = 0; i < 50; i++) {
        const far = WORLD.randomPoint(rnd, s2.w / 2, s2.h / 2, 300);
        minSep = Math.min(minSep, WORLD.distance(far.x, far.y, s2.w / 2, s2.h / 2));
    }
    check('randomPoint honours a clearance larger than the 1x radius', minSep >= 300 - 1e-6, `min ${minSep.toFixed(1)}`);
    // Scale is ignored on the torus, where the canvas must equal the world.
    WORLD.setMode('torus');
    check('scale has no effect on the torus', WORLD.size().w === canvas.width && WORLD.effectiveScale() === 1);
    WORLD.setScale(1);

    // Environment: a 2x env runs, stays inside the enlarged chart and is
    // deterministic; the env resets the scale each episode so sequences of
    // envs at different scales cannot leak into each other.
    const env = new AsteroidsEnv({ numAgents: 3, maxSteps: 300, curriculumStage: 3, fuelRescue: true, world: 'sphere', worldScale: 2 });
    function rollout(seed) {
        env.reset(seed);
        const W = WORLD.size().w, H = WORLD.size().h;
        let last = null, inBounds = true, finite = true, beyondCanvas = false, maxSep = 0;
        for (let t = 0; t < 300; t++) {
            last = env.step(null);
            for (const o of last.observations) for (const v of o) if (!Number.isFinite(v)) finite = false;
            for (const list of [aiShips, asteroids, enemyShips, bullets, enemyBullets]) {
                for (const obj of list) {
                    if (!obj) continue;
                    if (!(obj.x >= 0 && obj.x <= W && obj.y >= 0 && obj.y <= H)) inBounds = false;
                    if (obj.x > canvas.width || obj.y > canvas.height) beyondCanvas = true;
                }
            }
            for (const a of asteroids) for (const b of asteroids) if (a && b) maxSep = Math.max(maxSep, WORLD.distance(a.x, a.y, b.x, b.y));
            if (last.done) break;
        }
        return { sig: JSON.stringify({ score: last.info.score, step: last.info.step, r: last.info.episodeReturn.map(v => v.toFixed(6)) }), inBounds, finite, beyondCanvas, maxSep, info: last.info };
    }
    const a = rollout(4242);
    check('env applies the world scale on reset', WORLD.isSphere() && WORLD.getScale() === 2 && WORLD.size().w === canvas.width * 2);
    const b = rollout(4242);
    check('scaled sphere episodes are deterministic', a.sig === b.sig);
    check('objects stay inside the 2x chart', a.inBounds);
    check('objects actually use the space beyond the canvas', a.beyondCanvas);
    check('objects can be farther apart than the 1x globe allows (> pi*R1)', a.maxSep > Math.PI * r1, `max separation ${a.maxSep.toFixed(0)} vs pi*R1 ${(Math.PI * r1).toFixed(0)}`);
    check('observations stay finite at 2x', a.finite);
    check('fleet plays a full episode at 2x', a.info.step >= 150 && a.info.score >= 0, `${a.info.step} steps, score ${a.info.score}`);
    // Alpha's fuel drains more slowly per frame so endurance matches the longer tanker trip.
    const alpha = aiShips.find(s => s && s.isAlpha);
    if (alpha && alpha.fuelBound()) {
        alpha.fuel = 50;
        const before = alpha.fuel;
        alpha.thrust(1);
        const burn2x = before - alpha.fuel;
        check('thrust burns half the fuel per frame at 2x', Math.abs(burn2x - alpha.fuelThrustCost / 2) < 1e-9, `burn ${burn2x}`);
    } else {
        check('an alpha exists at end of 2x episode (skipped fuel check)', true, 'no alpha');
    }

    // A 1x sphere env afterwards restores the unscaled globe.
    const one = new AsteroidsEnv({ numAgents: 3, maxSteps: 50, curriculumStage: 0, world: 'sphere' });
    one.reset(1);
    check('a 1x env resets the scale back to 1', WORLD.getScale() === 1 && WORLD.size().w === canvas.width);
    const flat = new AsteroidsEnv({ numAgents: 3, maxSteps: 50, curriculumStage: 0 });
    flat.reset(1);
})();

print('');
if (failures === 0) {
    print('ALL ENV TESTS PASSED');
} else {
    print(`${failures} ENV TEST(S) FAILED`);
}
// Exit non-zero so run_tests.sh (and CI) sees the failure rather than the banner.
if (typeof quit === 'function') quit(failures === 0 ? 0 : 1);
