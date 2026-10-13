// PPO training entry point. Run under JavaScriptCore:
//
//   jsc training/train.js -- --stage 0 --iterations 40
//
// or through training/run_training.py, which captures metrics and checkpoints.
//
// Output protocol (consumed by run_training.py):
//   METRIC {json}          one per iteration
//   EVAL   {json}          one per evaluation round
//   MODEL_BEGIN / MODEL_END   checkpoint payload

load('sim/browser_shim.js');
load('rng.js');
load('theme.js');       // game.js reads GAME_THEME while constructing enemies, even headless
load('world.js');       // torus/sphere geometry; game.js branches on WORLD.isSphere()
load('ml_features.js');
load('entities.js');       // Ship, Bullet, Asteroid (extracted from game.js)
load('ai_ship.js');        // AIShip extends Ship
load('enemy_ship.js');     // EnemyBullet, EnemyShip
load('enemy_coordination.js'); // EnemyCoordinationSystem (references AIShip at runtime)
load('game.js');
load('sim/env.js');
load('training/nn.js');
load('training/ppo.js');
load('training/bc.js');
load('training/evaluate.js');

// Rendering is the single biggest cost we do not need; replacing draw() with a
// no-op is what turns 60 steps/sec of wall-clock into tens of thousands.
globalThis.draw = function () {};

function parseArgs(argv) {
    const args = {
        stage: 0,
        agents: 3,
        iterations: 40,
        rollout: 2048,
        maxSteps: 1000,
        seed: 'ppo-run-1',
        lr: 3e-4,
        criticLr: 1e-3,
        entropy: 0.01,
        epochs: 4,
        minibatch: 256,
        hidden: '64,64',
        evalEvery: 10,
        evalEpisodes: 30,
        bc: 0,
        bcSteps: 30000,
        bcEpochs: 8,
        bcLr: 1e-3,
        bcAlpha: 0.75,
        // Must match the default in run_training.py and sim/env.js. When these
        // disagreed, the same nominal command trained a different objective
        // depending on which entry point you used.
        aimShaping: 0.0,
        // 1 enables the alpha fuel-rescue mechanic: one ship becomes a scripted
        // tanker, the observation gains five fuel features (43 dims) and the
        // reward gains refuel / dock-death / low-fuel terms. Checkpoints trained
        // this way only load into a build that offers the fuel layout.
        fuelRescue: 0,
        // 'torus' (flat wrap-around, default) or 'sphere' (great-circle motion,
        // arc distances). Same observation layout; recorded in the checkpoint.
        world: 'torus',
        // Sphere only: globe size relative to the 1200x600 chart (R = 191 * scale).
        worldScale: 1,
        // Optional overrides of the stage's enemy / boss counts, for fleet-sized
        // training (-1 = use the stage default).
        enemies: -1,
        bosses: -1,
        quiet: 0
    };
    for (let i = 0; i < argv.length; i++) {
        const token = argv[i];
        if (!token.startsWith('--')) continue;
        const key = token.slice(2);
        const value = argv[i + 1];
        if (value === undefined || value.startsWith('--')) {
            args[key] = 1;
        } else {
            args[key] = isNaN(Number(value)) ? value : Number(value);
            i++;
        }
    }
    args.hidden = String(args.hidden).split(',').map(Number);
    return args;
}

const args = parseArgs(typeof arguments !== 'undefined' ? arguments : []);

print(`PPO training  stage=${args.stage}  agents=${args.agents}  iterations=${args.iterations}` +
    `  rollout=${args.rollout}  hidden=[${args.hidden}]  lr=${args.lr}  entropy=${args.entropy}` +
    (args.fuelRescue ? '  fuelRescue=on' : '') + (args.world === 'sphere' ? '  world=sphere x' + args.worldScale : '') +
    (args.enemies >= 0 || args.bosses >= 0 ? `  enemies=${args.enemies} bosses=${args.bosses}` : ''));

const env = new AsteroidsEnv({
    numAgents: args.agents,
    maxSteps: args.maxSteps,
    curriculumStage: args.stage,
    fuelRescue: !!args.fuelRescue,
    world: args.world,
    worldScale: args.worldScale,
    enemies: args.enemies >= 0 ? args.enemies : undefined,
    bosses: args.bosses >= 0 ? args.bosses : undefined,
    reward: { aimShaping: args.aimShaping }
});

const trainer = new PPOTrainer({
    obsDim: env.obsDim,
    featureNames: env.featureNames,
    actionSize: ACTION_SPACE_SIZE,
    hidden: args.hidden,
    lr: args.lr,
    criticLr: args.criticLr,
    entropyCoef: args.entropy,
    epochs: args.epochs,
    minibatchSize: args.minibatch,
    seed: args.seed
});

// Curriculum transfer: continue from a policy trained on an earlier stage
// instead of starting over. The observation includes curriculumStageNorm, so a
// single network can represent stage-conditioned behaviour.
if (args.init && args.init !== 'none' && args.init !== 0) {
    try {
        const restored = PPOTrainer.fromJSON(JSON.parse(read(args.init)));
        if (restored.obsDim !== env.obsDim || restored.actionSize !== ACTION_SPACE_SIZE) {
            throw new Error(
                `shape mismatch: checkpoint is ${restored.obsDim}x${restored.actionSize}, ` +
                `this run expects ${env.obsDim}x${ACTION_SPACE_SIZE}` +
                (args.fuelRescue ? ' (fuel features on)' : '')
            );
        }
        trainer.actor = restored.actor;
        trainer.critic = restored.critic;
        print(`Warm-started from ${args.init}`);
    } catch (e) {
        print(`ERROR: could not load --init ${args.init}: ${e}`);
        throw e;
    }
}

// Training seeds live far below the evaluation band (900000+), so the held-out
// set is genuinely held out.
let trainSeedCounter = 0;
function nextTrainSeed() {
    trainSeedCounter++;
    return `train-${args.seed}-${trainSeedCounter}`;
}

const EVAL_SEEDS = Evaluation.evalSeeds(args.evalEpisodes);
const evalRng = AsteroidsRNG.createStream('eval-random-baseline');

let cachedBaselines = null;

function runEvaluation(iteration) {
    // The baselines do not change between rounds, so compute them once.
    if (!cachedBaselines) {
        cachedBaselines = {
            heuristic: Evaluation.runEpisodes(env, null, EVAL_SEEDS),
            random: Evaluation.runEpisodes(env, Evaluation.randomPolicy(evalRng, ACTION_SPACE_SIZE), EVAL_SEEDS)
        };
    }
    const { heuristic, random } = cachedBaselines;

    // Report both action-selection modes. A policy imitating a controller that is
    // idle on 92% of frames can look degenerate under argmax while behaving
    // sensibly when sampled, so reporting only one of them is misleading.
    const greedy = Evaluation.runEpisodes(env, Evaluation.trainedPolicy(trainer, true), EVAL_SEEDS);
    const stochastic = Evaluation.runEpisodes(env, Evaluation.trainedPolicy(trainer, false), EVAL_SEEDS);

    const learned = greedy.summary.meanReturn >= stochastic.summary.meanReturn ? greedy : stochastic;
    const learnedMode = learned === greedy ? 'greedy' : 'stochastic';

    const vsHeuristic = Evaluation.pairedComparison(learned.returns, heuristic.returns);
    const vsRandom = Evaluation.pairedComparison(learned.returns, random.returns);

    print('');
    print(`  Evaluation @ iteration ${iteration}  (${EVAL_SEEDS.length} held-out seeds)`);
    print('  ' + Evaluation.formatSummary('random', random));
    print('  ' + Evaluation.formatSummary('heuristic (baseline)', heuristic));
    print('  ' + Evaluation.formatSummary('PPO (greedy)', greedy));
    print('  ' + Evaluation.formatSummary('PPO (stochastic)', stochastic));
    print(`  best PPO mode: ${learnedMode}`);
    print(`  PPO - heuristic: ${vsHeuristic.meanDiff >= 0 ? '+' : ''}${vsHeuristic.meanDiff.toFixed(2)}` +
        ` +/- ${vsHeuristic.ci95.toFixed(2)}  (t=${vsHeuristic.tStat.toFixed(2)}` +
        `, ${vsHeuristic.significant ? 'SIGNIFICANT' : 'not significant'})`);
    print('');

    print(`  PPO - random:    ${vsRandom.meanDiff >= 0 ? '+' : ''}${vsRandom.meanDiff.toFixed(2)}` +
        ` +/- ${vsRandom.ci95.toFixed(2)}  (t=${vsRandom.tStat.toFixed(2)}` +
        `, ${vsRandom.significant ? 'SIGNIFICANT' : 'not significant'})`);
    print('');

    print('EVAL ' + JSON.stringify({
        iteration,
        steps: trainer.totalSteps,
        random: random.summary,
        heuristic: heuristic.summary,
        greedy: greedy.summary,
        stochastic: stochastic.summary,
        learnedMode,
        vsHeuristic,
        vsRandom
    }));

    // Evaluation leaves the env mid-episode; force the next rollout to start fresh.
    trainer.obs = null;
    return { learned, learnedMode, heuristic, vsHeuristic, vsRandom };
}

print('');
print('Baseline evaluation before training:');
runEvaluation(0);

if (args.bc) {
    print('');
    print(`Behaviour-cloning warm start: collecting ${args.bcSteps} expert transitions...`);
    const t0 = Date.now();
    const expert = BC.collectExpertData(env, args.bcSteps, nextTrainSeed);
    print(`  collected ${expert.observations.length} transitions in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

    const histogram = BC.actionHistogram(expert.actions, ACTION_SPACE_SIZE);
    const top = histogram.slice(0, 6)
        .map(h => `${h.action}:${(h.share * 100).toFixed(1)}%`).join('  ');
    print(`  expert action distribution (top): ${top}`);
    print(`  distinct actions used by expert: ${histogram.length}/${ACTION_SPACE_SIZE}`);

    BC.pretrain(trainer.actor, expert, {
        epochs: args.bcEpochs,
        lr: args.bcLr,
        rng: trainer.rng,
        actionSize: ACTION_SPACE_SIZE,
        classWeightAlpha: args.bcAlpha,
        log: r => print(`  bc epoch ${r.epoch}/${args.bcEpochs}  loss ${r.loss.toFixed(4)}` +
            `  accuracy ${(r.accuracy * 100).toFixed(1)}%` +
            `  active-frame accuracy ${(r.activeAccuracy * 100).toFixed(1)}%`)
    });

    print('');
    print('Evaluation after behaviour cloning (before any RL):');
    runEvaluation(0);
    trainer.obs = null;
}

let best = { meanReturn: -Infinity, iteration: 0, json: null };
const startedAt = Date.now();

for (let iteration = 1; iteration <= args.iterations; iteration++) {
    const t0 = Date.now();
    const samples = trainer.collectRollout(env, args.rollout, nextTrainSeed);
    const rolloutMs = Date.now() - t0;

    const t1 = Date.now();
    const metrics = trainer.update(samples);
    const updateMs = Date.now() - t1;

    const episodes = trainer.drainEpisodeStats();
    const meanReturn = episodes.length ? Evaluation.mean(episodes.map(e => e.return)) : 0;
    const meanScore = episodes.length ? Evaluation.mean(episodes.map(e => e.score)) : 0;
    const meanLength = episodes.length ? Evaluation.mean(episodes.map(e => e.length)) : 0;

    const record = {
        iteration,
        steps: trainer.totalSteps,
        episodes: episodes.length,
        meanReturn,
        meanScore,
        meanLength,
        policyLoss: metrics.policyLoss,
        valueLoss: metrics.valueLoss,
        entropy: metrics.entropy,
        approxKL: metrics.approxKL,
        clipFraction: metrics.clipFraction,
        explainedVariance: metrics.explainedVariance,
        stoppedEarly: metrics.stoppedEarly,
        rolloutMs,
        updateMs,
        stepsPerSec: Math.round(args.rollout / (rolloutMs / 1000))
    };

    print(`iter ${String(iteration).padStart(3)}/${args.iterations}` +
        `  return ${meanReturn.toFixed(2).padStart(7)}` +
        `  score ${meanScore.toFixed(0).padStart(5)}` +
        `  len ${meanLength.toFixed(0).padStart(4)}` +
        `  H ${metrics.entropy.toFixed(3)}` +
        `  KL ${metrics.approxKL.toFixed(4)}` +
        `  clip ${metrics.clipFraction.toFixed(3)}` +
        `  EV ${metrics.explainedVariance.toFixed(3)}` +
        `  ${record.stepsPerSec} sps`);

    print('METRIC ' + JSON.stringify(record));

    // Entropy collapsing toward zero this early is the classic "this run is dead"
    // signal in PPO; surface it loudly rather than burning the remaining budget.
    if (metrics.entropy < 0.05 && iteration < args.iterations * 0.5) {
        print(`  WARNING: policy entropy collapsed to ${metrics.entropy.toFixed(4)} at iteration ${iteration}.`);
    }

    if (args.evalEvery > 0 && iteration % args.evalEvery === 0) {
        const result = runEvaluation(iteration);
        if (result.learned.summary.meanReturn > best.meanReturn) {
            best = {
                meanReturn: result.learned.summary.meanReturn,
                iteration,
                json: trainer.toJSON({
                    stage: args.stage,
                    agents: args.agents,
                    world: env.world,
                    worldScale: env.worldScale,
                    iteration,
                    trainedSteps: trainer.totalSteps,
                    evalMeanReturn: result.learned.summary.meanReturn,
                    evalMeanScore: result.learned.summary.meanScore,
                    actionSelection: result.learnedMode,
                    heuristicMeanReturn: result.heuristic.summary.meanReturn,
                    beatsHeuristic: result.vsHeuristic.meanDiff > 0 && result.vsHeuristic.significant,
                    beatsRandom: result.vsRandom.meanDiff > 0 && result.vsRandom.significant,
                    vsRandomDiff: result.vsRandom.meanDiff,
                    vsRandomT: result.vsRandom.tStat
                })
            };
        }
    }
}

const finalEval = runEvaluation(args.iterations);
if (finalEval.learned.summary.meanReturn > best.meanReturn) {
    best = {
        meanReturn: finalEval.learned.summary.meanReturn,
        iteration: args.iterations,
        json: trainer.toJSON({
            stage: args.stage,
            agents: args.agents,
            world: env.world,
            worldScale: env.worldScale,
            iteration: args.iterations,
            trainedSteps: trainer.totalSteps,
            evalMeanReturn: finalEval.learned.summary.meanReturn,
            evalMeanScore: finalEval.learned.summary.meanScore,
            actionSelection: finalEval.learnedMode,
            heuristicMeanReturn: finalEval.heuristic.summary.meanReturn,
            beatsHeuristic: finalEval.vsHeuristic.meanDiff > 0 && finalEval.vsHeuristic.significant,
            beatsRandom: finalEval.vsRandom.meanDiff > 0 && finalEval.vsRandom.significant,
            vsRandomDiff: finalEval.vsRandom.meanDiff,
            vsRandomT: finalEval.vsRandom.tStat
        })
    };
}

const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
print('');
print(`Training complete in ${elapsed}s. Best eval return ${best.meanReturn.toFixed(2)} at iteration ${best.iteration}.`);

print('MODEL_BEGIN');
print(JSON.stringify(best.json));
print('MODEL_END');
