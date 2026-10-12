// Fixed benchmark suite: named scenarios, held-out seeds, identical conditions
// for every policy. Run under JavaScriptCore via training/run_benchmark.py (which
// writes the report to disk and checks it against a stored baseline):
//
//   jsc training/benchmark.js -- --suite quick
//   jsc training/benchmark.js -- --suite standard --models models/policy_stage3.json
//
// Output protocol (JavaScriptCore has no file API):
//   BENCH {json}   one line per (scenario, policy)
//   PAIRED {json}  one line per (scenario, policy) vs the heuristic and random

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
load('training/nn.js');
load('training/ppo.js');
load('training/evaluate.js');

globalThis.draw = function () {};

// Scenario ids are stable: they key the stored baseline, so renaming one is a
// breaking change to the regression gate.
const SCENARIOS = {
    s0_torus:        { stage: 0, agents: 3, world: 'torus',  worldScale: 1, desc: 'asteroids only, flat torus' },
    s3_torus:        { stage: 3, agents: 3, world: 'torus',  worldScale: 1, desc: 'asteroids + enemies + boss, flat torus' },
    s3_sphere:       { stage: 3, agents: 3, world: 'sphere', worldScale: 2, desc: 'full stage on the x2 globe (app default world)' },
    swarm_sphere:    { stage: 3, agents: 6, world: 'sphere', worldScale: 2, enemies: 4, bosses: 2, desc: '6 agents vs 4 enemies + 2 bosses on the x2 globe' }
};

const SUITES = {
    quick:    { scenarios: ['s0_torus', 's3_sphere'],                                  episodes: 6,  maxSteps: 500 },
    standard: { scenarios: ['s0_torus', 's3_torus', 's3_sphere'],                      episodes: 20, maxSteps: 1000 },
    full:     { scenarios: ['s0_torus', 's3_torus', 's3_sphere', 'swarm_sphere'],      episodes: 40, maxSteps: 1000 }
};

function parseArgs(argv) {
    const args = { suite: 'standard', models: '', scenarios: '', episodes: 0, maxSteps: 0, seedBase: 900000 };
    for (let i = 0; i < argv.length; i++) {
        const token = argv[i];
        if (!token.startsWith('--')) continue;
        const key = token.slice(2);
        const value = argv[i + 1];
        if (value === undefined || value.startsWith('--')) { args[key] = 1; continue; }
        args[key] = isNaN(Number(value)) ? value : Number(value);
        i++;
    }
    return args;
}

const args = parseArgs(typeof arguments !== 'undefined' ? arguments : []);
const suite = SUITES[args.suite];
if (!suite) throw new Error('unknown suite: ' + args.suite + ' (quick|standard|full)');

const scenarioIds = args.scenarios ? String(args.scenarios).split(',') : suite.scenarios;
const episodes = args.episodes || suite.episodes;
const maxSteps = args.maxSteps || suite.maxSteps;
const seeds = Evaluation.evalSeeds(episodes, args.seedBase);

// Checkpoints that do not match the running build's observation layout or action
// space cannot be evaluated; say so instead of silently dropping them.
const checkpoints = [];
for (const path of String(args.models).split(',').filter(Boolean)) {
    try {
        const trainer = PPOTrainer.fromJSON(JSON.parse(read(path)));
        // Run directories all hold a file called policy.json, so name those after
        // the directory ("seed0_stage3") instead of colliding on "policy".
        const parts = path.replace(/\.json$/, '').split('/');
        let name = parts[parts.length - 1];
        if (name === 'policy' && parts.length > 1) name = parts[parts.length - 2];
        checkpoints.push({ name, path, trainer });
    } catch (e) {
        print('BENCH_SKIP ' + JSON.stringify({ path, reason: String(e) }));
    }
}

print(`Benchmark suite=${args.suite}  scenarios=[${scenarioIds}]  episodes=${episodes}  maxSteps=${maxSteps}` +
    `  checkpoints=${checkpoints.length}`);

function round(x, d) { const m = Math.pow(10, d); return Math.round(x * m) / m; }

function compact(summary) {
    const out = {
        meanReturn: round(summary.meanReturn, 3), ciReturn: round(summary.ciReturn, 3),
        meanScore: round(summary.meanScore, 1), ciScore: round(summary.ciScore, 1),
        meanLength: round(summary.meanLength, 1), meanSurvivors: round(summary.meanSurvivors, 3),
        episodes: summary.episodes
    };
    if (summary.killsPer1k) {
        out.killsPer1k = {
            asteroids: round(summary.killsPer1k.asteroids, 2),
            enemies: round(summary.killsPer1k.enemies, 2),
            bosses: round(summary.killsPer1k.bosses, 2)
        };
    }
    return out;
}

for (const id of scenarioIds) {
    const sc = SCENARIOS[id];
    if (!sc) throw new Error('unknown scenario: ' + id);

    const env = new AsteroidsEnv({
        numAgents: sc.agents, maxSteps, curriculumStage: sc.stage,
        world: sc.world, worldScale: sc.worldScale,
        enemies: sc.enemies, bosses: sc.bosses
    });

    const results = {};
    results.heuristic = Evaluation.runEpisodes(env, null, seeds);
    results.random = Evaluation.runEpisodes(env,
        Evaluation.randomPolicy(AsteroidsRNG.createStream('bench-random'), ACTION_SPACE_SIZE), seeds);

    for (const cp of checkpoints) {
        if (cp.trainer.obsDim !== env.obsDim || cp.trainer.actionSize !== ACTION_SPACE_SIZE) {
            print('BENCH_SKIP ' + JSON.stringify({ path: cp.path, scenario: id,
                reason: `observation ${cp.trainer.obsDim}x${cp.trainer.actionSize} != env ${env.obsDim}x${ACTION_SPACE_SIZE}` }));
            continue;
        }
        results[cp.name] = Evaluation.runEpisodes(env,
            Evaluation.trainedPolicy(cp.trainer, false, 'bench-policy-' + cp.name), seeds);
    }

    for (const policy of Object.keys(results)) {
        const r = results[policy];
        print('BENCH ' + JSON.stringify({ scenario: id, policy, desc: sc.desc, maxSteps, summary: compact(r.summary) }));
        print('  ' + Evaluation.formatSummary(id + ' / ' + policy, r));
        if (policy !== 'heuristic' && policy !== 'random') {
            const vh = Evaluation.pairedComparison(r.returns, results.heuristic.returns);
            const vr = Evaluation.pairedComparison(r.returns, results.random.returns);
            print('PAIRED ' + JSON.stringify({
                scenario: id, policy,
                vsHeuristic: { diff: round(vh.meanDiff, 3), t: round(vh.tStat, 2), significant: vh.significant },
                vsRandom: { diff: round(vr.meanDiff, 3), t: round(vr.tStat, 2), significant: vr.significant }
            }));
        }
    }
}
print('BENCH_DONE');
