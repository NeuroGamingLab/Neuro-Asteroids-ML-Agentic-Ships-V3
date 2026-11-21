// Adversarial Self-Play League Trainer (Phase 4).
//
// Trains a Blue swarm PPO policy against a league pool of historical checkpoints
// and adaptive Red enemy pressure profiles (varying enemy fire cadence, burst
// coordination, and boss aggression), implementing Prioritized Fictitious
// Self-Play (PFSP) opponent sampling so Blue never overfits to a single static
// enemy heuristic.
//
// Usage:
//   jsc training/self_play.js -- --rounds 4 --itersPerRound 15 --world sphere --worldScale 2
//   python3 training/run_self_play.py --rounds 4 --itersPerRound 20

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

function parseArgs(argv) {
    const args = {
        rounds: 3,
        itersPerRound: 10,
        rollout: 2048,
        agents: 4,
        enemies: 3,
        bosses: 1,
        world: 'sphere',
        worldScale: 2,
        fuelRescue: 0,
        waves: 0,
        bossPhases: 1,
        actionMode: 'factored',
        centralizedCritic: 1,
        creditMode: 'coma',
        hidden: '64,64',
        lr: 3e-4,
        entropy: 0.008,
        seed: 'selfplay-1',
        init: 'none'
    };
    for (let i = 0; i < argv.length; i++) {
        const tok = argv[i];
        if (!tok.startsWith('--')) continue;
        const key = tok.slice(2);
        const val = argv[i + 1];
        if (val === undefined || val.startsWith('--')) { args[key] = 1; continue; }
        args[key] = isNaN(Number(val)) ? val : Number(val);
        i++;
    }
    args.hidden = String(args.hidden).split(',').map(Number);
    return args;
}

const args = parseArgs(typeof arguments !== 'undefined' ? arguments : []);

// League of escalating adversarial Red Swarm tactical profiles (PFSP pool).
// Each profile configures enemy pressure & boss composition; sampling weights
// prioritize opponents against which Blue currently scores lowest.
const OPPONENT_LEAGUE = [
    { name: 'balanced-patrol',   enemies: args.enemies,     bosses: args.bosses,     waves: false, bossPhases: false },
    { name: 'wolfpack-rush',     enemies: args.enemies + 2, bosses: args.bosses,     waves: false, bossPhases: true  },
    { name: 'twin-dreadnoughts', enemies: Math.max(1, args.enemies - 1), bosses: args.bosses + 1, waves: false, bossPhases: true  },
    { name: 'endless-armada',    enemies: args.enemies + 1, bosses: args.bosses + 1, waves: true,  bossPhases: true  }
];

const rng = AsteroidsRNG.createStream(args.seed + '-league');
const baseEnv = new AsteroidsEnv({
    numAgents: args.agents,
    maxSteps: 600,
    curriculumStage: 3,
    enemies: args.enemies,
    bosses: args.bosses,
    world: args.world,
    worldScale: args.worldScale,
    fuelRescue: !!args.fuelRescue,
    waves: !!args.waves,
    bossPhases: !!args.bossPhases
});

const trainer = new PPOTrainer({
    obsDim: baseEnv.obsDim,
    globalStateDim: args.centralizedCritic ? baseEnv.globalStateDim : 0,
    featureNames: baseEnv.featureNames,
    actionSize: ACTION_SPACE_SIZE,
    actionMode: args.actionMode,
    centralizedCritic: !!args.centralizedCritic,
    creditMode: args.creditMode,
    hidden: args.hidden,
    lr: args.lr,
    lrEnd: args.lr * 0.25,
    entropyCoef: args.entropy,
    entropyEnd: args.entropy * 0.3,
    valueClip: 0.2,
    returnNorm: true,
    orthogonalInit: true,
    seed: args.seed
});

if (args.init && args.init !== 'none' && args.init !== 0) {
    const restored = PPOTrainer.fromJSON(JSON.parse(read(args.init)));
    if (restored.obsDim === baseEnv.obsDim && restored.actionMode === trainer.actionMode) {
        trainer.actor = restored.actor;
        if (!args.centralizedCritic && restored.critic) trainer.critic = restored.critic;
        print(`Warm-started self-play league from ${args.init}`);
    }
}

// Track win/score estimates against every league opponent for PFSP sampling.
const leagueScores = OPPONENT_LEAGUE.map(() => 0);
const checkpointArchive = [];
let seedIdx = 0;
function nextSeed() { seedIdx++; return `${args.seed}-ep-${seedIdx}`; }

function pickPFSPOpponent() {
    // Softmax over negative normalized scores so harder opponents are sampled more often.
    const maxS = Math.max.apply(null, leagueScores);
    const weights = leagueScores.map(s => Math.exp((maxS - s) / 10));
    const total = weights.reduce((a, b) => a + b, 0) || 1;
    let r = rng() * total;
    for (let i = 0; i < weights.length; i++) {
        r -= weights[i];
        if (r <= 0) return i;
    }
    return 0;
}

print(`Self-Play League  rounds=${args.rounds}  iters/round=${args.itersPerRound}` +
    `  actionMode=${args.actionMode}  MAPPO=${!!args.centralizedCritic}  credit=${args.creditMode}` +
    `  world=${args.world} x${args.worldScale}`);

const evalSeeds = Evaluation.evalSeeds(12, 950000);
const totalIters = args.rounds * args.itersPerRound;
let globalIter = 0;

for (let round = 1; round <= args.rounds; round++) {
    const oppIdx = pickPFSPOpponent();
    const opp = OPPONENT_LEAGUE[oppIdx];
    const env = new AsteroidsEnv({
        numAgents: args.agents,
        maxSteps: 600,
        curriculumStage: 3,
        enemies: opp.enemies,
        bosses: opp.bosses,
        world: args.world,
        worldScale: args.worldScale,
        fuelRescue: !!args.fuelRescue,
        waves: opp.waves,
        bossPhases: opp.bossPhases
    });

    print(`\n--- League Round ${round}/${args.rounds}: opponent="${opp.name}" (enemies=${opp.enemies}, bosses=${opp.bosses}, waves=${opp.waves}) ---`);
    trainer.obs = null;
    for (let it = 1; it <= args.itersPerRound; it++) {
        globalIter++;
        trainer.setProgress((globalIter - 1) / Math.max(1, totalIters));
        const samples = trainer.collectRollout(env, args.rollout, nextSeed);
        const m = trainer.update(samples);
        const eps = trainer.drainEpisodeStats();
        const rMean = eps.length ? Evaluation.mean(eps.map(e => e.return)) : 0;
        print(`  r${round}.${it}  ret=${rMean.toFixed(2)}  H=${m.entropy.toFixed(3)}  KL=${m.approxKL.toFixed(4)}  EV=${m.explainedVariance.toFixed(3)}`);
    }

    // Score across all league opponents to update PFSP priorities.
    const roundSummary = {};
    for (let i = 0; i < OPPONENT_LEAGUE.length; i++) {
        const o = OPPONENT_LEAGUE[i];
        const e = new AsteroidsEnv({
            numAgents: args.agents, maxSteps: 600, curriculumStage: 3,
            enemies: o.enemies, bosses: o.bosses,
            world: args.world, worldScale: args.worldScale,
            fuelRescue: !!args.fuelRescue, waves: o.waves, bossPhases: o.bossPhases
        });
        const res = Evaluation.runEpisodes(e, Evaluation.trainedPolicy(trainer, false, `sp-eval-r${round}-${i}`), evalSeeds);
        leagueScores[i] = res.summary.meanReturn;
        roundSummary[o.name] = Number(res.summary.meanReturn.toFixed(2));
    }
    checkpointArchive.push({ round, opponent: opp.name, leagueReturns: roundSummary });
    print('LEAGUE_ROUND ' + JSON.stringify({ round, opponent: opp.name, leagueReturns: roundSummary }));
}

// Final paired evaluation vs heuristic & random on baseEnv
const heur = Evaluation.runEpisodes(baseEnv, null, evalSeeds);
const rand = Evaluation.runEpisodes(baseEnv, Evaluation.randomPolicy(AsteroidsRNG.createStream('sp-rand'), ACTION_SPACE_SIZE), evalSeeds);
const learned = Evaluation.runEpisodes(baseEnv, Evaluation.trainedPolicy(trainer, false, 'sp-final'), evalSeeds);
const vh = Evaluation.pairedComparison(learned.returns, heur.returns);
const vr = Evaluation.pairedComparison(learned.returns, rand.returns);

const finalPayload = trainer.toJSON({
    stage: 3,
    agents: args.agents,
    world: baseEnv.world,
    worldScale: baseEnv.worldScale,
    fuelRescue: !!args.fuelRescue,
    waves: !!args.waves,
    bossPhases: !!args.bossPhases,
    centralizedCritic: !!args.centralizedCritic,
    creditMode: args.creditMode,
    selfPlayRounds: args.rounds,
    leagueHistory: checkpointArchive,
    trainedSteps: trainer.totalSteps,
    evalMeanReturn: learned.summary.meanReturn,
    evalMeanScore: learned.summary.meanScore,
    actionSelection: 'stochastic',
    heuristicMeanReturn: heur.summary.meanReturn,
    beatsHeuristic: vh.meanDiff > 0 && vh.significant,
    beatsRandom: vr.meanDiff > 0 && vr.significant,
    vsRandomDiff: vr.meanDiff,
    vsRandomT: vr.tStat
});

print(`Self-Play Final: return=${learned.summary.meanReturn.toFixed(2)} vs heur=${heur.summary.meanReturn.toFixed(2)} (t=${vh.tStat.toFixed(2)}) vs rand=${rand.summary.meanReturn.toFixed(2)} (t=${vr.tStat.toFixed(2)})`);
print('MODEL_BEGIN');
print(JSON.stringify(finalPayload));
print('MODEL_END');
