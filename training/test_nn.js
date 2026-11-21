// Gradient check for training/nn.js.
//
// Hand-written backprop is exactly the kind of code that looks right, runs
// without error, and quietly learns nothing. Every analytic gradient here is
// compared against a central finite-difference estimate; if this test does not
// pass, no training result from this repo means anything.

load('rng.js');
load('training/nn.js');

const { MLP, logSoftmax } = NN;

let failures = 0;
function check(name, condition, detail) {
    if (condition) {
        print(`  PASS  ${name}${detail ? '  (' + detail + ')' : ''}`);
    } else {
        print(`  FAIL  ${name}${detail ? '  (' + detail + ')' : ''}`);
        failures++;
    }
}

// Categorical cross-entropy loss on the network's logits, which is the same
// gradient path PPO uses for its policy term.
function lossAndGrad(net, x, targetIndex) {
    const logits = net.forward(x);
    const { logProbs, probs } = logSoftmax(logits);
    const loss = -logProbs[targetIndex];
    const dOut = new Float64Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
        dOut[i] = probs[i] - (i === targetIndex ? 1 : 0);
    }
    return { loss, dOut };
}

function lossOnly(net, x, targetIndex) {
    const logits = net.forward(x);
    const { logProbs } = logSoftmax(logits);
    return -logProbs[targetIndex];
}

function runGradCheck(label, hiddenActivation) {
    const stream = AsteroidsRNG.createStream(`gradcheck-${label}`);
    const net = new MLP(
        { inDim: 8, hidden: [10, 7], outDim: 5, hiddenActivation, outActivation: 'linear' },
        stream
    );

    const x = new Float64Array(8);
    for (let i = 0; i < x.length; i++) x[i] = stream() * 2 - 1;
    const target = 2;

    net.zeroGrad();
    const { dOut } = lossAndGrad(net, x, target);
    net.backward(dOut);

    // Central differences: (L(p+e) - L(p-e)) / 2e
    const eps = 1e-6;
    let maxRelErr = 0;
    let checked = 0;
    const params = net.getFlatParams();

    for (const { arr, grad } of params) {
        const stride = Math.max(1, Math.floor(arr.length / 12)); // sample the space
        for (let i = 0; i < arr.length; i += stride) {
            const original = arr[i];

            arr[i] = original + eps;
            const lPlus = lossOnly(net, x, target);
            arr[i] = original - eps;
            const lMinus = lossOnly(net, x, target);
            arr[i] = original;

            const numeric = (lPlus - lMinus) / (2 * eps);
            const analytic = grad[i];
            const denom = Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
            const relErr = Math.abs(numeric - analytic) / denom;
            if (relErr > maxRelErr) maxRelErr = relErr;
            checked++;
        }
    }

    check(
        `gradients match finite differences (${hiddenActivation})`,
        maxRelErr < 1e-5,
        `max rel err ${maxRelErr.toExponential(2)} over ${checked} params`
    );
}

print('Gradient check');
runGradCheck('tanh', 'tanh');
runGradCheck('relu', 'relu');

// A network that can actually fit something is the second half of the proof:
// correct gradients plus a working optimizer should drive a tiny supervised
// problem's loss to near zero.
print('Optimisation sanity');
(function learnsXor() {
    const stream = AsteroidsRNG.createStream('xor');
    const net = new MLP(
        { inDim: 2, hidden: [16], outDim: 2, hiddenActivation: 'tanh', outActivation: 'linear' },
        stream
    );
    const data = [
        { x: new Float64Array([0, 0]), y: 0 },
        { x: new Float64Array([0, 1]), y: 1 },
        { x: new Float64Array([1, 0]), y: 1 },
        { x: new Float64Array([1, 1]), y: 0 }
    ];

    let firstLoss = 0;
    let lastLoss = 0;
    for (let epoch = 0; epoch < 4000; epoch++) {
        net.zeroGrad();
        let total = 0;
        for (const d of data) {
            const { loss, dOut } = lossAndGrad(net, d.x, d.y);
            total += loss;
            net.backward(dOut);
        }
        net.applyGradients(0.02, { scale: 1 / data.length, clipNorm: 0 });
        if (epoch === 0) firstLoss = total / data.length;
        lastLoss = total / data.length;
    }

    let correct = 0;
    for (const d of data) {
        const logits = net.forward(d.x);
        let best = 0;
        for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        if (best === d.y) correct++;
    }

    check('XOR loss decreases', lastLoss < firstLoss * 0.05,
        `${firstLoss.toFixed(4)} -> ${lastLoss.toFixed(6)}`);
    check('XOR classified correctly', correct === 4, `${correct}/4`);
})();

// Serialisation must round-trip exactly, otherwise a policy that trained well
// scores differently once saved and reloaded in the browser.
print('Serialisation');
(function roundTrip() {
    const stream = AsteroidsRNG.createStream('serial');
    const net = new MLP({ inDim: 6, hidden: [8], outDim: 4, hiddenActivation: 'relu' }, stream);
    const x = new Float64Array([0.1, -0.2, 0.3, 0.9, -0.5, 0.4]);
    const before = Array.from(net.forward(x));
    const restored = MLP.fromJSON(JSON.parse(JSON.stringify(net.toJSON())));
    const after = Array.from(restored.forward(x));
    let maxDiff = 0;
    for (let i = 0; i < before.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(before[i] - after[i]));
    }
    check('JSON round-trip is exact', maxDiff === 0, `max diff ${maxDiff}`);
})();

// log-softmax must stay finite on extreme logits; a NaN here silently poisons
// the whole policy loss.
print('Numerical stability');
(function stability() {
    const big = new Float64Array([1000, -1000, 500, 0]);
    const { logProbs, probs } = logSoftmax(big);
    const finite = Array.from(logProbs).every(Number.isFinite) && Array.from(probs).every(Number.isFinite);
    let sum = 0;
    for (let i = 0; i < probs.length; i++) sum += probs[i];
    check('log-softmax finite on extreme logits', finite);
    check('probabilities sum to 1', Math.abs(sum - 1) < 1e-9, `sum=${sum}`);
})();

print('Multi-Agent RL (VDN & Counterfactual Credit)');
(function marlVDN() {
    load('training/ppo.js');
    const trainer = new PPOTrainer({ obsDim: 8, actionSize: 4, hidden: [16, 16], seed: 'marl-vdn-test' });
    const obs1 = new Float64Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    const obs2 = new Float64Array([-0.1, -0.2, 0.5, 0.1, 0.2, 0.3, -0.4, 0.9]);
    const decomp = trainer.decomposeTeamValue([obs1, obs2]);
    check('team value equals sum of individual values',
        Math.abs(decomp.teamValue - (decomp.individualValues[0] + decomp.individualValues[1])) < 1e-9);
    check('counterfactual credits sum to zero across team',
        Math.abs(decomp.counterfactualCredits[0] + decomp.counterfactualCredits[1]) < 1e-9);
    const metrics = trainer.computeSwarmMetrics([obs1, obs2]);
    check('swarm mean entropy is finite and positive', metrics.meanEntropy > 0 && Number.isFinite(metrics.meanEntropy));
    check('swarm action diversity is within [0, 1]', metrics.diversity >= 0 && metrics.diversity <= 1);
})();

print('Factored multi-discrete policy (11 logits -> 48 actions)');
(function factoredPolicyTests() {
    const stream = AsteroidsRNG.createStream('factored-grad');
    const net = new MLP({ inDim: 8, hidden: [12], outDim: NN.FACTOR_TOTAL_LOGITS, orthogonal: true, outGain: 0.01 }, stream);
    const x = new Float64Array(8);
    for (let i = 0; i < 8; i++) x[i] = stream() * 2 - 1;
    const action = 29; // rot=1, thrust=1, fire=2, shield=1
    const entropyCoef = 0.02;

    const dist = NN.factoredDistribution(net.forward(x));
    let pSum = 0;
    for (let i = 0; i < dist.probs.length; i++) pSum += dist.probs[i];
    check('factored joint probabilities sum to 1 over 48 actions', Math.abs(pSum - 1) < 1e-9, `sum=${pSum}`);

    function lossOnly() {
        const d = NN.factoredDistribution(net.forward(x));
        return -d.logProbs[action] - entropyCoef * d.entropy;
    }

    net.zeroGrad();
    const { dOut } = NN.factoredGrad(net.forward(x), action, -1, entropyCoef);
    net.backward(dOut);

    const eps = 1e-6;
    let maxRelErr = 0;
    let checked = 0;
    for (const { arr, grad } of net.getFlatParams()) {
        const stride = Math.max(1, Math.floor(arr.length / 10));
        for (let i = 0; i < arr.length; i += stride) {
            const orig = arr[i];
            arr[i] = orig + eps; const lp = lossOnly();
            arr[i] = orig - eps; const lm = lossOnly();
            arr[i] = orig;
            const num = (lp - lm) / (2 * eps);
            const ana = grad[i];
            const err = Math.abs(num - ana) / Math.max(1e-8, Math.abs(num) + Math.abs(ana));
            if (err > maxRelErr) maxRelErr = err;
            checked++;
        }
    }
    check('factored logProb + entropy gradients match finite differences',
        maxRelErr < 1e-5, `max rel err ${maxRelErr.toExponential(2)} over ${checked} params`);
})();

print('Entity-Attention encoder (permutation invariance & gradient check)');
(function entityAttentionTests() {
    const stream = AsteroidsRNG.createStream('entity-attn');
    const net = new NN.EntityAttentionNet({
        baseDim: 4, numTokens: 4, tokenDim: 3, embedDim: 6, hidden: [8], outDim: 5, orthogonal: true
    }, stream);

    // 3 present tokens + 1 absent pad token
    const x = new Float64Array([
        0.2, -0.4, 0.7, 0.1,
        1, 0.3, 0.8,
        1, 0.9, 0.2,
        0, 0.5, 0.5,
        1, 0.1, 0.6
    ]);
    // Permute token slots 0 and 1 -> output must be identical (permutation invariant!)
    const xPerm = new Float64Array([
        0.2, -0.4, 0.7, 0.1,
        1, 0.9, 0.2,
        1, 0.3, 0.8,
        0, 0.5, 0.5,
        1, 0.1, 0.6
    ]);
    const out1 = Array.from(net.forward(x));
    const attnSum = Array.from(net.lastAttentionWeights).reduce((a, b) => a + b, 0);
    const maskedZero = net.lastAttentionWeights[2] === 0;
    const out2 = Array.from(net.forward(xPerm));
    let permDiff = 0;
    for (let i = 0; i < out1.length; i++) permDiff = Math.max(permDiff, Math.abs(out1[i] - out2[i]));
    check('entity-attention is permutation invariant', permDiff < 1e-12, `max diff ${permDiff.toExponential(2)}`);
    check('attention weights sum to 1 and mask absent tokens', Math.abs(attnSum - 1) < 1e-9 && maskedZero);

    // Gradient check across encoder, scorer, and head MLP
    const target = 3;
    function lossOnly() {
        return -logSoftmax(net.forward(x)).logProbs[target];
    }
    net.zeroGrad();
    const { dOut } = lossAndGrad(net, x, target);
    net.backward(dOut);

    const eps = 1e-6;
    let maxRelErr = 0;
    let checked = 0;
    for (const { arr, grad } of net.getFlatParams()) {
        const stride = Math.max(1, Math.floor(arr.length / 8));
        for (let i = 0; i < arr.length; i += stride) {
            const orig = arr[i];
            arr[i] = orig + eps; const lp = lossOnly();
            arr[i] = orig - eps; const lm = lossOnly();
            arr[i] = orig;
            const num = (lp - lm) / (2 * eps);
            const ana = grad[i];
            const err = Math.abs(num - ana) / Math.max(1e-8, Math.abs(num) + Math.abs(ana));
            if (err > maxRelErr) maxRelErr = err;
            checked++;
        }
    }
    check('entity-attention gradients match finite differences',
        maxRelErr < 1e-5, `max rel err ${maxRelErr.toExponential(2)} over ${checked} params`);

    const restored = NN.networkFromJSON(JSON.parse(JSON.stringify(net.toJSON())));
    const after = Array.from(restored.forward(x));
    let serDiff = 0;
    for (let i = 0; i < out1.length; i++) serDiff = Math.max(serDiff, Math.abs(out1[i] - after[i]));
    check('entity-attention JSON round-trip is exact', serDiff === 0, `max diff ${serDiff}`);
})();

print('MAPPO centralized critic, annealing & COMA/VDN credit');
(function mappoTests() {
    const trainer = new PPOTrainer({
        obsDim: 16, globalStateDim: 6, actionSize: 48,
        actionMode: 'factored', centralizedCritic: true, creditMode: 'coma',
        hidden: [16], lr: 3e-4, lrEnd: 1e-4, entropyCoef: 0.01, entropyEnd: 0.002,
        valueClip: 0.2, returnNorm: true, orthogonalInit: true, seed: 'mappo-test'
    });
    trainer.setProgress(0.5);
    check('LR anneals linearly', Math.abs(trainer.currentLr() - 2e-4) < 1e-9);
    check('entropy coef anneals linearly', Math.abs(trainer.currentEntropyCoef() - 0.006) < 1e-9);
    const o = new Float64Array(16).fill(0.25);
    const g = new Float64Array(6).fill(0.5);
    const act = trainer.act(o, false);
    const val = trainer.value(o, g);
    check('MAPPO factored actor & centralized critic produce valid outputs',
        act.action >= 0 && act.action < 48 && Number.isFinite(val));
    const round = PPOTrainer.fromJSON(JSON.parse(JSON.stringify(trainer.toJSON({ stage: 3 }))));
    check('MAPPO checkpoint preserves actionMode and criticMode',
        round.actionMode === 'factored' && round.centralizedCritic === true && round.globalStateDim === 6);
})();

print('');
if (failures === 0) {
    print('ALL NN TESTS PASSED');
} else {
    print(`${failures} NN TEST(S) FAILED`);
}
