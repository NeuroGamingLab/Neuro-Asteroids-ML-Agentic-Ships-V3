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

print('');
if (failures === 0) {
    print('ALL NN TESTS PASSED');
} else {
    print(`${failures} NN TEST(S) FAILED`);
}
