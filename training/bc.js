// Behaviour cloning from the rule-based controller.
//
// Learning to shoot asteroids from scratch is a sparse-exploration problem: the
// agent has to both aim and fire before it sees any reward, and a uniform policy
// over 24 actions almost never does. The hand-written controller already solves
// that, so we use it as a teacher for initialisation only.
//
// This is deliberately *not* the end of the pipeline. Cloning a heuristic can at
// best reproduce the heuristic - the ceiling is the teacher. Its value is as a
// warm start that puts PPO somewhere useful in policy space, after which RL can
// exceed the teacher because it optimises reward rather than imitation.

(function (root) {
    'use strict';

    const { logSoftmax } = root.NN;

    // Roll out the rule-based controller and record (observation, action) pairs
    // in the policy's own action space.
    function collectExpertData(env, numSteps, nextSeed) {
        const observations = [];
        const actions = [];

        setExpertActionRecording(true);
        let obs = env.reset(nextSeed());

        while (observations.length < numSteps) {
            const preStepObs = obs.map(o => Float64Array.from(o));
            const res = env.step(null); // null => rule-based controller drives
            const expert = res.info.expertActions || [];

            for (let i = 0; i < env.numAgents; i++) {
                if (expert[i] === undefined || expert[i] === null) continue;
                observations.push(preStepObs[i]);
                actions.push(expert[i]);
            }

            obs = res.observations;
            if (res.done) obs = env.reset(nextSeed());
        }

        setExpertActionRecording(false);
        return { observations, actions };
    }

    // Cross-entropy on the actor's logits, with class weighting.
    //
    // The teacher is idle on ~89% of frames, so unweighted cross-entropy is
    // minimised by predicting "idle" always - it reaches ~91% accuracy while
    // learning nothing about the 11% of frames that actually score points. Class
    // weights of (1/frequency)^alpha rebalance the loss so the rare, decisive
    // actions carry comparable total weight to the common no-op.
    function pretrain(actor, data, options = {}) {
        const epochs = options.epochs || 8;
        const batchSize = options.batchSize || 256;
        const lr = options.lr || 1e-3;
        const rng = options.rng || Math.random;
        const log = options.log || function () {};
        const actionSize = options.actionSize || 24;
        const alpha = options.classWeightAlpha !== undefined ? options.classWeightAlpha : 0.75;

        const n = data.observations.length;
        const indices = new Int32Array(n);
        for (let i = 0; i < n; i++) indices[i] = i;

        const counts = new Array(actionSize).fill(0);
        for (const a of data.actions) counts[a]++;
        const weights = new Float64Array(actionSize);
        let weightSum = 0;
        for (let a = 0; a < actionSize; a++) {
            weights[a] = counts[a] > 0 ? Math.pow(n / counts[a], alpha) : 0;
            weightSum += weights[a] * counts[a];
        }
        // Normalise so the mean sample weight is 1 and the learning rate keeps
        // the same meaning as in the unweighted case.
        const normaliser = n / (weightSum || 1);
        for (let a = 0; a < actionSize; a++) weights[a] *= normaliser;

        const history = [];

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = n - 1; i > 0; i--) {
                const j = Math.floor(rng() * (i + 1));
                const tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
            }

            let lossSum = 0;
            let correct = 0;
            // Accuracy on the frames where the teacher actually did something is
            // the metric that matters; overall accuracy just tracks the base rate.
            let activeTotal = 0;
            let activeCorrect = 0;

            for (let start = 0; start < n; start += batchSize) {
                const end = Math.min(n, start + batchSize);
                const size = end - start;
                if (size === 0) continue;

                actor.zeroGrad();
                for (let k = start; k < end; k++) {
                    const idx = indices[k];
                    const obs = data.observations[idx];
                    const target = data.actions[idx];
                    const w = weights[target];

                    const logits = actor.forward(obs);
                    const { logProbs, probs } = logSoftmax(logits);
                    lossSum += -logProbs[target] * w;

                    let best = 0;
                    for (let a = 1; a < probs.length; a++) if (probs[a] > probs[best]) best = a;
                    if (best === target) correct++;

                    const decoded = decodeAction(target);
                    const isActive = decoded.rotation !== 0 || decoded.thrust || decoded.fire || decoded.shield;
                    if (isActive) {
                        activeTotal++;
                        if (best === target) activeCorrect++;
                    }

                    const dOut = new Float64Array(probs.length);
                    for (let a = 0; a < probs.length; a++) {
                        dOut[a] = w * (probs[a] - (a === target ? 1 : 0));
                    }
                    actor.backward(dOut);
                }
                actor.applyGradients(lr, { scale: 1 / size, clipNorm: 0.5 });
            }

            const record = {
                epoch: epoch + 1,
                loss: lossSum / n,
                accuracy: correct / n,
                activeAccuracy: activeTotal > 0 ? activeCorrect / activeTotal : 0
            };
            history.push(record);
            log(record);
        }

        return history;
    }

    // How often does the expert pick each action? A teacher that emits one action
    // 95% of the time produces a degenerate clone, so this is worth reporting.
    function actionHistogram(actions, actionSize) {
        const counts = new Array(actionSize).fill(0);
        for (const a of actions) counts[a]++;
        return counts.map((c, i) => ({ action: i, count: c, share: c / actions.length }))
            .filter(entry => entry.count > 0)
            .sort((a, b) => b.count - a.count);
    }

    root.BC = { collectExpertData, pretrain, actionHistogram };
})(typeof globalThis !== 'undefined' ? globalThis : this);
