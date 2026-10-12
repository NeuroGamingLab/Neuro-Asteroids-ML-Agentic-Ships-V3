// Web Worker half of the browser policy runner.
//
// The main thread owns the game state and therefore the feature extraction; it
// posts one batch of observations per frame. This worker owns the network
// weights and returns sampled actions, the full action distribution, entropy
// and the critic value for each agent. Moving the forward passes off the main
// thread keeps the render loop at 60 fps when many ships are policy-driven.
//
// The worker is built from a Blob: training/nn.js source is prepended to this
// file by PolicyRunner.enableWorker(), because the Streamlit page inlines all
// scripts into a sandboxed srcdoc iframe and has no URL to importScripts from.
//
// Protocol (main -> worker):
//   { type: 'load', actor, critic, seed }
//   { type: 'infer', frame, deterministic, agentIds: [...], obs: [[...], ...] }
// Protocol (worker -> main):
//   { type: 'ready' }
//   { type: 'result', frame, decisions: [{ agentId, action, probs, entropy, value }] }
//   { type: 'error', message }

(function (root) {
    'use strict';

    let actor = null;
    let critic = null;
    let rng = Math.random;

    // Same generator as rng.js, duplicated so the worker has no extra dependency.
    function mulberry32(seed) {
        let a = seed | 0;
        return function next() {
            a = (a + 0x6d2b79f5) | 0;
            let t = Math.imul(a ^ (a >>> 15), 1 | a);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }
    function hashSeed(value) {
        const str = String(value);
        let h = 2166136261 >>> 0;
        for (let i = 0; i < str.length; i++) {
            h ^= str.charCodeAt(i);
            h = Math.imul(h, 16777619);
        }
        return h >>> 0;
    }

    function argmax(probs) {
        let best = 0;
        for (let a = 1; a < probs.length; a++) if (probs[a] > probs[best]) best = a;
        return best;
    }

    function handle(message) {
        const NN = root.NN;
        if (!NN) {
            root.postMessage({ type: 'error', message: 'training/nn.js was not prepended to the worker source' });
            return;
        }
        if (message.type === 'load') {
            try {
                actor = NN.MLP.fromJSON(message.actor);
                critic = message.critic ? NN.MLP.fromJSON(message.critic) : null;
                rng = mulberry32(hashSeed(message.seed || 'policy-worker'));
                root.postMessage({ type: 'ready' });
            } catch (e) {
                root.postMessage({ type: 'error', message: String(e && e.message || e) });
            }
            return;
        }
        if (message.type === 'infer') {
            if (!actor) return;
            const decisions = [];
            for (let i = 0; i < message.obs.length; i++) {
                const observation = message.obs[i];
                const logits = actor.forward(observation);
                const { probs } = NN.logSoftmax(logits);
                const action = message.deterministic ? argmax(probs) : NN.sampleCategorical(probs, rng);
                decisions.push({
                    agentId: message.agentIds[i],
                    action,
                    probs: Array.from(probs),
                    entropy: NN.entropyOf(probs),
                    value: critic ? critic.forward(observation)[0] : null
                });
            }
            root.postMessage({ type: 'result', frame: message.frame, decisions });
        }
    }

    if (typeof root.onmessage !== 'undefined' || typeof root.postMessage === 'function') {
        root.onmessage = function (event) { handle(event.data); };
    }

    // Exposed for the headless test, which drives handle() directly.
    root.__policyWorkerHandle = handle;
})(typeof self !== 'undefined' ? self : (typeof globalThis !== 'undefined' ? globalThis : this));
