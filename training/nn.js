// Minimal dense neural network with real backpropagation and Adam.
//
// This exists because marl_system.js allocated actors, critics, target networks
// and a replay buffer, but trainAgent() was a console.log - the weights never
// moved. Everything here is plain JS with no native dependency, so the identical
// forward pass runs in the headless trainer and in the browser, and a trained
// policy is a JSON file rather than a converted model artifact.
//
// Layout: W is row-major [inDim x outDim], so W[i * outDim + j] is the weight
// from input i to output j.

(function (root) {
    'use strict';

    const ACTIVATIONS = {
        relu: {
            fn: z => (z > 0 ? z : 0),
            // derivative expressed in terms of the pre-activation z
            dfn: (z) => (z > 0 ? 1 : 0)
        },
        tanh: {
            fn: z => Math.tanh(z),
            dfn: (z) => { const t = Math.tanh(z); return 1 - t * t; }
        },
        linear: {
            fn: z => z,
            dfn: () => 1
        }
    };

    class Dense {
        constructor(inDim, outDim, activation, rngStream, skipInit) {
            this.inDim = inDim;
            this.outDim = outDim;
            this.activationName = activation;
            this.activation = ACTIVATIONS[activation];
            if (!this.activation) throw new Error(`unknown activation: ${activation}`);

            this.W = new Float64Array(inDim * outDim);
            this.b = new Float64Array(outDim);
            this.gW = new Float64Array(inDim * outDim);
            this.gb = new Float64Array(outDim);

            // Adam moments
            this.mW = new Float64Array(inDim * outDim);
            this.vW = new Float64Array(inDim * outDim);
            this.mb = new Float64Array(outDim);
            this.vb = new Float64Array(outDim);

            // He init for relu, Xavier otherwise. Getting this wrong is a silent
            // killer: with the wrong scale a deep-ish net starts saturated and the
            // policy collapses to a single action before learning anything.
            if (!skipInit) {
                const draw = rngStream || Math.random;
                const scale = activation === 'relu'
                    ? Math.sqrt(2 / inDim)
                    : Math.sqrt(1 / inDim);
                for (let i = 0; i < this.W.length; i++) {
                    this.W[i] = gaussian(draw) * scale;
                }
            }

            // forward cache
            this._x = null;
            this._z = new Float64Array(outDim);
            this._a = new Float64Array(outDim);
        }

        forward(x) {
            this._x = x;
            const { inDim, outDim, W, b, _z, _a } = this;
            for (let j = 0; j < outDim; j++) _z[j] = b[j];
            for (let i = 0; i < inDim; i++) {
                const xi = x[i];
                if (xi === 0) continue;
                const rowOffset = i * outDim;
                for (let j = 0; j < outDim; j++) {
                    _z[j] += xi * W[rowOffset + j];
                }
            }
            const fn = this.activation.fn;
            for (let j = 0; j < outDim; j++) _a[j] = fn(_z[j]);
            return _a;
        }

        // dA: gradient of loss w.r.t. this layer's activation output.
        // Accumulates parameter grads, returns gradient w.r.t. this layer's input.
        backward(dA, dXOut) {
            const { inDim, outDim, W, gW, gb, _x, _z } = this;
            const dfn = this.activation.dfn;

            const dZ = new Float64Array(outDim);
            for (let j = 0; j < outDim; j++) {
                dZ[j] = dA[j] * dfn(_z[j]);
                gb[j] += dZ[j];
            }

            for (let i = 0; i < inDim; i++) {
                const xi = _x[i];
                const rowOffset = i * outDim;
                let acc = 0;
                for (let j = 0; j < outDim; j++) {
                    gW[rowOffset + j] += xi * dZ[j];
                    acc += W[rowOffset + j] * dZ[j];
                }
                if (dXOut) dXOut[i] = acc;
            }
            return dXOut;
        }

        zeroGrad() {
            this.gW.fill(0);
            this.gb.fill(0);
        }
    }

    class MLP {
        constructor(spec, rngStream, skipInit) {
            // spec: { inDim, hidden: [128, 64], outDim, hiddenActivation, outActivation }
            this.layers = [];
            let prev = spec.inDim;
            const hidden = spec.hidden || [128, 64];
            const hiddenAct = spec.hiddenActivation || 'tanh';
            for (let i = 0; i < hidden.length; i++) {
                this.layers.push(new Dense(prev, hidden[i], hiddenAct, rngStream, skipInit));
                prev = hidden[i];
            }
            this.layers.push(new Dense(prev, spec.outDim, spec.outActivation || 'linear', rngStream, skipInit));
            this.spec = {
                inDim: spec.inDim,
                hidden: hidden.slice(),
                outDim: spec.outDim,
                hiddenActivation: hiddenAct,
                outActivation: spec.outActivation || 'linear'
            };
            this.adamStep = 0;
            this._scratch = this.layers.map(l => new Float64Array(l.inDim));
        }

        forward(x) {
            let a = x;
            for (let i = 0; i < this.layers.length; i++) {
                a = this.layers[i].forward(a);
            }
            return a;
        }

        // dOut: gradient w.r.t. network output. Accumulates into layer grads.
        backward(dOut) {
            let d = dOut;
            for (let i = this.layers.length - 1; i >= 0; i--) {
                const target = i > 0 ? this._scratch[i] : null;
                const returned = this.layers[i].backward(d, target || new Float64Array(this.layers[i].inDim));
                d = returned;
            }
            return d;
        }

        zeroGrad() {
            for (const l of this.layers) l.zeroGrad();
        }

        // Adam with decoupled gradient scaling (grads are summed over a minibatch,
        // so pass scale = 1/batchSize) and global-norm clipping.
        applyGradients(lr, options = {}) {
            const beta1 = options.beta1 !== undefined ? options.beta1 : 0.9;
            const beta2 = options.beta2 !== undefined ? options.beta2 : 0.999;
            const eps = options.eps !== undefined ? options.eps : 1e-8;
            const scale = options.scale !== undefined ? options.scale : 1;
            const clipNorm = options.clipNorm !== undefined ? options.clipNorm : 0.5;

            let sumSq = 0;
            for (const l of this.layers) {
                for (let i = 0; i < l.gW.length; i++) sumSq += (l.gW[i] * scale) ** 2;
                for (let i = 0; i < l.gb.length; i++) sumSq += (l.gb[i] * scale) ** 2;
            }
            const norm = Math.sqrt(sumSq);
            const clip = (clipNorm > 0 && norm > clipNorm) ? clipNorm / (norm + 1e-12) : 1;
            const effective = scale * clip;

            this.adamStep++;
            const bc1 = 1 - Math.pow(beta1, this.adamStep);
            const bc2 = 1 - Math.pow(beta2, this.adamStep);

            for (const l of this.layers) {
                stepAdam(l.W, l.gW, l.mW, l.vW, lr, beta1, beta2, eps, effective, bc1, bc2);
                stepAdam(l.b, l.gb, l.mb, l.vb, lr, beta1, beta2, eps, effective, bc1, bc2);
            }
            return norm;
        }

        toJSON() {
            return {
                spec: this.spec,
                layers: this.layers.map(l => ({
                    inDim: l.inDim,
                    outDim: l.outDim,
                    activation: l.activationName,
                    W: Array.from(l.W),
                    b: Array.from(l.b)
                }))
            };
        }

        static fromJSON(obj) {
            // skipInit: the weights are about to be overwritten, and running the
            // Gaussian initialiser with a degenerate stream would spin forever.
            const net = new MLP(obj.spec, null, true);
            obj.layers.forEach((ld, i) => {
                net.layers[i].W.set(ld.W);
                net.layers[i].b.set(ld.b);
            });
            return net;
        }

        // Flat parameter access, used by the numerical gradient check.
        getFlatParams() {
            const out = [];
            for (const l of this.layers) {
                out.push({ arr: l.W, grad: l.gW });
                out.push({ arr: l.b, grad: l.gb });
            }
            return out;
        }
    }

    function stepAdam(p, g, m, v, lr, beta1, beta2, eps, scale, bc1, bc2) {
        for (let i = 0; i < p.length; i++) {
            const grad = g[i] * scale;
            m[i] = beta1 * m[i] + (1 - beta1) * grad;
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
            const mHat = m[i] / bc1;
            const vHat = v[i] / bc2;
            p[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
    }

    function gaussian(draw) {
        let u = 0;
        let v = 0;
        while (u === 0) u = draw();
        while (v === 0) v = draw();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    // Numerically stable log-softmax. Returns { logProbs, probs }.
    function logSoftmax(logits) {
        const n = logits.length;
        let max = -Infinity;
        for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
        let sum = 0;
        const exp = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            exp[i] = Math.exp(logits[i] - max);
            sum += exp[i];
        }
        const logSum = Math.log(sum) + max;
        const logProbs = new Float64Array(n);
        const probs = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            logProbs[i] = logits[i] - logSum;
            probs[i] = exp[i] / sum;
        }
        return { logProbs, probs };
    }

    function sampleCategorical(probs, draw) {
        const r = draw();
        let acc = 0;
        for (let i = 0; i < probs.length; i++) {
            acc += probs[i];
            if (r < acc) return i;
        }
        return probs.length - 1;
    }

    function entropyOf(probs) {
        let h = 0;
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > 1e-12) h -= probs[i] * Math.log(probs[i]);
        }
        return h;
    }

    root.NN = { Dense, MLP, logSoftmax, sampleCategorical, entropyOf, gaussian };

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = root.NN;
    }
})(typeof globalThis !== 'undefined' ? globalThis : this);
