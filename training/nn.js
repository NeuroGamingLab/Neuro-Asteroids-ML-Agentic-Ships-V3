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

            // He init for relu, Xavier otherwise, or orthogonal (Gram-Schmidt) when requested.
            if (!skipInit) {
                const draw = rngStream || Math.random;
                if (typeof rngStream === 'object' && rngStream && rngStream.orthogonal) {
                    const d = rngStream.rng || Math.random;
                    const gain = rngStream.gain !== undefined ? rngStream.gain : (activation === 'relu' ? Math.SQRT2 : 1.0);
                    initOrthogonal(this.W, inDim, outDim, gain, d);
                } else {
                    const scale = activation === 'relu'
                        ? Math.sqrt(2 / inDim)
                        : Math.sqrt(1 / inDim);
                    for (let i = 0; i < this.W.length; i++) {
                        this.W[i] = gaussian(draw) * scale;
                    }
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
            // spec: { inDim, hidden: [128, 64], outDim, hiddenActivation, outActivation, orthogonal, outGain }
            this.layers = [];
            let prev = spec.inDim;
            const hidden = spec.hidden || [128, 64];
            const hiddenAct = spec.hiddenActivation || 'tanh';
            const useOrtho = !!spec.orthogonal;
            const hiddenGain = hiddenAct === 'relu' ? Math.SQRT2 : (5 / 3);
            const outGain = spec.outGain !== undefined ? spec.outGain : 1.0;
            for (let i = 0; i < hidden.length; i++) {
                const initArg = useOrtho ? { orthogonal: true, gain: hiddenGain, rng: rngStream } : rngStream;
                this.layers.push(new Dense(prev, hidden[i], hiddenAct, initArg, skipInit));
                prev = hidden[i];
            }
            const outInitArg = useOrtho ? { orthogonal: true, gain: outGain, rng: rngStream } : rngStream;
            this.layers.push(new Dense(prev, spec.outDim, spec.outActivation || 'linear', outInitArg, skipInit));
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

    // Orthogonal matrix initializer (modified Gram-Schmidt QR) scaled by `gain`.
    function initOrthogonal(W, rows, cols, gain, draw) {
        const flat = new Float64Array(rows * cols);
        for (let i = 0; i < flat.length; i++) flat[i] = gaussian(draw);
        if (rows >= cols) {
            // Orthonormalize columns
            for (let c = 0; c < cols; c++) {
                for (let prev = 0; prev < c; prev++) {
                    let dot = 0;
                    for (let r = 0; r < rows; r++) dot += flat[r * cols + c] * flat[r * cols + prev];
                    for (let r = 0; r < rows; r++) flat[r * cols + c] -= dot * flat[r * cols + prev];
                }
                let norm = 0;
                for (let r = 0; r < rows; r++) norm += flat[r * cols + c] * flat[r * cols + c];
                norm = Math.sqrt(norm) || 1;
                for (let r = 0; r < rows; r++) flat[r * cols + c] /= norm;
            }
        } else {
            // Orthonormalize rows
            for (let r = 0; r < rows; r++) {
                for (let prev = 0; prev < r; prev++) {
                    let dot = 0;
                    for (let c = 0; c < cols; c++) dot += flat[r * cols + c] * flat[prev * cols + c];
                    for (let c = 0; c < cols; c++) flat[r * cols + c] -= dot * flat[prev * cols + c];
                }
                let norm = 0;
                for (let c = 0; c < cols; c++) norm += flat[r * cols + c] * flat[r * cols + c];
                norm = Math.sqrt(norm) || 1;
                for (let c = 0; c < cols; c++) flat[r * cols + c] /= norm;
            }
        }
        for (let i = 0; i < W.length; i++) W[i] = flat[i] * gain;
    }

    // Factored multi-discrete distribution over the 48-action space:
    //   rotation (3) x thrust (2) x fire (4) x shield (2) => 11 logits total.
    const FACTOR_DIMS = [3, 2, 4, 2];
    const FACTOR_OFFSETS = [0, 3, 5, 9];
    const FACTOR_TOTAL_LOGITS = 11;

    function actionToFactors(action) {
        return [
            Math.floor(action / 16) % 3,
            (action >> 3) & 1,
            (action >> 1) & 3,
            action & 1
        ];
    }

    function factorsToAction(f0, f1, f2, f3) {
        return f0 * 16 + f1 * 8 + f2 * 2 + f3;
    }

    function factoredDistribution(logits11) {
        const factorProbs = [];
        const factorLogProbs = [];
        let entropy = 0;
        for (let f = 0; f < 4; f++) {
            const off = FACTOR_OFFSETS[f];
            const dim = FACTOR_DIMS[f];
            const slice = logits11.subarray ? logits11.subarray(off, off + dim) : logits11.slice(off, off + dim);
            const { logProbs, probs } = logSoftmax(slice);
            factorProbs.push(probs);
            factorLogProbs.push(logProbs);
            entropy += entropyOf(probs);
        }
        // Materialize full 48-way joint distribution for policy_viz / compatibility.
        const probs = new Float64Array(48);
        const logProbs = new Float64Array(48);
        const p0 = factorProbs[0], p1 = factorProbs[1], p2 = factorProbs[2], p3 = factorProbs[3];
        const lp0 = factorLogProbs[0], lp1 = factorLogProbs[1], lp2 = factorLogProbs[2], lp3 = factorLogProbs[3];
        let idx = 0;
        for (let r = 0; r < 3; r++) {
            for (let t = 0; t < 2; t++) {
                for (let fi = 0; fi < 4; fi++) {
                    for (let s = 0; s < 2; s++) {
                        probs[idx] = p0[r] * p1[t] * p2[fi] * p3[s];
                        logProbs[idx] = lp0[r] + lp1[t] + lp2[fi] + lp3[s];
                        idx++;
                    }
                }
            }
        }
        return { factorProbs, factorLogProbs, probs, logProbs, entropy };
    }

    function sampleFactored(logits11, draw, deterministic) {
        const dist = factoredDistribution(logits11);
        const picked = [0, 0, 0, 0];
        for (let f = 0; f < 4; f++) {
            const fp = dist.factorProbs[f];
            if (deterministic) {
                let best = 0;
                for (let i = 1; i < fp.length; i++) if (fp[i] > fp[best]) best = i;
                picked[f] = best;
            } else {
                picked[f] = sampleCategorical(fp, draw);
            }
        }
        const action = factorsToAction(picked[0], picked[1], picked[2], picked[3]);
        const logProb = dist.factorLogProbs[0][picked[0]] + dist.factorLogProbs[1][picked[1]]
            + dist.factorLogProbs[2][picked[2]] + dist.factorLogProbs[3][picked[3]];
        return { action, logProb, entropy: dist.entropy, probs: dist.probs, factorProbs: dist.factorProbs };
    }

    function factoredGrad(logits11, action, dLogp, entropyCoef) {
        const dist = factoredDistribution(logits11);
        const factors = actionToFactors(action);
        const dOut = new Float64Array(FACTOR_TOTAL_LOGITS);
        let jointLogProb = 0;
        for (let f = 0; f < 4; f++) {
            const off = FACTOR_OFFSETS[f];
            const dim = FACTOR_DIMS[f];
            const fp = dist.factorProbs[f];
            const flp = dist.factorLogProbs[f];
            const chosen = factors[f];
            jointLogProb += flp[chosen];
            const Hf = entropyOf(fp);
            for (let i = 0; i < dim; i++) {
                const dPol = dLogp * ((i === chosen ? 1 : 0) - fp[i]);
                const dEnt = entropyCoef * fp[i] * (Math.log(Math.max(fp[i], 1e-12)) + Hf);
                dOut[off + i] = dPol + dEnt;
            }
        }
        return { dOut, logProb: jointLogProb, entropy: dist.entropy };
    }

    // Permutation-invariant Entity-Attention Network (DeepSets + learned masked
    // soft attention over variable entity slots). Processes:
    //   x = [baseFeatures (baseDim), token_0 (tokenDim), ..., token_{K-1} (tokenDim)]
    // where token_k[0] > 0.5 indicates entity presence. Exposes lastAttentionWeights
    // for 3D sightline visualization and supports exact analytical backprop.
    class EntityAttentionNet {
        constructor(spec, rngStream, skipInit) {
            this.spec = {
                type: 'entity-attention',
                inDim: spec.baseDim + spec.numTokens * spec.tokenDim,
                baseDim: spec.baseDim,
                numTokens: spec.numTokens,
                tokenDim: spec.tokenDim,
                embedDim: spec.embedDim || 24,
                hidden: (spec.hidden || [64, 64]).slice(),
                outDim: spec.outDim,
                hiddenActivation: spec.hiddenActivation || 'tanh',
                outActivation: spec.outActivation || 'linear'
            };
            const useOrtho = !!spec.orthogonal;
            const encInit = useOrtho ? { orthogonal: true, gain: 5 / 3, rng: rngStream } : rngStream;
            const scrInit = useOrtho ? { orthogonal: true, gain: 1.0, rng: rngStream } : rngStream;
            this.encoder = new Dense(this.spec.tokenDim, this.spec.embedDim, 'tanh', encInit, skipInit);
            this.scorer = new Dense(this.spec.embedDim, 1, 'linear', scrInit, skipInit);
            this.head = new MLP({
                inDim: this.spec.baseDim + this.spec.embedDim,
                hidden: this.spec.hidden,
                outDim: this.spec.outDim,
                hiddenActivation: this.spec.hiddenActivation,
                outActivation: this.spec.outActivation,
                orthogonal: useOrtho,
                outGain: spec.outGain
            }, rngStream, skipInit);
            this.adamStep = 0;
            this.lastAttentionWeights = new Float64Array(this.spec.numTokens);

            // Forward caches for exact multi-token backprop
            this._toks = [];
            this._encZ = [];
            this._encH = [];
            this._present = new Uint8Array(this.spec.numTokens);
            this._scores = new Float64Array(this.spec.numTokens);
            this._alpha = new Float64Array(this.spec.numTokens);
            this._combined = new Float64Array(this.spec.baseDim + this.spec.embedDim);
            for (let k = 0; k < this.spec.numTokens; k++) {
                this._toks.push(new Float64Array(this.spec.tokenDim));
                this._encZ.push(new Float64Array(this.spec.embedDim));
                this._encH.push(new Float64Array(this.spec.embedDim));
            }
            this._dHScratch = new Float64Array(this.spec.embedDim);
            this._dHScore = new Float64Array(this.spec.embedDim);
        }

        forward(x) {
            const { baseDim, numTokens, tokenDim, embedDim } = this.spec;
            const combined = this._combined;
            for (let i = 0; i < baseDim; i++) combined[i] = x[i];

            let anyPresent = false;
            let maxScore = -Infinity;
            for (let k = 0; k < numTokens; k++) {
                const off = baseDim + k * tokenDim;
                const tok = this._toks[k];
                for (let j = 0; j < tokenDim; j++) tok[j] = x[off + j] || 0;
                const present = tok[0] > 0.5 ? 1 : 0;
                this._present[k] = present;
                if (!present) {
                    this._alpha[k] = 0;
                    continue;
                }
                anyPresent = true;
                const h = this.encoder.forward(tok);
                this._encZ[k].set(this.encoder._z);
                this._encH[k].set(h);
                const sc = this.scorer.forward(h)[0];
                this._scores[k] = sc;
                if (sc > maxScore) maxScore = sc;
            }

            for (let j = 0; j < embedDim; j++) combined[baseDim + j] = 0;

            if (anyPresent) {
                let sumExp = 0;
                for (let k = 0; k < numTokens; k++) {
                    if (!this._present[k]) continue;
                    const e = Math.exp(this._scores[k] - maxScore);
                    this._alpha[k] = e;
                    sumExp += e;
                }
                for (let k = 0; k < numTokens; k++) {
                    if (!this._present[k]) {
                        this.lastAttentionWeights[k] = 0;
                        continue;
                    }
                    const a = this._alpha[k] / sumExp;
                    this._alpha[k] = a;
                    this.lastAttentionWeights[k] = a;
                    const h = this._encH[k];
                    for (let j = 0; j < embedDim; j++) combined[baseDim + j] += a * h[j];
                }
            } else {
                this.lastAttentionWeights.fill(0);
            }

            return this.head.forward(combined);
        }

        backward(dOut) {
            const { baseDim, numTokens, embedDim } = this.spec;
            const dCombined = this.head.backward(dOut);
            const dC = dCombined.subarray(baseDim, baseDim + embedDim);

            let anyPresent = false;
            let meanDAlpha = 0;
            const dAlpha = new Float64Array(numTokens);
            for (let k = 0; k < numTokens; k++) {
                if (!this._present[k]) continue;
                anyPresent = true;
                const h = this._encH[k];
                let dot = 0;
                for (let j = 0; j < embedDim; j++) dot += dC[j] * h[j];
                dAlpha[k] = dot;
                meanDAlpha += this._alpha[k] * dot;
            }

            if (anyPresent) {
                const dScoreArr = new Float64Array(1);
                for (let k = 0; k < numTokens; k++) {
                    if (!this._present[k]) continue;
                    const a = this._alpha[k];
                    const dScore = a * (dAlpha[k] - meanDAlpha);

                    // Backprop through scorer for token k
                    this.scorer._x = this._encH[k];
                    this.scorer._z[0] = this._scores[k];
                    dScoreArr[0] = dScore;
                    this.scorer.backward(dScoreArr, this._dHScore);

                    // Combine direct pooling gradient + attention score gradient
                    for (let j = 0; j < embedDim; j++) {
                        this._dHScratch[j] = a * dC[j] + this._dHScore[j];
                    }

                    // Backprop through shared token encoder for token k
                    this.encoder._x = this._toks[k];
                    this.encoder._z.set(this._encZ[k]);
                    this.encoder.backward(this._dHScratch, null);
                }
            }
            return dCombined;
        }

        zeroGrad() {
            this.encoder.zeroGrad();
            this.scorer.zeroGrad();
            this.head.zeroGrad();
        }

        _allLayers() {
            return [this.encoder, this.scorer].concat(this.head.layers);
        }

        applyGradients(lr, options = {}) {
            const beta1 = options.beta1 !== undefined ? options.beta1 : 0.9;
            const beta2 = options.beta2 !== undefined ? options.beta2 : 0.999;
            const eps = options.eps !== undefined ? options.eps : 1e-8;
            const scale = options.scale !== undefined ? options.scale : 1;
            const clipNorm = options.clipNorm !== undefined ? options.clipNorm : 0.5;
            const layers = this._allLayers();

            let sumSq = 0;
            for (const l of layers) {
                for (let i = 0; i < l.gW.length; i++) sumSq += (l.gW[i] * scale) ** 2;
                for (let i = 0; i < l.gb.length; i++) sumSq += (l.gb[i] * scale) ** 2;
            }
            const norm = Math.sqrt(sumSq);
            const clip = (clipNorm > 0 && norm > clipNorm) ? clipNorm / (norm + 1e-12) : 1;
            const effective = scale * clip;

            this.adamStep++;
            const bc1 = 1 - Math.pow(beta1, this.adamStep);
            const bc2 = 1 - Math.pow(beta2, this.adamStep);
            for (const l of layers) {
                stepAdam(l.W, l.gW, l.mW, l.vW, lr, beta1, beta2, eps, effective, bc1, bc2);
                stepAdam(l.b, l.gb, l.mb, l.vb, lr, beta1, beta2, eps, effective, bc1, bc2);
            }
            return norm;
        }

        getFlatParams() {
            const out = [];
            for (const l of this._allLayers()) {
                out.push({ arr: l.W, grad: l.gW });
                out.push({ arr: l.b, grad: l.gb });
            }
            return out;
        }

        toJSON() {
            const serLayer = l => ({
                inDim: l.inDim,
                outDim: l.outDim,
                activation: l.activationName,
                W: Array.from(l.W),
                b: Array.from(l.b)
            });
            return {
                type: 'entity-attention',
                spec: this.spec,
                encoder: serLayer(this.encoder),
                scorer: serLayer(this.scorer),
                head: this.head.toJSON()
            };
        }

        static fromJSON(obj) {
            const net = new EntityAttentionNet(obj.spec, null, true);
            net.encoder.W.set(obj.encoder.W);
            net.encoder.b.set(obj.encoder.b);
            net.scorer.W.set(obj.scorer.W);
            net.scorer.b.set(obj.scorer.b);
            net.head = MLP.fromJSON(obj.head);
            return net;
        }
    }

    function networkFromJSON(obj) {
        if (!obj) return null;
        if (obj.type === 'entity-attention' || (obj.spec && obj.spec.type === 'entity-attention')) {
            return EntityAttentionNet.fromJSON(obj);
        }
        return MLP.fromJSON(obj);
    }

    root.NN = {
        Dense,
        MLP,
        EntityAttentionNet,
        networkFromJSON,
        logSoftmax,
        sampleCategorical,
        entropyOf,
        gaussian,
        FACTOR_DIMS,
        FACTOR_OFFSETS,
        FACTOR_TOTAL_LOGITS,
        actionToFactors,
        factorsToAction,
        factoredDistribution,
        sampleFactored,
        factoredGrad
    };

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = root.NN;
    }
})(typeof globalThis !== 'undefined' ? globalThis : this);
