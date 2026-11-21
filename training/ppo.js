// PPO / MAPPO with GAE, clipped surrogate objective, entropy bonus, optional
// factored multi-discrete head (11 logits), centralized critic (MAPPO),
// entity-attention encoder, VDN/COMA credit assignment, and PPO stabilizers.
//
// Why PPO and not the MADDPG the repo originally advertised: MADDPG extends
// DDPG, whose deterministic policy gradient differentiates the critic with
// respect to the action and therefore needs a *continuous* action space. This
// game's actions are discrete (rotation x thrust x fire x shield = 48 joint
// actions, or 4 factored heads of 3+2+4+2 = 11 logits).
//
// Multi-agent handling supports:
//   - IPPO (default): parameter-shared actor and local critic
//   - MAPPO (`centralizedCritic: true`): decentralized actor + centralized critic
//     conditioned on [local obs, global team state]
//   - VDN / COMA (`creditMode: 'vdn' | 'coma'`): team-level value decomposition
//     and counterfactual baseline shaping during advantage computation

(function (root) {
    'use strict';

    const {
        MLP, EntityAttentionNet, networkFromJSON,
        logSoftmax, sampleCategorical, entropyOf,
        FACTOR_TOTAL_LOGITS, factoredDistribution, sampleFactored, factoredGrad
    } = root.NN;

    const DEFAULTS = {
        hidden: [128, 64],
        hiddenActivation: 'tanh',
        lr: 3e-4,
        lrEnd: null,
        criticLr: 1e-3,
        gamma: 0.99,
        lambda: 0.95,
        clip: 0.2,
        valueClip: 0,
        returnNorm: false,
        orthogonalInit: false,
        entropyCoef: 0.01,
        entropyEnd: null,
        valueCoef: 0.5,
        epochs: 4,
        minibatchSize: 256,
        maxGradNorm: 0.5,
        targetKL: 0.02,
        actionMode: 'flat',
        centralizedCritic: false,
        entityTokens: false,
        creditMode: 'ippo',
        embedDim: 24
    };

    class PPOTrainer {
        constructor(options = {}) {
            const cfg = Object.assign({}, DEFAULTS, options);
            this.cfg = cfg;
            this.obsDim = options.obsDim;
            this.actionSize = options.actionSize;
            this.actionMode = cfg.actionMode === 'factored' ? 'factored' : 'flat';
            this.centralizedCritic = !!cfg.centralizedCritic;
            this.globalStateDim = this.centralizedCritic ? (options.globalStateDim || 12) : 0;
            this.entityTokens = !!cfg.entityTokens;
            this.creditMode = (cfg.creditMode === 'vdn' || cfg.creditMode === 'coma') ? cfg.creditMode : 'ippo';
            // Observation layout recorded in the checkpoint so the browser can
            // verify (and select) the matching feature extractor.
            this.featureNames = options.featureNames || root.RL_FEATURE_NAMES || null;

            this.rng = AsteroidsRNG.createStream(options.seed !== undefined ? options.seed : 'ppo');

            const actorOutDim = this.actionMode === 'factored' ? FACTOR_TOTAL_LOGITS : this.actionSize;
            const numTokens = options.numTokens || (typeof ENTITY_TOKEN_SLOTS !== 'undefined' ? ENTITY_TOKEN_SLOTS : 10);
            const tokenDim = options.tokenDim || (typeof ENTITY_TOKEN_DIM !== 'undefined' ? ENTITY_TOKEN_DIM : 6);
            this.numTokens = numTokens;
            this.tokenDim = tokenDim;

            if (this.entityTokens) {
                const baseDim = this.obsDim - numTokens * tokenDim;
                this.actor = new EntityAttentionNet({
                    baseDim, numTokens, tokenDim,
                    embedDim: cfg.embedDim,
                    hidden: cfg.hidden,
                    outDim: actorOutDim,
                    hiddenActivation: cfg.hiddenActivation,
                    outActivation: 'linear',
                    orthogonal: cfg.orthogonalInit,
                    outGain: cfg.orthogonalInit ? 0.01 : 1.0
                }, this.rng);
                this.critic = new EntityAttentionNet({
                    baseDim: baseDim + this.globalStateDim,
                    numTokens, tokenDim,
                    embedDim: cfg.embedDim,
                    hidden: cfg.hidden,
                    outDim: 1,
                    hiddenActivation: cfg.hiddenActivation,
                    outActivation: 'linear',
                    orthogonal: cfg.orthogonalInit,
                    outGain: 1.0
                }, this.rng);
            } else {
                this.actor = new MLP({
                    inDim: this.obsDim,
                    hidden: cfg.hidden,
                    outDim: actorOutDim,
                    hiddenActivation: cfg.hiddenActivation,
                    outActivation: 'linear',   // raw logits; softmax happens in the loss
                    orthogonal: cfg.orthogonalInit,
                    outGain: cfg.orthogonalInit ? 0.01 : 1.0
                }, this.rng);

                this.critic = new MLP({
                    inDim: this.obsDim + this.globalStateDim,
                    hidden: cfg.hidden,
                    outDim: 1,
                    hiddenActivation: cfg.hiddenActivation,
                    outActivation: 'linear',
                    orthogonal: cfg.orthogonalInit,
                    outGain: 1.0
                }, this.rng);
            }

            this.progress = 0;
            this.retMean = 0;
            this.retM2 = 1;
            this.retCount = 1;

            this.obs = null;
            this.episodeStats = [];
            this.totalSteps = 0;
        }

        // Annealing progress in [0, 1] for LR and entropy schedules.
        setProgress(frac) {
            this.progress = Math.max(0, Math.min(1, Number(frac) || 0));
        }

        currentLr() {
            if (this.cfg.lrEnd === null || this.cfg.lrEnd === undefined) return this.cfg.lr;
            return this.cfg.lr + (this.cfg.lrEnd - this.cfg.lr) * this.progress;
        }

        currentCriticLr() {
            if (this.cfg.lrEnd === null || this.cfg.lrEnd === undefined) return this.cfg.criticLr;
            const scale = this.cfg.lr > 0 ? (this.currentLr() / this.cfg.lr) : 1;
            return this.cfg.criticLr * scale;
        }

        currentEntropyCoef() {
            if (this.cfg.entropyEnd === null || this.cfg.entropyEnd === undefined) return this.cfg.entropyCoef;
            return this.cfg.entropyCoef + (this.cfg.entropyEnd - this.cfg.entropyCoef) * this.progress;
        }

        buildCriticInput(observation, globalState) {
            if (!this.centralizedCritic) return observation;
            const g = globalState || new Array(this.globalStateDim).fill(0);
            const out = new Float64Array(this.obsDim + this.globalStateDim);
            if (this.entityTokens) {
                const tokenSpan = this.numTokens * this.tokenDim;
                const baseDim = this.obsDim - tokenSpan;
                for (let i = 0; i < baseDim; i++) out[i] = observation[i];
                for (let i = 0; i < this.globalStateDim; i++) out[baseDim + i] = g[i] || 0;
                for (let i = 0; i < tokenSpan; i++) out[baseDim + this.globalStateDim + i] = observation[baseDim + i];
            } else {
                for (let i = 0; i < this.obsDim; i++) out[i] = observation[i];
                for (let i = 0; i < this.globalStateDim; i++) out[this.obsDim + i] = g[i] || 0;
            }
            return out;
        }

        // Greedy action for evaluation; sampled action for rollouts.
        act(observation, deterministic, rng) {
            const logits = this.actor.forward(observation);
            if (this.actionMode === 'factored') {
                const res = sampleFactored(logits, rng || this.rng, deterministic);
                return { action: res.action, logProb: res.logProb, entropy: res.entropy };
            }
            const { logProbs, probs } = logSoftmax(logits);
            let action;
            if (deterministic) {
                action = 0;
                for (let i = 1; i < probs.length; i++) if (probs[i] > probs[action]) action = i;
            } else {
                action = sampleCategorical(probs, rng || this.rng);
            }
            return { action, logProb: logProbs[action], entropy: entropyOf(probs) };
        }

        value(observation, globalState) {
            return this.critic.forward(this.buildCriticInput(observation, globalState))[0];
        }

        // Multi-Agent Value Decomposition (VDN) & Counterfactual Credit Assignment
        decomposeTeamValue(observations, globalState) {
            if (!Array.isArray(observations) || observations.length === 0) {
                return { teamValue: 0, individualValues: [], counterfactualCredits: [] };
            }
            const individualValues = observations.map(obs => obs ? this.value(obs, globalState) : 0);
            const teamValue = individualValues.reduce((sum, v) => sum + v, 0);
            const activeCount = individualValues.filter(v => v !== 0).length || 1;
            const meanValue = teamValue / activeCount;
            // Counterfactual marginal difference: agent contribution relative to team mean
            const counterfactualCredits = individualValues.map(v => v !== 0 ? v - meanValue : 0);
            return {
                teamValue,
                individualValues,
                counterfactualCredits
            };
        }

        // Swarm policy coordination and entropy metrics
        computeSwarmMetrics(observations) {
            if (!Array.isArray(observations) || observations.length === 0) {
                return { meanEntropy: 0, diversity: 0, jointConfidence: 0 };
            }
            let totalEntropy = 0;
            let totalConf = 0;
            const actions = [];
            let count = 0;
            for (const obs of observations) {
                if (!obs) continue;
                const logits = this.actor.forward(obs);
                let probs, ent;
                if (this.actionMode === 'factored') {
                    const dist = factoredDistribution(logits);
                    probs = dist.probs;
                    ent = dist.entropy;
                } else {
                    probs = logSoftmax(logits).probs;
                    ent = entropyOf(probs);
                }
                totalEntropy += ent;
                let maxP = 0, bestA = 0;
                for (let a = 0; a < probs.length; a++) {
                    if (probs[a] > maxP) { maxP = probs[a]; bestA = a; }
                }
                totalConf += maxP;
                actions.push(bestA);
                count++;
            }
            if (count === 0) return { meanEntropy: 0, diversity: 0, jointConfidence: 0 };
            const uniqueActions = new Set(actions).size;
            return {
                meanEntropy: totalEntropy / count,
                diversity: uniqueActions / count,
                jointConfidence: totalConf / count
            };
        }

        closeSegment(segment, bootstrapValue, out) {
            let advantage = 0;
            let nextValue = bootstrapValue;
            for (let t = segment.length - 1; t >= 0; t--) {
                const nonTerminal = segment[t].done ? 0 : 1;
                const delta = segment[t].reward + this.cfg.gamma * nextValue * nonTerminal - segment[t].value;
                advantage = delta + this.cfg.gamma * this.cfg.lambda * nonTerminal * advantage;
                // Optional COMA / VDN counterfactual baseline blending
                let finalAdv = advantage;
                if (this.creditMode === 'coma' && Number.isFinite(segment[t].cfBaseline)) {
                    finalAdv = advantage - 0.25 * segment[t].cfBaseline;
                } else if (this.creditMode === 'vdn' && Number.isFinite(segment[t].teamDelta)) {
                    finalAdv = 0.7 * advantage + 0.3 * segment[t].teamDelta;
                }
                segment[t].advantage = finalAdv;
                segment[t].target = advantage + segment[t].value;
                nextValue = segment[t].value;
                out.push(segment[t]);
            }
        }

        collectRollout(env, numSteps, nextSeed) {
            const samples = [];
            const pending = [];
            for (let i = 0; i < env.numAgents; i++) pending.push([]);

            if (!this.obs) this.obs = env.reset(nextSeed());

            let collected = 0;
            while (collected < numSteps) {
                const actions = new Array(env.numAgents).fill(null);
                const staged = new Array(env.numAgents).fill(null);
                const gState = this.centralizedCritic && typeof env.globalState === 'function'
                    ? Float64Array.from(env.globalState()) : null;

                let stepValSum = 0;
                let stepAlive = 0;
                for (let i = 0; i < env.numAgents; i++) {
                    if (!env.alive[i]) continue;
                    const o = this.obs[i];
                    const { action, logProb } = this.act(o, false);
                    const cObs = this.centralizedCritic ? this.buildCriticInput(o, gState) : null;
                    const v = this.centralizedCritic ? this.critic.forward(cObs)[0] : this.value(o);
                    stepValSum += v;
                    stepAlive++;
                    // Copy the observation: the env reuses feature arrays.
                    staged[i] = {
                        obs: Float64Array.from(o),
                        criticObs: cObs,
                        action,
                        logProb,
                        value: v
                    };
                    actions[i] = action;
                }

                // Store per-step counterfactual baseline (mean of teammates' V) for COMA/VDN.
                if (stepAlive > 1 && (this.creditMode === 'coma' || this.creditMode === 'vdn')) {
                    for (let i = 0; i < env.numAgents; i++) {
                        if (!staged[i]) continue;
                        const othersMean = (stepValSum - staged[i].value) / (stepAlive - 1);
                        staged[i].cfBaseline = staged[i].value - othersMean;
                    }
                }

                const res = env.step(actions);
                collected++;
                this.totalSteps++;

                let stepRewardSum = 0;
                let stepRewardCount = 0;
                for (let i = 0; i < env.numAgents; i++) {
                    if (!staged[i]) continue;
                    staged[i].reward = res.rewards[i];
                    staged[i].done = res.dones[i];
                    stepRewardSum += res.rewards[i];
                    stepRewardCount++;
                    pending[i].push(staged[i]);
                }
                if (this.creditMode === 'vdn' && stepRewardCount > 0) {
                    const teamMeanR = stepRewardSum / stepRewardCount;
                    for (let i = 0; i < env.numAgents; i++) {
                        if (staged[i]) staged[i].teamDelta = teamMeanR;
                    }
                }

                this.obs = res.observations;
                const nextGState = this.centralizedCritic && typeof env.globalState === 'function'
                    ? Float64Array.from(env.globalState()) : null;

                // An agent that died this step has a genuinely terminal segment:
                // bootstrap value 0.
                for (let i = 0; i < env.numAgents; i++) {
                    if (res.dones[i] && pending[i].length) {
                        this.closeSegment(pending[i], 0, samples);
                        pending[i] = [];
                    }
                }

                if (res.done) {
                    for (let i = 0; i < env.numAgents; i++) {
                        if (!pending[i].length) continue;
                        const boot = (res.info.timeLimit && env.alive[i])
                            ? this.value(this.obs[i], nextGState) : 0;
                        this.closeSegment(pending[i], boot, samples);
                        pending[i] = [];
                    }
                    const returns = res.info.episodeReturn;
                    this.episodeStats.push({
                        return: returns.reduce((a, b) => a + b, 0) / returns.length,
                        score: res.info.score,
                        length: res.info.step,
                        survivors: res.info.aliveCount
                    });
                    this.obs = env.reset(nextSeed());
                }
            }

            // Rollout boundary mid-episode: bootstrap the open segments.
            const endGState = this.centralizedCritic && typeof env.globalState === 'function'
                ? Float64Array.from(env.globalState()) : null;
            for (let i = 0; i < env.numAgents; i++) {
                if (!pending[i].length) continue;
                const boot = env.alive[i] ? this.value(this.obs[i], endGState) : 0;
                this.closeSegment(pending[i], boot, samples);
                pending[i] = [];
            }

            return samples;
        }

        update(samples) {
            const cfg = this.cfg;
            const n = samples.length;
            if (n === 0) return null;

            // Optional running return variance tracker for stabilized value gradients.
            let retScale = 1.0;
            if (cfg.returnNorm) {
                for (const s of samples) {
                    this.retCount++;
                    const d = s.target - this.retMean;
                    this.retMean += d / this.retCount;
                    this.retM2 += d * (s.target - this.retMean);
                }
                retScale = Math.max(1.0, Math.sqrt(this.retM2 / Math.max(1, this.retCount)));
            }

            // Advantage normalisation, computed once over the whole batch.
            let mean = 0;
            for (const s of samples) mean += s.advantage;
            mean /= n;
            let variance = 0;
            for (const s of samples) variance += (s.advantage - mean) ** 2;
            const std = Math.sqrt(variance / n);
            for (const s of samples) s.normAdv = (s.advantage - mean) / (std + 1e-8);

            const indices = new Int32Array(n);
            for (let i = 0; i < n; i++) indices[i] = i;

            const activeLr = this.currentLr();
            const activeCriticLr = this.currentCriticLr();
            const activeEntCoef = this.currentEntropyCoef();

            let policyLoss = 0;
            let valueLoss = 0;
            let entropySum = 0;
            let klSum = 0;
            let clipped = 0;
            let counted = 0;
            let stoppedEarly = false;

            for (let epoch = 0; epoch < cfg.epochs && !stoppedEarly; epoch++) {
                shuffle(indices, this.rng);

                for (let start = 0; start < n; start += cfg.minibatchSize) {
                    const end = Math.min(n, start + cfg.minibatchSize);
                    const size = end - start;
                    if (size === 0) continue;

                    this.actor.zeroGrad();
                    this.critic.zeroGrad();

                    let mbKL = 0;

                    for (let k = start; k < end; k++) {
                        const s = samples[indices[k]];

                        // ---- policy ----
                        const logits = this.actor.forward(s.obs);
                        const lo = 1 - cfg.clip;
                        const hi = 1 + cfg.clip;
                        const adv = s.normAdv;
                        let logp, H, dOut;

                        if (this.actionMode === 'factored') {
                            const dist = factoredDistribution(logits);
                            logp = dist.logProbs[s.action];
                            const ratio = Math.exp(logp - s.logProb);
                            const surr1 = ratio * adv;
                            const surr2 = Math.min(Math.max(ratio, lo), hi) * adv;
                            const insideClipRegion = ratio > lo && ratio < hi;
                            const dLogp = (surr1 <= surr2 || insideClipRegion) ? -ratio * adv : 0;
                            if (!insideClipRegion) clipped++;
                            const gRes = factoredGrad(logits, s.action, dLogp, activeEntCoef);
                            dOut = gRes.dOut;
                            H = gRes.entropy;
                            policyLoss += -Math.min(surr1, surr2);
                        } else {
                            const { logProbs, probs } = logSoftmax(logits);
                            logp = logProbs[s.action];
                            const ratio = Math.exp(logp - s.logProb);
                            const surr1 = ratio * adv;
                            const surr2 = Math.min(Math.max(ratio, lo), hi) * adv;
                            const insideClipRegion = ratio > lo && ratio < hi;
                            const dLogp = (surr1 <= surr2 || insideClipRegion) ? -ratio * adv : 0;
                            if (!insideClipRegion) clipped++;
                            H = entropyOf(probs);
                            dOut = new Float64Array(this.actionSize);
                            for (let i = 0; i < this.actionSize; i++) {
                                const dPolicy = dLogp * ((i === s.action ? 1 : 0) - probs[i]);
                                const dEntropy = activeEntCoef * probs[i] * (Math.log(Math.max(probs[i], 1e-12)) + H);
                                dOut[i] = dPolicy + dEntropy;
                            }
                            policyLoss += -Math.min(surr1, surr2);
                        }
                        this.actor.backward(dOut);

                        entropySum += H;
                        const kl = s.logProb - logp;
                        klSum += kl;
                        mbKL += kl;
                        counted++;

                        // ---- value (optional PPO value clipping + return scaling) ----
                        const cIn = s.criticObs || s.obs;
                        const v = this.critic.forward(cIn)[0];
                        const diff = (v - s.target) / retScale;
                        let vGrad = diff;
                        let vLoss = 0.5 * diff * diff;
                        if (cfg.valueClip > 0) {
                            const vClipped = s.value + Math.max(-cfg.valueClip, Math.min(cfg.valueClip, v - s.value));
                            const diffClipped = (vClipped - s.target) / retScale;
                            const lossClipped = 0.5 * diffClipped * diffClipped;
                            if (lossClipped > vLoss) {
                                vLoss = lossClipped;
                                // Flat gradient outside clip boundary
                                const insideV = Math.abs(v - s.value) < cfg.valueClip;
                                vGrad = insideV ? diffClipped : 0;
                            }
                        }
                        valueLoss += vLoss;
                        this.critic.backward(new Float64Array([cfg.valueCoef * vGrad / retScale]));
                    }

                    this.actor.applyGradients(activeLr, { scale: 1 / size, clipNorm: cfg.maxGradNorm });
                    this.critic.applyGradients(activeCriticLr, { scale: 1 / size, clipNorm: cfg.maxGradNorm });

                    if (cfg.targetKL > 0 && (mbKL / size) > cfg.targetKL * 1.5) {
                        stoppedEarly = true;
                        break;
                    }
                }
            }

            let targetMean = 0;
            for (const s of samples) targetMean += s.target;
            targetMean /= n;
            let varTarget = 0;
            let varResidual = 0;
            for (const s of samples) {
                const v = this.critic.forward(s.criticObs || s.obs)[0];
                varTarget += (s.target - targetMean) ** 2;
                varResidual += (s.target - v) ** 2;
            }
            const explainedVariance = varTarget > 0 ? 1 - varResidual / varTarget : 0;

            return {
                policyLoss: policyLoss / Math.max(1, counted),
                valueLoss: valueLoss / Math.max(1, counted),
                entropy: entropySum / Math.max(1, counted),
                approxKL: klSum / Math.max(1, counted),
                clipFraction: clipped / Math.max(1, counted),
                explainedVariance,
                stoppedEarly,
                samples: n,
                lr: activeLr,
                entropyCoef: activeEntCoef
            };
        }

        drainEpisodeStats() {
            const stats = this.episodeStats;
            this.episodeStats = [];
            return stats;
        }

        toJSON(meta) {
            return {
                format: 'asteroids-ppo-policy',
                version: 1,
                createdAt: new Date().toISOString(),
                obsDim: this.obsDim,
                actionSize: this.actionSize,
                actionMode: this.actionMode,
                networkType: this.entityTokens ? 'entity-attention' : 'mlp',
                criticMode: this.centralizedCritic ? 'centralized' : 'local',
                globalStateDim: this.globalStateDim,
                creditMode: this.creditMode,
                featureNames: this.featureNames,
                config: this.cfg,
                meta: meta || {},
                actor: this.actor.toJSON(),
                critic: this.critic.toJSON()
            };
        }

        static fromJSON(obj) {
            const trainer = new PPOTrainer({
                obsDim: obj.obsDim,
                actionSize: obj.actionSize,
                actionMode: obj.actionMode || 'flat',
                entityTokens: obj.networkType === 'entity-attention',
                centralizedCritic: obj.criticMode === 'centralized',
                globalStateDim: obj.globalStateDim || 0,
                creditMode: obj.creditMode || 'ippo',
                featureNames: obj.featureNames || null,
                hidden: obj.config ? obj.config.hidden : undefined,
                seed: 'restored'
            });
            trainer.actor = networkFromJSON(obj.actor);
            if (obj.critic) trainer.critic = networkFromJSON(obj.critic);
            return trainer;
        }
    }

    function shuffle(indices, rng) {
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(rng() * (i + 1));
            const tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
    }

    root.PPOTrainer = PPOTrainer;
    root.PPO_DEFAULTS = DEFAULTS;
})(typeof globalThis !== 'undefined' ? globalThis : this);
