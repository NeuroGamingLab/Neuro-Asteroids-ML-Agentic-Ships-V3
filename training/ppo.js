// PPO with GAE, a clipped surrogate objective and an entropy bonus.
//
// Why PPO and not the MADDPG the repo advertises: MADDPG extends DDPG, whose
// deterministic policy gradient differentiates the critic with respect to the
// action and therefore needs a *continuous* action space. This game's actions are
// discrete (rotation x thrust x fire x shield = 24 joint actions). Making MADDPG
// work there requires a Gumbel-Softmax relaxation that is fiddly and unstable,
// whereas a 24-way categorical PPO policy is a direct fit and is the stronger
// baseline on discrete multi-agent benchmarks anyway.
//
// Multi-agent handling is IPPO with full parameter sharing: the agents are
// interchangeable, so one network trains on numAgents times the data instead of
// N networks each training on their own slice.

(function (root) {
    'use strict';

    const { MLP, logSoftmax, sampleCategorical, entropyOf } = root.NN;

    const DEFAULTS = {
        hidden: [128, 64],
        hiddenActivation: 'tanh',
        lr: 3e-4,
        criticLr: 1e-3,
        gamma: 0.99,
        lambda: 0.95,
        clip: 0.2,
        entropyCoef: 0.01,
        valueCoef: 0.5,
        epochs: 4,
        minibatchSize: 256,
        maxGradNorm: 0.5,
        targetKL: 0.02
    };

    class PPOTrainer {
        constructor(options = {}) {
            const cfg = Object.assign({}, DEFAULTS, options);
            this.cfg = cfg;
            this.obsDim = options.obsDim;
            this.actionSize = options.actionSize;
            // Observation layout recorded in the checkpoint so the browser can
            // verify (and select) the matching feature extractor.
            this.featureNames = options.featureNames || root.RL_FEATURE_NAMES || null;

            this.rng = AsteroidsRNG.createStream(options.seed !== undefined ? options.seed : 'ppo');

            this.actor = new MLP({
                inDim: this.obsDim,
                hidden: cfg.hidden,
                outDim: this.actionSize,
                hiddenActivation: cfg.hiddenActivation,
                outActivation: 'linear'   // raw logits; softmax happens in the loss
            }, this.rng);

            this.critic = new MLP({
                inDim: this.obsDim,
                hidden: cfg.hidden,
                outDim: 1,
                hiddenActivation: cfg.hiddenActivation,
                outActivation: 'linear'
            }, this.rng);

            this.obs = null;
            this.episodeStats = [];
            this.totalSteps = 0;
        }

        // Greedy action for evaluation; sampled action for rollouts.
        // `rng` overrides the training stream. Evaluation passes its own freshly
        // seeded stream so that sampling actions does not consume the training
        // RNG and, more importantly, so that evaluating one policy twice gives
        // the same answer twice.
        act(observation, deterministic, rng) {
            const logits = this.actor.forward(observation);
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

        value(observation) {
            return this.critic.forward(observation)[0];
        }

        // Multi-Agent Value Decomposition (VDN) & Counterfactual Credit Assignment
        decomposeTeamValue(observations) {
            if (!Array.isArray(observations) || observations.length === 0) {
                return { teamValue: 0, individualValues: [], counterfactualCredits: [] };
            }
            const individualValues = observations.map(obs => obs ? this.value(obs) : 0);
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
                const { probs } = logSoftmax(logits);
                const ent = entropyOf(probs);
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
                segment[t].advantage = advantage;
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

                for (let i = 0; i < env.numAgents; i++) {
                    if (!env.alive[i]) continue;
                    const o = this.obs[i];
                    const { action, logProb } = this.act(o, false);
                    // Copy the observation: the env reuses feature arrays.
                    staged[i] = {
                        obs: Float64Array.from(o),
                        action,
                        logProb,
                        value: this.value(o)
                    };
                    actions[i] = action;
                }

                const res = env.step(actions);
                collected++;
                this.totalSteps++;

                for (let i = 0; i < env.numAgents; i++) {
                    if (!staged[i]) continue;
                    staged[i].reward = res.rewards[i];
                    staged[i].done = res.dones[i];
                    pending[i].push(staged[i]);
                }

                this.obs = res.observations;

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
                        // Hitting the step limit is truncation, not failure. Treating
                        // it as terminal teaches the agent that the world ends at
                        // t=1000, so bootstrap from the value function instead.
                        const boot = (res.info.timeLimit && env.alive[i]) ? this.value(this.obs[i]) : 0;
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
            for (let i = 0; i < env.numAgents; i++) {
                if (!pending[i].length) continue;
                const boot = env.alive[i] ? this.value(this.obs[i]) : 0;
                this.closeSegment(pending[i], boot, samples);
                pending[i] = [];
            }

            return samples;
        }

        update(samples) {
            const cfg = this.cfg;
            const n = samples.length;
            if (n === 0) return null;

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
                        const { logProbs, probs } = logSoftmax(logits);
                        const logp = logProbs[s.action];
                        const ratio = Math.exp(logp - s.logProb);
                        const adv = s.normAdv;

                        const surr1 = ratio * adv;
                        const lo = 1 - cfg.clip;
                        const hi = 1 + cfg.clip;
                        const surr2 = Math.min(Math.max(ratio, lo), hi) * adv;

                        const insideClipRegion = ratio > lo && ratio < hi;
                        // d(-min(surr1, surr2))/d logp. In the clipped region the
                        // objective is flat, so the gradient is exactly zero.
                        const dLogp = (surr1 <= surr2 || insideClipRegion) ? -ratio * adv : 0;
                        if (!insideClipRegion) clipped++;

                        const H = entropyOf(probs);
                        const dOut = new Float64Array(this.actionSize);
                        for (let i = 0; i < this.actionSize; i++) {
                            const dPolicy = dLogp * ((i === s.action ? 1 : 0) - probs[i]);
                            // d(-entropyCoef * H)/dz_i  with  dH/dz_i = -p_i (log p_i + H)
                            const dEntropy = cfg.entropyCoef * probs[i] * (Math.log(Math.max(probs[i], 1e-12)) + H);
                            dOut[i] = dPolicy + dEntropy;
                        }
                        this.actor.backward(dOut);

                        policyLoss += -Math.min(surr1, surr2);
                        entropySum += H;
                        const kl = s.logProb - logp;
                        klSum += kl;
                        mbKL += kl;
                        counted++;

                        // ---- value ----
                        const v = this.critic.forward(s.obs)[0];
                        const diff = v - s.target;
                        valueLoss += 0.5 * diff * diff;
                        this.critic.backward(new Float64Array([cfg.valueCoef * diff]));
                    }

                    this.actor.applyGradients(cfg.lr, { scale: 1 / size, clipNorm: cfg.maxGradNorm });
                    this.critic.applyGradients(cfg.criticLr, { scale: 1 / size, clipNorm: cfg.maxGradNorm });

                    // Early stop on KL blow-up: the standard guard against a single
                    // update destroying the policy.
                    if (cfg.targetKL > 0 && (mbKL / size) > cfg.targetKL * 1.5) {
                        stoppedEarly = true;
                        break;
                    }
                }
            }

            // Explained variance of the value function: 1 means the critic fully
            // explains the returns, <= 0 means it is no better than predicting
            // the mean, which is the usual sign that value learning has failed.
            let targetMean = 0;
            for (const s of samples) targetMean += s.target;
            targetMean /= n;
            let varTarget = 0;
            let varResidual = 0;
            for (const s of samples) {
                const v = this.critic.forward(s.obs)[0];
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
                samples: n
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
                featureNames: obj.featureNames || null,
                hidden: obj.config ? obj.config.hidden : undefined,
                seed: 'restored'
            });
            trainer.actor = MLP.fromJSON(obj.actor);
            if (obj.critic) trainer.critic = MLP.fromJSON(obj.critic);
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
