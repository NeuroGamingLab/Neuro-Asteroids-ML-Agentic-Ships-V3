// Runs a policy trained by training/train.js inside the browser.
//
// There is no model conversion step and no TensorFlow.js dependency on this
// path: the trainer and the browser execute the same forward pass from
// training/nn.js, over the same feature extractor, into the same
// applyAgentAction(). That removes the usual class of train/serve bugs where the
// deployed model sees differently ordered or differently scaled inputs than the
// one that was trained.

(function (root) {
    'use strict';

    // game.js and ml_features.js publish their layout constants onto the global
    // object explicitly. Top-level `const` would otherwise be invisible here:
    // it lives in the global lexical environment, not on globalThis.
    function resolveGlobal(name) {
        return root[name];
    }

    class PolicyRunner {
        constructor() {
            this.policy = null;
            this.actor = null;
            this.critic = null;
            this.actionMode = 'flat';
            this.criticMode = 'local';
            this.globalStateDim = 0;
            this.enabled = false;
            this.deterministic = false;
            this.meta = null;
            this.curriculumStage = 0;
            this.lastDecisions = [];
            this.inferenceMs = 0;
            this.rng = (root.AsteroidsRNG ? root.AsteroidsRNG.createStream('policy-runner') : Math.random);

            // Observation layout the checkpoint was trained on. Checkpoints
            // trained with the fuel-rescue features carry five extra inputs.
            this.featureOptions = {};

            // Optional Web Worker. When active, tick() posts observations and
            // applies the most recently returned actions (one frame of latency).
            this.worker = null;
            this.workerReady = false;
            this.workerPendingActions = null;
            this.workerFrame = 0;
            this.workerLastResultFrame = -1;
        }

        async loadFromUrl(url) {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`failed to fetch policy: ${response.status} ${response.statusText}`);
            }
            return this.loadFromObject(await response.json());
        }

        loadFromObject(payload) {
            if (!payload || payload.format !== 'asteroids-ppo-policy') {
                throw new Error('not an asteroids-ppo-policy file');
            }

            const expectedDim = resolveGlobal('RL_FEATURE_DIM');
            const expectedActions = resolveGlobal('ACTION_SPACE_SIZE');
            const expectedNames = resolveGlobal('RL_FEATURE_NAMES');
            const fuelNames = resolveGlobal('RL_FEATURE_NAMES_FUEL');
            const entityNames = resolveGlobal('RL_FEATURE_NAMES_ENTITY');

            if (expectedDim === undefined || expectedActions === undefined || !expectedNames) {
                throw new Error(
                    'cannot verify this policy: the game build did not expose its ' +
                    'feature/action layout, so a mismatch could not be detected'
                );
            }
            if (payload.actionSize !== expectedActions) {
                throw new Error(
                    `policy has ${payload.actionSize} actions but this build has ${expectedActions}`
                );
            }

            // Three observation layouts are known: base RL (38), fuel RL (43),
            // and entity-token RL (103). The checkpoint's featureNames choose
            // the matching extractor options; anything else is refused.
            const names = payload.featureNames ? payload.featureNames.join() : '';
            if (names === expectedNames.join() && payload.obsDim === expectedDim) {
                this.featureOptions = {};
            } else if (fuelNames && names === fuelNames.join() && payload.obsDim === fuelNames.length) {
                this.featureOptions = { fuelFeatures: true };
            } else if (entityNames && names === entityNames.join() && payload.obsDim === entityNames.length) {
                this.featureOptions = { fuelFeatures: true, entityTokens: true };
            } else if (payload.obsDim !== expectedDim) {
                throw new Error(
                    `policy expects ${payload.obsDim} features but this build produces ${expectedDim}` +
                    (fuelNames ? ` (or ${fuelNames.length} fuel / ${entityNames ? entityNames.length : '?'} entity)` : '')
                );
            } else {
                throw new Error('policy feature order does not match this build');
            }

            const loader = root.NN.networkFromJSON || root.NN.MLP.fromJSON;
            this.policy = payload;
            this.actor = loader(payload.actor);
            this.critic = payload.critic ? loader(payload.critic) : null;
            this.actionMode = payload.actionMode === 'factored' ? 'factored' : 'flat';
            this.criticMode = payload.criticMode === 'centralized' ? 'centralized' : 'local';
            this.globalStateDim = payload.globalStateDim || 0;
            this.meta = payload.meta || {};

            // curriculumStageNorm is one of the observation features, so it has to
            // carry the value the policy was trained with. Reading the browser's
            // marlCurriculumStage instead would hand a stage-0 policy a feature of
            // 1.0 whenever the UI sits at stage 3 — a train/serve mismatch that
            // changes the input without producing any visible error.
            this.curriculumStage = (this.meta.stage !== undefined) ? this.meta.stage : 0;
            console.log('[Policy] loaded', this.meta);
            return this;
        }

        setEnabled(enabled) {
            this.enabled = !!enabled && !!this.actor;
            if (!this.enabled) {
                setPolicyControl(false, null);
                this.lastDecisions = [];
            }
            return this.enabled;
        }

        setDeterministic(deterministic) {
            this.deterministic = !!deterministic;
        }

        // A ship the policy is not responsible for: the scripted fuel tanker.
        isScripted(shipRef) {
            return !!(shipRef && shipRef.role === 'fuel' &&
                typeof fuelRescueEnabled !== 'undefined' && fuelRescueEnabled &&
                typeof alphaAttackEnabled !== 'undefined' && alphaAttackEnabled);
        }

        observe() {
            const agentIds = [];
            const obs = [];
            const entityRefs = [];
            for (let i = 0; i < aiShips.length; i++) {
                const shipRef = aiShips[i];
                if (!shipRef || this.isScripted(shipRef)) continue;
                const agentId = shipRef.agentId !== undefined ? shipRef.agentId : i;
                const features = MLFeatureExtractor.extractRL(
                    shipRef, asteroids, aiShips, ship, enemyShips, enemyBullets,
                    Object.assign({ curriculumStage: this.curriculumStage }, this.featureOptions)
                );
                agentIds.push(agentId);
                obs.push(features._vector);
                entityRefs.push(features._entities || null);
            }
            const globalState = (this.criticMode === 'centralized' && typeof MLFeatureExtractor.extractGlobalState === 'function')
                ? MLFeatureExtractor.extractGlobalState(aiShips, asteroids, enemyShips, enemyBullets, { numAgents: aiShips.length })
                : null;
            return { agentIds, obs, entityRefs, globalState };
        }

        buildCriticInput(observation, globalState) {
            if (this.criticMode !== 'centralized') return observation;
            const g = globalState || (typeof MLFeatureExtractor.extractGlobalState === 'function'
                ? MLFeatureExtractor.extractGlobalState(aiShips, asteroids, enemyShips, enemyBullets, { numAgents: aiShips.length })
                : new Array(this.globalStateDim).fill(0));
            const out = new Float64Array(observation.length + this.globalStateDim);
            if (this.featureOptions.entityTokens) {
                const tokenSpan = (resolveGlobal('ENTITY_TOKEN_SLOTS') || 10) * (resolveGlobal('ENTITY_TOKEN_DIM') || 6);
                const baseDim = observation.length - tokenSpan;
                for (let i = 0; i < baseDim; i++) out[i] = observation[i];
                for (let i = 0; i < this.globalStateDim; i++) out[baseDim + i] = g[i] || 0;
                for (let i = 0; i < tokenSpan; i++) out[baseDim + this.globalStateDim + i] = observation[baseDim + i];
            } else {
                for (let i = 0; i < observation.length; i++) out[i] = observation[i];
                for (let i = 0; i < this.globalStateDim; i++) out[observation.length + i] = g[i] || 0;
            }
            return out;
        }

        decorate(agentId, action, probs, entropy, value, attentionWeights, attendedEntities) {
            return {
                agentId,
                action,
                decoded: decodeAction(action),
                confidence: probs[action],
                entropy,
                value,
                probs,
                topActions: topK(probs, 3),
                attentionWeights: attentionWeights || null,
                attendedEntities: attendedEntities || null
            };
        }

        // Called once per frame from update().
        tick() {
            if (!this.enabled || !this.actor) return;

            const start = (typeof performance !== 'undefined') ? performance.now() : Date.now();
            const { agentIds, obs, entityRefs, globalState } = this.observe();

            if (this.worker && this.workerReady) {
                if (this.workerPendingActions) {
                    setPolicyControl(true, this.workerPendingActions);
                } else {
                    setPolicyControl(true, []);
                }
                this.workerFrame++;
                this._lastEntityRefs = { agentIds, entityRefs };
                this.worker.postMessage({
                    type: 'infer',
                    frame: this.workerFrame,
                    deterministic: this.deterministic,
                    agentIds,
                    obs,
                    globalState
                });
                this.inferenceMs = ((typeof performance !== 'undefined') ? performance.now() : Date.now()) - start;
                return;
            }

            const actions = [];
            const decisions = [];
            for (let k = 0; k < obs.length; k++) {
                const observation = obs[k];
                const agentId = agentIds[k];
                const logits = this.actor.forward(observation);
                const attn = this.actor.lastAttentionWeights ? Array.from(this.actor.lastAttentionWeights) : null;
                let action, probs, entropy;

                if (this.actionMode === 'factored') {
                    const sampled = root.NN.sampleFactored(logits, this.rng, this.deterministic);
                    action = sampled.action;
                    probs = Array.from(sampled.probs);
                    entropy = sampled.entropy;
                } else {
                    const dist = root.NN.logSoftmax(logits);
                    probs = Array.from(dist.probs);
                    entropy = root.NN.entropyOf(dist.probs);
                    if (this.deterministic) {
                        action = 0;
                        for (let a = 1; a < probs.length; a++) if (probs[a] > probs[action]) action = a;
                    } else {
                        action = root.NN.sampleCategorical(dist.probs, this.rng);
                    }
                }

                const val = this.critic ? this.critic.forward(this.buildCriticInput(observation, globalState))[0] : null;
                actions[agentId] = action;
                decisions[agentId] = this.decorate(
                    agentId, action, probs, entropy, val, attn, entityRefs[k]
                );
            }

            this.inferenceMs = ((typeof performance !== 'undefined') ? performance.now() : Date.now()) - start;
            this.lastDecisions = decisions;
            setPolicyControl(true, actions);
        }

        // Evaluate the critic for an arbitrary observation (used by the value
        // heatmap). Runs on the main thread regardless of the worker: the
        // heatmap is throttled and only needs a handful of forwards per frame.
        valueOf(observation) {
            return this.critic ? this.critic.forward(this.buildCriticInput(observation, null))[0] : null;
        }

        // Move inference into a Web Worker. `nnSource` is the text of
        // training/nn.js and `workerSource` the text of policy_worker.js; both
        // are inlined into the page by app.py, so a Blob URL is the only way to
        // construct the worker inside the sandboxed iframe. Returns false (and
        // keeps the synchronous path) when workers are unavailable.
        enableWorker(nnSource, workerSource) {
            if (typeof Worker === 'undefined' || typeof Blob === 'undefined' || typeof URL === 'undefined') {
                return false;
            }
            if (!this.actor || !this.policy) return false;
            try {
                const blob = new Blob([nnSource, '\n;\n', workerSource], { type: 'application/javascript' });
                const url = URL.createObjectURL(blob);
                const worker = new Worker(url);
                worker.onmessage = (event) => this.onWorkerMessage(event.data);
                worker.onerror = (err) => {
                    console.warn('[Policy] worker failed, falling back to main-thread inference:', err.message || err);
                    this.disableWorker();
                };
                worker.postMessage({
                    type: 'load',
                    actor: this.policy.actor,
                    critic: this.policy.critic || null,
                    actionMode: this.actionMode,
                    criticMode: this.criticMode,
                    globalStateDim: this.globalStateDim,
                    entityTokens: !!this.featureOptions.entityTokens,
                    seed: 'policy-worker-' + Date.now()
                });
                this.worker = worker;
                this.workerReady = false;
                this.workerPendingActions = null;
                return true;
            } catch (e) {
                console.warn('[Policy] could not start worker:', e.message || e);
                this.disableWorker();
                return false;
            }
        }

        disableWorker() {
            if (this.worker) {
                try { this.worker.terminate(); } catch (e) { /* ignore */ }
            }
            this.worker = null;
            this.workerReady = false;
            this.workerPendingActions = null;
        }

        onWorkerMessage(message) {
            if (!message) return;
            if (message.type === 'ready') {
                this.workerReady = true;
                console.log('[Policy] inference running in a Web Worker');
                return;
            }
            if (message.type === 'error') {
                console.warn('[Policy] worker error, falling back:', message.message);
                this.disableWorker();
                return;
            }
            if (message.type === 'result') {
                if (message.frame <= this.workerLastResultFrame) return; // stale
                this.workerLastResultFrame = message.frame;
                const actions = [];
                const decisions = [];
                for (let idx = 0; idx < message.decisions.length; idx++) {
                    const d = message.decisions[idx];
                    actions[d.agentId] = d.action;
                    const refs = (this._lastEntityRefs && this._lastEntityRefs.entityRefs)
                        ? this._lastEntityRefs.entityRefs[idx] : null;
                    decisions[d.agentId] = this.decorate(d.agentId, d.action, d.probs, d.entropy, d.value, d.attentionWeights, refs);
                }
                this.workerPendingActions = actions;
                this.lastDecisions = decisions;
            }
        }

        getStatus() {
            return {
                loaded: !!this.actor,
                enabled: this.enabled,
                deterministic: this.deterministic,
                meta: this.meta,
                inferenceMs: this.inferenceMs,
                worker: !!(this.worker && this.workerReady),
                fuelFeatures: !!this.featureOptions.fuelFeatures,
                decisions: this.lastDecisions
            };
        }
    }

    function topK(probs, k) {
        const indexed = [];
        for (let i = 0; i < probs.length; i++) indexed.push({ action: i, p: probs[i] });
        indexed.sort((a, b) => b.p - a.p);
        return indexed.slice(0, k);
    }

    const FIRE_LABELS = ['hold', 'fwd', 'ast', 'enemy'];

    // Human-readable summary of what an agent decided and how sure it was.
    function describeDecision(decision) {
        if (!decision) return '';
        const d = decision.decoded;
        const parts = [];
        if (d.rotation === -1) parts.push('rot L');
        else if (d.rotation === 1) parts.push('rot R');
        if (d.thrust) parts.push('thrust');
        if (d.fire) parts.push(`fire:${FIRE_LABELS[d.fire]}`);
        if (d.shield) parts.push('shield');
        if (!parts.length) parts.push('idle');
        return `${parts.join(' ')}  p=${decision.confidence.toFixed(2)}`;
    }

    let policyRunner = null;

    function initializePolicyRunner() {
        if (!policyRunner) policyRunner = new PolicyRunner();
        return policyRunner;
    }

    // Hook called from game.js update().
    function policyRunnerTick() {
        if (policyRunner && policyRunner.enabled) policyRunner.tick();
    }

    async function loadTrainedPolicy(url) {
        const runner = initializePolicyRunner();
        await runner.loadFromUrl(url);
        return runner.getStatus();
    }

    function setTrainedPolicyEnabled(enabled, deterministic) {
        const runner = initializePolicyRunner();
        runner.setDeterministic(deterministic);
        return runner.setEnabled(enabled);
    }

    function getPolicyStatus() {
        return initializePolicyRunner().getStatus();
    }

    function enablePolicyWorker(nnSource, workerSource) {
        return initializePolicyRunner().enableWorker(nnSource, workerSource);
    }

    root.PolicyRunner = PolicyRunner;
    root.enablePolicyWorker = enablePolicyWorker;
    root.initializePolicyRunner = initializePolicyRunner;
    root.policyRunnerTick = policyRunnerTick;
    root.loadTrainedPolicy = loadTrainedPolicy;
    root.setTrainedPolicyEnabled = setTrainedPolicyEnabled;
    root.getPolicyStatus = getPolicyStatus;
    root.describePolicyDecision = describeDecision;
    root.getPolicyRunner = () => policyRunner;
})(typeof globalThis !== 'undefined' ? globalThis : this);
