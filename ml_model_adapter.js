// Multi-task ML adapter with batched inference and safety bounds

class MLModelAdapter {
    constructor(loader) {
        this.loader = loader;
        this.cache = new Map();
        this.cacheFrame = 0;
        this.cacheInterval = 5;
        this.inferenceMs = 0;
        this.useMultiTask = true;
        this.pendingBatch = null;
    }

    scaleParameterOutputs(values) {
        return {
            detectionRadius: 50 + values[0] * 150,
            firingRange: 100 + values[1] * 300,
            flockWeight: 0.1 + values[2] * 0.7,
            thrustFrequency: 0.01 + values[3] * 0.09,
            enemyFiringRange: 150 + values[4] * 250
        };
    }

    scalePriorityOutputs(values) {
        return {
            avoidShips: 1.0,
            fireAtEnemies: 0.5 + values[0] * 0.5,
            protectAllies: 0.3 + values[1] * 0.5,
            fireAtAsteroids: 0.2 + values[2] * 0.4,
            avoidAsteroids: 0.2 + values[3] * 0.3,
            randomNav: 0.05 + values[4] * 0.15
        };
    }

    clampParameters(params) {
        return {
            detectionRadius: Math.max(50, Math.min(200, params.detectionRadius)),
            firingRange: Math.max(100, Math.min(400, params.firingRange)),
            flockWeight: Math.max(0.1, Math.min(0.8, params.flockWeight)),
            thrustFrequency: Math.max(0.01, Math.min(0.1, params.thrustFrequency)),
            enemyFiringRange: Math.max(150, Math.min(400, params.enemyFiringRange))
        };
    }

    clampPriorities(weights) {
        return {
            avoidShips: 1.0,
            fireAtEnemies: Math.max(0.5, Math.min(1.0, weights.fireAtEnemies)),
            protectAllies: Math.max(0.3, Math.min(0.8, weights.protectAllies)),
            fireAtAsteroids: Math.max(0.2, Math.min(0.6, weights.fireAtAsteroids)),
            avoidAsteroids: Math.max(0.2, Math.min(0.5, weights.avoidAsteroids)),
            randomNav: Math.max(0.05, Math.min(0.2, weights.randomNav))
        };
    }

    async _predictTensor(model, featureVector) {
        const start = performance.now();
        const input = tf.tensor2d([featureVector]);
        const output = model.predict(input);
        const data = await output.data();
        input.dispose();
        output.dispose();
        this.inferenceMs = performance.now() - start;
        return Array.from(data);
    }

    async predict(features, shipId = 'default') {
        const vector = MLFeatureExtractor.vectorFromObject(features);
        const cacheKey = `${shipId}_params_${this.cacheFrame}`;

        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        let result = null;
        const multiTask = this.loader.getModel('multitask');
        const paramModel = this.loader.getModel('parameters');

        if (multiTask && this.useMultiTask) {
            const outputs = await this._predictTensor(multiTask, vector);
            result = this.clampParameters(this.scaleParameterOutputs(outputs.slice(0, 5)));
        } else if (paramModel) {
            const outputs = await this._predictTensor(paramModel, vector.slice(0, 13));
            result = this.clampParameters(this.scaleParameterOutputs(outputs));
        }

        if (result) this.cache.set(cacheKey, result);
        return result;
    }

    async predictPriorityWeights(features, shipId = 'default') {
        const vector = MLFeatureExtractor.vectorFromObject(features);
        const cacheKey = `${shipId}_prio_${this.cacheFrame}`;

        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        let result = null;
        const multiTask = this.loader.getModel('multitask');
        const prioModel = this.loader.getModel('priorities');

        if (multiTask && this.useMultiTask) {
            const outputs = await this._predictTensor(multiTask, vector);
            result = this.clampPriorities(this.scalePriorityOutputs(outputs.slice(5, 10)));
        } else if (prioModel) {
            const outputs = await this._predictTensor(prioModel, vector.slice(0, 13));
            result = this.clampPriorities(this.scalePriorityOutputs(outputs));
        }

        if (result) this.cache.set(cacheKey, result);
        return result;
    }

    // Index at which the optional action head starts in the multitask output.
    // The default supervised model (train_multitask_model.py) emits 10 values:
    // 5 parameters + 5 priority weights, and no action head.
    static get ACTION_HEAD_OFFSET() { return 10; }
    static get ACTION_HEAD_WIDTH() { return 5; }

    async predictAction(features, shipId = 'default') {
        const vector = MLFeatureExtractor.vectorFromObject(features);
        const multiTask = this.loader.getModel('multitask');
        const actionModel = this.loader.getModel('actions');

        const offset = MLModelAdapter.ACTION_HEAD_OFFSET;
        const width = MLModelAdapter.ACTION_HEAD_WIDTH;

        let outputs = null;
        if (multiTask && this.useMultiTask) {
            const all = await this._predictTensor(multiTask, vector);
            // A 10-output model has no action head. Slicing anyway yields an
            // empty array and silently produces NaN actions, so bail out and let
            // the caller fall back to the rule-based controller.
            if (all.length < offset + width) {
                if (!this._warnedNoActionHead) {
                    console.warn(
                        `[ML] Multitask model exposes ${all.length} outputs; an action head needs ` +
                        `${offset + width}. Falling back to rule-based actions.`
                    );
                    this._warnedNoActionHead = true;
                }
                return null;
            }
            outputs = all.slice(offset, offset + width);
        } else if (actionModel) {
            outputs = await this._predictTensor(actionModel, vector.slice(0, 13));
        }

        if (!outputs || outputs.length < width) return null;
        if (outputs.some(v => !Number.isFinite(v))) return null;

        // Outputs are sigmoid-activated, i.e. in [0, 1]. Decode on that scale.
        const actionTypes = ['rotate', 'thrust', 'fire', 'shield', 'idle'];
        const actionIndex = Math.min(actionTypes.length - 1, Math.max(0, Math.floor(outputs[0] * actionTypes.length)));

        return {
            action: actionTypes[actionIndex],
            rotation: outputs[1] > 0.66 ? 1 : (outputs[1] < 0.33 ? -1 : 0),
            target: features._canvas ? {
                x: outputs[2] * features._canvas.width,
                y: outputs[3] * features._canvas.height
            } : null,
            priority: outputs[4]
        };
    }

    async batchPredict(shipsData) {
        const start = performance.now();
        const results = [];

        for (let i = 0; i < shipsData.length; i++) {
            const { features, shipId } = shipsData[i];
            const params = await this.predict(features, shipId || `ship_${i}`);
            const priorities = await this.predictPriorityWeights(features, shipId || `ship_${i}`);
            results.push({ params, priorities });
        }

        this.inferenceMs = performance.now() - start;
        return results;
    }

    tickCache(frame) {
        if (frame % this.cacheInterval === 0) {
            this.cache.clear();
            this.cacheFrame = frame;
        }
    }

    getInferenceMs() {
        return this.inferenceMs;
    }
}

let mlModelAdapter = null;

function initializeMLModelAdapter(loader) {
    if (!mlModelAdapter) {
        mlModelAdapter = new MLModelAdapter(loader || initializeMLModelLoader());
    }
    return mlModelAdapter;
}

if (typeof window !== 'undefined') {
    window.MLModelAdapter = MLModelAdapter;
    window.initializeMLModelAdapter = initializeMLModelAdapter;
}
