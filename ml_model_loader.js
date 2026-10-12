// TensorFlow.js model loader with versioning and WASM backend support

class MLModelLoader {
    constructor() {
        this.models = {};
        this.metadata = {};
        this.backend = 'webgl';
        this.loaded = false;
        this.loadError = null;
    }

    async setBackend(preferWasm = true) {
        if (typeof tf === 'undefined') {
            console.warn('[ML Loader] TensorFlow.js not loaded');
            return false;
        }

        try {
            if (preferWasm && tf.setBackend) {
                await tf.setBackend('wasm');
                this.backend = 'wasm';
            } else {
                await tf.setBackend('webgl');
                this.backend = 'webgl';
            }
            await tf.ready();
            console.log(`[ML Loader] Backend: ${tf.getBackend()}`);
            return true;
        } catch (err) {
            console.warn('[ML Loader] Backend fallback to webgl:', err);
            try {
                await tf.setBackend('webgl');
                await tf.ready();
                this.backend = 'webgl';
                return true;
            } catch (e) {
                this.loadError = e;
                return false;
            }
        }
    }

    async loadModel(key, modelPath, metadata = {}) {
        if (typeof tf === 'undefined') {
            throw new Error('TensorFlow.js not available');
        }

        try {
            const model = await tf.loadLayersModel(modelPath);
            this.models[key] = model;
            this.metadata[key] = {
                path: modelPath,
                version: metadata.version || '1.0.0',
                loadedAt: Date.now(),
                type: metadata.type || key
            };
            console.log(`[ML Loader] Loaded ${key} from ${modelPath}`);
            return model;
        } catch (err) {
            console.warn(`[ML Loader] Failed to load ${key}:`, err.message);
            this.loadError = err;
            return null;
        }
    }

    async loadModels(config = {}) {
        const {
            multiTaskModelPath = null,
            parameterModelPath = null,
            priorityModelPath = null,
            actionModelPath = null,
            preferWasm = true,
            version = '1.0.0'
        } = config;

        await this.setBackend(preferWasm);

        const loads = [];

        if (multiTaskModelPath && multiTaskModelPath !== 'none') {
            loads.push(this.loadModel('multitask', multiTaskModelPath, { version, type: 'multitask' }));
        }
        if (parameterModelPath && parameterModelPath !== 'none') {
            loads.push(this.loadModel('parameters', parameterModelPath, { version, type: 'parameters' }));
        }
        if (priorityModelPath && priorityModelPath !== 'none') {
            loads.push(this.loadModel('priorities', priorityModelPath, { version, type: 'priorities' }));
        }
        if (actionModelPath && actionModelPath !== 'none') {
            loads.push(this.loadModel('actions', actionModelPath, { version, type: 'actions' }));
        }

        await Promise.all(loads);
        this.loaded = Object.keys(this.models).length > 0;
        return this.getStatus();
    }

    getModel(key) {
        return this.models[key] || null;
    }

    getStatus() {
        return {
            loaded: this.loaded,
            backend: this.backend,
            models: Object.keys(this.models),
            metadata: this.metadata,
            error: this.loadError ? this.loadError.message : null
        };
    }
}

let mlModelLoader = null;

function initializeMLModelLoader() {
    if (!mlModelLoader) {
        mlModelLoader = new MLModelLoader();
    }
    return mlModelLoader;
}

if (typeof window !== 'undefined') {
    window.MLModelLoader = MLModelLoader;
    window.initializeMLModelLoader = initializeMLModelLoader;
}
