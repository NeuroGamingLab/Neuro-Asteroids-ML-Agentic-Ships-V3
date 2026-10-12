// Reliable training data collection with IndexedDB persistence
// Supports imitation labels, MARL rollouts, and expert replay samples

class MARLDataCollector {
    constructor(options = {}) {
        this.collecting = false;
        this.dbName = options.dbName || 'asteroids_marl_training';
        this.storeName = options.storeName || 'samples';
        this.maxMemoryBuffer = options.maxMemoryBuffer || 500;
        this.autoSaveInterval = options.autoSaveInterval || 100;
        this.autoExportIntervalMs = options.autoExportIntervalMs || 0;
        this.memoryBuffer = [];
        this.sampleCount = 0;
        this.db = null;
        this.highScoreThreshold = options.highScoreThreshold || 500;
        this.lastAutoSaveCount = 0;
        this._autoExportTimer = null;
    }

    async init() {
        if (!window.indexedDB) {
            console.warn('[DataCollector] IndexedDB unavailable, using memory only');
            return false;
        }

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, 1);
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    db.createObjectStore(this.storeName, { keyPath: 'id', autoIncrement: true });
                }
            };
            request.onsuccess = (event) => {
                this.db = event.target.result;
                console.log('[DataCollector] IndexedDB ready');
                resolve(true);
            };
            request.onerror = () => reject(request.error);
        });
    }

    setCollecting(enabled) {
        this.collecting = enabled;
        if (enabled) {
            console.log('[DataCollector] Collection ENABLED');
            if (this.autoExportIntervalMs > 0) {
                this._autoExportTimer = setInterval(() => this.flushToIndexedDB(), this.autoExportIntervalMs);
            }
        } else if (this._autoExportTimer) {
            clearInterval(this._autoExportTimer);
            this._autoExportTimer = null;
        }
    }

    buildOptimalParameters(features, ship) {
        const base = {
            detectionRadius: ship.baseDetectionRadius || 100,
            firingRange: ship.baseFiringRange || 200,
            flockWeight: ship.baseFlockWeight || 0.3,
            thrustFrequency: ship.baseThrustFrequency || 0.02,
            enemyFiringRange: ship.baseEnemyFiringRange || 250
        };

        if (features.threatLevel > 0.7) {
            base.detectionRadius = 150;
            base.thrustFrequency = 0.04;
        } else if (features.threatLevel < 0.3) {
            base.detectionRadius = 80;
            base.thrustFrequency = 0.015;
        }
        if (features.enemyCount > 0.5) {
            base.firingRange = 250;
            base.enemyFiringRange = 300;
            base.flockWeight = 0.6;
        }
        if (features.bulletThreat > 0) {
            base.thrustFrequency = 0.05;
        }
        return base;
    }

    buildOptimalPriorityWeights(features) {
        const weights = {
            avoidShips: 1.0,
            fireAtEnemies: 0.8,
            protectAllies: 0.6,
            fireAtAsteroids: 0.4,
            avoidAsteroids: 0.3,
            randomNav: 0.1
        };

        if (features.enemyCount > 0.5 || features.nearestEnemyDistance < 0.5) {
            weights.fireAtEnemies = 0.9;
            weights.protectAllies = 0.7;
            weights.fireAtAsteroids = 0.3;
        }
        if (features.threatLevel > 0.7 || features.bulletThreat > 0) {
            weights.avoidAsteroids = 0.45;
            weights.randomNav = 0.05;
        }
        if (features.health < 0.5) {
            weights.avoidAsteroids = 0.5;
            weights.fireAtEnemies = 0.65;
        }
        if (features.bossCount > 0) {
            weights.fireAtEnemies = 0.95;
            weights.protectAllies = 0.75;
        }
        return weights;
    }

    collectSample(context) {
        if (!this.collecting) return;

        const {
            agentIndex = 0,
            ship,
            features,
            marlObservation = null,
            marlAction = null,
            reward = 0,
            mlParams = null,
            mlPriorities = null,
            strategicDecision = null,
            score = 0,
            curriculumStage = 0,
            labelSource = 'expert',
            done = false
        } = context;

        if (!ship || !features) return;

        const optimalParameters = mlParams || this.buildOptimalParameters(features, ship);
        const optimalPriorityWeights = mlPriorities || this.buildOptimalPriorityWeights(features);

        const isHighPerformance = score >= this.highScoreThreshold && (ship.health || 0) > 0;
        const sample = {
            timestamp: Date.now(),

            // Episode delimitation. A flat pool of (state, action) rows cannot be
            // turned back into trajectories: returns, n-step targets and GAE all
            // need to know where one episode stops and the next begins, and the
            // seed is what makes a logged episode replayable.
            episodeId: typeof episodeId !== 'undefined' ? episodeId : null,
            stepIndex: typeof episodeStepCount !== 'undefined' ? episodeStepCount : null,
            done: !!done,
            seed: typeof getGameSeed === 'function' ? getGameSeed() : null,

            agentIndex,
            curriculumStage,
            labelSource: isHighPerformance ? 'imitation' : labelSource,
            score,
            reward,
            marlObservation,
            marlAction,
            mlFeatures: MLFeatureExtractor.vectorFromObject(features),
            featureNames: CANONICAL_FEATURE_NAMES,
            optimalParameters,
            optimalPriorityWeights,
            optimalAction: marlAction || {
                rotation: 0,
                thrust: features.threatLevel > 0.5 ? 1 : 0,
                fire: features.nearestEnemyDistance < 0.5 ? 1 : 0,
                shield: features.health < 0.4 ? 1 : 0
            },
            strategicDecision,
            shipState: {
                x: ship.x,
                y: ship.y,
                angle: ship.angle,
                velocityX: ship.velocityX,
                velocityY: ship.velocityY,
                health: ship.health,
                phase: ship.aiPhase,
                role: ship.role,
                isAlpha: ship.isAlpha
            }
        };

        this.memoryBuffer.push(sample);
        this.sampleCount++;

        if (this.memoryBuffer.length > this.maxMemoryBuffer) {
            this.memoryBuffer.shift();
        }

        if (this.sampleCount - this.lastAutoSaveCount >= this.autoSaveInterval) {
            this.flushToIndexedDB();
        }
    }

    async flushToIndexedDB() {
        if (!this.db || this.memoryBuffer.length === 0) return;

        const batch = this.memoryBuffer.splice(0, this.memoryBuffer.length);
        const tx = this.db.transaction(this.storeName, 'readwrite');
        const store = tx.objectStore(this.storeName);

        batch.forEach(sample => store.add(sample));
        this.lastAutoSaveCount = this.sampleCount;

        return new Promise((resolve, reject) => {
            tx.oncomplete = () => {
                console.log(`[DataCollector] Saved ${batch.length} samples to IndexedDB (total: ${this.sampleCount})`);
                resolve(batch.length);
            };
            tx.onerror = () => reject(tx.error);
        });
    }

    async getAllSamples() {
        const samples = [...this.memoryBuffer];
        if (!this.db) return samples;

        return new Promise((resolve, reject) => {
            const tx = this.db.transaction(this.storeName, 'readonly');
            const store = tx.objectStore(this.storeName);
            const request = store.getAll();
            request.onsuccess = () => resolve(samples.concat(request.result || []));
            request.onerror = () => reject(request.error);
        });
    }

    async exportJSON() {
        await this.flushToIndexedDB();
        const samples = await this.getAllSamples();
        // v3.0 adds episodeId / stepIndex / done / seed to every sample.
        // The preprocessor rejects v2.0 exports because they cannot be segmented
        // into episodes.
        const payload = {
            version: '3.0',
            exportedAt: new Date().toISOString(),
            featureDim: CANONICAL_FEATURE_DIM,
            featureNames: CANONICAL_FEATURE_NAMES,
            sampleCount: samples.length,
            episodeCount: new Set(samples.map(s => s.episodeId).filter(Boolean)).size,
            samples
        };
        return JSON.stringify(payload, null, 2);
    }

    downloadExport(filename = null) {
        return this.exportJSON().then(data => {
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename || `marl_training_data_${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            console.log('[DataCollector] Export downloaded');
            return data;
        });
    }

    async clear() {
        this.memoryBuffer = [];
        this.sampleCount = 0;
        if (!this.db) return;
        const tx = this.db.transaction(this.storeName, 'readwrite');
        tx.objectStore(this.storeName).clear();
    }

    getStats() {
        return {
            collecting: this.collecting,
            memoryBuffer: this.memoryBuffer.length,
            totalSamples: this.sampleCount
        };
    }
}

let marlDataCollector = null;

function initializeMARLDataCollector(options = {}) {
    if (!marlDataCollector) {
        marlDataCollector = new MARLDataCollector(options);
        marlDataCollector.init().catch(err => console.warn('[DataCollector] Init failed:', err));
    }
    return marlDataCollector;
}

function setMARLDataCollection(enabled) {
    const collector = initializeMARLDataCollector();
    collector.setCollecting(enabled);
    if (typeof window !== 'undefined') {
        window.marlDataCollectionEnabled = enabled;
    }
}

async function exportMARLTrainingData() {
    const collector = initializeMARLDataCollector();
    return collector.downloadExport();
}

function getMARLDataCollectorStats() {
    const collector = marlDataCollector || initializeMARLDataCollector();
    return collector.getStats();
}

if (typeof window !== 'undefined') {
    window.MARLDataCollector = MARLDataCollector;
    window.initializeMARLDataCollector = initializeMARLDataCollector;
    window.setMARLDataCollection = setMARLDataCollection;
    window.exportMARLTrainingData = exportMARLTrainingData;
    window.getMARLDataCollectorStats = getMARLDataCollectorStats;
}
