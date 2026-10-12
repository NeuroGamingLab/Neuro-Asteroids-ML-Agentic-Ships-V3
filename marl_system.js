// Multi-Agent Reinforcement Learning System - IN-BROWSER SCAFFOLDING ONLY.
//
// IMPORTANT: despite the naming, this file does not implement MADDPG and does
// not learn. trainAgent() below has always been a console.log; the actor,
// critic and target networks it allocates are never updated by gradient
// descent, and trainEpisode() has no callers. Enabling "MARL training mode" in
// the UI collects experience into a replay buffer that nothing consumes.
//
// Two further problems with the original design:
//   1. MADDPG is built on DDPG, whose deterministic policy gradient
//      differentiates the critic with respect to the action. That requires a
//      continuous action space. This game's actions are discrete, so MADDPG is
//      the wrong algorithm here without a Gumbel-Softmax relaxation.
//   2. The centralized critic that CTDE depends on was allocated and then never
//      referenced. It has been removed rather than left as evidence of a
//      training path that does not exist.
//
// The working implementation lives in training/ and runs headless:
//   training/ppo.js   - PPO with GAE, clipped surrogate, entropy bonus
//   training/nn.js    - dense network with gradient-checked backpropagation
//   sim/env.js        - Gym-style environment over the real game loop
//   training/train.js - entry point:  jsc training/train.js -- --stage 0
//
// This file is kept because the UI and app.py reference it, and because its
// inference path (getActionsForInference) is still used to drive ships from a
// loaded policy. Treat it as inference glue, not as a trainer.

class MARLSystem {
    constructor(config = {}) {
        this.numAgents = config.numAgents || 3;
        this.training = config.training !== false;
        this.modelPath = config.modelPath || null;
        
        // Environment
        this.env = new MARLEnvironment(this.numAgents);
        
        // Training parameters
        this.learningRate = config.learningRate || 0.001;
        this.gamma = config.gamma || 0.99; // Discount factor
        this.tau = config.tau || 0.01; // Soft update parameter
        this.batchSize = config.batchSize || 64;
        this.bufferSize = config.bufferSize || 10000;
        
        // Experience replay buffer
        this.replayBuffer = [];
        
        // Neural networks (simplified - using simple feedforward networks)
        // In production, would use TensorFlow.js for proper neural networks
        this.actors = []; // Policy networks
        this.critics = []; // Value networks
        this.targetActors = []; // Target policy networks
        this.targetCritics = []; // Target value networks

        // Must be set before initializeNetworks() below, which reads it to size
        // each agent's input layer -- it used to run in the other order, so
        // this was always undefined when the networks were actually built,
        // silently producing zero-row weight matrices (randomWeights(undefined,
        // cols) loops zero times) that threw "Cannot read properties of
        // undefined (reading 'length')" the moment forward() tried to use them.
        this.observationSize = (typeof CANONICAL_FEATURE_DIM !== 'undefined') ? CANONICAL_FEATURE_DIM : 30;

        // Initialize networks for each agent
        this.initializeNetworks();

        // Training statistics
        this.episodeCount = 0;
        this.totalRewards = [];
        this.trainingHistory = [];
        this.curriculumStage = 0;
        
        // Action space
        this.actionSpace = {
            rotation: [-1, 0, 1], // Left, none, right
            thrust: [0, 1],       // No thrust, thrust
            fire: [0, 1],         // Don't fire, fire
            shield: [0, 1]        // Don't activate, activate
        };
    }
    
    // Initialize neural networks (simplified version)
    // In production, use TensorFlow.js for proper implementation
    initializeNetworks() {
        const obsSize = this.observationSize;

        for (let i = 0; i < this.numAgents; i++) {
            this.actors[i] = this.createSimpleNetwork(obsSize, [64, 32], 7);
            this.critics[i] = this.createSimpleNetwork(obsSize + 4, [64, 32], 1);
            this.targetActors[i] = this.createSimpleNetwork(obsSize, [64, 32], 7);
            this.targetCritics[i] = this.createSimpleNetwork(obsSize + 4, [64, 32], 1);
            this.copyWeights(this.actors[i], this.targetActors[i]);
            this.copyWeights(this.critics[i], this.targetCritics[i]);
        }
    }

    setCurriculumStage(stage) {
        this.curriculumStage = stage;
        if (this.env) this.env.setCurriculumStage(stage);
    }
    
    // Create a simple feedforward network (placeholder)
    // In production, replace with TensorFlow.js models
    createSimpleNetwork(inputSize, hiddenSizes, outputSize) {
        const network = {
            layers: [],
            inputSize,
            outputSize
        };
        
        // Input layer
        let prevSize = inputSize;
        hiddenSizes.forEach(hiddenSize => {
            network.layers.push({
                weights: this.randomWeights(prevSize, hiddenSize),
                biases: new Array(hiddenSize).fill(0)
            });
            prevSize = hiddenSize;
        });
        
        // Output layer
        network.layers.push({
            weights: this.randomWeights(prevSize, outputSize),
            biases: new Array(outputSize).fill(0)
        });
        
        return network;
    }
    
    // Random weight initialization
    randomWeights(rows, cols) {
        const weights = [];
        const limit = Math.sqrt(6.0 / (rows + cols));
        for (let i = 0; i < rows; i++) {
            weights[i] = [];
            for (let j = 0; j < cols; j++) {
                weights[i][j] = (gameRandom() * 2 - 1) * limit;
            }
        }
        return weights;
    }
    
    // Copy weights from source to target network
    copyWeights(source, target) {
        for (let i = 0; i < source.layers.length; i++) {
            target.layers[i] = {
                weights: source.layers[i].weights.map(row => [...row]),
                biases: [...source.layers[i].biases]
            };
        }
    }
    
    // Forward pass through network
    forward(network, input) {
        let output = input;
        
        for (let i = 0; i < network.layers.length; i++) {
            const layer = network.layers[i];
            const newOutput = [];
            
            for (let j = 0; j < layer.weights[0].length; j++) {
                let sum = layer.biases[j];
                for (let k = 0; k < output.length; k++) {
                    sum += output[k] * layer.weights[k][j];
                }
                // ReLU activation for hidden layers, sigmoid for output
                if (i < network.layers.length - 1) {
                    newOutput[j] = Math.max(0, sum); // ReLU
                } else {
                    newOutput[j] = 1 / (1 + Math.exp(-sum)); // Sigmoid
                }
            }
            
            output = newOutput;
        }
        
        return output;
    }
    
    // Get action from policy network (with exploration)
    getAction(agentIndex, observation, epsilon = 0.1) {
        if (!observation || observation.length === 0) {
            return this.getRandomAction();
        }

        // Pad/truncate observation to expected size
        if (observation.length < this.observationSize) {
            observation = [...observation, ...new Array(this.observationSize - observation.length).fill(0)];
        } else if (observation.length > this.observationSize) {
            observation = observation.slice(0, this.observationSize);
        }
        
        // Epsilon-greedy exploration
        if (gameRandom() < epsilon && this.training) {
            return this.getRandomAction();
        }
        
        // Get action probabilities from actor network
        const actionProbs = this.forward(this.actors[agentIndex], observation);
        
        // Convert probabilities to discrete actions
        const action = {
            rotation: this.sampleDiscrete([-1, 0, 1], [actionProbs[0] || 0.33, actionProbs[1] || 0.34, actionProbs[2] || 0.33]),
            thrust: (actionProbs[3] || 0) > 0.5 ? 1 : 0,
            fire: (actionProbs[4] || 0) > 0.5 ? 1 : 0,
            shield: (actionProbs[5] || 0) > 0.5 ? 1 : 0,
            communication: actionProbs[6] || 0
        };
        
        return action;
    }
    
    // Get random action for exploration
    getRandomAction() {
        return {
            rotation: this.actionSpace.rotation[Math.floor(gameRandom() * 3)],
            thrust: gameRandom() > 0.5 ? 1 : 0,
            fire: gameRandom() > 0.1 ? 1 : 0, // Fire less frequently
            shield: gameRandom() > 0.95 ? 1 : 0, // Shield rarely
            communication: gameRandom()
        };
    }
    
    // Sample from discrete distribution
    sampleDiscrete(values, probabilities) {
        const rand = gameRandom();
        let sum = 0;
        for (let i = 0; i < probabilities.length; i++) {
            sum += probabilities[i];
            if (rand < sum) {
                return values[i];
            }
        }
        return values[values.length - 1];
    }
    
    // Store experience in replay buffer
    storeExperience(observations, actions, rewards, nextObservations, done) {
        const experience = {
            observations,
            actions,
            rewards,
            nextObservations,
            done
        };
        
        this.replayBuffer.push(experience);
        
        // Limit buffer size
        if (this.replayBuffer.length > this.bufferSize) {
            this.replayBuffer.shift();
        }
    }
    
    // Train on a batch of experiences
    trainStep() {
        if (this.replayBuffer.length < this.batchSize) {
            return; // Not enough experiences
        }
        
        // Sample random batch
        const batch = this.sampleBatch(this.batchSize);
        
        // Train each agent
        for (let agentIndex = 0; agentIndex < this.numAgents; agentIndex++) {
            this.trainAgent(agentIndex, batch);
        }
        
        // Soft update target networks
        this.softUpdateTargets();
    }
    
    // Sample random batch from replay buffer
    sampleBatch(batchSize) {
        const batch = [];
        for (let i = 0; i < batchSize; i++) {
            const randomIndex = Math.floor(gameRandom() * this.replayBuffer.length);
            batch.push(this.replayBuffer[randomIndex]);
        }
        return batch;
    }
    
    // NOT IMPLEMENTED. This does not update any weights.
    //
    // It is left in place so the call sites keep working, but it warns once
    // rather than logging every batch, because silently doing nothing while
    // printing "Training agent N" is how this was mistaken for a trainer.
    // Use training/train.js for actual learning.
    trainAgent(agentIndex, batch) {
        if (!MARLSystem._warnedNotTraining) {
            MARLSystem._warnedNotTraining = true;
            console.warn(
                '[MARL] trainAgent() is a no-op: this file does not implement learning. ' +
                'No weights are being updated. Train with: jsc training/train.js -- --stage 0'
            );
        }
    }
    
    // Soft update target networks
    softUpdateTargets() {
        for (let i = 0; i < this.numAgents; i++) {
            // Soft update: target = tau * main + (1 - tau) * target
            this.softUpdateNetwork(this.actors[i], this.targetActors[i], this.tau);
            this.softUpdateNetwork(this.critics[i], this.targetCritics[i], this.tau);
        }
    }
    
    // Soft update a single network
    softUpdateNetwork(source, target, tau) {
        for (let i = 0; i < source.layers.length; i++) {
            for (let j = 0; j < source.layers[i].weights.length; j++) {
                for (let k = 0; k < source.layers[i].weights[j].length; k++) {
                    target.layers[i].weights[j][k] = 
                        tau * source.layers[i].weights[j][k] + 
                        (1 - tau) * target.layers[i].weights[j][k];
                }
            }
            for (let j = 0; j < source.layers[i].biases.length; j++) {
                target.layers[i].biases[j] = 
                    tau * source.layers[i].biases[j] + 
                    (1 - tau) * target.layers[i].biases[j];
            }
        }
    }
    
    // Train for one episode
    trainEpisode(aiShips, asteroids, enemyShips, playerShip) {
        // Initialize environment
        this.env.initialize(aiShips, asteroids, enemyShips, playerShip);
        let observations = this.env.reset();
        let episodeReward = 0;
        let done = false;
        let step = 0;
        
        while (!done && step < this.env.maxEpisodeSteps) {
            // Get actions for all agents
            const actions = [];
            for (let i = 0; i < this.numAgents; i++) {
                if (observations[i]) {
                    // Epsilon decreases over time
                    const epsilon = Math.max(0.01, 0.1 * (1 - this.episodeCount / 1000));
                    actions.push(this.getAction(i, observations[i], epsilon));
                } else {
                    actions.push(this.getRandomAction());
                }
            }
            
            // Execute actions (will be handled by game loop)
            // For now, we'll simulate the step
            
            // Get next observations and rewards
            const stepResult = this.env.step(actions);
            const nextObservations = stepResult.observations;
            const rewards = stepResult.rewards;
            done = stepResult.done;
            
            // Store experience
            this.storeExperience(observations, actions, rewards, nextObservations, done);
            
            // Train on batch
            if (this.replayBuffer.length >= this.batchSize) {
                this.trainStep();
            }
            
            // Update for next step
            observations = nextObservations;
            episodeReward += rewards.reduce((a, b) => a + b, 0);
            step++;
        }
        
        // Update statistics
        this.episodeCount++;
        this.totalRewards.push(episodeReward);
        const stats = this.env.getEpisodeStats();
        
        this.trainingHistory.push({
            episode: this.episodeCount,
            totalReward: episodeReward,
            averageReward: episodeReward / this.numAgents,
            steps: step,
            ...stats
        });
        
        // Log progress
        if (this.episodeCount % 10 === 0) {
            const avgReward = this.totalRewards.slice(-10).reduce((a, b) => a + b, 0) / 10;
            console.log(`[MARL] Episode ${this.episodeCount}: Avg Reward = ${avgReward.toFixed(2)}`);
        }
        
        return episodeReward;
    }
    
    // Get actions for inference (no exploration)
    getActionsForInference(observations) {
        const actions = [];
        for (let i = 0; i < this.numAgents; i++) {
            if (observations[i]) {
                actions.push(this.getAction(i, observations[i], 0)); // No exploration
            } else {
                actions.push(this.getRandomAction());
            }
        }
        return actions;
    }
    
    // Save model (simplified - would serialize network weights)
    saveModel(path) {
        const modelData = {
            actors: this.actors,
            critics: this.critics,
            episodeCount: this.episodeCount,
            trainingHistory: this.trainingHistory
        };
        
        // In production, would save to file or localStorage
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem('marl_model', JSON.stringify(modelData));
            console.log('[MARL] Model saved to localStorage');
        }
    }
    
    // Load model
    loadModel(path) {
        if (typeof localStorage !== 'undefined') {
            const modelData = localStorage.getItem('marl_model');
            if (modelData) {
                const data = JSON.parse(modelData);
                this.actors = data.actors;
                this.critics = data.critics;
                this.episodeCount = data.episodeCount || 0;
                this.trainingHistory = data.trainingHistory || [];
                
                // Update target networks
                for (let i = 0; i < this.numAgents; i++) {
                    this.copyWeights(this.actors[i], this.targetActors[i]);
                    this.copyWeights(this.critics[i], this.targetCritics[i]);
                }
                
                console.log('[MARL] Model loaded from localStorage');
                return true;
            }
        }
        return false;
    }
    
    // Get training statistics
    getStats() {
        if (this.totalRewards.length === 0) {
            return {
                episodes: 0,
                averageReward: 0,
                bestReward: 0
            };
        }
        
        const avgReward = this.totalRewards.reduce((a, b) => a + b, 0) / this.totalRewards.length;
        const bestReward = Math.max(...this.totalRewards);
        
        return {
            episodes: this.episodeCount,
            averageReward: avgReward,
            bestReward: bestReward,
            recentAverage: this.totalRewards.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, this.totalRewards.length)
        };
    }

    // Value Decomposition Network (VDN) joint value projection and counterfactual credit
    decomposeJointValue(observations) {
        if (!Array.isArray(observations) || observations.length === 0) {
            return { teamValue: 0, individualValues: [], counterfactualCredits: [] };
        }
        const values = [];
        for (let i = 0; i < observations.length; i++) {
            const obs = observations[i];
            if (!obs) {
                values.push(0);
                continue;
            }
            if (this.critics[i]) {
                const dummyAction = [0, 0, 0, 0];
                const input = obs.concat(dummyAction);
                const out = this.forwardNetwork(this.critics[i], input);
                values.push(out[0] || 0);
            } else {
                values.push(0);
            }
        }
        const teamValue = values.reduce((acc, v) => acc + v, 0);
        const activeCount = values.filter(v => v !== 0).length || 1;
        const meanVal = teamValue / activeCount;
        const counterfactualCredits = values.map(v => v !== 0 ? v - meanVal : 0);
        return {
            teamValue,
            individualValues: values,
            counterfactualCredits
        };
    }

    // Generate real-time MARL Swarm telemetry for mission control
    getMARLTelemetry(aiShips, observations) {
        let runner = null;
        if (typeof window !== 'undefined' && typeof window.getPolicyRunner === 'function') {
            runner = window.getPolicyRunner();
        }

        let teamValue = 0;
        let credits = [];
        let swarmEntropy = 0.5;

        // If trained PPO policy is running, utilize its verified autograd critic for exact decomposition
        if (runner && runner.enabled && runner.policy && typeof runner.policy.decomposeTeamValue === 'function' && observations) {
            const decomp = runner.policy.decomposeTeamValue(observations);
            teamValue = decomp.teamValue;
            credits = decomp.counterfactualCredits;
            if (typeof runner.policy.computeSwarmMetrics === 'function') {
                const metrics = runner.policy.computeSwarmMetrics(observations);
                swarmEntropy = metrics.meanEntropy;
            }
        } else if (observations && observations.length) {
            const decomp = this.decomposeJointValue(observations);
            teamValue = decomp.teamValue;
            credits = decomp.counterfactualCredits;
        }

        // Swarm spatial coordination index (distance from ideal formation slot)
        let coherenceSum = 0;
        let activeShips = 0;
        if (Array.isArray(aiShips)) {
            aiShips.forEach(s => {
                if (!s || s.health <= 0) return;
                activeShips++;
                if (s.formationOffset) {
                    const offsetMag = Math.hypot(s.formationOffset.x || 0, s.formationOffset.y || 0);
                    coherenceSum += Math.max(0, 1 - (offsetMag / 300));
                } else {
                    coherenceSum += 0.8;
                }
            });
        }
        const coordinationIndex = activeShips > 0 ? (coherenceSum / activeShips) : 1.0;

        const creditBreakdown = [];
        if (Array.isArray(aiShips)) {
            aiShips.forEach((s, idx) => {
                if (!s) return;
                creditBreakdown.push({
                    agentId: s.agentId !== undefined ? s.agentId : idx,
                    role: s.role || 'scout',
                    isAlpha: !!s.isAlpha,
                    fuelPct: s.maxFuel ? Math.round((s.fuel / s.maxFuel) * 100) : 100,
                    creditScore: credits[idx] !== undefined ? credits[idx].toFixed(2) : '0.00'
                });
            });
        }

        return {
            algorithm: runner && runner.enabled ? 'MAPPO + VDN' : 'Cooperative Swarm VDN',
            teamValue: teamValue.toFixed(2),
            coordinationIndex: Math.round(coordinationIndex * 100),
            swarmEntropy: swarmEntropy.toFixed(3),
            credits: creditBreakdown
        };
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MARLSystem;
} else {
    window.MARLSystem = MARLSystem;
}

