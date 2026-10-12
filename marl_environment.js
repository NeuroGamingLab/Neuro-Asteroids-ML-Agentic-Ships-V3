// Multi-Agent Reinforcement Learning Environment Wrapper
// For cooperative AI ship behaviors

class MARLEnvironment {
    constructor(numAgents = 3) {
        this.numAgents = numAgents;
        this.agents = [];
        this.episodeStep = 0;
        this.maxEpisodeSteps = 1000;
        this.episodeRewards = [];
        this.curriculumStage = 0; // 0=asteroids, 1=enemies, 2=boss, 3=full
        this.previousKills = 0;
        this.friendlyFireCount = 0;
        
        // State normalization parameters
        this.stateBounds = {
            position: { min: -1000, max: 1000 },
            velocity: { min: -10, max: 10 },
            angle: { min: 0, max: Math.PI * 2 },
            distance: { min: 0, max: 2000 },
            health: { min: 0, max: 3 }
        };
    }
    
    // Initialize environment with agents
    initialize(aiShips, asteroids, enemyShips, playerShip) {
        this.aiShips = aiShips;
        this.asteroids = asteroids;
        this.enemyShips = enemyShips;
        this.playerShip = playerShip;
        this.episodeStep = 0;
        this.episodeRewards = new Array(this.numAgents).fill(0);
        
        // Store initial states for reward calculation
        this.previousStates = this.aiShips.map(ship => ({
            health: ship.health,
            position: { x: ship.x, y: ship.y }
        }));
    }
    
    setCurriculumStage(stage) {
        this.curriculumStage = Math.max(0, Math.min(3, stage));
    }

    getCurriculumConfig() {
        const configs = [
            { name: 'asteroids', enemyWeight: 0, bossWeight: 0, asteroidWeight: 1.5 },
            { name: 'enemies', enemyWeight: 1.5, bossWeight: 0, asteroidWeight: 0.5 },
            { name: 'boss', enemyWeight: 1.0, bossWeight: 2.0, asteroidWeight: 0.3 },
            { name: 'full', enemyWeight: 1.2, bossWeight: 1.5, asteroidWeight: 1.0 }
        ];
        return configs[this.curriculumStage] || configs[3];
    }

    // Get observation for a specific agent (canonical 30-dim when MLFeatureExtractor available)
    getObservation(agentIndex) {
        if (!this.aiShips || agentIndex >= this.aiShips.length) {
            return null;
        }
        
        const agent = this.aiShips[agentIndex];
        if (!agent) return null;

        if (typeof MLFeatureExtractor !== 'undefined') {
            const features = MLFeatureExtractor.extract(
                agent,
                this.asteroids,
                this.aiShips,
                this.playerShip,
                this.enemyShips,
                typeof enemyBullets !== 'undefined' ? enemyBullets : [],
                { curriculumStage: this.curriculumStage }
            );
            return MLFeatureExtractor.vectorFromObject(features);
        }
        
        const normalize = (value, min, max) => (value - min) / (max - min);
        const nearestAsteroid = this.findNearest(agent, this.asteroids);
        const nearestEnemy = this.findNearest(agent, this.enemyShips);
        const nearestAlly = this.findNearestAlly(agent, this.aiShips, agentIndex);
        const totalAllyHealth = this.aiShips
            .filter((s, i) => i !== agentIndex && s)
            .reduce((sum, s) => sum + (s.health || 0), 0);
        const allyCount = Math.max(1, this.aiShips.length - 1);

        return [
            normalize(agent.x, this.stateBounds.position.min, this.stateBounds.position.max) * 2 - 1,
            normalize(agent.y, this.stateBounds.position.min, this.stateBounds.position.max) * 2 - 1,
            normalize(agent.angle, 0, Math.PI * 2),
            normalize(agent.velocityX, this.stateBounds.velocity.min, this.stateBounds.velocity.max),
            normalize(agent.velocityY, this.stateBounds.velocity.min, this.stateBounds.velocity.max),
            normalize(agent.health, 0, 3),
            (agent.aiPhase - 1) / 2,
            agent.shieldActive ? 1 : 0,
            nearestAsteroid ? normalize(nearestAsteroid.distance, 0, 200) : 1.0,
            nearestAsteroid ? normalize(nearestAsteroid.angle, 0, Math.PI * 2) : 0,
            nearestAsteroid ? normalize(nearestAsteroid.size || 20, 0, 50) : 0,
            nearestEnemy ? normalize(nearestEnemy.distance, 0, 400) : 1.0,
            nearestEnemy ? normalize(nearestEnemy.angle, 0, Math.PI * 2) : 0,
            nearestEnemy ? normalize(nearestEnemy.health || 1, 0, 5) : 0,
            nearestAlly ? normalize(nearestAlly.distance, 0, 300) : 1.0,
            nearestAlly ? normalize(nearestAlly.angle, 0, Math.PI * 2) : 0,
            nearestAlly ? normalize(nearestAlly.health || 3, 0, 3) : 1.0,
            Math.min(this.asteroids.length / 10, 1.0),
            Math.min(this.enemyShips.length / 5, 1.0),
            Math.min((this.aiShips.length - 1) / 5, 1.0),
            normalize(totalAllyHealth / allyCount, 0, 3)
        ];
    }
    
    // Get observations for all agents
    getAllObservations() {
        const observations = [];
        for (let i = 0; i < this.numAgents; i++) {
            observations.push(this.getObservation(i));
        }
        return observations;
    }
    
    // Find nearest entity to agent
    findNearest(agent, entities) {
        let nearest = null;
        let nearestDistance = Infinity;
        
        entities.forEach(entity => {
            if (!entity) return;
            const dx = entity.x - agent.x;
            const dy = entity.y - agent.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < nearestDistance) {
                nearestDistance = distance;
                const angle = Math.atan2(dy, dx);
                nearest = { distance, angle, ...entity };
            }
        });
        
        return nearest;
    }
    
    // Find nearest ally (excluding self)
    findNearestAlly(agent, allies, agentIndex) {
        let nearest = null;
        let nearestDistance = Infinity;
        
        allies.forEach((ally, index) => {
            if (index === agentIndex || !ally) return;
            const dx = ally.x - agent.x;
            const dy = ally.y - agent.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < nearestDistance) {
                nearestDistance = distance;
                const angle = Math.atan2(dy, dx);
                nearest = { distance, angle, health: ally.health };
            }
        });
        
        return nearest;
    }
    
    // Calculate rewards for all agents (cooperative, curriculum-aware)
    calculateRewards(previousScore, currentScore) {
        const rewards = [];
        const curriculum = this.getCurriculumConfig();
        const scoreDiff = currentScore - previousScore;
        const killsGained = Math.max(0, Math.floor(scoreDiff / 10));
        
        for (let i = 0; i < this.numAgents; i++) {
            if (!this.aiShips[i]) {
                rewards.push(0);
                continue;
            }
            
            const agent = this.aiShips[i];
            const prevState = this.previousStates[i];
            let reward = 0;
            
            reward += 0.1; // Survival
            
            if (agent.health < prevState.health) {
                reward -= 10 * curriculum.enemyWeight;
            }
            if (agent.health <= 0) {
                reward -= 100;
            }
            
            const nearbyAllies = this.countNearbyAllies(agent, i, 100);
            if (nearbyAllies > 0) {
                reward += 0.5 * Math.min(nearbyAllies, 2);
            }
            
            const lowHealthAlly = this.findLowHealthAlly(agent, i);
            if (lowHealthAlly && this.isProtecting(agent, lowHealthAlly)) {
                reward += 5 * curriculum.enemyWeight;
            }
            
            reward += 0.2 * Math.min(nearbyAllies, 3);

            // Formation cohesion reward
            if (typeof MLFeatureExtractor !== 'undefined') {
                const features = MLFeatureExtractor.extract(
                    agent, this.asteroids, this.aiShips, this.playerShip, this.enemyShips
                );
                reward += features.formationCohesion * 0.5;
                if (features.bulletThreat > 0 && agent.shieldActive) {
                    reward += 1.0;
                }
            }

            // Idle thrust near threats penalty
            const speed = Math.sqrt(agent.velocityX ** 2 + agent.velocityY ** 2);
            if (speed < 0.1 && this.enemyShips.length > 0) {
                reward -= 0.2;
            }
            
            rewards.push(reward);
            this.episodeRewards[i] += reward;
        }
        
        if (scoreDiff > 0) {
            const sharedReward = scoreDiff * 2 * curriculum.asteroidWeight;
            rewards.forEach((r, i) => {
                rewards[i] = r + sharedReward / this.numAgents;
            });
        }

        if (killsGained > 0) {
            const killReward = killsGained * 3 * curriculum.enemyWeight;
            rewards.forEach((r, i) => {
                rewards[i] = r + killReward / this.numAgents;
            });
        }

        this.previousKills += killsGained;
        
        // Update previous states
        this.previousStates = this.aiShips.map(ship => ({
            health: ship.health,
            position: { x: ship.x, y: ship.y }
        }));
        
        return rewards;
    }
    
    // Count nearby allies
    countNearbyAllies(agent, agentIndex, radius) {
        let count = 0;
        this.aiShips.forEach((ally, index) => {
            if (index === agentIndex || !ally) return;
            const dx = ally.x - agent.x;
            const dy = ally.y - agent.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < radius) count++;
        });
        return count;
    }
    
    // Find low-health ally
    findLowHealthAlly(agent, agentIndex) {
        let lowHealthAlly = null;
        let minHealth = 3;
        
        this.aiShips.forEach((ally, index) => {
            if (index === agentIndex || !ally) return;
            if (ally.health < minHealth && ally.health > 0) {
                minHealth = ally.health;
                lowHealthAlly = ally;
            }
        });
        
        return lowHealthAlly;
    }
    
    // Check if agent is protecting an ally
    isProtecting(agent, ally) {
        const dx = ally.x - agent.x;
        const dy = ally.y - agent.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Check if agent is between ally and nearest threat
        const nearestEnemy = this.findNearest(ally, this.enemyShips);
        if (!nearestEnemy) return false;
        
        // Agent is protecting if close to ally and facing threat direction
        return distance < 80 && distance > 20;
    }
    
    // Check if episode is done
    isDone() {
        // Episode ends if:
        // 1. Max steps reached
        if (this.episodeStep >= this.maxEpisodeSteps) return true;
        
        // 2. All agents dead
        const aliveAgents = this.aiShips.filter(ship => ship && ship.health > 0).length;
        if (aliveAgents === 0) return true;
        
        return false;
    }
    
    // Step environment
    step(actions) {
        this.episodeStep++;
        
        // Execute actions (will be handled by game loop)
        // Actions format: [{rotation, thrust, fire, shield, communication}, ...]
        
        return {
            observations: this.getAllObservations(),
            rewards: this.calculateRewards(0, 0), // Score tracking will be added
            done: this.isDone(),
            info: {
                step: this.episodeStep,
                aliveAgents: this.aiShips.filter(s => s && s.health > 0).length
            }
        };
    }
    
    // Reset environment
    reset() {
        this.episodeStep = 0;
        this.episodeRewards = new Array(this.numAgents).fill(0);
        this.previousStates = [];
        
        return this.getAllObservations();
    }
    
    // Get episode statistics
    getEpisodeStats() {
        return {
            totalReward: this.episodeRewards.reduce((a, b) => a + b, 0),
            averageReward: this.episodeRewards.reduce((a, b) => a + b, 0) / this.numAgents,
            steps: this.episodeStep,
            aliveAgents: this.aiShips.filter(s => s && s.health > 0).length
        };
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MARLEnvironment;
} else {
    window.MARLEnvironment = MARLEnvironment;
}

