// MARL Integration with Game
// Connects MARL system to the Asteroids game with CTDE, curriculum, and data collection

let marlSystem = null;
let marlEnabled = false;
let marlTrainingMode = false;
let marlCurriculumStage = 3;
let marlLastObservations = null;
let marlLastActions = null;
let marlPreviousScore = 0;

function initializeMARL(numAgents = 3, training = false) {
    if (marlSystem) {
        marlSystem.numAgents = numAgents;
        marlSystem.training = training;
        return;
    }

    marlSystem = new MARLSystem({
        numAgents: numAgents,
        training: training
    });

    marlTrainingMode = training;
    marlEnabled = true;
    marlSystem.setCurriculumStage(marlCurriculumStage);

    console.log(`[MARL] System initialized with ${numAgents} agents, training=${training}, curriculum=${marlCurriculumStage}`);

    if (!training) {
        marlSystem.loadModel();
    }
}

function setMARLMode(enabled, training = false) {
    marlEnabled = enabled;
    marlTrainingMode = training;

    if (typeof window !== 'undefined') {
        window.marlEnabled = enabled;
        window.marlTrainingMode = training;
    }

    if (marlSystem) {
        marlSystem.training = training;
    } else if (enabled && typeof aiShips !== 'undefined' && aiShips.length > 0) {
        initializeMARL(Math.min(aiShips.length, 5), training);
    }

    console.log(`[MARL] Mode set: enabled=${enabled}, training=${training}`);
}

function setMARLCurriculumStage(stage) {
    marlCurriculumStage = Math.max(0, Math.min(3, stage));
    if (typeof window !== 'undefined') {
        window.marlCurriculumStage = marlCurriculumStage;
    }
    if (marlSystem) {
        marlSystem.setCurriculumStage(marlCurriculumStage);
    }
    const names = ['Asteroids Only', 'Enemies', 'Boss Fight', 'Full Chaos'];
    console.log(`[MARL] Curriculum stage ${marlCurriculumStage}: ${names[marlCurriculumStage]}`);
}

function updateMARL() {
    if (!marlEnabled || !marlSystem || typeof aiShips === 'undefined' || aiShips.length === 0) {
        return null;
    }

    marlSystem.env.initialize(aiShips, asteroids, enemyShips, ship);
    const observations = marlSystem.env.getAllObservations();
    const actions = marlSystem.getActionsForInference(observations);

    marlLastObservations = observations;
    marlLastActions = actions;

    return actions;
}

function executeMARLAction(aiShip, action) {
    if (!action) return;

    if (action.rotation === -1) aiShip.rotate(-1);
    else if (action.rotation === 1) aiShip.rotate(1);

    if (action.thrust === 1) aiShip.thrust();

    if (action.fire === 1 && aiShip.shootCooldown <= 0) aiShip.fire();

    if (action.shield === 1 && aiShip.shieldCooldown <= 0 && aiShip.health < aiShip.maxHealth) {
        aiShip.shieldActive = true;
        aiShip.shieldDuration = 60;
        aiShip.shieldCooldown = 300;
    }
}

function integrateMARLWithAIShip(aiShip, agentIndex) {
    if (!marlEnabled || !marlSystem) return null;

    const observation = marlSystem.env.getObservation(agentIndex);
    if (!observation) return null;

    const epsilon = marlTrainingMode ? Math.max(0.01, 0.1 * (1 - marlSystem.episodeCount / 1000)) : 0;
    return marlSystem.getAction(agentIndex, observation, epsilon);
}

function storeMARLExperience(previousScore, currentScore) {
    if (!marlEnabled || !marlTrainingMode || !marlSystem) return;

    const observations = marlLastObservations || marlSystem.env.getAllObservations();
    const actions = marlLastActions || [];
    const rewards = marlSystem.env.calculateRewards(previousScore, currentScore);
    const nextObservations = marlSystem.env.getAllObservations();
    const done = marlSystem.env.isDone();

    if (observations && actions.length > 0) {
        marlSystem.storeExperience(observations, actions, rewards, nextObservations, done);
    }

    // Collect training data for NN pipeline
    if (typeof initializeMARLDataCollector === 'function') {
        const collector = initializeMARLDataCollector();
        if (collector.collecting) {
            aiShips.forEach((aiShip, index) => {
                if (!aiShip) return;
                const features = typeof MLFeatureExtractor !== 'undefined'
                    ? MLFeatureExtractor.extract(aiShip, asteroids, aiShips, ship, enemyShips, enemyBullets, { curriculumStage: marlCurriculumStage })
                    : null;
                if (!features) return;

                collector.collectSample({
                    agentIndex: index,
                    ship: aiShip,
                    features,
                    marlObservation: observations[index],
                    marlAction: actions[index],
                    reward: rewards[index] || 0,
                    score: currentScore,
                    curriculumStage: marlCurriculumStage,
                    labelSource: 'marl',
                    strategicDecision: typeof initializeMLHierarchy === 'function'
                        ? initializeMLHierarchy().getState()
                        : null
                });
            });
        }
    }

    marlPreviousScore = currentScore;
}

function trainMARL() {
    if (!marlEnabled || !marlTrainingMode || !marlSystem) return;

    if (marlSystem.replayBuffer.length >= marlSystem.batchSize) {
        marlSystem.trainStep();
    }
}

function getMARLStats() {
    if (!marlSystem) return null;
    return {
        episodeCount: marlSystem.episodeCount,
        bufferSize: marlSystem.replayBuffer.length,
        curriculumStage: marlCurriculumStage,
        envStats: marlSystem.env ? marlSystem.env.getEpisodeStats() : null
    };
}

window.initializeMARL = initializeMARL;
window.setMARLMode = setMARLMode;
window.setMARLCurriculumStage = setMARLCurriculumStage;
window.updateMARL = updateMARL;
window.executeMARLAction = executeMARLAction;
window.integrateMARLWithAIShip = integrateMARLWithAIShip;
window.storeMARLExperience = storeMARLExperience;
window.trainMARL = trainMARL;
window.getMARLStats = getMARLStats;

let marlInitialized = false;
function autoInitializeMARL() {
    if (marlInitialized || !marlEnabled) return;

    if (typeof aiShips !== 'undefined' && aiShips.length > 0) {
        const numAgents = Math.min(aiShips.length, 5);
        initializeMARL(numAgents, marlTrainingMode);
        marlInitialized = true;
    }
}

setTimeout(autoInitializeMARL, 1000);
