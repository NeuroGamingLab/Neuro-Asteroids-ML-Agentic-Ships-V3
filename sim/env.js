// Gym-style environment wrapper around the real game.js update loop.
//
// reset(seed) -> observations
// step(actions) -> { observations, rewards, dones, done, info }
//
// Agents are addressed by a stable `agentId`, never by array position: dead ships
// are spliced out of `aiShips`, so position is not an identity.

(function (root) {
    'use strict';

    // Curriculum stages match marl_environment.js so browser and headless agree.
    const CURRICULUM = [
        { name: 'asteroids', enemies: 0, bosses: 0, enemyWeight: 0.0, bossWeight: 0.0, asteroidWeight: 1.5 },
        { name: 'enemies', enemies: 2, bosses: 0, enemyWeight: 1.5, bossWeight: 0.0, asteroidWeight: 0.5 },
        { name: 'boss', enemies: 1, bosses: 1, enemyWeight: 1.0, bossWeight: 2.0, asteroidWeight: 0.3 },
        { name: 'full', enemies: 2, bosses: 1, enemyWeight: 1.2, bossWeight: 1.5, asteroidWeight: 1.0 }
    ];

    // Reward weights.
    //
    // The original shaping terms could earn ~1700 per episode for sitting still
    // next to allies, while the entire score term was worth ~67. That makes
    // passive camping the optimal policy. Here the objective (kills) dominates and
    // shaping is capped at a few percent of a typical return.
    // Calibrated against the two reference policies on stage 0: the rule-based
    // controller scores ~1120 per episode and a random policy ~20, so a reward
    // function that measures the right thing must rank them far apart. The first
    // version of this table did not - random actually scored *higher* than the
    // heuristic, because survival and shaping swamped the kill term. Objective
    // terms now dominate and shaping is worth a few percent of a typical return.
    const DEFAULT_REWARD = {
        survival: 0.002,       // base per-step; survivalScaling adds a late-game ramp on top
        asteroidKill: 2.0,
        enemyKill: 5.0,
        bossKill: 15.0,
        damageTaken: -1.0,     // per health point
        death: -5.0,
        teamShare: 0.3,        // fraction of kill value pooled across the team
        allyProximity: 0.002,
        maxAllyProximity: 2,
        formationCohesion: 0.002,
        idlePenalty: -0.005,
        idleSpeedThreshold: 0.1,

        // Survival scaling: extra bonus that grows linearly with episode progress.
        // At step 0 it adds 0; at the final step it adds survivalScaling once more.
        // Total extra over a 1000-step episode = survivalScaling * 1000 / 2 = +1.5,
        // still far below a single asteroid kill (2.0) so kill-dominance is preserved.
        survivalScaling: 0.003,

        // Center-camping penalty: charged to the alpha when it hovers within
        // centerCampRadius px of world center AND speed is below the idle threshold
        // AND at least one threat is on the map. Forces the alpha to lead the swarm
        // into the action instead of sitting in the safe center-park position.
        // Weight is tiny (same order as idlePenalty) so it does not dominate.
        centerCampingPenalty: -0.003,
        centerCampRadius: 80,  // px from world center that triggers the penalty

        // Potential-based shaping on aim alignment, off by default.
        //
        // F(s,s') = gamma * PHI(s') - PHI(s) with PHI = aim alignment. Ng, Harada
        // and Russell (1999) show this form leaves the optimal policy unchanged -
        // it telescopes to a constant over any episode - so it densifies the
        // learning signal without changing what the agent is being asked to do.
        // That invariance is also why turning it on barely moves measured returns.
        //
        // Default 0 because the published stage-0 result was trained without it;
        // enable with --aimShaping to experiment. Whether it helps here has not
        // been isolated, since the runs that differed in shaping also differed in
        // entropy, rollout size and network width.
        aimShaping: 0.0,
        shapingGamma: 0.99,

        // Fuel rescue (only active when the env is built with fuelRescue: true).
        //
        // The alpha burns fuel to thrust and fire; below 35% it pings HELP and a
        // scripted green tanker flies in to dock. While docked neither ship can
        // move and an asteroid strike kills both. The terms below are sized so
        // that one refuel is worth about a boss kill's team share and a dock
        // death costs more than an ordinary death: the interesting behaviour to
        // learn is guarding the dock, not avoiding fuel use.
        refuelSuccess: 3.0,    // paid to every alive policy agent when a transfer completes
        dockDeath: -8.0,       // extra penalty to the alpha (and pooled to the team) on a linked kill
        lowFuel: -0.01         // per frame the alpha is pinging HELP, paid to the alpha
    };

    class AsteroidsEnv {
        constructor(options = {}) {
            this.numAgents = options.numAgents || 3;
            this.maxSteps = options.maxSteps || 1000;
            this.curriculumStage = options.curriculumStage || 0;
            this.reward = Object.assign({}, DEFAULT_REWARD, options.reward || {});
            // With fuel rescue on, one of the ships becomes the scripted tanker and
            // the observation gains the five fuel features.
            this.fuelRescue = !!options.fuelRescue;
            this.featureOptions = this.fuelRescue ? { fuelFeatures: true } : {};
            this.obsDim = this.fuelRescue ? RL_FEATURE_DIM_FUEL : RL_FEATURE_DIM;
            this.featureNames = this.fuelRescue ? RL_FEATURE_NAMES_FUEL : RL_FEATURE_NAMES;
            this.actionSize = ACTION_SPACE_SIZE;
            this.fuelStats = { refuels: 0, dockDeaths: 0, lowFuelFrames: 0 };
            // 'torus' (default) or 'sphere'. Requires world.js to be loaded; the
            // observation layout is unchanged, but the chart-space relative
            // features are distorted near the poles, so policies trained on one
            // world should be evaluated on the same one.
            this.world = options.world === 'sphere' ? 'sphere' : 'torus';
            if (this.world === 'sphere' && typeof WORLD === 'undefined') {
                throw new Error('AsteroidsEnv: world="sphere" needs world.js loaded before game.js');
            }
            // Sphere only: logical world = worldScale x the 1200x600 chart
            // (R = 191 * worldScale). Ignored on the torus.
            const ws = Number(options.worldScale);
            this.worldScale = (isFinite(ws) && ws > 0) ? ws : 1;

            this.stepCount = 0;
            this.prevHealth = [];
            this.alive = [];
            this.episodeReturn = [];
            this.episodeTaskReturn = [];
            this.prevPotential = [];
            this.lastFeatures = [];
        }

        get stageConfig() {
            return CURRICULUM[Math.max(0, Math.min(3, this.curriculumStage))];
        }

        setCurriculumStage(stage) {
            this.curriculumStage = Math.max(0, Math.min(3, stage));
        }

        reset(seed) {
            const stage = this.stageConfig;

            // The player ship must be inactive: its death handler calls initGame()
            // in the middle of update(), which would silently reset the world
            // mid-episode and corrupt the trajectory.
            playerShipActive = false;

            numAIShips = this.numAgents;
            numEnemyShips = stage.enemies;
            numBossShips = stage.bosses;
            if (typeof root.marlCurriculumStage !== 'undefined') {
                root.marlCurriculumStage = this.curriculumStage;
            }

            marlEnabledSafeOff();

            // The fuel mechanic requires the alpha formation. Set both flags
            // explicitly either way so a stale browser default cannot leak in.
            alphaAttackEnabled = this.fuelRescue;
            fuelRescueEnabled = this.fuelRescue;
            fuelRescueDispatchCooldown = 0;

            // World geometry is process-global (like the other game.js flags), so
            // set it on every reset in case another env instance changed it.
            if (typeof WORLD !== 'undefined') { WORLD.setMode(this.world); WORLD.setScale(this.worldScale); }

            initGame(seed);

            if (this.fuelRescue) {
                // Pick the alpha and convert one wingman into the tanker now, so the
                // first observation already reflects who is who.
                ensureAlphaAssignment(aiShips);
            }

            this.stepCount = 0;
            this.prevHealth = [];
            this.alive = [];
            this.episodeReturn = [];
            this.episodeTaskReturn = [];
            this.lastFeatures = [];
            this.fuelStats = { refuels: 0, dockDeaths: 0, lowFuelFrames: 0 };
            for (let i = 0; i < this.numAgents; i++) {
                const shipRef = this.shipFor(i);
                this.prevHealth.push(shipRef ? shipRef.health : 0);
                // The scripted tanker is never "alive" as far as the learner is
                // concerned: no observations are collected for it and it is never
                // marked done, so PPO simply skips it.
                this.alive.push(!!shipRef && !this.isScripted(shipRef));
                this.episodeReturn.push(0);
                this.episodeTaskReturn.push(0);
                this.lastFeatures.push(null);
            }
            consumeAgentCredit(); // discard anything accrued during setup
            consumeFuelEvents();

            const obs = this.observations();
            this.prevPotential = [];
            for (let i = 0; i < this.numAgents; i++) this.prevPotential.push(this.potential(i));
            return obs;
        }

        shipFor(agentId) {
            for (let i = 0; i < aiShips.length; i++) {
                if (aiShips[i] && aiShips[i].agentId === agentId) return aiShips[i];
            }
            return null;
        }

        isScripted(shipRef) {
            return this.fuelRescue && !!shipRef && shipRef.role === 'fuel';
        }

        observation(agentId) {
            const shipRef = this.shipFor(agentId);
            if (!shipRef || this.isScripted(shipRef)) {
                this.lastFeatures[agentId] = null;
                return new Array(this.obsDim).fill(0);
            }
            const features = MLFeatureExtractor.extractRL(
                shipRef, asteroids, aiShips, ship, enemyShips, enemyBullets,
                Object.assign({ curriculumStage: this.curriculumStage }, this.featureOptions)
            );
            this.lastFeatures[agentId] = features;
            return features._vector;
        }

        observations() {
            const out = [];
            for (let i = 0; i < this.numAgents; i++) out.push(this.observation(i));
            return out;
        }

        // Shaping potential: how well the agent is currently aimed at whatever it
        // should be shooting at for this curriculum stage.
        potential(agentId) {
            const f = this.lastFeatures[agentId];
            if (!f) return 0;
            const stage = this.stageConfig;
            const asteroidTerm = (f.aimAlignedAsteroid || 0) * stage.asteroidWeight;
            const enemyTerm = (f.aimAlignedEnemy || 0) * (stage.enemyWeight + stage.bossWeight);
            const weightSum = stage.asteroidWeight + stage.enemyWeight + stage.bossWeight;
            if (weightSum <= 0) return 0;
            return (asteroidTerm + enemyTerm) / weightSum;
        }

        step(actions) {
            const stage = this.stageConfig;
            const R = this.reward;

            // actions === null means "let the rule-based controller drive". That is
            // how the heuristic baseline is measured on exactly the same seeds,
            // reward function and termination rules as a trained policy.
            if (actions) {
                setPolicyControl(true, actions);
            } else {
                setPolicyControl(false, null);
            }
            update();
            this.stepCount++;

            // Refresh observations before computing rewards: shaping needs the
            // potential of the *next* state.
            const nextObservations = this.observations();

            const credit = consumeAgentCredit();
            const fuel = this.fuelRescue ? consumeFuelEvents() : null;
            if (fuel) {
                this.fuelStats.refuels += fuel.refuels.length;
                this.fuelStats.dockDeaths += fuel.dockDeaths.length;
                this.fuelStats.lowFuelFrames += fuel.lowFuelFrames;
            }

            // Team pool: a share of everyone's kill value, so cooperation still
            // pays, but the majority of a kill goes to whoever landed it.
            let teamValue = 0;
            for (let i = 0; i < credit.length; i++) {
                if (!credit[i]) continue;
                teamValue += credit[i].asteroidKills * R.asteroidKill * stage.asteroidWeight
                    + credit[i].enemyKills * R.enemyKill * stage.enemyWeight
                    + credit[i].bossKills * R.bossKill * stage.bossWeight;
            }
            const teamPerAgent = (teamValue * R.teamShare) / Math.max(1, this.numAgents);

            // Fuel events are team events: a refuel keeps the leader alive for
            // everyone, and a dock death costs the swarm its alpha and its tanker.
            let fuelTeam = 0;
            if (fuel) {
                fuelTeam += fuel.refuels.length * R.refuelSuccess;
                fuelTeam += fuel.dockDeaths.length * R.dockDeath * R.teamShare;
            }

            const rewards = new Array(this.numAgents).fill(0);
            const taskRewards = new Array(this.numAgents).fill(0);
            const dones = new Array(this.numAgents).fill(false);

            for (let i = 0; i < this.numAgents; i++) {
                const shipRef = this.shipFor(i);
                const wasAlive = this.alive[i];
                let r = 0;

                if (shipRef && this.isScripted(shipRef)) {
                    // A wingman converted into the tanker leaves the agent set
                    // without a death penalty; its segment simply ends here.
                    if (wasAlive) dones[i] = true;
                    this.alive[i] = false;
                    this.prevPotential[i] = 0;
                    rewards[i] = 0;
                    continue;
                }

                if (shipRef) {
                    // Base survival + late-game scaling ramp.
                    // The ramp rewards lasting into the difficult late-episode without
                    // changing the kill-dominance ordering: max extra over 1000 steps = +1.5.
                    const progressFrac = Math.min(1, this.stepCount / Math.max(1, this.maxSteps));
                    r += R.survival + R.survivalScaling * progressFrac;
                    r += fuelTeam;
                    if (fuel && shipRef.isAlpha && shipRef.helpPingActive) {
                        r += R.lowFuel;
                    }

                    const c = credit[i];
                    if (c) {
                        const own = c.asteroidKills * R.asteroidKill * stage.asteroidWeight
                            + c.enemyKills * R.enemyKill * stage.enemyWeight
                            + c.bossKills * R.bossKill * stage.bossWeight;
                        r += own * (1 - R.teamShare);
                    }
                    r += teamPerAgent;

                    const lost = this.prevHealth[i] - shipRef.health;
                    if (lost > 0) r += lost * R.damageTaken;
                    this.prevHealth[i] = shipRef.health;

                    // Bounded shaping.
                    let nearbyAllies = 0;
                    for (let j = 0; j < aiShips.length; j++) {
                        const other = aiShips[j];
                        if (!other || other === shipRef) continue;
                        const dx = other.x - shipRef.x;
                        const dy = other.y - shipRef.y;
                        if (Math.sqrt(dx * dx + dy * dy) < 100) nearbyAllies++;
                    }
                    r += R.allyProximity * Math.min(nearbyAllies, R.maxAllyProximity);

                    const speed = Math.sqrt(
                        (shipRef.velocityX || 0) ** 2 + (shipRef.velocityY || 0) ** 2
                    );
                    const threatsPresent = enemyShips.length > 0 || asteroids.length > 0;
                    if (speed < R.idleSpeedThreshold && threatsPresent) {
                        r += R.idlePenalty;
                    }

                    // Center-camping penalty: only charged to the alpha, and only when
                    // it is parked at the safe centre while threats are active.
                    if (shipRef.isAlpha && threatsPresent && speed < R.idleSpeedThreshold) {
                        const cx = worldW() / 2;
                        const cy = worldH() / 2;
                        const dCx = shipRef.x - cx;
                        const dCy = shipRef.y - cy;
                        if (Math.sqrt(dCx * dCx + dCy * dCy) < R.centerCampRadius) {
                            r += R.centerCampingPenalty;
                        }
                    }

                    // Everything above this point is task reward. Shaping is added
                    // afterwards and tracked separately so that evaluation can
                    // report what the agent actually achieved rather than how well
                    // it collected its own shaping bonus.
                    taskRewards[i] = r;

                    if (R.aimShaping > 0) {
                        const phiNext = this.potential(i);
                        r += R.aimShaping * (R.shapingGamma * phiNext - this.prevPotential[i]);
                        this.prevPotential[i] = phiNext;
                    }

                    this.alive[i] = true;
                } else {
                    if (wasAlive) {
                        r += R.death;      // paid once, on the transition into death
                        if (fuel) {
                            // Dying docked is worse than dying: it takes the tanker too.
                            for (const ev of fuel.dockDeaths) {
                                if (ev.a === i || ev.b === i) r += R.dockDeath;
                            }
                        }
                        this.alive[i] = false;
                        dones[i] = true;
                    }
                    taskRewards[i] = r;
                    this.prevPotential[i] = 0;
                    // Already-dead agents accrue nothing further.
                }

                rewards[i] = r;
                this.episodeReturn[i] += r;
                this.episodeTaskReturn[i] += taskRewards[i];
            }

            setPolicyControl(false, null);

            const anyAlive = this.alive.some(Boolean);
            const done = !anyAlive || this.stepCount >= this.maxSteps || !gameRunning;
            episodeDone = done;

            return {
                observations: nextObservations,
                rewards,
                dones,
                done,
                info: {
                    step: this.stepCount,
                    expertActions: expertActionRecording ? readExpertActions() : null,
                    score,
                    aliveCount: this.alive.filter(Boolean).length,
                    asteroids: asteroids.length,
                    enemies: enemyShips.length,
                    episodeReturn: this.episodeReturn.slice(),
                    episodeTaskReturn: this.episodeTaskReturn.slice(),
                    timeLimit: this.stepCount >= this.maxSteps && anyAlive,
                    fuel: this.fuelRescue ? Object.assign({}, this.fuelStats) : null
                }
            };
        }
    }

    // The MARL stack overrides ship control if it happens to be enabled; make sure
    // it is off so the policy under test is the only thing driving the ships.
    function marlEnabledSafeOff() {
        if (typeof root.marlEnabled !== 'undefined') root.marlEnabled = false;
        if (typeof root.marlTrainingMode !== 'undefined') root.marlTrainingMode = false;
    }

    root.AsteroidsEnv = AsteroidsEnv;
    root.ENV_CURRICULUM = CURRICULUM;
    root.DEFAULT_REWARD_CONFIG = DEFAULT_REWARD;
})(typeof globalThis !== 'undefined' ? globalThis : this);
