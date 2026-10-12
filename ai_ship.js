// Extracted from game.js as part of a module split.
// Relies on globals defined in game.js/world.js/rng.js/fx.js loaded earlier
// (canvas, ctx, worldW/worldH/sphereWorld, gameRandom, FXL, ACTION_SPACE_SIZE,
// decodeAction, keys, etc.) — same plain-script global scope as before, just
// split across files.

// AI Ship class - extends Ship with AI navigation
class AIShip extends Ship {
    constructor(x, y) {
        super();
        this.x = x || worldW() * 0.25;
        this.y = y || worldH() * 0.25;
        this.velocityX = 0;
        this.velocityY = 0;
        this.angle = Math.PI / 2; // Start facing up
        this.detectionRadius = 100; // How far ahead to look for asteroids
        this.avoidanceForce = 0.3; // How strongly to avoid asteroids
        this.thrustFrequency = 0.02; // Probability of thrusting each frame
        this.targetAngle = this.angle;
        // Firing parameters
        this.shootCooldown = 0;
        this.firingRange = 200; // Increased maximum distance to consider firing
        this.imminentThreatDistance = 80; // Increased distance for emergency firing
        this.collisionAngleThreshold = Math.PI / 3; // 60 degrees - wider firing arc
        this.minAsteroidSize = 15; // Fire at smaller asteroids too
        this.rapidFireCooldown = 3; // Very short cooldown for rapid continuous firing
        // Enemy detection and combat
        this.enemyDetectionRadius = 300; // Range to detect enemy ships
        this.enemyFiringRange = 250; // Range to fire at enemy ships
        // Flocking parameters
        this.flockRadius = 150; // Distance to consider for flocking
        this.separationDistance = 60; // Minimum distance to maintain from other AI ships
        this.alignmentRadius = 100; // Distance for alignment behavior
        this.cohesionRadius = 120; // Distance for cohesion behavior
        this.flockWeight = 0.3; // Weight of flocking behavior
        // Defensive flocking (when enemies present) - ENHANCED
        this.defensiveSeparationDistance = 25; // Much tighter spacing when enemies present (was 40)
        this.defensiveCohesionRadius = 60; // Tighter group when enemies present (was 80)
        this.defensiveFlockWeight = 0.8; // Stronger flocking when enemies present (was 0.6)
        this.defensiveFormationType = 'circle'; // 'circle' or 'line' formation
        this.mutualProtectionEnabled = true; // Enable mutual protection for low-health allies
        
        // Health and combat system (similar to boss ships)
        this.maxHealth = 3; // AI ships have 3 health points
        this.health = this.maxHealth;
        this.originalMaxHealth = 3; // Store original maxHealth for alpha bonus restoration
        this.aiPhase = 1; // 1, 2, or 3 based on health
        this.shieldActive = false;
        this.shieldCooldown = 0;
        this.shieldDuration = 0;
        this.phaseTransitionTimer = 0;
        
        // Healing system
        this.healingCooldown = 0; // Cooldown between healing contacts (frames)
        this.healingCooldownMax = 180; // 3 seconds at 60fps
        this.lastHealTime = 0; // Frame when last healed
        this.healingRange = 30; // Distance for healing contact
        this.seekHealingRange = 400; // Range to detect Phase 1 allies for seeking

        // Fuel rescue system (Alpha HELP ping + green fuel ship refuel)
        this.maxFuel = 100;
        this.fuel = 100;
        this.fuelHelpThreshold = 0.35;
        this.helpPingActive = false;
        this.fuelTransferActive = false;
        this.fuelTransferPartner = null;
        this.fuelTransferRange = 36;
        this.fuelTransferRate = 2.2;
        this.fuelTransferTimer = 0;
        this.fuelTransferDurationMax = 120;
        // Burn rates are sized against the tanker's round trip (~8-10 s: fly
        // in, transfer, 2 s dock hold, depart, ~8 s regen at standby). The AI
        // fires every 3 frames, so at the earlier 0.2/shot the alpha went from
        // full to HELP in ~5 s and could never be kept fuelled; it now takes
        // ~20 s of normal combat, which the tanker can sustain.
        // On a scaled sphere every trip is worldScale() times longer, so the
        // burn sites divide by it to keep the ~20 s endurance / ~10 s round-trip
        // ratio that the tanker can sustain.
        this.fuelPassiveDrain = 0.02;
        this.fuelThrustCost = 0.08;
        this.fuelFireCost = 0.03;
        this.fuelDocked = false;
        this.fuelDockHoldTimer = 0;
        this.fuelDockHoldFrames = 120; // 2 seconds at 60fps
        this.fuelStandby = true; // Away from swarm until Alpha HELP ping
        this.fuelDeparting = false;
        this.fuelDepartTimer = 0;
        this.fuelDepartDuration = 150;
        this.standbyTarget = null;
        
        // Alpha Attack Pattern System
        this.isAlpha = false; // Is this ship the alpha leader?
        this.alphaShip = null; // Reference to the alpha leader ship
        this.formationPosition = null; // Position in V-formation (left, right, or center)
        this.formationAngle = 0; // Angle offset for V-formation
        this.formationDistance = 80; // Distance behind alpha in formation
        this.formationSpread = 60; // Spread angle for V-formation (degrees)
        this.alphaAttackTarget = null; // Target enemy for alpha attack
        this.alphaAttackCooldown = 0; // Cooldown between alpha attacks
        this.alphaAttackCooldownMax = 300; // 5 seconds at 60fps
        
        // ALPHA ESCAPE SYSTEM: Escape/evasion behavior for alpha ship
        this.escapeMode = false; // Whether alpha is in escape mode
        this.threatLevel = 0; // Calculated threat level (0-1)
        this.escapeTarget = null; // {x, y} target escape position
        this.escapeTimer = 0; // How long in escape mode
        this.escapeCooldown = 0; // Cooldown before can escape again
        this.escapeCooldownMax = 180; // 3 seconds cooldown
        this.escapeSpeed = 3; // Speed multiplier during escape
        this.threatThreshold = 0.4; // Threat level threshold to trigger escape (0-1) - Lowered from 0.6 to 0.4
        this.returnToCenterDistance = 50; // Distance threshold to return to center
        this.centerPosition = {x: worldW() / 2, y: worldH() / 2}; // Center position
        this.nearbyEnemyCount = 0; // Count of nearby enemies
        this.incomingBulletCount = 0; // Count of incoming bullets
        this.escapePath = null; // Calculated escape path
        
        // Phase 2: Dynamic Formation Types
        this.formationType = (typeof formationType !== 'undefined' ? formationType : 'arrowhead'); // 'arrowhead', 'line', 'circle', 'diamond', 'wedge'
        this.formationTypeChanged = false; // Flag for visual update
        
        // Phase 2: Role Specialization
        this.role = null; // 'scout', 'tank', 'support', 'dps', 'interceptor', null (auto-assigned)
        this.roleAssigned = false; // Whether role has been assigned
        this.roleAbilities = {
            scout: { speed: 1.3, detection: 1.5, health: 0.9 },
            tank: { speed: 0.8, detection: 0.9, health: 1.5, shield: 1.3 },
            support: { speed: 1.0, detection: 1.2, health: 1.1, healing: 1.5 },
            fuel: { speed: 1.15, detection: 1.1, health: 1.0 },
            dps: { speed: 1.1, detection: 1.0, health: 0.9, damage: 1.5, fireRate: 1.3 },
            interceptor: { speed: 1.4, detection: 1.3, health: 0.95 }
        };
        
        // Phase 2: Formation Abilities
        this.formationShieldActive = false; // Combined shield when in formation
        this.formationBoostActive = false; // Speed boost when in formation
        this.formationHealActive = false; // Healing aura when in formation
        this.formationRadarActive = false; // Extended detection in formation
        this.formationAbilityCooldown = 0;
        this.formationAbilityCooldownMax = 600; // 10 seconds at 60fps
        
        // Phase 2: Flanking Maneuvers
        this.flankingMode = 'none'; // 'none', 'side', 'pincer', 'encirclement'
        this.flankingPosition = null; // Target position for flanking
        this.flankingCooldown = 0;
        this.flankingCooldownMax = 450; // 7.5 seconds at 60fps
        
        // Phase 3: Adaptive Formations
        this.adaptiveFormationEnabled = false; // Auto-switch formations based on situation
        this.lastFormationSwitch = 0; // Frame when last switched
        this.formationSwitchCooldown = 300; // 5 seconds cooldown between switches
        
        // Phase 3: Multi-Target Coordination
        this.assignedTarget = null; // Specific target assigned to this ship
        
        // ALPHA ESCAPE: Covering fire mode for wingmen
        this.coveringFireMode = false; // Whether wingman should provide covering fire
        this.targetPriority = 0; // Priority of assigned target
        this.multiTargetMode = 'focus'; // 'focus', 'split', 'prioritize'
        
        // Phase 3: Escort & Protection Modes
        this.escortMode = 'none'; // 'none', 'escort', 'guard', 'patrol', 'intercept', 'cover'
        this.escortTarget = null; // Target to escort/protect (player ship or ally)
        this.escortDistance = 100; // Distance to maintain from escort target
        this.escortAngle = 0; // Angle offset for escort position
        this.patrolCenter = null; // Center point for patrol mode
        this.patrolRadius = 200; // Patrol radius
        this.interceptTarget = null; // Target to intercept
        
        // Advanced Alpha Attack Patterns: Tactical Attack Sequences
        this.attackSequence = 'none'; // 'none', 'alphaStrike', 'wave'
        this.attackSequenceTimer = 0; // Timer for attack sequence
        this.attackSequencePhase = 0; // Current phase in sequence (0, 1, 2, ...)
        this.waveNumber = 0; // Wave number for wave attacks
        
        // Advanced Alpha Attack Patterns: Dynamic Formation Transitions
        this.formationMorphing = false; // Is formation currently morphing?
        this.formationMorphTarget = null; // Target formation type to morph to
        this.formationMorphProgress = 0; // 0-1, progress of morph
        this.splitFormation = false; // Is formation split?
        this.splitGroups = []; // Array of split groups
        this.mergeTimer = 0; // Timer for merge after split
        
        // Advanced Alpha Attack Patterns: Advanced Flanking Patterns
        this.flankPattern = 'none'; // 'none', 'hammerAnvil', 'scissors'
        this.flankPosition = null; // Target position for flanking
        this.flankTimer = 0; // Timer for flanking maneuver
        
        // Phase 5.1: ML/AI Integration - Parameter Tuning
        this.mlEnabled = false; // Toggle for ML mode
        this.mlModel = null; // Will be loaded from trained model (placeholder for now)
        this.useMLParameters = false; // Use ML-suggested parameters
        
        // Phase 5.2: ML Priority Weights
        this.useMLPriorities = false; // Use ML-suggested priority weights
        
        // Phase 5.3: Full ML Decision Making
        this.useMLDecisions = false; // Use ML for direct action prediction
        this.mlActionCache = null; // Cache ML action predictions
        this.mlActionCacheFrame = 0; // Frame when cache was created
        this.mlActionCacheInterval = 5; // Update ML action every 5 frames (performance)
        
        // Base parameters (rule-based fallback)
        this.baseDetectionRadius = 100;
        this.baseFiringRange = 200;
        this.baseFlockWeight = 0.3;
        this.baseThrustFrequency = 0.02;
        this.baseEnemyFiringRange = 250;
        
        // Base priority weights (rule-based fallback)
        this.basePriorityWeights = {
            avoidShips: 1.0,      // Always highest (safety)
            fireAtEnemies: 0.8,   // High priority for combat
            protectAllies: 0.6,   // Medium-high priority
            fireAtAsteroids: 0.4,  // Medium priority
            avoidAsteroids: 0.3,   // Medium-low priority
            randomNav: 0.1        // Low priority
        };
        
        // ML-suggested parameters (updated by ML model)
        this.mlDetectionRadius = this.baseDetectionRadius;
        this.mlFiringRange = this.baseFiringRange;
        this.mlFlockWeight = this.baseFlockWeight;
        this.mlThrustFrequency = this.baseThrustFrequency;
        this.mlEnemyFiringRange = this.baseEnemyFiringRange;
        
        // ML-suggested priority weights (updated by ML model)
        this.mlPriorityWeights = {...this.basePriorityWeights};
        
        // ML inference caching (run every N frames for performance)
        this.mlInferenceFrame = 0;
        this.mlInferenceInterval = 5; // Run ML every 5 frames (12 times per second)
        
        // Phase 5.1: Data collection for ML training (optional)
        this.collectTrainingData = false; // Enable via Streamlit if needed
        this.trainingDataBuffer = []; // Buffer to store training samples
        this.maxTrainingSamples = 1000; // Max samples to keep in memory
    }

    // Find nearest asteroid in path
    findNearestAsteroid(asteroids) {
        let nearest = null;
        let nearestDistance = Infinity;
        
        asteroids.forEach(asteroid => {
            const dx = asteroid.x - this.x;
            const dy = asteroid.y - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Check if asteroid is in front of ship (within detection radius)
            const angleToAsteroid = Math.atan2(dy, dx);
            const angleDiff = Math.abs(this.normalizeAngle(angleToAsteroid - this.angle));
            
            // Consider asteroids in front (within 120 degrees) and within detection radius
            if (distance < this.detectionRadius && angleDiff < Math.PI * 2/3) {
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearest = { asteroid, distance, angle: angleToAsteroid };
                }
            }
        });
        
        return nearest;
    }

    // Find nearest ship (AI or player) that's too close
    findNearestShip(aiShips, playerShip) {
        let nearest = null;
        let nearestDistance = Infinity;
        const avoidanceDistance = 80; // Distance to start avoiding other ships
        
        // Check player ship
        if (playerShip && playerShipActive) {
            const dx = playerShip.x - this.x;
            const dy = playerShip.y - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < avoidanceDistance && distance > 0) {
                const angleToShip = Math.atan2(dy, dx);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearest = { ship: playerShip, distance, angle: angleToShip, isPlayer: true };
                }
            }
        }
        
        // Check other AI ships
        aiShips.forEach(aiShip => {
            if (aiShip && aiShip !== this) { // Don't check self
                const dx = aiShip.x - this.x;
                const dy = aiShip.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < avoidanceDistance && distance > 0) {
                    const angleToShip = Math.atan2(dy, dx);
                    if (distance < nearestDistance) {
                        nearestDistance = distance;
                        nearest = { ship: aiShip, distance, angle: angleToShip, isPlayer: false };
                    }
                }
            }
        });
        
        return nearest;
    }

    // Normalize angle to -PI to PI range
    normalizeAngle(angle) {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }
    
    // Find nearest Phase 1 ally for healing
    findNearestPhase1Ally(aiShips) {
        if (this.aiPhase !== 3) return null; // Only Phase 3 ships seek healing
        
        let nearest = null;
        let nearestDistance = Infinity;
        
        aiShips.forEach(ally => {
            if (ally && ally !== this && ally.aiPhase === 1) { // Only Phase 1 allies
                const dx = ally.x - this.x;
                const dy = ally.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < this.seekHealingRange && distance < nearestDistance) {
                    nearestDistance = distance;
                    const angleToAlly = Math.atan2(dy, dx);
                    nearest = { ally, distance, angle: angleToAlly, dx, dy };
                }
            }
        });
        
        return nearest;
    }
    
    // ALPHA ATTACK: Select alpha ship from available AI ships
    static selectAlphaShip(aiShips) {
        if (!alphaAttackEnabled || aiShips.length === 0) return null;
        
        // IMPROVED: Enhanced alpha ship selection with multiple factors
        let bestCandidate = null;
        let bestScore = -Infinity;

        // The scripted tanker is only promoted when it is the last ship left:
        // an alpha that is also the only fuel source can never be refuelled.
        const isTanker = ship => fuelRescueEnabled && ship.role === 'fuel';
        const combatShips = aiShips.filter(ship => ship && !isTanker(ship));
        const candidates = combatShips.length > 0 ? combatShips : aiShips;

        candidates.forEach(ship => {
            if (!ship) return;
            
            let score = 0;
            
            // Factor 1: Phase preference (strongest factor)
            if (ship.aiPhase === 1) {
                score += 200; // Phase 1 ships are ideal (was 100)
            } else if (ship.aiPhase === 2) {
                score += 50; // Phase 2 ships are acceptable
            } else if (ship.aiPhase === 3) {
                score -= 30; // Phase 3 penalty reduced (was -50) - still selectable when needed
            }
            
            // Factor 2: Health level (scaled by maxHealth for fairness)
            const healthRatio = ship.health / (ship.maxHealth || 3);
            score += healthRatio * 100; // 0-100 points based on health percentage
            
            // Factor 3: Absolute health value (prefer ships with more HP)
            score += ship.health * 15; // Increased from 10 to 15
            
            // Factor 4: Distance from center (alpha stays at center, so closer is better)
            const centerX = worldW() / 2;
            const centerY = worldH() / 2;
            const dx = ship.x - centerX;
            const dy = ship.y - centerY;
            const distanceFromCenter = Math.sqrt(dx * dx + dy * dy);
            const maxDistance = Math.sqrt(centerX * centerX + centerY * centerY);
            const centerProximity = 1.0 - Math.min(1.0, distanceFromCenter / maxDistance);
            score += centerProximity * 30; // Bonus for being near center (0-30 points)
            
            // Factor 5: Shield status (prefer ships with shields available)
            if (ship.shieldCooldown <= 0) {
                score += 20; // Shield ready bonus
            }
            
            // Factor 6: Role consideration (prefer DPS or balanced roles for alpha)
            if (ship.role === 'dps' || ship.role === null) {
                score += 10; // DPS or no role is good for alpha
            } else if (ship.role === 'support') {
                score += 5; // Support can be alpha but less ideal
            }
            
            // Factor 7: Avoid ships that are currently seeking healing (they're in trouble)
            if (ship.healingCooldown > 0) {
                score -= 15; // Penalty for ships that recently needed healing
            }
            
            // Factor 8: Prefer ships not currently in formation (easier transition)
            if (!ship.alphaShip) {
                score += 5; // Not following anyone is a small bonus
            }
            
            // IMPROVED: Better handling of all-low-health scenario
            // When all ships are Phase 3, still select the best one
            if (ship.aiPhase === 3) {
                // Additional scoring for Phase 3 ships to differentiate them
                // Higher health Phase 3 ships are still better than lower health ones
                score += (ship.health - 0.5) * 20; // Extra weight on health when in Phase 3
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestCandidate = ship;
            }
        });
        
        // IMPROVED: Enhanced fallback logic
        if (!bestCandidate && aiShips.length > 0) {
            // Fallback 1: Select ship with highest health
            bestCandidate = aiShips.reduce((best, ship) => {
                if (!ship) return best;
                if (!best) return ship;
                return ship.health > best.health ? ship : best;
            }, null);
            
            // Fallback 2: If still no candidate, select first available
            if (!bestCandidate) {
                bestCandidate = aiShips.find(ship => ship);
            }
        }

        // A promoted tanker becomes a combat ship again.
        if (bestCandidate && isTanker(bestCandidate)) {
            bestCandidate.role = null;
            bestCandidate.roleAssigned = false;
            bestCandidate.fuelStandby = false;
            bestCandidate.fuelDeparting = false;
            bestCandidate.fuelDepartTimer = 0;
            bestCandidate.standbyTarget = null;
        }
        
        return bestCandidate;
    }
    
    // ALPHA ATTACK: Find alpha ship and assign roles
    findAlphaShip(aiShips) {
        if (!alphaAttackEnabled) {
            // Restore original maxHealth when alpha attack is disabled
            if (this.isAlpha && this.originalMaxHealth) {
                this.maxHealth = this.originalMaxHealth;
                this.health = Math.min(this.health, this.maxHealth);
                this.originalMaxHealth = null;
            }
            this.isAlpha = false;
            this.alphaShip = null;
            return null;
        }
        
        // Find or select alpha
        let alpha = null;
        aiShips.forEach(ship => {
            if (ship && ship.isAlpha) {
                alpha = ship;
            }
        });
        
        // If no alpha exists, select one
        if (!alpha) {
            alpha = AIShip.selectAlphaShip(aiShips);
            if (alpha) {
                // Store original maxHealth if not already stored
                if (!alpha.originalMaxHealth) {
                    alpha.originalMaxHealth = alpha.maxHealth;
                }
                // Apply 25% bonus to alpha ship
                alpha.maxHealth = Math.ceil(alpha.originalMaxHealth * 1.25);
                alpha.health = Math.min(alpha.health, alpha.maxHealth); // Cap current health to new max
                alpha.isAlpha = true;
                alpha.alphaShip = null; // Alpha doesn't follow itself
                // Position alpha ship at center of screen
                alpha.x = worldW() / 2;
                alpha.y = worldH() / 2;
                alpha.velocityX = 0;
                alpha.velocityY = 0;
                alpha.maxFuel = 100;
                alpha.fuel = 100;
                updateAlphaHelpPing(alpha);
            }
        }
        
        assignFuelShipRole(aiShips);
        
        // Update this ship's alpha reference
        if (alpha === this) {
            // Store original maxHealth if not already stored
            if (!this.originalMaxHealth) {
                this.originalMaxHealth = this.maxHealth;
            }
            // Apply 25% bonus to alpha ship
            this.maxHealth = Math.ceil(this.originalMaxHealth * 1.25);
            this.health = Math.min(this.health, this.maxHealth); // Cap current health to new max
            // Fuel is only topped up on the frame a ship *becomes* alpha. This
            // block runs every frame, so refilling unconditionally here kept
            // the alpha at 100% forever and the rescue mechanic never fired
            // under rule-based control.
            if (!this.isAlpha) {
                this.maxFuel = 100;
                this.fuel = 100;
            }
            this.isAlpha = true;
            this.alphaShip = null;
            // Position alpha ship at center of screen
            this.x = worldW() / 2;
            this.y = worldH() / 2;
            this.velocityX = 0;
            this.velocityY = 0;
            updateAlphaHelpPing(this);
        } else {
            // Restore original maxHealth when no longer alpha
            if (this.isAlpha && this.originalMaxHealth) {
                this.maxHealth = this.originalMaxHealth;
                this.health = Math.min(this.health, this.maxHealth);
                this.originalMaxHealth = null; // Clear stored value
            }
            this.isAlpha = false;
            this.alphaShip = alpha;
        }
        
        return alpha;
    }
    
    // ALPHA ESCAPE SYSTEM: Assess threat level
    assessThreatLevel(enemyShips, enemyBullets) {
        if (!this.isAlpha || !alphaAttackEnabled) return 0;
        
        let threatScore = 0;
        this.nearbyEnemyCount = 0;
        this.incomingBulletCount = 0;
        
        // Count nearby enemies - INCREASED THREAT RADIUS for better detection
        const threatRadius = 300; // Increased from 250 to 300
        if (enemyShips && Array.isArray(enemyShips)) {
            enemyShips.forEach(enemy => {
                if (!enemy) return;
                const dx = enemy.x - this.x;
                const dy = enemy.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < threatRadius) {
                    this.nearbyEnemyCount++;
                    // Closer enemies = higher threat - INCREASED WEIGHT
                    const proximityThreat = 1.0 - (distance / threatRadius);
                    threatScore += proximityThreat * 0.4; // Increased from 0.3 to 0.4
                    
                    // Enemy type multiplier - INCREASED
                    if (enemy.type === 'boss') threatScore += 0.3; // Increased from 0.2
                    else if (enemy.type === 'advanced') threatScore += 0.15; // Increased from 0.1
                    else threatScore += 0.05; // Basic enemies also contribute
                }
            });
        }
        
        // Count incoming bullets - INCREASED RANGE AND WEIGHT
        if (enemyBullets && Array.isArray(enemyBullets)) {
            enemyBullets.forEach(bullet => {
                if (!bullet) return;
                const dx = bullet.x - this.x;
                const dy = bullet.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 200) { // Increased from 150 to 200
                    this.incomingBulletCount++;
                    // Calculate if bullet is heading toward alpha
                    const bulletAngle = Math.atan2(dy, dx);
                    const bulletTravelAngle = bullet.angle;
                    const angleDiff = Math.abs(bulletAngle - bulletTravelAngle);
                    if (angleDiff < Math.PI / 3 || angleDiff > Math.PI * 2 - Math.PI / 3) {
                        threatScore += 0.2; // Increased from 0.15 - bullets are more threatening
                    } else {
                        threatScore += 0.05; // Even bullets not directly heading toward alpha add some threat
                    }
                }
            });
        }
        
        // Health factor (lower health = higher threat perception) - INCREASED WEIGHTS
        const healthRatio = this.health / this.maxHealth;
        if (healthRatio < 0.3) threatScore += 0.4; // Increased from 0.3
        else if (healthRatio < 0.5) threatScore += 0.3; // Increased from 0.2
        else if (healthRatio < 0.7) threatScore += 0.15; // Increased from 0.1
        
        // ADDITIONAL: Multiple enemies nearby = exponential threat increase
        if (this.nearbyEnemyCount >= 3) {
            threatScore += 0.2; // Bonus for 3+ enemies
        } else if (this.nearbyEnemyCount >= 2) {
            threatScore += 0.1; // Bonus for 2 enemies
        }
        
        // Normalize threat score to 0-1
        this.threatLevel = Math.min(1.0, threatScore);
        
        // DEBUG LOGGING: Log threat assessment details
        if (this.isAlpha && (this.threatLevel > 0.3 || this.escapeMode)) {
            console.log(`[ALPHA ESCAPE] Threat Assessment:`, {
                threatLevel: this.threatLevel.toFixed(2),
                threshold: this.threatThreshold.toFixed(2),
                nearbyEnemies: this.nearbyEnemyCount,
                incomingBullets: this.incomingBulletCount,
                healthRatio: (this.health / this.maxHealth).toFixed(2),
                escapeMode: this.escapeMode,
                escapeCooldown: this.escapeCooldown
            });
        }
        
        return this.threatLevel;
    }
    
    // ALPHA ESCAPE SYSTEM: Calculate escape position
    calculateEscapePosition(enemyShips, asteroids) {
        if (!this.isAlpha || !alphaAttackEnabled) {
            console.log(`[ALPHA ESCAPE] calculateEscapePosition: Not alpha or alpha attack disabled`);
            return null;
        }
        
        // Find direction away from largest enemy concentration
        let avgEnemyX = 0;
        let avgEnemyY = 0;
        let enemyCount = 0;
        const threatRadius = 250;
        
        if (enemyShips && Array.isArray(enemyShips)) {
            enemyShips.forEach(enemy => {
                if (!enemy) return;
                const dx = enemy.x - this.x;
                const dy = enemy.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < threatRadius) {
                    avgEnemyX += enemy.x;
                    avgEnemyY += enemy.y;
                    enemyCount++;
                }
            });
        }
        
        if (enemyCount === 0) {
            console.log(`[ALPHA ESCAPE] calculateEscapePosition: No enemies within ${threatRadius}px radius`);
            return null;
        }
        
        console.log(`[ALPHA ESCAPE] calculateEscapePosition: Found ${enemyCount} enemies within threat radius`);
        
        avgEnemyX /= enemyCount;
        avgEnemyY /= enemyCount;
        
        // Calculate escape direction (away from enemy concentration)
        const escapeAngle = Math.atan2(this.y - avgEnemyY, this.x - avgEnemyX);
        const escapeDistance = 200; // Distance to escape
        
        let escapeX = this.x + Math.cos(escapeAngle) * escapeDistance;
        let escapeY = this.y + Math.sin(escapeAngle) * escapeDistance;
        
        // Check for asteroids and adjust path
        if (asteroids && Array.isArray(asteroids)) {
            let bestEscapeX = escapeX;
            let bestEscapeY = escapeY;
            let bestScore = -1;
            
            // Try multiple escape angles
            for (let angleOffset = -Math.PI / 2; angleOffset <= Math.PI / 2; angleOffset += Math.PI / 6) {
                const testAngle = escapeAngle + angleOffset;
                const testX = this.x + Math.cos(testAngle) * escapeDistance;
                const testY = this.y + Math.sin(testAngle) * escapeDistance;
                
                // Check if path is clear of asteroids
                let pathClear = true;
                let asteroidDistance = Infinity;
                asteroids.forEach(asteroid => {
                    if (!asteroid) return;
                    const dx = asteroid.x - testX;
                    const dy = asteroid.y - testY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < asteroid.radius + this.size + 20) {
                        pathClear = false;
                    }
                    asteroidDistance = Math.min(asteroidDistance, distance);
                });
                
                // Score this escape position
                const score = pathClear ? 1.0 : (asteroidDistance / 100);
                if (score > bestScore) {
                    bestScore = score;
                    bestEscapeX = testX;
                    bestEscapeY = testY;
                }
            }
            
            escapeX = bestEscapeX;
            escapeY = bestEscapeY;
        }
        
        // Keep within screen bounds
        escapeX = Math.max(this.size, Math.min(worldW() - this.size, escapeX));
        escapeY = Math.max(this.size, Math.min(worldH() - this.size, escapeY));
        
        const escapePos = {x: escapeX, y: escapeY};
        console.log(`[ALPHA ESCAPE] Final escape position: (${escapeX.toFixed(0)}, ${escapeY.toFixed(0)}), Distance from current: ${Math.sqrt((escapeX - this.x) ** 2 + (escapeY - this.y) ** 2).toFixed(0)}px`);
        
        return escapePos;
    }
    
    // ALPHA ESCAPE SYSTEM: Execute escape behavior
    executeEscapeBehavior(enemyShips, enemyBullets, asteroids, aiShips) {
        if (!this.isAlpha || !alphaAttackEnabled) {
            if (this.escapeMode) {
                console.log(`[ALPHA ESCAPE] executeEscapeBehavior: alphaAttackEnabled is false`);
            }
            return;
        }
        if (this.fuelDocked || this.fuelTransferActive) {
            this.escapeMode = false;
            return;
        }
        if (this.helpPingActive && fuelRescueEnabled) {
            this.escapeMode = false;
            return;
        }
        
        // Update escape cooldown
        if (this.escapeCooldown > 0) {
            this.escapeCooldown--;
        }
        
        // Assess threat level
        const threatLevel = this.assessThreatLevel(enemyShips, enemyBullets);
        
        // DEBUG: Log escape behavior execution
        if (this.escapeMode || threatLevel > 0.3) {
            console.log(`[ALPHA ESCAPE] executeEscapeBehavior called:`, {
                escapeMode: this.escapeMode,
                threatLevel: threatLevel.toFixed(2),
                threshold: this.threatThreshold.toFixed(2),
                cooldown: this.escapeCooldown,
                hasEscapeTarget: !!this.escapeTarget,
                currentPos: `(${this.x.toFixed(0)}, ${this.y.toFixed(0)})`
            });
        }
        
        // Check if should enter escape mode
        if (!this.escapeMode && threatLevel >= this.threatThreshold && this.escapeCooldown === 0) {
            // Enter escape mode
            console.log(`[ALPHA ESCAPE]  ENTERING ESCAPE MODE! Threat: ${threatLevel.toFixed(2)}, Threshold: ${this.threatThreshold.toFixed(2)}`);
            this.escapeMode = true;
            this.escapeTimer = 0;
            this.escapeTarget = this.calculateEscapePosition(enemyShips, asteroids);
            
            if (this.escapeTarget) {
                console.log(`[ALPHA ESCAPE] Escape target calculated: (${this.escapeTarget.x.toFixed(0)}, ${this.escapeTarget.y.toFixed(0)})`);
            } else {
                console.log(`[ALPHA ESCAPE]  WARNING: No escape target calculated!`);
            }
            
            // Signal wingmen to provide covering fire
            if (aiShips && Array.isArray(aiShips)) {
                const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
                console.log(`[ALPHA ESCAPE] Signaling ${wingmen.length} wingmen for covering fire`);
                wingmen.forEach(wingman => {
                    if (wingman) {
                        wingman.coveringFireMode = true; // Signal for covering fire
                    }
                });
            }
        } else if (!this.escapeMode && threatLevel >= this.threatThreshold && this.escapeCooldown > 0) {
            // DEBUG: Log when escape would trigger but cooldown is active
            console.log(`[ALPHA ESCAPE]  Escape would trigger but cooldown active: ${this.escapeCooldown} frames remaining`);
        }
        
        // Execute escape movement
        if (this.escapeMode) {
            this.escapeTimer++;
            
            if (this.escapeTarget) {
                // Move toward escape position
                const dx = this.escapeTarget.x - this.x;
                const dy = this.escapeTarget.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance > 10) {
                    // Still moving to escape position
                    const angle = Math.atan2(dy, dx);
                    const speed = this.escapeSpeed;
                    // Set velocity directly (override friction during escape)
                    this.velocityX = Math.cos(angle) * speed;
                    this.velocityY = Math.sin(angle) * speed;
                    // Also rotate toward escape direction
                    this.targetAngle = angle;
                    let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
                    if (Math.abs(angleDiff) > 0.1) {
                        this.rotate(angleDiff > 0 ? 1 : -1);
                    }
                    
                    // DEBUG: Log escape movement every 30 frames to avoid spam
                    if (this.escapeTimer % 30 === 0) {
                        console.log(`[ALPHA ESCAPE] Moving to escape: Distance ${distance.toFixed(0)}px, Velocity: (${this.velocityX.toFixed(2)}, ${this.velocityY.toFixed(2)}), Timer: ${this.escapeTimer}`);
                    }
                } else {
                    // Reached escape position, check if threat reduced
                    const currentThreat = this.assessThreatLevel(enemyShips, enemyBullets);
                    const exitThreshold = this.threatThreshold * 0.7;
                    if (currentThreat < exitThreshold || this.escapeTimer > 180) {
                        // Threat reduced or timeout, exit escape mode
                        console.log(`[ALPHA ESCAPE]  EXITING ESCAPE MODE. Reason: ${currentThreat < exitThreshold ? 'Threat reduced' : 'Timeout'}, Current Threat: ${currentThreat.toFixed(2)}, Exit Threshold: ${exitThreshold.toFixed(2)}, Timer: ${this.escapeTimer}`);
                        this.escapeMode = false;
                        this.escapeTimer = 0;
                        this.escapeCooldown = this.escapeCooldownMax;
                        this.escapeTarget = null;
                        
                        // Signal wingmen to return to formation
                        if (aiShips && Array.isArray(aiShips)) {
                            const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
                            console.log(`[ALPHA ESCAPE] Returning ${wingmen.length} wingmen to formation`);
                            wingmen.forEach(wingman => {
                                if (wingman) {
                                    wingman.coveringFireMode = false;
                                }
                            });
                        }
                    } else {
                        // Still threatened, calculate new escape position
                        console.log(`[ALPHA ESCAPE] Still threatened (${currentThreat.toFixed(2)}), recalculating escape position`);
                        this.escapeTarget = this.calculateEscapePosition(enemyShips, asteroids);
                    }
                }
            }
        }
    }
    
    // ALPHA ATTACK: Calculate V-formation position
    calculateFormationPosition(aiShips) {
        if (fuelRescueEnabled && this.role === 'fuel') {
            this.formationPosition = null;
            return null;
        }
        if (!this.alphaShip || !this.alphaShip.isAlpha) {
            this.formationPosition = null;
            return null;
        }
        
        const alpha = this.alphaShip;
        
        // Count wingmen (non-alpha ships)
        const wingmen = getAlphaCombatWingmen(aiShips, alpha);
        const myIndex = wingmen.indexOf(this);
        
        if (myIndex < 0) {
            this.formationPosition = null;
            return null;
        }
        
        // Assign formation positions: left, right, left, right, etc.
        const isLeft = myIndex % 2 === 0;
        const row = Math.floor(myIndex / 2);
        
        // Calculate formation angle offset
        const spreadRad = (this.formationSpread * Math.PI) / 180; // Convert to radians
        const angleOffset = isLeft ? -spreadRad : spreadRad;
        
        // Calculate target position behind alpha
        const formationDist = this.formationDistance + (row * 40); // Stagger rows
        const targetAngle = alpha.angle + angleOffset;
        
        const targetX = alpha.x - Math.cos(targetAngle) * formationDist;
        const targetY = alpha.y - Math.sin(targetAngle) * formationDist;
        
        return {
            x: targetX,
            y: targetY,
            angle: alpha.angle, // Match alpha's angle
            distance: Math.sqrt(
                (targetX - this.x) ** 2 + (targetY - this.y) ** 2
            )
        };
    }
    
    // ALPHA ATTACK: Execute alpha attack behavior
    executeAlphaAttack(enemyShips) {
        if (!alphaAttackEnabled || !this.alphaShip) return null;
        
        const alpha = this.alphaShip;
        
        // Alpha selects target
        if (this.isAlpha) {
            // Find best enemy target
            let bestTarget = null;
            let bestScore = -1;
            
            enemyShips.forEach(enemy => {
                if (!enemy) return;
                
                const dx = enemy.x - this.x;
                const dy = enemy.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < this.enemyDetectionRadius) {
                    // Score based on proximity and threat
                    let score = 1 / (distance + 1);
                    // Prefer boss enemies (higher threat)
                    if (enemy.aiType === 'boss') score *= 1.5;
                    // Prefer lower health enemies
                    score *= (1 + (enemy.maxHealth - enemy.health) / enemy.maxHealth);
                    
                    if (score > bestScore) {
                        bestScore = score;
                        bestTarget = enemy;
                    }
                }
            });
            
            if (bestTarget && this.alphaAttackCooldown === 0) {
                this.alphaAttackTarget = bestTarget;
                return bestTarget;
            }
        } else {
            // Wingmen follow alpha's target
            if (alpha.alphaAttackTarget) {
                this.alphaAttackTarget = alpha.alphaAttackTarget;
                return alpha.alphaAttackTarget;
            }
        }
        
        return null;
    }
    
    // PHASE 2: Assign role to ship (auto-assign or manual)
    assignRole(aiShips) {
        if (!alphaAttackEnabled || this.roleAssigned) return;
        
        if (autoAssignRoles) {
            // Auto-assign based on ship characteristics and position
            const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this.alphaShip && ship.role !== 'fuel');
            const myIndex = wingmen.indexOf(this);
            
            if (this.isAlpha) {
                // Alpha is usually a balanced role, but can be specialized
                this.role = 'dps'; // Default alpha to DPS for offensive leadership
            } else {
                // Assign roles based on position in formation
                const roles = ['scout', 'tank', 'support', 'dps', 'interceptor'];
                this.role = roles[myIndex % roles.length];
            }
        }
        
        // Apply real stat differences now that the role is set
        this.applyRoleStats();
        
        this.roleAssigned = true;
    }

    // PHASE 2: Apply role-specific stat modifiers to real ship fields.
    // Called exactly once per role assignment (guarded by roleAssigned in assignRole).
    // Stats are expressed as absolute values so the effects are predictable regardless
    // of construction order, and so the HUD / kill-feed reflects actual differences.
    applyRoleStats() {
        if (!this.role || !this.roleAbilities[this.role]) return;

        // Reset to base values first so re-assignment on a new episode is clean
        this.maxHealth         = 3;
        this.maxVelocity       = 8;
        this.firingRange       = 200;
        this.enemyFiringRange  = 250;
        this.rapidFireCooldown = 3;
        this.detectionRadius   = this.baseDetectionRadius || 100;

        switch (this.role) {
            case 'scout':
                // Fast, long detection, fragile
                this.maxHealth        = 2;
                this.maxVelocity      = 10;    // +25% vs base 8
                this.firingRange      = 220;   // slightly wider range
                this.detectionRadius  = 150;   // +50% detection
                break;

            case 'tank':
                // Tough, slow, shield-first
                this.maxHealth        = 5;     // +67%
                this.maxVelocity      = 6;     // -25%
                this.shieldCooldown   = Math.max(0, (this.shieldCooldown || 0) - 60); // shield recharges faster
                break;

            case 'support':
                // Balanced, longer healing range
                this.maxHealth        = 3;
                this.maxVelocity      = 8;
                this.healingRange     = (this.healingRange || 30) * 1.5; // 45px contact heal
                this.seekHealingRange = (this.seekHealingRange || 400) * 1.2;
                break;

            case 'dps':
                // Glass cannon: fast fire, lower health
                this.maxHealth        = 2;
                this.maxVelocity      = 9;
                this.firingRange      = 250;   // +25%
                this.enemyFiringRange = 300;
                this.rapidFireCooldown = 2;    // fires 33% more often than default (3)
                break;

            case 'interceptor':
                // Pure speed — fastest in the swarm
                this.maxHealth        = 2;
                this.maxVelocity      = 11;    // +38%
                this.firingRange      = 230;
                this.enemyFiringRange = 280;
                this.detectionRadius  = 130;
                break;

            case 'fuel':
                // Scripted tanker — leave all stats unchanged
                break;
        }

        // Clamp current health to new maxHealth in case it was higher
        this.health = Math.min(this.health, this.maxHealth);
    }

    
    // PHASE 2: Calculate formation position based on formation type
    calculateFormationPositionAdvanced(aiShips) {
        if (fuelRescueEnabled && this.role === 'fuel') {
            this.formationPosition = null;
            return null;
        }
        if (!this.alphaShip || !this.alphaShip.isAlpha) {
            this.formationPosition = null;
            return null;
        }
        
        const alpha = this.alphaShip;
        const currentFormationType = this.formationType || formationType || 'arrowhead';
        const wingmen = getAlphaCombatWingmen(aiShips, alpha);
        const myIndex = wingmen.indexOf(this);
        
        if (myIndex < 0) {
            this.formationPosition = null;
            return null;
        }
        
        let targetX, targetY, targetAngle;
        
        switch (currentFormationType) {
            case 'arrowhead': // V-formation (existing)
                const isLeft = myIndex % 2 === 0;
                const row = Math.floor(myIndex / 2);
                const spreadRad = (this.formationSpread * Math.PI) / 180;
                const angleOffset = isLeft ? -spreadRad : spreadRad;
                const formationDist = this.formationDistance + (row * 40);
                targetAngle = alpha.angle + angleOffset;
                targetX = alpha.x - Math.cos(targetAngle) * formationDist;
                targetY = alpha.y - Math.sin(targetAngle) * formationDist;
                targetAngle = alpha.angle;
                break;
                
            case 'line': // Horizontal line formation
                const lineSpacing = 60;
                const lineOffset = (myIndex - (wingmen.length - 1) / 2) * lineSpacing;
                const perpAngle = alpha.angle + Math.PI / 2;
                targetX = alpha.x + Math.cos(perpAngle) * lineOffset;
                targetY = alpha.y + Math.sin(perpAngle) * lineOffset;
                targetAngle = alpha.angle;
                break;
                
            case 'circle': // Circle formation around alpha
                const circleRadius = 100;
                const angleStep = (2 * Math.PI) / Math.max(wingmen.length, 1);
                const circleAngle = alpha.angle + Math.PI + (angleStep * myIndex);
                targetX = alpha.x + Math.cos(circleAngle) * circleRadius;
                targetY = alpha.y + Math.sin(circleAngle) * circleRadius;
                targetAngle = circleAngle + Math.PI; // Face outward
                break;
                
            case 'diamond': // Diamond formation
                const diamondSize = 80;
                let diamondX = 0, diamondY = 0;
                if (myIndex === 0) { diamondX = 0; diamondY = -diamondSize; } // Top
                else if (myIndex === 1) { diamondX = diamondSize; diamondY = 0; } // Right
                else if (myIndex === 2) { diamondX = 0; diamondY = diamondSize; } // Bottom
                else if (myIndex === 3) { diamondX = -diamondSize; diamondY = 0; } // Left
                else {
                    // Additional ships form outer ring
                    const outerAngle = (myIndex - 4) * (2 * Math.PI / (wingmen.length - 4));
                    diamondX = Math.cos(outerAngle) * (diamondSize * 1.5);
                    diamondY = Math.sin(outerAngle) * (diamondSize * 1.5);
                }
                targetX = alpha.x + Math.cos(alpha.angle) * diamondX - Math.sin(alpha.angle) * diamondY;
                targetY = alpha.y + Math.sin(alpha.angle) * diamondX + Math.cos(alpha.angle) * diamondY;
                targetAngle = alpha.angle;
                break;
                
            case 'wedge': // Tight wedge formation
                const wedgeSpread = 30 * Math.PI / 180; // 30 degrees
                const wedgeDist = 60 + (Math.floor(myIndex / 2) * 30);
                const wedgeOffset = (myIndex % 2 === 0 ? -1 : 1) * wedgeSpread;
                targetAngle = alpha.angle + wedgeOffset;
                targetX = alpha.x - Math.cos(targetAngle) * wedgeDist;
                targetY = alpha.y - Math.sin(targetAngle) * wedgeDist;
                targetAngle = alpha.angle;
                break;
                
            default: // Fallback to arrowhead
                return this.calculateFormationPosition(aiShips);
        }
        
        return {
            x: targetX,
            y: targetY,
            angle: targetAngle,
            distance: Math.sqrt(
                (targetX - this.x) ** 2 + (targetY - this.y) ** 2
            )
        };
    }
    
    // PHASE 2: Execute flanking maneuver
    executeFlankingManeuver(enemyShips, aiShips) {
        if (!alphaAttackEnabled || !this.alphaShip || this.flankingCooldown > 0) return null;
        
        const alpha = this.alphaShip;
        if (!alpha.alphaAttackTarget) return null;
        
        const target = alpha.alphaAttackTarget;
        const dx = target.x - alpha.x;
        const dy = target.y - alpha.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Determine flanking mode based on situation
        if (distance < 200 && enemyShips.length > 0) {
            const wingmen = aiShips.filter(ship => ship && ship !== alpha && ship.alphaShip === alpha && ship.role !== 'fuel');
            const myIndex = wingmen.indexOf(this);
            
            if (myIndex >= 0) {
                // Side attack: split to left/right
                if (this.flankingMode === 'none' || this.flankingMode === 'side') {
                    this.flankingMode = 'side';
                    const flankAngle = Math.atan2(dy, dx) + (myIndex % 2 === 0 ? -Math.PI / 2 : Math.PI / 2);
                    const flankDistance = 150;
                    this.flankingPosition = {
                        x: target.x + Math.cos(flankAngle) * flankDistance,
                        y: target.y + Math.sin(flankAngle) * flankDistance,
                        angle: Math.atan2(target.y - this.y, target.x - this.x)
                    };
                    return this.flankingPosition;
                }
            }
        }
        
        return null;
    }
    
    // PHASE 3: Adaptive Formation - Auto-switch based on situation
    calculateAdaptiveFormation(enemyShips, aiShips) {
        if (!adaptiveFormationEnabled || !alphaAttackEnabled || !this.isAlpha) return null;
        
        if (this.formationSwitchCooldown > 0) {
            this.formationSwitchCooldown--;
            return null;
        }
        
        const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
        const enemyCount = enemyShips ? enemyShips.length : 0;
        const threatLevel = this.hasNearbyEnemies(enemyShips) ? 1.0 : 0.0;
        const avgHealth = wingmen.length > 0 ? 
            wingmen.reduce((sum, w) => sum + (w.health / w.maxHealth), 0) / wingmen.length : 1.0;
        
        let recommendedFormation = this.formationType;
        
        // Threat-based: Use defensive formations under heavy fire
        if (threatLevel > 0.7 && enemyCount >= 2) {
            recommendedFormation = 'circle'; // Defensive circle
        }
        // Enemy count: Use line for many enemies, diamond for few
        else if (enemyCount >= 3) {
            recommendedFormation = 'line'; // Broadside attack
        } else if (enemyCount === 1) {
            recommendedFormation = 'diamond'; // Surround single target
        }
        // Health-based: Use tighter formations when ships are damaged
        else if (avgHealth < 0.5) {
            recommendedFormation = 'circle'; // Defensive circle for protection
        }
        // Phase-based: Different formations for different phases
        else if (this.aiPhase === 3) {
            recommendedFormation = 'circle'; // Defensive when critical
        } else if (this.aiPhase === 2) {
            recommendedFormation = 'diamond'; // Balanced defense
        }
        // Default: Arrowhead for normal operations
        else {
            recommendedFormation = 'arrowhead';
        }
        
        // Switch formation if different
        if (recommendedFormation !== this.formationType) {
            this.formationType = recommendedFormation;
            this.formationTypeChanged = true;
            this.formationSwitchCooldown = 300; // 5 second cooldown
            // Update all wingmen
            wingmen.forEach(w => {
                if (w) {
                    w.formationType = recommendedFormation;
                    w.formationTypeChanged = true;
                }
            });
            return recommendedFormation;
        }
        
        return null;
    }
    
    // PHASE 3: Multi-Target Coordination
    coordinateMultiTarget(enemyShips, aiShips) {
        if (!alphaAttackEnabled || !this.isAlpha || !enemyShips || enemyShips.length === 0) {
            this.assignedTarget = null;
            return null;
        }
        
        const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
        
        if (multiTargetMode === 'focus') {
            // Focus Fire: All ships target the same high-priority enemy
            let bestTarget = null;
            let bestScore = -1;
            
            enemyShips.forEach(enemy => {
                if (!enemy) return;
                const dx = enemy.x - this.x;
                const dy = enemy.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < this.enemyDetectionRadius) {
                    let score = 1 / (distance + 1);
                    if (enemy.aiType === 'boss') score *= 2.0; // Prioritize bosses
                    score *= (1 + (enemy.maxHealth - enemy.health) / enemy.maxHealth);
                    
                    if (score > bestScore) {
                        bestScore = score;
                        bestTarget = enemy;
                    }
                }
            });
            
            if (bestTarget) {
                this.assignedTarget = bestTarget;
                this.alphaAttackTarget = bestTarget;
                // Assign same target to all wingmen
                wingmen.forEach(w => {
                    if (w) {
                        w.assignedTarget = bestTarget;
                        w.alphaAttackTarget = bestTarget;
                    }
                });
                return bestTarget;
            }
        } else if (multiTargetMode === 'split') {
            // Split Formation: Alpha and wingmen target different enemies
            const availableTargets = enemyShips.filter(enemy => {
                if (!enemy) return false;
                const dx = enemy.x - this.x;
                const dy = enemy.y - this.y;
                return Math.sqrt(dx * dx + dy * dy) < this.enemyDetectionRadius;
            });
            
            if (availableTargets.length > 1) {
                // Alpha targets highest priority
                let bestTarget = null;
                let bestScore = -1;
                availableTargets.forEach(enemy => {
                    let score = 1 / (Math.sqrt((enemy.x - this.x) ** 2 + (enemy.y - this.y) ** 2) + 1);
                    if (enemy.aiType === 'boss') score *= 1.5;
                    if (score > bestScore) {
                        bestScore = score;
                        bestTarget = enemy;
                    }
                });
                
                this.assignedTarget = bestTarget;
                this.alphaAttackTarget = bestTarget;
                
                // Assign different targets to wingmen
                wingmen.forEach((w, index) => {
                    if (w && availableTargets.length > index + 1) {
                        const target = availableTargets[(index + 1) % availableTargets.length];
                        w.assignedTarget = target;
                        w.alphaAttackTarget = target; // Still follow alpha's primary target for formation
                    }
                });
                return bestTarget;
            }
        } else if (multiTargetMode === 'prioritize') {
            // Prioritize: Assign targets based on threat and position
            const targets = enemyShips
                .filter(enemy => {
                    if (!enemy) return false;
                    const dx = enemy.x - this.x;
                    const dy = enemy.y - this.y;
                    return Math.sqrt(dx * dx + dy * dy) < this.enemyDetectionRadius;
                })
                .map(enemy => {
                    const dx = enemy.x - this.x;
                    const dy = enemy.y - this.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    let priority = 1 / (distance + 1);
                    if (enemy.aiType === 'boss') priority *= 2.0;
                    priority *= (1 + (enemy.maxHealth - enemy.health) / enemy.maxHealth);
                    return { enemy, priority, distance };
                })
                .sort((a, b) => b.priority - a.priority);
            
            if (targets.length > 0) {
                // Alpha gets highest priority target
                this.assignedTarget = targets[0].enemy;
                this.alphaAttackTarget = targets[0].enemy;
                
                // Distribute remaining targets to wingmen
                wingmen.forEach((w, index) => {
                    if (w && targets.length > index + 1) {
                        w.assignedTarget = targets[index + 1].enemy;
                        w.alphaAttackTarget = targets[0].enemy; // Formation still follows alpha
                    }
                });
                return targets[0].enemy;
            }
        }
        
        return null;
    }
    
    // PHASE 3: Escort & Protection Modes
    executeEscortMode(playerShip, aiShips, enemyShips) {
        if (!alphaAttackEnabled || escortMode === 'none') {
            this.escortMode = 'none';
            this.escortTarget = null;
            return null;
        }
        
        if (escortMode === 'escort' && playerShip && playerShipActive) {
            // Escort player ship
            this.escortTarget = playerShip;
            const dx = playerShip.x - this.x;
            const dy = playerShip.y - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Maintain escort distance
            if (distance > this.escortDistance * 1.5) {
                // Too far, move closer
                return {
                    x: playerShip.x - Math.cos(playerShip.angle) * this.escortDistance,
                    y: playerShip.y - Math.sin(playerShip.angle) * this.escortDistance,
                    angle: playerShip.angle,
                    mode: 'escort'
                };
            }
        } else if (escortMode === 'guard') {
            // Guard mode: Defensive formation around target
            if (this.isAlpha && this.alphaAttackTarget) {
                const target = this.alphaAttackTarget;
                const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
                const guardRadius = 120;
                const guardAngle = (2 * Math.PI * wingmen.indexOf(this)) / Math.max(wingmen.length, 1);
                
                return {
                    x: target.x + Math.cos(guardAngle) * guardRadius,
                    y: target.y + Math.sin(guardAngle) * guardRadius,
                    angle: guardAngle + Math.PI, // Face outward
                    mode: 'guard'
                };
            }
        } else if (escortMode === 'patrol') {
            // Patrol mode: Circle around patrol center
            if (!this.patrolCenter) {
                this.patrolCenter = { x: this.x, y: this.y };
            }
            
            const time = Date.now() / 1000;
            const angle = (time * 0.5) % (Math.PI * 2); // Slow rotation
            return {
                x: this.patrolCenter.x + Math.cos(angle) * this.patrolRadius,
                y: this.patrolCenter.y + Math.sin(angle) * this.patrolRadius,
                angle: angle + Math.PI / 2,
                mode: 'patrol'
            };
        } else if (escortMode === 'intercept') {
            // Intercept mode: Intercept incoming threats
            if (enemyShips && enemyShips.length > 0) {
                let nearestThreat = null;
                let nearestDistance = Infinity;
                
                enemyShips.forEach(enemy => {
                    if (!enemy) return;
                    const dx = enemy.x - this.x;
                    const dy = enemy.y - this.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    // Check if enemy is heading toward player or alpha
                    const playerDx = (playerShip ? playerShip.x : this.x) - enemy.x;
                    const playerDy = (playerShip ? playerShip.y : this.y) - enemy.y;
                    const playerDist = Math.sqrt(playerDx * playerDx + playerDy * playerDy);
                    
                    if (distance < 300 && playerDist < 200) {
                        if (distance < nearestDistance) {
                            nearestDistance = distance;
                            nearestThreat = enemy;
                        }
                    }
                });
                
                if (nearestThreat) {
                    this.interceptTarget = nearestThreat;
                    // Calculate intercept point
                    const interceptTime = nearestDistance / 5; // Estimate
                    return {
                        x: nearestThreat.x + (nearestThreat.velocityX || 0) * interceptTime,
                        y: nearestThreat.y + (nearestThreat.velocityY || 0) * interceptTime,
                        angle: Math.atan2(nearestThreat.y - this.y, nearestThreat.x - this.x),
                        mode: 'intercept'
                    };
                }
            }
        } else if (escortMode === 'cover') {
            // Cover mode: Provide covering fire for retreat
            if (playerShip && playerShipActive) {
                const dx = playerShip.x - this.x;
                const dy = playerShip.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Position between player and enemies
                return {
                    x: playerShip.x - Math.cos(playerShip.angle) * 150,
                    y: playerShip.y - Math.sin(playerShip.angle) * 150,
                    angle: playerShip.angle + Math.PI, // Face away from player
                    mode: 'cover'
                };
            }
        }
        
        return null;
    }
    
    // ADVANCED PATTERNS: Tactical Attack Sequences
    executeAlphaStrikeCombo(enemyShips, aiShips) {
        if (!alphaAttackEnabled || !this.isAlpha || !enemyShips || enemyShips.length === 0) {
            this.attackSequence = 'none';
            return null;
        }
        
        const target = this.alphaAttackTarget || this.findNearestEnemy(enemyShips)?.enemyShip;
        if (!target) return null;
        
        // Initialize sequence if not started
        if (this.attackSequence !== 'alphaStrike') {
            this.attackSequence = 'alphaStrike';
            this.attackSequenceTimer = 0;
            this.attackSequencePhase = 0;
        }
        
        this.attackSequenceTimer++;
        
        // Phase 0: Alpha fires first (0-30 frames)
        if (this.attackSequencePhase === 0) {
            if (this.attackSequenceTimer < 30) {
                // Alpha fires
                if (this.shootCooldown <= 0 && this.shouldFireAtEnemy(target)) {
                    this.fire(target);
                }
            } else {
                // Move to phase 1: Wingmen fire
                this.attackSequencePhase = 1;
                this.attackSequenceTimer = 0;
            }
        }
        // Phase 1: Wingmen fire (30-60 frames)
        else if (this.attackSequencePhase === 1) {
            if (this.attackSequenceTimer < 30) {
                // Wingmen fire (handled in their makeDecision)
                // Signal wingmen to fire
                const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
                wingmen.forEach(w => {
                    if (w && w.shootCooldown <= 0 && w.shouldFireAtEnemy(target)) {
                        w.fire(target);
                    }
                });
            } else {
                // Move to phase 2: Alpha fires again
                this.attackSequencePhase = 2;
                this.attackSequenceTimer = 0;
            }
        }
        // Phase 2: Alpha fires again (60-90 frames)
        else if (this.attackSequencePhase === 2) {
            if (this.attackSequenceTimer < 30) {
                // Alpha fires again
                if (this.shootCooldown <= 0 && this.shouldFireAtEnemy(target)) {
                    this.fire(target);
                }
            } else {
                // Sequence complete, reset
                this.attackSequence = 'none';
                this.attackSequenceTimer = 0;
                this.attackSequencePhase = 0;
            }
        }
        
        return target;
    }
    
    executeWaveAttack(enemyShips, aiShips) {
        if (!alphaAttackEnabled || !this.isAlpha || !enemyShips || enemyShips.length === 0) {
            this.attackSequence = 'none';
            return null;
        }
        
        const target = this.alphaAttackTarget || this.findNearestEnemy(enemyShips)?.enemyShip;
        if (!target) return null;
        
        // Initialize wave attack if not started
        if (this.attackSequence !== 'wave') {
            this.attackSequence = 'wave';
            this.attackSequenceTimer = 0;
            this.waveNumber = 0;
        }
        
        this.attackSequenceTimer++;
        const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
        const totalWaves = Math.min(3, Math.floor(wingmen.length / 2) + 1);
        
        // Wave 0: Alpha leads (0-60 frames)
        if (this.waveNumber === 0) {
            if (this.attackSequenceTimer < 60) {
                // Alpha engages
                if (this.shootCooldown <= 0 && this.shouldFireAtEnemy(target)) {
                    this.fire(target);
                }
            } else {
                // Move to wave 1
                this.waveNumber = 1;
                this.attackSequenceTimer = 0;
            }
        }
        // Wave 1: First group of wingmen (60-120 frames)
        else if (this.waveNumber === 1 && wingmen.length >= 2) {
            if (this.attackSequenceTimer < 60) {
                const wave1Ships = wingmen.slice(0, Math.ceil(wingmen.length / 2));
                wave1Ships.forEach(w => {
                    if (w && w.shootCooldown <= 0 && w.shouldFireAtEnemy(target)) {
                        w.fire(target);
                    }
                });
            } else {
                if (totalWaves > 2) {
                    this.waveNumber = 2;
                    this.attackSequenceTimer = 0;
                } else {
                    // Sequence complete
                    this.attackSequence = 'none';
                    this.attackSequenceTimer = 0;
                    this.waveNumber = 0;
                }
            }
        }
        // Wave 2: Second group provides cover (120-180 frames)
        else if (this.waveNumber === 2) {
            if (this.attackSequenceTimer < 60) {
                const wave2Ships = wingmen.slice(Math.ceil(wingmen.length / 2));
                wave2Ships.forEach(w => {
                    if (w && w.shootCooldown <= 0 && w.shouldFireAtEnemy(target)) {
                        w.fire(target);
                    }
                });
            } else {
                // Sequence complete
                this.attackSequence = 'none';
                this.attackSequenceTimer = 0;
                this.waveNumber = 0;
            }
        }
        
        return target;
    }
    
    // ADVANCED PATTERNS: Dynamic Formation Transitions
    morphFormation(targetFormation, aiShips) {
        if (!alphaAttackEnabled || !this.isAlpha) return false;
        
        if (this.formationType === targetFormation) {
            this.formationMorphing = false;
            return false;
        }
        
        // Start morphing
        if (!this.formationMorphing) {
            this.formationMorphing = true;
            this.formationMorphTarget = targetFormation;
            this.formationMorphProgress = 0;
        }
        
        // Smooth transition over 60 frames (1 second)
        this.formationMorphProgress += 1 / 60;
        
        if (this.formationMorphProgress >= 1.0) {
            // Morph complete
            this.formationType = targetFormation;
            this.formationMorphing = false;
            this.formationMorphProgress = 0;
            
            // Update all wingmen
            const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
            wingmen.forEach(w => {
                if (w) {
                    w.formationType = targetFormation;
                    w.formationTypeChanged = true;
                }
            });
            
            return true;
        }
        
        return false;
    }
    
    splitAndMergeFormation(enemyShips, aiShips) {
        if (!alphaAttackEnabled || !this.isAlpha || !enemyShips || enemyShips.length === 0) {
            this.splitFormation = false;
            this.mergeTimer = 0;
            return null;
        }
        
        const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
        if (wingmen.length < 2) return null;
        
        // Check if should split (when multiple enemies present)
        if (!this.splitFormation && enemyShips.length >= 2) {
            this.splitFormation = true;
            this.mergeTimer = 0;
            
            // Divide wingmen into two groups
            const midPoint = Math.ceil(wingmen.length / 2);
            this.splitGroups = [
                wingmen.slice(0, midPoint),
                wingmen.slice(midPoint)
            ];
            
            // Assign different targets to each group
            const targets = enemyShips.slice(0, 2);
            if (targets[0]) this.alphaAttackTarget = targets[0];
            if (targets[1] && this.splitGroups[1].length > 0) {
                this.splitGroups[1].forEach(w => {
                    if (w) w.assignedTarget = targets[1];
                });
            }
        }
        
        // Merge back after 180 frames (3 seconds) or when enemies reduced
        if (this.splitFormation) {
            this.mergeTimer++;
            
            if (this.mergeTimer >= 180 || enemyShips.length < 2) {
                this.splitFormation = false;
                this.mergeTimer = 0;
                this.splitGroups = [];
                
                // Reset assigned targets
                wingmen.forEach(w => {
                    if (w) w.assignedTarget = null;
                });
            }
        }
        
        return this.splitFormation;
    }
    
    // ADVANCED PATTERNS: Advanced Flanking Patterns
    executeHammerAndAnvil(enemyShips, aiShips) {
        if (!alphaAttackEnabled || !this.isAlpha || !enemyShips || enemyShips.length === 0) {
            this.flankPattern = 'none';
            return null;
        }
        
        const target = this.alphaAttackTarget || this.findNearestEnemy(enemyShips)?.enemyShip;
        if (!target) return null;
        
        const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
        if (wingmen.length < 2) return null;
        
        // Initialize hammer and anvil
        if (this.flankPattern !== 'hammerAnvil') {
            this.flankPattern = 'hammerAnvil';
            this.flankTimer = 0;
        }
        
        this.flankTimer++;
        
        // Alpha (Hammer) approaches from front
        const dx = target.x - this.x;
        const dy = target.y - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > 150) {
            // Alpha moves toward target (front approach)
            this.targetAngle = Math.atan2(dy, dx);
            if (gameRandom() < 0.5) {
                this.thrust(1.2);
            }
        }
        
        // Wingmen (Anvil) approach from rear
        wingmen.forEach((w, index) => {
            if (w) {
                const rearAngle = Math.atan2(target.y - w.y, target.x - w.x) + Math.PI; // Opposite direction
                const flankDistance = 120;
                const flankAngle = rearAngle + (index % 2 === 0 ? -Math.PI / 4 : Math.PI / 4);
                
                w.flankPosition = {
                    x: target.x + Math.cos(flankAngle) * flankDistance,
                    y: target.y + Math.sin(flankAngle) * flankDistance,
                    angle: Math.atan2(target.y - w.y, target.x - w.x)
                };
                
                // Move toward flank position
                const dxFlank = w.flankPosition.x - w.x;
                const dyFlank = w.flankPosition.y - w.y;
                w.targetAngle = Math.atan2(dyFlank, dxFlank);
                
                if (gameRandom() < 0.4) {
                    w.thrust(1.1);
                }
                
                // Fire when in position
                const distToTarget = Math.sqrt((target.x - w.x) ** 2 + (target.y - w.y) ** 2);
                if (distToTarget < w.enemyFiringRange && w.shouldFireAtEnemy(target)) {
                    if (w.shootCooldown <= 0) {
                        w.fire(target);
                    }
                }
            }
        });
        
        return target;
    }
    
    executeScissorsAttack(enemyShips, aiShips) {
        if (!alphaAttackEnabled || !this.isAlpha || !enemyShips || enemyShips.length === 0) {
            this.flankPattern = 'none';
            return null;
        }
        
        const target = this.alphaAttackTarget || this.findNearestEnemy(enemyShips)?.enemyShip;
        if (!target) return null;
        
        const wingmen = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel');
        if (wingmen.length < 2) return null;
        
        // Initialize scissors attack
        if (this.flankPattern !== 'scissors') {
            this.flankPattern = 'scissors';
            this.flankTimer = 0;
        }
        
        this.flankTimer++;
        
        // Divide wingmen into two groups for scissors
        const midPoint = Math.ceil(wingmen.length / 2);
        const leftGroup = wingmen.slice(0, midPoint);
        const rightGroup = wingmen.slice(midPoint);
        
        // Calculate approach angles (from opposite sides)
        const targetAngle = Math.atan2(target.y - this.y, target.x - this.x);
        const leftAngle = targetAngle - Math.PI / 2; // Left side
        const rightAngle = targetAngle + Math.PI / 2; // Right side
        
        // Left group approaches from left
        leftGroup.forEach((w, index) => {
            if (w) {
                const approachDistance = 150 + (index * 30);
                w.flankPosition = {
                    x: target.x + Math.cos(leftAngle) * approachDistance,
                    y: target.y + Math.sin(leftAngle) * approachDistance,
                    angle: targetAngle
                };
                
                const dx = w.flankPosition.x - w.x;
                const dy = w.flankPosition.y - w.y;
                w.targetAngle = Math.atan2(dy, dx);
                
                if (gameRandom() < 0.4) {
                    w.thrust(1.1);
                }
                
                // Fire when in range
                const distToTarget = Math.sqrt((target.x - w.x) ** 2 + (target.y - w.y) ** 2);
                if (distToTarget < w.enemyFiringRange && w.shouldFireAtEnemy(target)) {
                    if (w.shootCooldown <= 0) {
                        w.fire(target);
                    }
                }
            }
        });
        
        // Right group approaches from right
        rightGroup.forEach((w, index) => {
            if (w) {
                const approachDistance = 150 + (index * 30);
                w.flankPosition = {
                    x: target.x + Math.cos(rightAngle) * approachDistance,
                    y: target.y + Math.sin(rightAngle) * approachDistance,
                    angle: targetAngle
                };
                
                const dx = w.flankPosition.x - w.x;
                const dy = w.flankPosition.y - w.y;
                w.targetAngle = Math.atan2(dy, dx);
                
                if (gameRandom() < 0.4) {
                    w.thrust(1.1);
                }
                
                // Fire when in range
                const distToTarget = Math.sqrt((target.x - w.x) ** 2 + (target.y - w.y) ** 2);
                if (distToTarget < w.enemyFiringRange && w.shouldFireAtEnemy(target)) {
                    if (w.shootCooldown <= 0) {
                        w.fire(target);
                    }
                }
            }
        });
        
        // Alpha can also engage
        const dx = target.x - this.x;
        const dy = target.y - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < this.enemyFiringRange && this.shouldFireAtEnemy(target)) {
            if (this.shootCooldown <= 0) {
                this.fire(target);
            }
        }
        
        return target;
    }
    
    // PHASE 2: Apply formation abilities
    applyFormationAbilities(aiShips) {
        if (!alphaAttackEnabled || !this.alphaShip) {
            // Reset abilities when not in formation
            this.formationShieldActive = false;
            this.formationBoostActive = false;
            this.formationHealActive = false;
            this.formationRadarActive = false;
            if (this.isAlpha) {
                this.detectionRadius = this.baseDetectionRadius;
            }
            return;
        }
        
        const alpha = this.alphaShip;
        const wingmen = aiShips.filter(ship => ship && ship !== alpha && ship.alphaShip === alpha && ship.role !== 'fuel');
        const inFormation = wingmen.length >= 1;
        
        if (inFormation) {
            // Formation Shield: Combined shield when 3+ ships in formation
            if (wingmen.length >= 2) {
                this.formationShieldActive = true;
                if (alpha) alpha.formationShieldActive = true;
                wingmen.forEach(w => {
                    if (w) w.formationShieldActive = true;
                });
            } else {
                this.formationShieldActive = false;
            }
            
            // Formation Boost: Speed boost when in formation
            this.formationBoostActive = true;
            
            // Formation Radar: Extended detection when in formation
            if (this.isAlpha) {
                this.formationRadarActive = true;
                this.detectionRadius = this.baseDetectionRadius * 1.3;
            }
            
            // Formation Heal: Healing aura for support role
            if (this.role === 'support' || (alpha && alpha.role === 'support')) {
                this.formationHealActive = true;
            } else {
                this.formationHealActive = false;
            }
        } else {
            // Reset abilities when formation breaks
            this.formationShieldActive = false;
            this.formationBoostActive = false;
            this.formationHealActive = false;
            this.formationRadarActive = false;
            if (this.isAlpha) {
                this.detectionRadius = this.baseDetectionRadius;
            }
        }
    }

    // Check if should fire at asteroid (more aggressive/continuous firing)
    shouldFire(asteroid) {
        const dx = asteroid.x - this.x;
        const dy = asteroid.y - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const angleToAsteroid = Math.atan2(dy, dx);
        const angleDiff = Math.abs(this.normalizeAngle(angleToAsteroid - this.angle));
        
        // Calculate relative velocity for collision prediction
        const relativeVelX = asteroid.velocityX - this.velocityX;
        const relativeVelY = asteroid.velocityY - this.velocityY;
        const relativeSpeed = Math.sqrt(relativeVelX * relativeVelX + relativeVelY * relativeVelY);
        const timeToCollision = relativeSpeed > 0 ? distance / relativeSpeed : Infinity;
        
        // Condition 1: Imminent collision threat (more lenient)
        if (distance < this.imminentThreatDistance && 
            angleDiff < this.collisionAngleThreshold &&
            timeToCollision < 90) { // Less than 1.5 seconds at 60fps
            return true;
        }
        
        // Condition 2: Asteroid in path and blocking (more aggressive)
        if (distance < this.firingRange && 
            angleDiff < Math.PI / 3 && // 60 degrees - wider arc
            asteroid.size >= this.minAsteroidSize &&
            timeToCollision < 180) { // Less than 3 seconds
            return true;
        }
        
        // Condition 3: Large asteroid directly ahead (more lenient)
        if (distance < this.firingRange &&
            angleDiff < this.collisionAngleThreshold &&
            asteroid.size >= 30) { // Medium/large asteroids
            return true;
        }
        
        // Condition 4: Any asteroid in front within range (continuous firing)
        if (distance < this.firingRange &&
            angleDiff < Math.PI / 2 && // 90 degrees - very wide arc
            asteroid.size >= this.minAsteroidSize) {
            return true;
        }
        
        return false;
    }

    // Find nearest enemy ship
    findNearestEnemy(enemyShips) {
        let nearest = null;
        let nearestDistance = Infinity;
        
        enemyShips.forEach(enemyShip => {
            if (enemyShip) {
                const dx = enemyShip.x - this.x;
                const dy = enemyShip.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < this.enemyDetectionRadius && distance < nearestDistance) {
                    const angleToEnemy = Math.atan2(dy, dx);
                    const angleDiff = Math.abs(this.normalizeAngle(angleToEnemy - this.angle));
                    
                    // Consider enemies in front (within 150 degrees)
                    if (angleDiff < Math.PI * 5/6) {
                        nearestDistance = distance;
                        nearest = { enemyShip, distance, angle: angleToEnemy };
                    }
                }
            }
        });
        
        return nearest;
    }

    // Check if should fire at enemy ship
    shouldFireAtEnemy(enemyShip) {
        if (!enemyShip) return false;
        const dx = enemyShip.x - this.x;
        const dy = enemyShip.y - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const angleToEnemy = Math.atan2(dy, dx);
        const angleDiff = Math.abs(this.normalizeAngle(angleToEnemy - this.angle));
        
        // Fire if enemy is in range and in front
        if (distance < this.enemyFiringRange && angleDiff < Math.PI / 2) {
            return true;
        }
        return false;
    }

    // Predict enemy position for leading shots
    predictEnemyPosition(enemyShip) {
        if (!enemyShip) return null;
        const dx = enemyShip.x - this.x;
        const dy = enemyShip.y - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > this.enemyFiringRange) return null;
        
        // Calculate time for bullet to reach enemy
        const bulletSpeed = 8; // Player/AI bullet speed
        const timeToReach = distance / bulletSpeed;
        
        // Predict future position based on enemy velocity
        const predictedX = enemyShip.x + enemyShip.velocityX * timeToReach;
        const predictedY = enemyShip.y + enemyShip.velocityY * timeToReach;
        
        // Wrap predicted position
        const wrappedX = ((predictedX % worldW()) + worldW()) % worldW();
        const wrappedY = ((predictedY % worldH()) + worldH()) % worldH();
        
        return {
            x: wrappedX,
            y: wrappedY,
            angle: Math.atan2(wrappedY - this.y, wrappedX - this.x)
        };
    }

    // Coordinate firing with other AI ships (avoid over-firing on same target)
    coordinateFiring(enemyShip, aiShips) {
        if (!enemyShip) return true; // No coordination needed if no target
        
        let targetingCount = 0;
        const coordinationRadius = 200; // Range to check for other AI ships
        
        aiShips.forEach(aiShip => {
            if (aiShip && aiShip !== this) {
                const dx = aiShip.x - this.x;
                const dy = aiShip.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < coordinationRadius) {
                    // Check if this AI ship is also targeting the same enemy
                    const dxToEnemy = enemyShip.x - aiShip.x;
                    const dyToEnemy = enemyShip.y - aiShip.y;
                    const distToEnemy = Math.sqrt(dxToEnemy * dxToEnemy + dyToEnemy * dyToEnemy);
                    const angleToEnemy = Math.atan2(dyToEnemy, dxToEnemy);
                    const angleDiff = Math.abs(aiShip.normalizeAngle(angleToEnemy - aiShip.angle));
                    
                    // If AI ship is close to enemy and aiming at it, count as targeting
                    if (distToEnemy < aiShip.enemyFiringRange && angleDiff < Math.PI / 3) {
                        targetingCount++;
                    }
                }
            }
        });
        
        // Allow firing if less than 3 other ships are targeting the same enemy
        // This prevents over-firing while still allowing coordinated attacks
        return targetingCount < 3;
    }

    // Check if enemies are present nearby
    hasNearbyEnemies(enemyShips) {
        if (!enemyShips || enemyShips.length === 0) return false;
        
        for (let enemy of enemyShips) {
            if (enemy) {
                const dx = enemy.x - this.x;
                const dy = enemy.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // If any enemy is within defensive range, return true
                if (distance < this.enemyDetectionRadius) {
                    return true;
                }
            }
        }
        return false;
    }

    // Predictive Avoidance: Predict enemy bullet trajectories and avoid them
    predictBulletCollision(enemyBullets) {
        if (!enemyBullets || enemyBullets.length === 0) return null;
        
        let nearestThreat = null;
        let minTimeToCollision = Infinity;
        
        enemyBullets.forEach(bullet => {
            if (!bullet) return;
            
            // Calculate relative position and velocity
            const dx = bullet.x - this.x;
            const dy = bullet.y - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Skip if bullet is too far away
            if (distance > 300) return;
            
            // Calculate relative velocity
            const relativeVx = bullet.velocityX - this.velocityX;
            const relativeVy = bullet.velocityY - this.velocityY;
            const relativeSpeed = Math.sqrt(relativeVx * relativeVx + relativeVy * relativeVy);
            
            if (relativeSpeed < 0.1) return; // Bullet moving too slowly relative to ship
            
            // Predict time to collision (simplified: assume linear trajectory)
            const dotProduct = dx * relativeVx + dy * relativeVy;
            const timeToCollision = -dotProduct / (relativeSpeed * relativeSpeed);
            
            // Check if collision is predicted within reasonable time (2 seconds = 120 frames)
            if (timeToCollision > 0 && timeToCollision < 120 && timeToCollision < minTimeToCollision) {
                // Calculate predicted collision point
                const predictedX = bullet.x + bullet.velocityX * timeToCollision;
                const predictedY = bullet.y + bullet.velocityY * timeToCollision;
                const predictedDistance = Math.sqrt(
                    (predictedX - this.x) ** 2 + (predictedY - this.y) ** 2
                );
                
                // Check if predicted collision is within ship's size
                if (predictedDistance < this.size * 2) {
                    minTimeToCollision = timeToCollision;
                    nearestThreat = {
                        bullet,
                        timeToCollision,
                        distance,
                        angle: Math.atan2(dy, dx),
                        threatLevel: 1 / (timeToCollision + 1) // Higher threat = closer collision
                    };
                }
            }
        });
        
        return nearestThreat;
    }
    
    // Flocking behavior: separation, alignment, cohesion (with ENHANCED defensive mode)
    calculateFlocking(aiShips, enemyShips) {
        // Check if enemies are present - use defensive parameters
        const enemiesPresent = this.hasNearbyEnemies(enemyShips);
        const separationDist = enemiesPresent ? this.defensiveSeparationDistance : this.separationDistance;
        const cohesionRad = enemiesPresent ? this.defensiveCohesionRadius : this.cohesionRadius;
        const flockWeight = enemiesPresent ? this.defensiveFlockWeight : this.flockWeight;
        
        // ENHANCED: Mutual protection - prioritize protecting low-health allies
        let lowHealthAlly = null;
        let lowestHealth = Infinity;
        if (enemiesPresent && this.mutualProtectionEnabled) {
            aiShips.forEach(ally => {
                if (ally && ally !== this) {
                    const dx = ally.x - this.x;
                    const dy = ally.y - this.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < this.flockRadius && ally.health < lowestHealth && ally.health < ally.maxHealth * 0.5) {
                        lowestHealth = ally.health;
                        lowHealthAlly = { ally, distance, dx, dy };
                    }
                }
            });
        }
        
        let separationX = 0, separationY = 0;
        let alignmentX = 0, alignmentY = 0;
        let cohesionX = 0, cohesionY = 0;
        let separationCount = 0;
        let alignmentCount = 0;
        let cohesionCount = 0;
        
        aiShips.forEach(aiShip => {
            if (aiShip && aiShip !== this) {
                const dx = aiShip.x - this.x;
                const dy = aiShip.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance > 0 && distance < this.flockRadius) {
                    // Separation: avoid crowding neighbors (tighter when enemies present)
                    if (distance < separationDist) {
                        const separationForce = enemiesPresent ? (1 / distance) * 0.5 : (1 / distance);
                        separationX -= (dx / distance) * separationForce;
                        separationY -= (dy / distance) * separationForce;
                        separationCount++;
                    }
                    
                    // Alignment: steer towards average heading of neighbors
                    if (distance < this.alignmentRadius) {
                        alignmentX += Math.cos(aiShip.angle);
                        alignmentY += Math.sin(aiShip.angle);
                        alignmentCount++;
                    }
                    
                    // Cohesion: steer towards average position of neighbors (stronger when enemies present)
                    if (distance < cohesionRad) {
                        const cohesionWeight = enemiesPresent ? 1.5 : 1.0; // Stronger cohesion when enemies present
                        cohesionX += aiShip.x * cohesionWeight;
                        cohesionY += aiShip.y * cohesionWeight;
                        cohesionCount++;
                    }
                }
            }
        });
        
        // Normalize and combine behaviors
        let flockAngle = this.angle;
        
        // ENHANCED: Mutual protection - prioritize moving to protect low-health allies
        if (lowHealthAlly && enemiesPresent) {
            const protectionAngle = Math.atan2(lowHealthAlly.dy, lowHealthAlly.dx);
            // Strong priority to move toward low-health ally to protect them
            flockAngle = protectionAngle;
        } else if (separationCount > 0) {
            const separationAngle = Math.atan2(separationY, separationX);
            // When enemies present, separation is less aggressive (allows closer grouping)
            flockAngle = enemiesPresent ? 
                this.angle + (separationAngle - this.angle) * 0.2 : // Even less aggressive (was 0.3)
                separationAngle;
        } else if (cohesionCount > 0) {
            // Cohesion has higher priority when enemies present
            cohesionX /= cohesionCount;
            cohesionY /= cohesionCount;
            const dx = cohesionX - this.x;
            const dy = cohesionY - this.y;
            const cohesionAngle = Math.atan2(dy, dx);
            
            if (enemiesPresent) {
                // ENHANCED: Defensive formations (circle or line)
                if (this.defensiveFormationType === 'circle' && cohesionCount >= 2) {
                    // Circle formation: position around center point
                    const centerX = cohesionX;
                    const centerY = cohesionY;
                    const angleToCenter = Math.atan2(centerY - this.y, centerX - this.x);
                    const formationRadius = 50; // Radius of circle formation
                    const desiredAngle = angleToCenter + Math.PI / 2; // Perpendicular to center
                    flockAngle = desiredAngle;
                } else if (this.defensiveFormationType === 'line' && cohesionCount >= 2) {
                    // Line formation: align with average heading
                    if (alignmentCount > 0) {
                        const alignmentAngle = Math.atan2(alignmentY, alignmentX);
                        flockAngle = alignmentAngle;
                    } else {
                        flockAngle = cohesionAngle;
                    }
                } else {
                    // Default: Strong cohesion when enemies present - move closer together
                    flockAngle = cohesionAngle;
                }
            } else if (alignmentCount > 0) {
                const alignmentAngle = Math.atan2(alignmentY, alignmentX);
                flockAngle = alignmentAngle;
            } else {
                flockAngle = cohesionAngle;
            }
        } else if (alignmentCount > 0) {
            const alignmentAngle = Math.atan2(alignmentY, alignmentX);
            flockAngle = alignmentAngle;
        }
        
        return { angle: flockAngle, weight: flockWeight };
    }

    // Protect nearby AI ships from enemies (enhanced defensive behavior)
    protectAllies(aiShips, enemyShips) {
        if (!enemyShips || enemyShips.length === 0) return null;
        
        let protectionAngle = null;
        let maxThreat = 0;
        let closestAllyDistance = Infinity;
        let closestAlly = null;
        
        // First, find closest ally
        aiShips.forEach(ally => {
            if (ally && ally !== this) {
                const dx = ally.x - this.x;
                const dy = ally.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < this.flockRadius && distance < closestAllyDistance) {
                    closestAllyDistance = distance;
                    closestAlly = { ally, distance, dx, dy };
                }
            }
        });
        
        // Check if any ally is threatened
        aiShips.forEach(ally => {
            if (ally && ally !== this) {
                const dx = ally.x - this.x;
                const dy = ally.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Check if ally is within protection range
                if (distance < this.flockRadius) {
                    // Check if any enemy is threatening this ally
                    enemyShips.forEach(enemy => {
                        if (enemy) {
                            const dxToAlly = ally.x - enemy.x;
                            const dyToAlly = ally.y - enemy.y;
                            const distToAlly = Math.sqrt(dxToAlly * dxToAlly + dyToAlly * dyToAlly);
                            
                            // If enemy is close to ally, this is a threat
                            if (distToAlly < enemy.attackRange) {
                                const threatLevel = 1 / (distToAlly + 1);
                                if (threatLevel > maxThreat) {
                                    maxThreat = threatLevel;
                                    // Move towards ally to protect them
                                    protectionAngle = Math.atan2(dy, dx);
                                }
                            }
                        }
                    });
                }
            }
        });
        
        // If no immediate threat but enemies present, move closer to nearest ally
        if (protectionAngle === null && closestAlly && this.hasNearbyEnemies(enemyShips)) {
            // Move toward nearest ally for mutual protection
            protectionAngle = Math.atan2(closestAlly.dy, closestAlly.dx);
        }
        
        return protectionAngle;
    }

    // Per-frame timer housekeeping.
    //
    // This used to live inside makeDecision(), which meant it only ran when the
    // rule-based controller was driving. Any other controller - the MARL path or
    // a trained policy - left shootCooldown pinned at its post-shot value, so the
    // ship fired once and was then silently unable to fire for the rest of the
    // episode. It has to run for every ship on every frame, whoever is deciding.
    updateCooldowns() {
        if (this.flankingCooldown > 0) this.flankingCooldown--;
        if (this.formationAbilityCooldown > 0) this.formationAbilityCooldown--;
        if (this.formationSwitchCooldown > 0) this.formationSwitchCooldown--;
        if (this.flankTimer > 0 && this.flankPattern === 'none') this.flankTimer = 0;
        if (this.mergeTimer > 0 && !this.splitFormation) this.mergeTimer = 0;
        if (this.shootCooldown > 0) this.shootCooldown--;
        if (this.healingCooldown > 0) this.healingCooldown--;
        if (this.alphaAttackCooldown > 0) this.alphaAttackCooldown--;
        if (this.fuelTransferTimer > 0 && !this.fuelTransferActive) this.fuelTransferTimer = 0;
    }

    // An alpha on fuel rescue duty is "fuel-bound": it cannot thrust or fire
    // with an empty tank, nor while linked to the tanker (the dock freezes both
    // ships, and burning fuel during the hold defeated the refuel).
    fuelBound() {
        return fuelRescueEnabled && alphaAttackEnabled && this.isAlpha;
    }

    fuelStarved() {
        return this.fuelBound() && (this.fuel <= 0 || this.fuelDocked || this.fuelTransferActive);
    }

    thrust(power = 1) {
        if (this.fuelStarved()) return;
        if (FXL) {
            this.thrustColor = (this.isAlpha && alphaAttackEnabled) ? GAME_THEME.gold
                : (this.role === 'fuel' && fuelRescueEnabled && alphaAttackEnabled) ? GAME_THEME.fuelShip
                : GAME_THEME.orange;
        }
        super.thrust(power);
        if (this.fuelBound()) {
            this.fuel = Math.max(0, this.fuel - this.fuelThrustCost * power / worldScale());
            updateAlphaHelpPing(this);
        }
    }

    tickFuel() {
        if (!fuelRescueEnabled || !alphaAttackEnabled) return;
        if (this.isAlpha) {
            this.fuel = Math.max(0, this.fuel - this.fuelPassiveDrain / worldScale());
            updateAlphaHelpPing(this);
        }
    }

    findAlphaNeedingFuel(aiShips) {
        if (!fuelRescueEnabled || this.role !== 'fuel' || this.fuel <= 0) return null;
        if (this.fuelStandby || this.fuelDeparting || this.fuelDocked || this.fuelTransferActive) return null;
        const alpha = aiShips.find(s => s && s.isAlpha && s.helpPingActive);
        if (!alpha) return null;
        const dx = alpha.x - this.x;
        const dy = alpha.y - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        return { alpha, distance, angle: Math.atan2(dy, dx) };
    }

    executeFuelStandby() {
        if (!this.standbyTarget) {
            this.standbyTarget = {
                x: Math.max(60, worldW() * 0.12),
                y: Math.max(60, worldH() * 0.88)
            };
        }
        const dx = this.standbyTarget.x - this.x;
        const dy = this.standbyTarget.y - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance > 35) {
            this.targetAngle = Math.atan2(dy, dx);
            const angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
            this.rotate(angleDiff > 0 ? 1 : -1);
            if (gameRandom() < 0.4) this.thrust(1.0);
        } else {
            this.velocityX *= 0.82;
            this.velocityY *= 0.82;
            // Replenish the reserve while parked (full again after ~11 s), so a
            // second HELP call does not find an empty tanker and starve the swarm.
            this.fuel = Math.min(this.maxFuel, this.fuel + 0.15);
        }
    }

    executeFuelDepart(aiShips) {
        const alpha = aiShips.find(s => s && s.isAlpha);
        if (alpha) {
            const dx = this.x - alpha.x;
            const dy = this.y - alpha.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            this.targetAngle = Math.atan2(dy, dx);
            const angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
            this.rotate(angleDiff > 0 ? 1 : -1);
            if (gameRandom() < 0.8) this.thrust(1.6);
        }
        this.fuelDepartTimer++;
        if (this.fuelDepartTimer >= this.fuelDepartDuration) {
            this.fuelDeparting = false;
            this.fuelStandby = true;
            this.fuelDepartTimer = 0;
        }
    }

    // Fuel tanker: never follows the wingman swarm — only HELP response, dock, depart, standby.
    handleFuelShipBehavior(aiShips) {
        if (this.fuelTransferActive || this.fuelDocked) {
            return true;
        }
        if (this.fuelDeparting) {
            this.executeFuelDepart(aiShips);
            return true;
        }

        const alpha = aiShips.find(s => s && s.isAlpha);
        if (this.fuelStandby) {
            if (alpha && alpha.helpPingActive) {
                this.fuelStandby = false;
            } else {
                this.executeFuelStandby();
                return true;
            }
        }

        const alphaNeed = this.findAlphaNeedingFuel(aiShips);
        if (alphaNeed) {
            this.targetAngle = alphaNeed.angle;
            const angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
            this.rotate(angleDiff > 0 ? 1 : -1);
            if (gameRandom() < 0.65) this.thrust(1.45);
            return true;
        }

        this.fuelStandby = true;
        this.executeFuelStandby();
        return true;
    }

    // Fire bullet at enemy or asteroid
    fire(targetEnemy = null) {
        if (this.fuelStarved()) return;
        if (this.shootCooldown <= 0) {
            let fireAngle = this.angle;
            
            // If firing at enemy, use predictive aiming
            if (targetEnemy) {
                const predicted = this.predictEnemyPosition(targetEnemy);
                if (predicted) {
                    fireAngle = predicted.angle;
                } else {
                    const dx = targetEnemy.x - this.x;
                    const dy = targetEnemy.y - this.y;
                    fireAngle = Math.atan2(dy, dx);
                }
            }
            
            const bulletX = this.x + Math.cos(fireAngle) * this.size;
            const bulletY = this.y + Math.sin(fireAngle) * this.size;
            bullets.push(new Bullet(bulletX, bulletY, fireAngle, true, this)); // true = AI bullet
            if (this._actionRecord) {
                // Record which fire mode the teacher used, so the clone learns
                // target selection rather than just "fire".
                if (!targetEnemy) {
                    this._actionRecord.fire = FIRE_FORWARD;
                } else if (typeof asteroids !== 'undefined' && asteroids.indexOf(targetEnemy) >= 0) {
                    this._actionRecord.fire = FIRE_AT_ASTEROID;
                } else {
                    this._actionRecord.fire = FIRE_AT_ENEMY;
                }
            }
            this.shootCooldown = this.rapidFireCooldown; // Very short cooldown for rapid fire
            if (this.fuelBound()) {
                this.fuel = Math.max(0, this.fuel - this.fuelFireCost / worldScale());
                updateAlphaHelpPing(this);
            }
            playFireSound(true); // Play AI firing sound
        }
    }

    // Check if cornered (near edges with limited escape)
    isCornered() {
        const edgeThreshold = 100;
        const nearLeft = this.x < edgeThreshold;
        const nearRight = this.x > worldW() - edgeThreshold;
        const nearTop = this.y < edgeThreshold;
        const nearBottom = this.y > worldH() - edgeThreshold;
        return (nearLeft || nearRight) && (nearTop || nearBottom);
    }

    // Calculate threat level (0-1) based on nearby threats
    calculateThreatLevel(asteroids, enemyShips) {
        let threatLevel = 0;
        const nearestAsteroid = this.findNearestAsteroid(asteroids);
        const nearestEnemy = this.findNearestEnemy(enemyShips);
        
        if (nearestAsteroid && nearestAsteroid.distance < 150) {
            threatLevel += 0.3 * (1 - nearestAsteroid.distance / 150);
        }
        if (nearestEnemy && nearestEnemy.distance < 300) {
            threatLevel += 0.4 * (1 - nearestEnemy.distance / 300);
        }
        
        return Math.min(1.0, threatLevel);
    }
    
    // Phase 5.1: Extract features for ML model (canonical 30-dim)
    extractFeatures(asteroids, aiShips, playerShip, enemyShips, enemyBullets = []) {
        if (typeof MLFeatureExtractor !== 'undefined') {
            const features = MLFeatureExtractor.extract(
                this, asteroids, aiShips, playerShip, enemyShips, enemyBullets,
                { curriculumStage: typeof window !== 'undefined' ? window.marlCurriculumStage : 0 }
            );
            return MLFeatureExtractor.toLegacy13(features);
        }

        const nearestAsteroid = this.findNearestAsteroid(asteroids);
        const nearestEnemy = this.findNearestEnemy(enemyShips);
        const threatLevel = this.calculateThreatLevel(asteroids, enemyShips);
        let nearbyAllies = 0;
        aiShips.forEach(otherShip => {
            if (otherShip !== this) {
                const dx = otherShip.x - this.x;
                const dy = otherShip.y - this.y;
                if (Math.sqrt(dx * dx + dy * dy) < this.flockRadius) nearbyAllies++;
            }
        });
        const flockDensity = Math.min(1.0, nearbyAllies / 9);
        return {
            health: this.health / this.maxHealth,
            phase: this.aiPhase / 3.0,
            shieldActive: this.shieldActive ? 1 : 0,
            velocity: Math.min(1.0, Math.sqrt(this.velocityX ** 2 + this.velocityY ** 2) / 5.0),
            asteroidCount: Math.min(1.0, asteroids.length / 10.0),
            nearestAsteroidDistance: nearestAsteroid ? Math.min(1.0, nearestAsteroid.distance / 200.0) : 1.0,
            enemyCount: Math.min(1.0, enemyShips.length / 10.0),
            nearestEnemyDistance: nearestEnemy ? Math.min(1.0, nearestEnemy.distance / 400.0) : 1.0,
            allyCount: Math.min(1.0, nearbyAllies / 9.0),
            threatLevel,
            flockDensity,
            hasNearbyEnemies: this.hasNearbyEnemies(enemyShips) ? 1 : 0
        };
    }

    extractCanonicalFeatures(asteroids, aiShips, playerShip, enemyShips, enemyBullets = []) {
        if (typeof MLFeatureExtractor !== 'undefined') {
            return MLFeatureExtractor.extract(
                this, asteroids, aiShips, playerShip, enemyShips, enemyBullets,
                { curriculumStage: typeof window !== 'undefined' ? window.marlCurriculumStage : 0 }
            );
        }
        return this.extractFeatures(asteroids, aiShips, playerShip, enemyShips, enemyBullets);
    }
    
    // Phase 5.1: Get ML-suggested parameters with safety bounds
    getMLParameters(features) {
        if (!this.mlEnabled || !this.useMLParameters) {
            // Fallback to rule-based base parameters
            return {
                detectionRadius: this.baseDetectionRadius,
                firingRange: this.baseFiringRange,
                flockWeight: this.baseFlockWeight,
                thrustFrequency: this.baseThrustFrequency,
                enemyFiringRange: this.baseEnemyFiringRange
            };
        }
        
        const adapter = (typeof mlModelAdapter !== 'undefined' && mlModelAdapter) ? mlModelAdapter : null;
        if (adapter && mlUseNeuralNetwork && this._cachedMLParams) {
            return this._cachedMLParams;
        }

        // Legacy single model interface
        if (this.mlModel && typeof this.mlModel.predict === 'function' && !mlUseNeuralNetwork) {
            const mlSuggestions = this.mlModel.predict(features);
            
            // Apply safety bounds (rule-based constraints)
            return {
                detectionRadius: Math.max(50, Math.min(200, mlSuggestions.detectionRadius || this.baseDetectionRadius)),
                firingRange: Math.max(100, Math.min(400, mlSuggestions.firingRange || this.baseFiringRange)),
                flockWeight: Math.max(0.1, Math.min(0.8, mlSuggestions.flockWeight || this.baseFlockWeight)),
                thrustFrequency: Math.max(0.01, Math.min(0.1, mlSuggestions.thrustFrequency || this.baseThrustFrequency)),
                enemyFiringRange: Math.max(150, Math.min(400, mlSuggestions.enemyFiringRange || this.baseEnemyFiringRange))
            };
        }
        
        // Simple rule-based "ML model" (placeholder until real model is trained)
        // This adapts parameters based on game state
        let detectionRadius = this.baseDetectionRadius;
        let firingRange = this.baseFiringRange;
        let flockWeight = this.baseFlockWeight;
        let thrustFrequency = this.baseThrustFrequency;
        let enemyFiringRange = this.baseEnemyFiringRange;
        
        // Adaptive detection radius based on threat level
        if (features.threatLevel > 0.7) {
            detectionRadius = 150; // Increase detection when high threat
        } else if (features.threatLevel < 0.3) {
            detectionRadius = 80; // Decrease when low threat (save computation)
        }
        
        // Adaptive firing range based on enemy presence
        if (features.enemyCount > 0.5) {
            firingRange = 250; // Increase range when many enemies
            enemyFiringRange = 300; // Increase enemy firing range
        } else if (features.asteroidCount > 0.5) {
            firingRange = 180; // Focus on closer asteroids
        }
        
        // Adaptive flocking weight based on flock density and enemies
        if (features.hasNearbyEnemies) {
            flockWeight = 0.6; // Stronger flocking when enemies present (defensive mode)
        } else if (features.flockDensity > 0.5) {
            flockWeight = 0.4; // Moderate flocking when many allies
        } else {
            flockWeight = 0.2; // Lighter flocking when few allies
        }
        
        // Adaptive thrust frequency based on threat
        if (features.threatLevel > 0.6) {
            thrustFrequency = 0.04; // More frequent thrust when threatened
        } else {
            thrustFrequency = 0.015; // Less frequent when safe
        }
        
        // Apply safety bounds
        return {
            detectionRadius: Math.max(50, Math.min(200, detectionRadius)),
            firingRange: Math.max(100, Math.min(400, firingRange)),
            flockWeight: Math.max(0.1, Math.min(0.8, flockWeight)),
            thrustFrequency: Math.max(0.01, Math.min(0.1, thrustFrequency)),
            enemyFiringRange: Math.max(150, Math.min(400, enemyFiringRange))
        };
    }
    
    // Phase 5.2: Get ML-suggested priority weights with safety bounds
    getMLPriorityWeights(features) {
        if (!this.mlEnabled || !this.useMLPriorities) {
            // Fallback to rule-based base priority weights
            return {...this.basePriorityWeights};
        }
        
        const adapter = (typeof mlModelAdapter !== 'undefined' && mlModelAdapter) ? mlModelAdapter : null;
        if (adapter && mlUseNeuralNetwork && this._cachedMLPriorities) {
            return this._cachedMLPriorities;
        }

        if (this.mlModel && typeof this.mlModel.predictPriorityWeights === 'function' && !mlUseNeuralNetwork) {
            const mlWeights = this.mlModel.predictPriorityWeights(features);
            
            // Apply safety bounds (rule-based constraints)
            // avoidShips must always be highest priority (safety)
            return {
                avoidShips: 1.0,  // Never reduce (safety)
                fireAtEnemies: Math.max(0.5, Math.min(1.0, mlWeights.fireAtEnemies || this.basePriorityWeights.fireAtEnemies)),
                protectAllies: Math.max(0.3, Math.min(0.8, mlWeights.protectAllies || this.basePriorityWeights.protectAllies)),
                fireAtAsteroids: Math.max(0.2, Math.min(0.6, mlWeights.fireAtAsteroids || this.basePriorityWeights.fireAtAsteroids)),
                avoidAsteroids: Math.max(0.2, Math.min(0.5, mlWeights.avoidAsteroids || this.basePriorityWeights.avoidAsteroids)),
                randomNav: Math.max(0.05, Math.min(0.2, mlWeights.randomNav || this.basePriorityWeights.randomNav))
            };
        }
        
        // Simple rule-based "ML model" (placeholder until real model is trained)
        // This adapts priority weights based on game state
        let fireAtEnemies = this.basePriorityWeights.fireAtEnemies;
        let protectAllies = this.basePriorityWeights.protectAllies;
        let fireAtAsteroids = this.basePriorityWeights.fireAtAsteroids;
        let avoidAsteroids = this.basePriorityWeights.avoidAsteroids;
        let randomNav = this.basePriorityWeights.randomNav;
        
        // Adapt priorities based on situation
        
        // High enemy count -> prioritize combat
        if (features.enemyCount > 0.5) {
            fireAtEnemies = 0.9; // Very high priority
            protectAllies = 0.7; // Higher priority for protection
            fireAtAsteroids = 0.3; // Lower priority
            avoidAsteroids = 0.25; // Lower priority
        }
        
        // High threat level -> prioritize survival
        if (features.threatLevel > 0.7) {
            avoidAsteroids = 0.4; // Higher priority
            fireAtAsteroids = 0.35; // Medium priority
            randomNav = 0.05; // Very low priority
        }
        
        // High flock density -> prioritize coordination
        if (features.flockDensity > 0.5) {
            protectAllies = 0.65; // Higher priority
            fireAtEnemies = 0.75; // Slightly lower (coordinate instead of solo)
        }
        
        // Low health -> prioritize survival over combat
        if (features.health < 0.5) {
            avoidAsteroids = 0.45; // Higher priority
            fireAtEnemies = 0.65; // Lower priority
            protectAllies = 0.5; // Lower priority
        }
        
        // Nearby enemies -> prioritize combat and protection
        if (features.hasNearbyEnemies) {
            fireAtEnemies = 0.85; // Very high priority
            protectAllies = 0.7; // High priority
            fireAtAsteroids = 0.3; // Lower priority
        }
        
        // Apply safety bounds
        return {
            avoidShips: 1.0,  // Always highest (safety)
            fireAtEnemies: Math.max(0.5, Math.min(1.0, fireAtEnemies)),
            protectAllies: Math.max(0.3, Math.min(0.8, protectAllies)),
            fireAtAsteroids: Math.max(0.2, Math.min(0.6, fireAtAsteroids)),
            avoidAsteroids: Math.max(0.2, Math.min(0.5, avoidAsteroids)),
            randomNav: Math.max(0.05, Math.min(0.2, randomNav))
        };
    }

    // Phase 5.3: Get ML-predicted action
    getMLAction(features) {
        const canonical = features._vector
            ? features
            : (typeof MLFeatureExtractor !== 'undefined'
                ? MLFeatureExtractor.extract(this, asteroids, aiShips, ship, enemyShips, enemyBullets)
                : features);

        if (this._cachedMLAction) return this._cachedMLAction;

        const adapter = (typeof mlModelAdapter !== 'undefined' && mlModelAdapter) ? mlModelAdapter : null;
        if (adapter && mlUseNeuralNetwork) {
            return null; // Async - use cached action from batch inference
        }

        if (this.mlModel && typeof this.mlModel.predictAction === 'function') {
            return this.mlModel.predictAction(canonical);
        }

        // Rule-based action fallback for Phase 5.3 sandbox
        if (canonical.threatLevel > 0.7 || canonical.bulletThreat > 0) {
            return { action: 'shield', rotation: 0, target: null, priority: 0.9 };
        }
        if (canonical.nearestEnemyDistance < 0.5) {
            return { action: 'fire', rotation: 0, target: null, priority: 0.8 };
        }
        if (canonical.threatLevel > 0.4) {
            return { action: 'thrust', rotation: 0, target: null, priority: 0.6 };
        }
        return { action: 'idle', rotation: 0, target: null, priority: 0.3 };
    }

    isSafeMLAction(mlAction, asteroids, aiShips, playerShip, enemyShips) {
        if (!mlAction) return false;

        if (mlAction.action === 'fire' && mlAction.target) {
            for (const ally of aiShips) {
                if (!ally || ally === this) continue;
                const dx = mlAction.target.x - ally.x;
                const dy = mlAction.target.y - ally.y;
                if (Math.sqrt(dx * dx + dy * dy) < 40) return false;
            }
        }

        if (mlAction.action === 'thrust') {
            const nextX = this.x + Math.cos(this.angle) * 20;
            const nextY = this.y + Math.sin(this.angle) * 20;
            for (const asteroid of asteroids) {
                const dx = nextX - asteroid.x;
                const dy = nextY - asteroid.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < (asteroid.radius || 20) + this.radius) return false;
            }
        }

        if (mlAction.action === 'shield' && this.shieldCooldown > 0) return false;

        return true;
    }

    executeMLAction(mlAction, asteroids, aiShips, playerShip, enemyShips) {
        if (!mlAction) return;

        switch (mlAction.action) {
            case 'rotate':
                if (mlAction.rotation === -1) this.rotate(-1);
                else if (mlAction.rotation === 1) this.rotate(1);
                break;
            case 'thrust':
                this.thrust();
                break;
            case 'fire':
                if (this.shootCooldown <= 0) this.fire();
                break;
            case 'shield':
                if (this.shieldCooldown <= 0) {
                    this.shieldActive = true;
                    this.shieldDuration = 60;
                    this.shieldCooldown = 300;
                }
                break;
            default:
                break;
        }
    }
    
    // AI decision making - ENHANCED with predictive avoidance
    makeDecision(asteroids, aiShips, playerShip, enemyShips, enemyBullets = []) {
        if (fuelRescueEnabled && alphaAttackEnabled && this.role === 'fuel') {
            if (this.handleFuelShipBehavior(aiShips)) {
                return;
            }
        }

        // Phase 5.1: Update ML parameters (every N frames for performance)
        this.mlInferenceFrame++;
        if (this.mlInferenceFrame >= this.mlInferenceInterval) {
            this.mlInferenceFrame = 0;
            
            // Extract features and get ML-suggested parameters
            const features = this.extractCanonicalFeatures(asteroids, aiShips, playerShip, enemyShips, enemyBullets);
            const mlParams = this._cachedMLParams || this.getMLParameters(features);
            
            // Phase 5.2: Get ML-suggested priority weights
            if (this.useMLPriorities) {
                this.mlPriorityWeights = this.getMLPriorityWeights(features);
            }
            
            // Apply ML parameters
            if (this.useMLParameters) {
                const oldParams = {
                    detectionRadius: this.detectionRadius,
                    firingRange: this.firingRange,
                    flockWeight: this.flockWeight
                };
                
                this.detectionRadius = mlParams.detectionRadius;
                this.firingRange = mlParams.firingRange;
                this.flockWeight = mlParams.flockWeight;
                this.thrustFrequency = mlParams.thrustFrequency;
                this.enemyFiringRange = mlParams.enemyFiringRange;
                
                // Update defensive flocking weight based on ML suggestion
                if (features.hasNearbyEnemies) {
                    this.defensiveFlockWeight = Math.max(0.4, Math.min(0.8, mlParams.flockWeight * 1.2));
                }
                
                // Debug logging (only log when parameters change significantly)
                if (Math.abs(oldParams.detectionRadius - mlParams.detectionRadius) > 10 ||
                    Math.abs(oldParams.firingRange - mlParams.firingRange) > 20 ||
                    Math.abs(oldParams.flockWeight - mlParams.flockWeight) > 0.1) {
                    console.log('[ML] Parameters updated:', {
                        detectionRadius: mlParams.detectionRadius.toFixed(1),
                        firingRange: mlParams.firingRange.toFixed(1),
                        flockWeight: mlParams.flockWeight.toFixed(2),
                        threatLevel: features.threatLevel.toFixed(2),
                        enemyCount: features.enemyCount.toFixed(2)
                    });
                }
            } else {
                // Use base parameters
                this.detectionRadius = this.baseDetectionRadius;
                this.firingRange = this.baseFiringRange;
                this.flockWeight = this.baseFlockWeight;
                this.thrustFrequency = this.baseThrustFrequency;
                this.enemyFiringRange = this.baseEnemyFiringRange;
            }
            
            // Phase 5.1: Collect training data (if enabled)
            if (this.collectTrainingData || mlDataCollectionEnabled) {
                this.collectTrainingSample(features, mlParams);
            }
            
            // Phase 5.3: Full ML Decision Making (if enabled)
            if (this.useMLDecisions && this.mlEnabled) {
                const mlAction = this.getMLAction(features);
                
                // Validate ML action for safety
                if (mlAction && this.isSafeMLAction(mlAction, asteroids, aiShips, playerShip, enemyShips)) {
                    // Execute ML action
                    this.executeMLAction(mlAction, asteroids, aiShips, playerShip, enemyShips);
                    return; // Skip rule-based decision making
                } else {
                    // ML action unsafe or unavailable - fallback to rule-based
                    // Continue with normal decision making below
                }
            }
        }
        
        const nearestAsteroid = this.findNearestAsteroid(asteroids);
        const nearestShip = this.findNearestShip(aiShips, playerShip);
        const nearestEnemy = this.findNearestEnemy(enemyShips);
        
        // HEALING SYSTEM: Find nearest Phase 1 ally for healing (Phase 3 ships only)
        const nearestPhase1Ally = this.findNearestPhase1Ally(aiShips);
        
        // ALPHA ATTACK: Find alpha ship and assign roles
        const alpha = this.findAlphaShip(aiShips);
        
        // ALPHA ESCAPE SYSTEM: Execute escape behavior (alpha only)
        if (this.isAlpha) {
            this.executeEscapeBehavior(enemyShips, enemyBullets, asteroids, aiShips);
        }
        
        // PHASE 2: Assign role to ship
        this.assignRole(aiShips);
        
        // PHASE 2: Calculate formation position based on formation type
        const formationPos = this.calculateFormationPositionAdvanced(aiShips) || this.calculateFormationPosition(aiShips);
        
        // ALPHA ATTACK: Execute alpha attack (select target)
        const alphaAttackTarget = this.executeAlphaAttack(enemyShips);
        
        // ALPHA ROTATION FIX: Always rotate alpha ship toward target when it has one (even during escape)
        // This ensures the alpha ship rotates even if it doesn't get the alphaAttack action
        if (this.isAlpha && alphaAttackTarget && !this.escapeMode) {
            const targetToUse = this.assignedTarget || alphaAttackTarget;
            if (targetToUse && targetToUse.x !== undefined && targetToUse.y !== undefined) {
                const dx = targetToUse.x - this.x;
                const dy = targetToUse.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance > 0) {
                    this.targetAngle = Math.atan2(dy, dx);
                    let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
                    // Always rotate, even for small differences, for smooth tracking
                    if (Math.abs(angleDiff) > 0.01) {
                        this.rotate(angleDiff > 0 ? 1 : -1);
                    }
                }
            }
        }
        
        // PHASE 2: Execute flanking maneuver
        const flankingPos = this.executeFlankingManeuver(enemyShips, aiShips);
        
        // PHASE 2: Apply formation abilities
        this.applyFormationAbilities(aiShips);
        
        // PHASE 3: Adaptive Formation (alpha only)
        if (this.isAlpha) {
            this.calculateAdaptiveFormation(enemyShips, aiShips);
        }
        
        // PHASE 3: Multi-Target Coordination (alpha only)
        if (this.isAlpha) {
            this.coordinateMultiTarget(enemyShips, aiShips);
        }
        
        // PHASE 3: Escort & Protection Modes
        const escortPos = this.executeEscortMode(playerShip, aiShips, enemyShips);
        
        // ADVANCED PATTERNS: Tactical Attack Sequences (alpha only)
        if (tacticalAttackSequencesEnabled && this.isAlpha && alphaAttackEnabled && enemyShips && enemyShips.length > 0) {
            // Choose attack sequence based on situation
            const enemyCount = enemyShips.length;
            const wingmenCount = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel').length;
            
            if (enemyCount === 1 && wingmenCount >= 2) {
                // Single enemy: Use alpha strike combo
                this.executeAlphaStrikeCombo(enemyShips, aiShips);
            } else if (enemyCount >= 2 && wingmenCount >= 4) {
                // Multiple enemies: Use wave attack
                this.executeWaveAttack(enemyShips, aiShips);
            }
        }
        
        // ADVANCED PATTERNS: Dynamic Formation Transitions (alpha only)
        if (dynamicFormationTransitionsEnabled && this.isAlpha && alphaAttackEnabled) {
            // Formation morphing based on threat
            const threatLevel = this.calculateThreatLevel(asteroids, enemyShips);
            const enemyCount = enemyShips ? enemyShips.length : 0;
            
            if (threatLevel > 0.7 && this.formationType !== 'circle') {
                // High threat: Morph to defensive circle
                this.morphFormation('circle', aiShips);
            } else if (enemyCount >= 3 && this.formationType !== 'line') {
                // Many enemies: Morph to line formation
                this.morphFormation('line', aiShips);
            } else if (enemyCount === 1 && this.formationType !== 'diamond') {
                // Single enemy: Morph to diamond
                this.morphFormation('diamond', aiShips);
            }
            
            // Split and merge formation
            this.splitAndMergeFormation(enemyShips, aiShips);
        }
        
        // ADVANCED PATTERNS: Advanced Flanking Patterns (alpha only)
        if (advancedFlankingEnabled && this.isAlpha && alphaAttackEnabled && enemyShips && enemyShips.length > 0) {
            const wingmenCount = aiShips.filter(ship => ship && ship !== this && ship.alphaShip === this && ship.role !== 'fuel').length;
            const target = this.alphaAttackTarget || this.findNearestEnemy(enemyShips)?.enemyShip;
            
            if (target && wingmenCount >= 2) {
                // Choose flanking pattern based on situation
                const dx = target.x - this.x;
                const dy = target.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance > 200 && wingmenCount >= 2) {
                    // Far target: Use hammer and anvil
                    this.executeHammerAndAnvil(enemyShips, aiShips);
                } else if (distance < 300 && wingmenCount >= 2) {
                    // Close target: Use scissors attack
                    this.executeScissorsAttack(enemyShips, aiShips);
                }
            }
        }
        
        // Get current priority weights (ML or base)
        const priorityWeights = this.useMLPriorities ? this.mlPriorityWeights : this.basePriorityWeights;
        
        // ENHANCED: Predictive Avoidance - Check for incoming bullets
        const bulletThreat = this.predictBulletCollision(enemyBullets);
        
        // Phase 5.2: Weighted priority decision making
        // Calculate weighted scores for each possible action
        const actionScores = {
            avoidShips: 0,
            avoidBullets: 0, // NEW: Predictive bullet avoidance
            seekHealing: 0, // NEW: Seek Phase 1 ally for healing
            alphaAttack: 0, // NEW: Alpha attack pattern
            fireAtEnemies: 0,
            protectAllies: 0,
            fireAtAsteroids: 0,
            avoidAsteroids: 0,
            randomNav: 0
        };
        
        // Score: Avoid ships (always highest if ship is very close)
        if (nearestShip && nearestShip.distance < 60) {
            actionScores.avoidShips = priorityWeights.avoidShips * (1.0 - nearestShip.distance / 60);
        }
        
        // ENHANCED: Score: Avoid bullets (high priority when threat detected)
        if (bulletThreat) {
            // Higher threat = higher priority (inverse of time to collision)
            const threatScore = bulletThreat.threatLevel;
            actionScores.avoidBullets = Math.min(1.0, threatScore * 1.5); // Very high priority for bullets
        }
        
        // HEALING SYSTEM: Score: Seek healing (Phase 3 ships only, high priority when critical)
        if (nearestPhase1Ally && this.aiPhase === 3 && this.healingCooldown === 0) {
            // Higher priority when closer to Phase 1 ally
            const healingDistance = Math.min(1.0, nearestPhase1Ally.distance / this.seekHealingRange);
            // Very high priority for healing when in Phase 3 (0.9 weight)
            actionScores.seekHealing = 0.9 * (1.0 - healingDistance * 0.3); // Higher score when closer
        }

        // ADVANCED PATTERNS: Score: Advanced flanking patterns (high priority when enabled)
        if (advancedFlankingEnabled && alphaAttackEnabled && this.isAlpha && this.flankPattern !== 'none') {
            const targetToUse = this.alphaAttackTarget || (nearestEnemy ? nearestEnemy.enemyShip : null);
            if (targetToUse) {
                const targetDistance = Math.min(1.0, Math.sqrt(
                    (targetToUse.x - this.x) ** 2 + (targetToUse.y - this.y) ** 2
                ) / this.enemyDetectionRadius);
                actionScores.alphaAttack = 0.9 * (1.0 - targetDistance * 0.4); // High priority for advanced flanking
            }
        }
        // ALPHA ATTACK: Score: Alpha attack pattern (high priority when enabled and target available)
        else if (alphaAttackTarget && alphaAttackEnabled) {
            if (this.isAlpha) {
                // Alpha: High priority to lead attack
                const targetDistance = Math.min(1.0, Math.sqrt(
                    (alphaAttackTarget.x - this.x) ** 2 + (alphaAttackTarget.y - this.y) ** 2
                ) / this.enemyDetectionRadius);
                actionScores.alphaAttack = 0.85 * (1.0 - targetDistance * 0.5);
            } else if (this.alphaShip) {
                // PHASE 2: Flanking has higher priority than standard formation
                if (flankingPos && this.flankingMode !== 'none') {
                    const flankDistance = Math.min(1.0, Math.sqrt(
                        (flankingPos.x - this.x) ** 2 + (flankingPos.y - this.y) ** 2
                    ) / 200);
                    actionScores.alphaAttack = 0.9 * (1.0 - flankDistance * 0.3); // Higher priority for flanking
                } else if (formationPos) {
                    // Wingmen: High priority to follow alpha in formation
                    const formationDistance = Math.min(1.0, formationPos.distance / 200);
                    actionScores.alphaAttack = 0.8 * (1.0 - formationDistance * 0.4);
                }
            }
        }
        
        // Score: Fire at enemies
        if (nearestEnemy && this.shouldFireAtEnemy(nearestEnemy.enemyShip)) {
            const enemyDistance = Math.min(1.0, nearestEnemy.distance / this.enemyFiringRange);
            actionScores.fireAtEnemies = priorityWeights.fireAtEnemies * (1.0 - enemyDistance * 0.5);
        }
        
        // Score: Protect allies
        const protectionAngle = this.protectAllies(aiShips, enemyShips);
        if (protectionAngle !== null) {
            actionScores.protectAllies = priorityWeights.protectAllies * (this.hasNearbyEnemies(enemyShips) ? 1.0 : 0.6);
        }
        
        // Score: Fire at asteroids
        let shouldFireAtAsteroid = false;
        let targetAsteroid = null;
        asteroids.forEach(asteroid => {
            if (this.shouldFire(asteroid)) {
                const dx = asteroid.x - this.x;
                const dy = asteroid.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (!targetAsteroid || distance < Math.sqrt(
                    (targetAsteroid.x - this.x) ** 2 + (targetAsteroid.y - this.y) ** 2
                )) {
                    targetAsteroid = asteroid;
                    shouldFireAtAsteroid = true;
                }
            }
        });
        if (targetAsteroid && shouldFireAtAsteroid) {
            const asteroidDistance = Math.min(1.0, Math.sqrt(
                (targetAsteroid.x - this.x) ** 2 + (targetAsteroid.y - this.y) ** 2
            ) / this.firingRange);
            actionScores.fireAtAsteroids = priorityWeights.fireAtAsteroids * (1.0 - asteroidDistance * 0.5);
        }
        
        // Score: Avoid asteroids
        if (nearestAsteroid) {
            const asteroidDistance = Math.min(1.0, nearestAsteroid.distance / this.detectionRadius);
            actionScores.avoidAsteroids = priorityWeights.avoidAsteroids * (1.0 - asteroidDistance);
        }
        
        // Score: Random navigation (always low, but present)
        actionScores.randomNav = priorityWeights.randomNav * 0.1;
        
        // Find the action with the highest weighted score
        let bestAction = 'randomNav';
        let bestScore = actionScores.randomNav;
        for (const [action, score] of Object.entries(actionScores)) {
            if (score > bestScore) {
                bestScore = score;
                bestAction = action;
            }
        }
        
        // Execute the best action based on weighted priorities
        // Priority 0: Avoid bullets (highest priority - predictive avoidance)
        if (bestAction === 'avoidBullets' && bulletThreat) {
            // Calculate evasion angle (perpendicular to bullet trajectory)
            const bulletAngle = Math.atan2(bulletThreat.bullet.velocityY, bulletThreat.bullet.velocityX);
            const evasionAngle = bulletAngle + Math.PI / 2; // Perpendicular (90 degrees)
            
            // Choose the direction that moves away from bullet
            const dx = this.x - bulletThreat.bullet.x;
            const dy = this.y - bulletThreat.bullet.y;
            const angleToBullet = Math.atan2(dy, dx);
            
            // Evade perpendicular to bullet path, away from bullet
            this.targetAngle = evasionAngle;
            if (Math.abs(this.normalizeAngle(evasionAngle - angleToBullet)) > Math.PI / 2) {
                this.targetAngle = evasionAngle + Math.PI; // Reverse direction if needed
            }
            
            // Apply extra thrust for evasion (weighted by threat level)
            const thrustChance = 0.6 * bulletThreat.threatLevel;
            if (gameRandom() < thrustChance) {
                this.thrust(1.5); // Strong evasion thrust
            }
        }
        // Priority 1: Avoid other ships (highest priority - immediate threat, safety-critical)
        else if (bestAction === 'avoidShips' || (nearestShip && nearestShip.distance < 60)) {
            // Ship too close - move away immediately
            const avoidanceAngle = nearestShip.angle + Math.PI; // Opposite direction
            this.targetAngle = avoidanceAngle;
            
            // Apply extra thrust to get away quickly (weighted by priority)
            const thrustChance = 0.3 * priorityWeights.avoidShips;
            if (gameRandom() < thrustChance) {
                this.thrust(1.5); // Extra thrust to escape faster
            }
        }
        // HEALING SYSTEM: Priority 1.5: Seek healing (Phase 3 ships only)
        else if (bestAction === 'seekHealing' && nearestPhase1Ally && this.aiPhase === 3 && this.healingCooldown === 0) {
            // Move toward Phase 1 ally for healing
            this.targetAngle = nearestPhase1Ally.angle;
            
            // Rotate toward target
            let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
            this.rotate(angleDiff > 0 ? 1 : -1);
            
            // Apply thrust to move toward ally (weighted by priority)
            const thrustChance = 0.4; // Higher chance to move toward healing
            if (gameRandom() < thrustChance) {
                this.thrust(1.2); // Moderate thrust to reach ally
            }
        }
        // ADVANCED PATTERNS: Priority 1.5: Advanced flanking patterns
        else if (bestAction === 'alphaAttack' && advancedFlankingEnabled && alphaAttackEnabled && this.isAlpha && this.flankPattern !== 'none') {
            // Execute advanced flanking (hammer & anvil or scissors)
            if (this.flankPattern === 'hammerAnvil') {
                this.executeHammerAndAnvil(enemyShips, aiShips);
            } else if (this.flankPattern === 'scissors') {
                this.executeScissorsAttack(enemyShips, aiShips);
            }
        }
        // ALPHA ATTACK: Priority 1.6: Alpha attack pattern (coordinated V-formation attack)
        // PHASE 3: Escort mode takes precedence over standard alpha attack
        else if (bestAction === 'alphaAttack' && escortPos && escortMode !== 'none' && alphaAttackEnabled) {
            // PHASE 3: Execute escort/protection mode
            const dx = escortPos.x - this.x;
            const dy = escortPos.y - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            this.targetAngle = Math.atan2(dy, dx);
            
            // Rotate toward escort position
            let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
            this.rotate(angleDiff > 0 ? 1 : -1);
            
            // Fire at threats when in escort mode
            if (escortMode === 'intercept' && this.interceptTarget) {
                const target = this.interceptTarget;
                const dxToTarget = target.x - this.x;
                const dyToTarget = target.y - this.y;
                const distToTarget = Math.sqrt(dxToTarget * dxToTarget + dyToTarget * dyToTarget);
                
                if (distToTarget < this.enemyFiringRange) {
                    const targetAngle = Math.atan2(dyToTarget, dxToTarget);
                    const targetAngleDiff = this.normalizeAngle(targetAngle - this.angle);
                    
                    if (Math.abs(targetAngleDiff) < 0.5 && this.shouldFireAtEnemy(target)) {
                        if (this.coordinateFiring(target, aiShips)) {
                            this.fire(target);
                        }
                    }
                }
            } else if (escortMode === 'guard' && this.alphaAttackTarget) {
                // Guard mode: Fire at threats to protected target
                const target = this.alphaAttackTarget;
                const dxToTarget = target.x - this.x;
                const dyToTarget = target.y - this.y;
                const distToTarget = Math.sqrt(dxToTarget * dxToTarget + dyToTarget * dyToTarget);
                
                if (distToTarget < this.enemyFiringRange) {
                    const targetAngle = Math.atan2(dyToTarget, dxToTarget);
                    const targetAngleDiff = this.normalizeAngle(targetAngle - this.angle);
                    
                    if (Math.abs(targetAngleDiff) < 0.5 && this.shouldFireAtEnemy(target)) {
                        if (this.coordinateFiring(target, aiShips)) {
                            this.fire(target);
                        }
                    }
                }
            }
            
            // Apply thrust to maintain escort position
            const thrustChance = 0.5;
            if (gameRandom() < thrustChance || distance > 50) {
                this.thrust(1.2); // Moderate thrust for escort
            }
        } else if (bestAction === 'alphaAttack' && alphaAttackTarget && alphaAttackEnabled) {
            if (this.isAlpha) {
                // Alpha: Lead the attack toward target (or assigned target in multi-target mode)
                const targetToUse = this.assignedTarget || alphaAttackTarget;
                const dx = targetToUse.x - this.x;
                const dy = targetToUse.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                this.targetAngle = Math.atan2(dy, dx);
                
                // ALWAYS rotate toward target (even if not in alphaAttack action)
                let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
                if (Math.abs(angleDiff) > 0.1) {
                    this.rotate(angleDiff > 0 ? 1 : -1);
                }
                
                // Fire at target when in range and aligned
                if (distance < this.enemyFiringRange && Math.abs(angleDiff) < 0.4) {
                    if (this.shouldFireAtEnemy(targetToUse) && this.coordinateFiring(targetToUse, aiShips)) {
                        this.fire(targetToUse);
                    }
                }
                
                // Don't apply thrust during escape mode (movement handled by escape system)
                if (!this.escapeMode) {
                    // Apply thrust to approach target (only when not escaping)
                    const thrustChance = 0.5;
                    if (gameRandom() < thrustChance) {
                        this.thrust(1.3); // Strong thrust for alpha
                    }
                }
            } else if (this.alphaShip && formationPos) {
                // Wingmen: Follow alpha in formation (or assigned target in multi-target mode)
                const targetToUse = this.assignedTarget || (this.alphaShip.alphaAttackTarget);
                
                const dx = formationPos.x - this.x;
                const dy = formationPos.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                this.targetAngle = Math.atan2(dy, dx);
                
                // Rotate toward formation position
                let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
                this.rotate(angleDiff > 0 ? 1 : -1);
                
                // Match alpha's angle for formation
                const alphaAngleDiff = this.normalizeAngle(this.alphaShip.angle - this.angle);
                this.rotate(alphaAngleDiff > 0 ? 0.5 : -0.5);
                
                // Fire at assigned target or alpha's target when in range
                if (targetToUse) {
                    const dxToTarget = targetToUse.x - this.x;
                    const dyToTarget = targetToUse.y - this.y;
                    const distToTarget = Math.sqrt(dxToTarget * dxToTarget + dyToTarget * dyToTarget);
                    
                    if (distToTarget < this.enemyFiringRange) {
                        const targetAngle = Math.atan2(dyToTarget, dxToTarget);
                        const targetAngleDiff = this.normalizeAngle(targetAngle - this.angle);
                        
                        if (Math.abs(targetAngleDiff) < 0.5 && this.shouldFireAtEnemy(targetToUse)) {
                            if (this.coordinateFiring(targetToUse, aiShips)) {
                                this.fire(targetToUse);
                            }
                        }
                    }
                }
                
                // Apply thrust to maintain formation
                const thrustChance = 0.4;
                if (gameRandom() < thrustChance || distance > 100) {
                    this.thrust(1.2); // Moderate thrust to maintain formation
                }
            }
        }
        // Priority 2: Fire at enemy ships (weighted priority - combat)
        else if (bestAction === 'fireAtEnemies' && nearestEnemy && this.shouldFireAtEnemy(nearestEnemy.enemyShip)) {
            // ALPHA ESCAPE: Covering fire mode - prioritize firing when alpha is escaping
            if (this.coveringFireMode && this.alphaShip && this.alphaShip.escapeMode) {
                // Aggressive covering fire - fire more frequently
                const predicted = this.predictEnemyPosition(nearestEnemy.enemyShip);
                if (predicted) {
                    this.targetAngle = predicted.angle;
                } else {
                    this.targetAngle = nearestEnemy.angle;
                }
                
                let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
                if (Math.abs(angleDiff) < 0.5) { // Wider tolerance for covering fire
                    this.fire(nearestEnemy.enemyShip);
                }
                this.rotate(angleDiff > 0 ? 1 : -1);
                
                // Increase fire rate during covering fire
                if (this.shootCooldown > 0) {
                    this.shootCooldown = Math.max(0, this.shootCooldown - 2); // Faster cooldown
                }
            } else {
                // Normal firing behavior
                // Check coordination before firing
                if (this.coordinateFiring(nearestEnemy.enemyShip, aiShips)) {
                    // Rotate toward enemy to aim
                    const predicted = this.predictEnemyPosition(nearestEnemy.enemyShip);
                    if (predicted) {
                        this.targetAngle = predicted.angle;
                    } else {
                        this.targetAngle = nearestEnemy.angle;
                    }
                    
                    // Rotate toward target
                    let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
                    if (Math.abs(angleDiff) < 0.4) { // Wider angle tolerance for rapid fire
                        this.fire(nearestEnemy.enemyShip);
                    }
                    this.rotate(angleDiff > 0 ? 1 : -1);
                } else {
                    // Other ships are already targeting, just rotate toward enemy
                    this.targetAngle = nearestEnemy.angle;
                }
            }
        }
        // Priority 3: Protect allies (weighted priority - flocking and protection)
        else if (bestAction === 'protectAllies') {
            const protectionAngle = this.protectAllies(aiShips, enemyShips);
            const flockResult = this.calculateFlocking(aiShips, enemyShips);
            const flockAngle = flockResult.angle;
            const flockWeight = flockResult.weight;
            
            if (protectionAngle !== null) {
                // Protect nearby allies from enemies (weighted by priority)
                const enemiesPresent = this.hasNearbyEnemies(enemyShips);
                if (enemiesPresent) {
                    // When enemies present, prioritize protection more strongly
                    this.targetAngle = protectionAngle;
                    // Apply extra thrust to move closer to allies faster (weighted)
                    const thrustChance = 0.4 * priorityWeights.protectAllies;
                    if (gameRandom() < thrustChance) {
                        this.thrust(1.2);
                    }
                } else {
                    this.targetAngle = protectionAngle;
                }
            } else if (flockAngle !== this.angle) {
                // Apply flocking behavior (weighted with current target and priority)
                const currentAngle = this.targetAngle;
                const angleDiff = this.normalizeAngle(flockAngle - currentAngle);
                // Combine flock weight with priority weight
                const combinedWeight = flockWeight * priorityWeights.protectAllies;
                this.targetAngle = currentAngle + angleDiff * combinedWeight;
                
                // When enemies present, apply more thrust to maintain formation (weighted)
                if (this.hasNearbyEnemies(enemyShips) && gameRandom() < 0.3 * priorityWeights.protectAllies) {
                    this.thrust(1.1);
                }
            }
        }
        // Priority 4: Fire at threatening asteroids (weighted priority)
        else if (bestAction === 'fireAtAsteroids' && targetAsteroid && shouldFireAtAsteroid) {
            // Rotate toward target asteroid to aim
            const dx = targetAsteroid.x - this.x;
            const dy = targetAsteroid.y - this.y;
            const angleToTarget = Math.atan2(dy, dx);
            this.targetAngle = angleToTarget;
            
            // Rotate toward target (angle tolerance weighted by priority)
            let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
            const angleTolerance = 0.4 * priorityWeights.fireAtAsteroids;
            if (Math.abs(angleDiff) < angleTolerance) {
                this.fire();
            }
            this.rotate(angleDiff > 0 ? 1 : -1);
        }
        // Priority 5: Avoid asteroids (weighted priority)
        else if (bestAction === 'avoidAsteroids' && nearestAsteroid) {
            // Asteroid detected - calculate avoidance angle
            const angleToAsteroid = nearestAsteroid.angle;
            const avoidanceAngle = angleToAsteroid + Math.PI; // Opposite direction
            
            // Rotate towards avoidance direction (thrust weighted by priority)
            this.targetAngle = avoidanceAngle;
            if (gameRandom() < 0.2 * priorityWeights.avoidAsteroids) {
                this.thrust(1.1);
            }
        }
        // Priority 6: Avoid ships at medium distance (safety fallback)
        else if (nearestShip && nearestShip.distance < 120) {
            // Ship at medium distance - maintain separation
            const avoidanceAngle = nearestShip.angle + Math.PI;
            this.targetAngle = avoidanceAngle;
        }
        // Priority 7: Random navigation (weighted priority)
        else {
            // No immediate threat - random navigation (weighted by priority)
            const randomChance = 0.01 * priorityWeights.randomNav;
            if (gameRandom() < randomChance) {
                // Occasionally change direction
                this.targetAngle = gameRandom() * Math.PI * 2;
            }
        }
        
        // FINAL ROTATION: Ensure alpha ship always rotates toward target (even if not in alphaAttack action)
        // This ensures smooth rotation when shooting, regardless of which action was selected
        if (this.isAlpha && alphaAttackEnabled && !this.escapeMode) {
            const alphaAttackTarget = this.executeAlphaAttack(enemyShips);
            if (alphaAttackTarget) {
                const targetToUse = this.assignedTarget || alphaAttackTarget;
                if (targetToUse && targetToUse.x !== undefined && targetToUse.y !== undefined) {
                    const dx = targetToUse.x - this.x;
                    const dy = targetToUse.y - this.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance > 0) {
                        this.targetAngle = Math.atan2(dy, dx);
                        let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
                        // Always rotate toward target for smooth tracking
                        if (Math.abs(angleDiff) > 0.01) {
                            this.rotate(angleDiff > 0 ? 1 : -1);
                        }
                    }
                }
            }
        }

        // Rotate towards target angle
        let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
        if (Math.abs(angleDiff) > 0.1) {
            this.rotate(angleDiff > 0 ? 1 : -1);
        }

        // Thrust occasionally to maintain movement
        if (gameRandom() < this.thrustFrequency) {
            this.thrust();
        }
        
        // Note: updateAIPhase() and updateAIShield() are now called separately
        // in the main update loop to work with both MARL and rule-based systems
    }
    
    // Phase 5.1: Collect training data sample for ML training
    collectTrainingSample(features, mlParams) {
        const canonical = features._vector ? features : null;
        const priorities = this.mlPriorityWeights || this.basePriorityWeights;

        if (typeof initializeMARLDataCollector === 'function') {
            const collector = initializeMARLDataCollector();
            collector.collectSample({
                agentIndex: aiShips.indexOf(this),
                ship: this,
                features: canonical || features,
                mlParams,
                mlPriorities: priorities,
                score: typeof score !== 'undefined' ? score : 0,
                curriculumStage: typeof window !== 'undefined' ? window.marlCurriculumStage : 0,
                labelSource: 'expert'
            });
            return;
        }

        const sample = {
            timestamp: Date.now(),
            features: { ...features },
            parameters: { ...mlParams },
            priorityWeights: { ...priorities },
            shipState: {
                x: this.x, y: this.y, angle: this.angle,
                velocityX: this.velocityX, velocityY: this.velocityY,
                health: this.health, phase: this.aiPhase
            }
        };
        this.trainingDataBuffer.push(sample);
        if (this.trainingDataBuffer.length > this.maxTrainingSamples) {
            this.trainingDataBuffer.shift();
        }
    }
    
    // Phase 5.1: Export training data (for ML training)
    exportTrainingData() {
        if (this.trainingDataBuffer.length === 0) {
            return null;
        }
        
        // Convert to JSON string for download
        const dataStr = JSON.stringify(this.trainingDataBuffer, null, 2);
        return dataStr;
    }
    
    // Update AI ship phase based on health
    updateAIPhase() {
        if (this.maxHealth <= 0) return; // Safety check
        const healthPercent = Math.max(0, Math.min(1, this.health / this.maxHealth));
        const oldPhase = this.aiPhase;
        
        if (healthPercent > 0.66) {
            this.aiPhase = 1;
        } else if (healthPercent > 0.33) {
            this.aiPhase = 2;
        } else {
            this.aiPhase = 3;
        }
        
        // Phase transition effects - ENHANCED with temporary invulnerability
        if (oldPhase !== this.aiPhase) {
            this.phaseTransitionTimer = 60; // 1 second transition effect
            this.shieldActive = true;
            // Phase 3: Longer invulnerability on phase transition (temporary invulnerability)
            if (this.aiPhase === 3) {
                this.shieldDuration = 45; // 0.75 seconds of invulnerability (was 30 = 0.5s)
            } else {
                this.shieldDuration = 30; // Brief invincibility on phase change (0.5s)
            }
        }
    }
    
    // AI ship shield system - ENHANCED with Phase-Based Defense
    updateAIShield() {
        if (this.shieldDuration > 0) {
            this.shieldDuration--;
            this.shieldActive = true;
        } else {
            this.shieldActive = false;
        }
        
        // Phase-Based Defense Enhancements:
        // Phase 2: Increased shield frequency (1% chance per frame)
        // Phase 3: Maximum shield frequency (2% chance per frame) + longer duration
        if (this.shieldCooldown > 0) {
            this.shieldCooldown--;
        } else if (this.aiPhase >= 2 && !this.shieldActive) {
            let shieldChance = 0.005; // Base 0.5% for Phase 2
            let shieldDuration = 60; // 1 second base duration
            
            if (this.aiPhase === 2) {
                shieldChance = 0.01; // 1% chance per frame in Phase 2
                shieldDuration = 75; // 1.25 seconds
            } else if (this.aiPhase === 3) {
                shieldChance = 0.02; // 2% chance per frame in Phase 3 (maximum)
                shieldDuration = 90; // 1.5 seconds (longer)
            }
            
            if (gameRandom() < shieldChance) {
                this.shieldActive = true;
                this.shieldDuration = shieldDuration;
                this.shieldCooldown = this.aiPhase === 3 ? 240 : 300; // Shorter cooldown in Phase 3
            }
        }
    }

    // Override draw to make AI ship visually distinct
    draw() {
        ctx.save();
        ctx.translate(this.x, this.y);
        
        // AI ship shield visual effect
        if (this.shieldActive) {
            ctx.strokeStyle = GAME_THEME.cyan;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.6;
            ctx.beginPath();
            const shieldRadius = this.size + 8 + Math.sin(Date.now() / 50) * 3;
            ctx.arc(0, 0, shieldRadius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
        
        // Phase transition effect
        if (this.phaseTransitionTimer > 0) {
            ctx.strokeStyle = GAME_THEME.text;
            ctx.lineWidth = 3;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            ctx.arc(0, 0, this.size + 10, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
            this.phaseTransitionTimer--;
        }
        
        // HEALING SYSTEM: Healing visual effect (green pulsing circle)
        if (this.healingEffect && this.healingEffectTimer > 0) {
            ctx.strokeStyle = GAME_THEME.primary;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.7 + Math.sin(Date.now() / 100) * 0.3;
            ctx.beginPath();
            const healingRadius = this.size + 5 + Math.sin(Date.now() / 50) * 5;
            ctx.arc(0, 0, healingRadius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
            this.healingEffectTimer--;
            if (this.healingEffectTimer <= 0) {
                this.healingEffect = false;
            }
        }
        
        ctx.rotate(this.angle);

        // ALPHA ATTACK: Visual indicator for alpha ship (golden glow)
        if (this.isAlpha && alphaAttackEnabled) {
            const drawHelpPing = this.helpPingActive && fuelRescueEnabled &&
                !this.fuelTransferActive && !this.fuelDocked;

            // ALPHA ESCAPE: Different visual when in escape mode
            if (this.escapeMode) {
                // Escape mode: Red/orange pulsing glow
                ctx.strokeStyle = GAME_THEME.escape; // Red-orange for escape
                ctx.lineWidth = 4;
                ctx.globalAlpha = 0.9 + Math.sin(Date.now() / 80) * 0.1;
                ctx.beginPath();
                const escapeGlowRadius = this.size + 12 + Math.sin(Date.now() / 60) * 6;
                ctx.arc(0, 0, escapeGlowRadius, 0, Math.PI * 2);
                ctx.stroke();
                
                // Draw escape path indicator
                if (this.escapeTarget) {
                    ctx.restore();
                    ctx.save();
                    ctx.strokeStyle = GAME_THEME.orange;
                    ctx.lineWidth = 2;
                    ctx.globalAlpha = 0.5;
                    ctx.setLineDash([5, 5]);
                    ctx.beginPath();
                    ctx.moveTo(this.x, this.y);
                    ctx.lineTo(this.escapeTarget.x, this.escapeTarget.y);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.globalAlpha = 1.0;
                    ctx.restore();
                    ctx.save();
                    ctx.translate(this.x, this.y);
                }
                
                // Threat level indicator
                ctx.fillStyle = GAME_THEME.danger;
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.globalAlpha = 0.8;
                ctx.fillText(`ESCAPE! (${Math.round(this.threatLevel * 100)}%)`, 0, -this.size - 20);
                ctx.globalAlpha = 1.0;
            } else {
                // Normal mode: Golden glow
                ctx.strokeStyle = GAME_THEME.gold; // Gold color for alpha
                ctx.lineWidth = 3;
                ctx.globalAlpha = 0.8;
                ctx.beginPath();
                const alphaGlowRadius = this.size + 8 + Math.sin(Date.now() / 100) * 4;
                ctx.arc(0, 0, alphaGlowRadius, 0, Math.PI * 2);
                ctx.stroke();
                
                // Threat level indicator (subtle when not escaping)
                if (this.threatLevel > 0.4) {
                    ctx.fillStyle = GAME_THEME.amber;
                    ctx.font = '8px Arial';
                    ctx.textAlign = 'center';
                    ctx.globalAlpha = 0.6;
                    ctx.fillText(`Threat: ${Math.round(this.threatLevel * 100)}%`, 0, -this.size - 15);
                    ctx.globalAlpha = 1.0;
                }
            }

            if (drawHelpPing) {
                if (FXL && mlGlobalFrame % 40 === 0) FXL.emitRipple(this.x, this.y, GAME_THEME.helpPing);
                const pingRadius = this.size + 18 + Math.sin(Date.now() / 120) * 10;
                ctx.strokeStyle = GAME_THEME.helpPing;
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.55 + Math.sin(Date.now() / 180) * 0.25;
                ctx.beginPath();
                ctx.arc(0, 0, pingRadius, 0, Math.PI * 2);
                ctx.stroke();
                ctx.fillStyle = GAME_THEME.helpPing;
                ctx.font = 'bold 11px Arial';
                ctx.textAlign = 'center';
                ctx.globalAlpha = 0.95;
                ctx.fillText('HELP!', 0, -this.size - (this.escapeMode ? 36 : 24));
                ctx.globalAlpha = 1.0;
            }
            ctx.globalAlpha = 1.0;
        }
        
        // PHASE 2: Formation Shield visual effect
        if (this.formationShieldActive && alphaAttackEnabled) {
            ctx.strokeStyle = GAME_THEME.cyan;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.5;
            ctx.beginPath();
            const shieldRadius = this.size + 12 + Math.sin(Date.now() / 80) * 3;
            ctx.arc(0, 0, shieldRadius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
        
        // PHASE 2: Formation Boost visual effect (speed lines)
        if (this.formationBoostActive && alphaAttackEnabled) {
            ctx.strokeStyle = GAME_THEME.primary;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.6;
            for (let i = 0; i < 3; i++) {
                ctx.beginPath();
                const boostX = -this.size - 10 - (i * 8);
                ctx.moveTo(boostX, -this.size / 3);
                ctx.lineTo(boostX - 5, 0);
                ctx.lineTo(boostX, this.size / 3);
                ctx.stroke();
            }
            ctx.globalAlpha = 1.0;
        }
        
        // PHASE 2: Role indicator (small icon/color)
        if (this.role && alphaAttackEnabled) {
            const roleColors = {
                scout: GAME_THEME.roleScout,
                tank: GAME_THEME.roleTank,
                support: GAME_THEME.roleSupport,
                fuel: GAME_THEME.fuelShip,
                dps: GAME_THEME.roleDps,
                interceptor: GAME_THEME.roleInterceptor
            };
            ctx.fillStyle = roleColors[this.role] || GAME_THEME.text;
            ctx.globalAlpha = 0.7;
            ctx.beginPath();
            ctx.arc(this.size + 5, -this.size - 5, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1.0;
        }
        
        // Draw triangular ship (blue color for AI, intensity based on phase)
        let intensity = 1.0;
        if (this.aiPhase === 2) intensity = 1.2;
        if (this.aiPhase === 3) intensity = 1.5;

        // Phase 3: flicker effect — randomly dim the ship to show failing systems
        if (this.aiPhase === 3) {
            ctx.globalAlpha = 0.6 + Math.random() * 0.4;
        }
        
        // Phase-based color intensity
        let blueValue = Math.floor(170 / intensity);
        // ALPHA ATTACK: Alpha ship has golden tint; fuel ships are green
        // Glows are cached radial sprites (fx.js) rather than ctx.shadowBlur,
        // which re-rasterises the blur on every stroke.
        if (this.role === 'fuel' && fuelRescueEnabled && alphaAttackEnabled) {
            ctx.strokeStyle = GAME_THEME.fuelShip;
            ctx.lineWidth = 3;
            if (FXL) FXL.glow(ctx, 0, 0, this.size * 1.6, GAME_THEME.fuelShip, 0.45);
        } else if (this.isAlpha && alphaAttackEnabled) {
            ctx.strokeStyle = `rgba(255, 215, 0, 1)`; // Gold for alpha
            ctx.lineWidth = 3;
            if (FXL) FXL.glow(ctx, 0, 0, this.size * 1.8, 'rgba(255, 215, 0, 1)', 0.35);
        } else {
            ctx.strokeStyle = `rgba(0, ${blueValue}, 255, 1)`;
            ctx.lineWidth = 2;
            if (FXL) FXL.glow(ctx, 0, 0, this.size * 1.2, `rgba(0, ${blueValue}, 255, 1)`, 0.25);
        }
        ctx.beginPath();
        ctx.moveTo(this.size, 0);
        ctx.lineTo(-this.size / 2, -this.size / 2);
        ctx.lineTo(-this.size / 2, this.size / 2);
        ctx.closePath();
        ctx.stroke();

        // Phase 2+: hull scoring — diagonal battle-damage lines across the hull
        if (this.aiPhase >= 2) {
            const dmgColor = this.aiPhase === 3 ? 'rgba(255,80,80,0.7)' : 'rgba(255,200,50,0.55)';
            ctx.strokeStyle = dmgColor;
            ctx.lineWidth = 1;
            // Three random-ish but deterministic scoring marks
            ctx.beginPath(); ctx.moveTo(-4, -8); ctx.lineTo(6, -2); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(2, 5);  ctx.lineTo(-8, 9);  ctx.stroke();
            if (this.aiPhase === 3) {
                ctx.beginPath(); ctx.moveTo(8, -4); ctx.lineTo(0, 4); ctx.stroke();
            }
        }

        // Phase 3: emit occasional sparks via the FX layer
        if (this.aiPhase === 3 && FXL && Math.random() < 0.15) {
            const sparkAngle = Math.random() * Math.PI * 2;
            const r = this.size * 0.4;
            // emitMuzzle repurposed as a tiny spark burst
            FXL.emitMuzzle(
                Math.cos(sparkAngle) * r,
                Math.sin(sparkAngle) * r,
                sparkAngle,
                'rgba(255,120,0,0.9)'
            );
        }

        ctx.globalAlpha = 1.0;

        // Draw detection radius indicator (subtle)
        ctx.strokeStyle = GAME_THEME.aiBlue;
        ctx.globalAlpha = 0.2;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(0, 0, this.detectionRadius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        // Draw thruster effect when moving
        if (Math.abs(this.velocityX) > 0.1 || Math.abs(this.velocityY) > 0.1) {
            ctx.strokeStyle = GAME_THEME.orange;
            ctx.beginPath();
            ctx.moveTo(-this.size / 2, -this.size / 4);
            ctx.lineTo(-this.size - 5, 0);
            ctx.lineTo(-this.size / 2, this.size / 4);
            ctx.stroke();
        }

        ctx.restore();
        
        // Draw health bar and phase indicator
        ctx.save();
        ctx.translate(this.x, this.y - this.size - 10);
        const barWidth = this.size * 1.5;
        const barHeight = 3;
        const healthPercent = this.maxHealth > 0 ? Math.max(0, Math.min(1, this.health / this.maxHealth)) : 0;
        
        // Background
        ctx.fillStyle = GAME_THEME.barBg;
        ctx.fillRect(-barWidth / 2, -barHeight / 2, barWidth, barHeight);
        
        // Health bar color based on phase
        if (this.aiPhase === 1) ctx.fillStyle = GAME_THEME.healthHigh; // Green
        else if (this.aiPhase === 2) ctx.fillStyle = GAME_THEME.healthMid; // Yellow
        else ctx.fillStyle = GAME_THEME.healthLow; // Red
        
        ctx.fillRect(-barWidth / 2, -barHeight / 2, barWidth * healthPercent, barHeight);
        
        // Border
        ctx.strokeStyle = GAME_THEME.text;
        ctx.lineWidth = 1;
        ctx.strokeRect(-barWidth / 2, -barHeight / 2, barWidth, barHeight);
        
        // Phase indicator
        ctx.fillStyle = GAME_THEME.text;
        ctx.font = '8px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Phase ${this.aiPhase}`, 0, -8);

        // Fuel bar (alpha + fuel ships)
        if (fuelRescueEnabled && alphaAttackEnabled && (this.isAlpha || this.role === 'fuel')) {
            const fuelPercent = this.maxFuel > 0 ? Math.max(0, Math.min(1, this.fuel / this.maxFuel)) : 0;
            const fuelBarY = 4;
            ctx.fillStyle = GAME_THEME.barBg;
            ctx.fillRect(-barWidth / 2, fuelBarY, barWidth, 2);
            ctx.fillStyle = this.role === 'fuel' ? GAME_THEME.fuelShip : GAME_THEME.amber;
            ctx.fillRect(-barWidth / 2, fuelBarY, barWidth * fuelPercent, 2);
            if (this.fuelTransferActive) {
                ctx.fillStyle = GAME_THEME.fuelTransfer;
                ctx.font = '7px Arial';
                ctx.fillText('FUEL LINK', 0, fuelBarY + 10);
            } else if (this.fuelDocked) {
                ctx.fillStyle = GAME_THEME.fuelShip;
                ctx.font = 'bold 7px Arial';
                ctx.fillText('DOCKED', 0, fuelBarY + 10);
            }
        }
        
        // PHASE 3: Escort mode indicator
        if (escortMode !== 'none' && this.escortTarget) {
            ctx.fillStyle = GAME_THEME.primary;
            ctx.font = '7px Arial';
            ctx.fillText(escortMode.toUpperCase(), 0, -15);
        }
        
        // PHASE 3: Multi-target mode indicator (alpha only)
        if (this.isAlpha && multiTargetMode !== 'focus' && this.assignedTarget) {
            ctx.fillStyle = GAME_THEME.amber;
            ctx.font = '7px Arial';
            ctx.fillText(`TGT:${multiTargetMode}`, 0, -22);
        }
        
        // PHASE 3: Adaptive formation indicator (alpha only)
        if (this.isAlpha && adaptiveFormationEnabled && this.formationTypeChanged) {
            ctx.fillStyle = GAME_THEME.cyan;
            ctx.font = '7px Arial';
            ctx.fillText(`FORM:${this.formationType}`, 0, -29);
        }
        
        // PHASE 5.3: ML Decision mode indicator
        if (this.useMLDecisions && this.mlEnabled) {
            ctx.fillStyle = GAME_THEME.magenta;
            ctx.font = '7px Arial';
            ctx.fillText('ML:ON', 0, -36);
        }
        
        // ADVANCED PATTERNS: Attack sequence indicator (alpha only)
        if (this.isAlpha && this.attackSequence !== 'none') {
            ctx.fillStyle = GAME_THEME.orange;
            ctx.font = '7px Arial';
            const sequenceNames = {
                'alphaStrike': 'COMBO',
                'wave': 'WAVE'
            };
            ctx.fillText(sequenceNames[this.attackSequence] || 'SEQ', 0, -43);
        }
        
        // ADVANCED PATTERNS: Formation morphing indicator (alpha only)
        if (this.isAlpha && this.formationMorphing) {
            ctx.fillStyle = GAME_THEME.cyan;
            ctx.font = '7px Arial';
            ctx.fillText(`MORPH:${this.formationMorphTarget}`, 0, -50);
        }
        
        // ADVANCED PATTERNS: Split formation indicator (alpha only)
        if (this.isAlpha && this.splitFormation) {
            ctx.fillStyle = GAME_THEME.highlight;
            ctx.font = '7px Arial';
            ctx.fillText('SPLIT', 0, -57);
        }
        
        // ADVANCED PATTERNS: Flanking pattern indicator (alpha only)
        if (this.isAlpha && this.flankPattern !== 'none') {
            ctx.fillStyle = GAME_THEME.magenta;
            ctx.font = '7px Arial';
            const patternNames = {
                'hammerAnvil': 'HAMMER',
                'scissors': 'SCISSORS'
            };
            ctx.fillText(patternNames[this.flankPattern] || 'FLANK', 0, -64);
        }
        
        ctx.restore();
        
        // PHASE 3: Draw escort connection line
        if (escortMode !== 'none' && this.escortTarget) {
            ctx.save();
            ctx.strokeStyle = GAME_THEME.primary;
            ctx.globalAlpha = 0.4;
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(this.x, this.y);
            ctx.lineTo(this.escortTarget.x, this.escortTarget.y);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1.0;
            ctx.restore();
        }
        
        // PHASE 3: Draw patrol path (patrol mode)
        if (escortMode === 'patrol' && this.patrolCenter) {
            ctx.save();
            ctx.strokeStyle = GAME_THEME.aiBlue;
            ctx.globalAlpha = 0.3;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(this.patrolCenter.x, this.patrolCenter.y, this.patrolRadius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
            ctx.restore();
        }
    }
}
