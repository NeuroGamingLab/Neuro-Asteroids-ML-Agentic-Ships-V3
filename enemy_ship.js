// Extracted from game.js as part of a module split.
// Relies on globals defined in game.js/world.js/rng.js/fx.js loaded earlier
// (canvas, ctx, worldW/worldH/sphereWorld, gameRandom, FXL, ACTION_SPACE_SIZE,
// decodeAction, keys, etc.) — same plain-script global scope as before, just
// split across files.

// Enemy Bullet class (red/orange bullets that can hit player)
class EnemyBullet {
    constructor(x, y, angle, isSeeking = false, target = null) {
        this.x = x;
        this.y = y;
        this.angle = angle;
        this.speed = 7;
        this.radius = 3;
        this.lifetime = 90;
        
        // SEEKING MISSILES: Advanced projectile tracking
        this.isSeeking = isSeeking;
        this.target = target; // Target ship to track
        this.turnRate = 0.08; // How fast the missile can turn (radians per frame)
        this.seekRange = 500; // Maximum range to seek target
        this.seekActive = true; // Whether seeking is still active
    }

    update() {
        // SEEKING MISSILES: Track target if seeking
        if (this.isSeeking && this.seekActive && this.target) {
            const dx = this.target.x - this.x;
            const dy = this.target.y - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < this.seekRange && distance > 0) {
                // Calculate desired angle to target
                const desiredAngle = Math.atan2(dy, dx);
                
                // Calculate angle difference
                let angleDiff = desiredAngle - this.angle;
                // Normalize to [-PI, PI]
                while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
                while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
                
                // Turn toward target (limited by turn rate)
                if (Math.abs(angleDiff) > this.turnRate) {
                    this.angle += angleDiff > 0 ? this.turnRate : -this.turnRate;
                } else {
                    this.angle = desiredAngle;
                }
                
                // Stop seeking if target is destroyed or out of range
                if (distance > this.seekRange || (this.target.health !== undefined && this.target.health <= 0)) {
                    this.seekActive = false;
                }
            } else {
                this.seekActive = false;
            }
        }
        
        // Move in current direction
        this.lifetime--;
        if (sphereWorld()) {
            WORLD_GEO.moveAlongHeading(this, this.speed);
            return;
        }
        this.x += Math.cos(this.angle) * this.speed;
        this.y += Math.sin(this.angle) * this.speed;

        if (this.x < 0) this.x = worldW();
        if (this.x > worldW()) this.x = 0;
        if (this.y < 0) this.y = worldH();
        if (this.y > worldH()) this.y = 0;
    }

    draw() {
        // SEEKING MISSILES: Different visual for seeking missiles
        if (this.isSeeking && this.seekActive) {
            // Glowing effect for seeking missiles
            ctx.fillStyle = GAME_THEME.danger;
            if (FXL) FXL.glow(ctx, this.x, this.y, this.radius * 4, GAME_THEME.orange, 0.7);
        } else {
            ctx.fillStyle = GAME_THEME.enemy;
            if (FXL) FXL.glow(ctx, this.x, this.y, this.radius * 3, GAME_THEME.orange, 0.5);
        }
        
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw trail for seeking missiles
        if (this.isSeeking && this.seekActive) {
            ctx.strokeStyle = GAME_THEME.orange;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.5;
            ctx.beginPath();
            ctx.moveTo(this.x, this.y);
            ctx.lineTo(
                this.x - Math.cos(this.angle) * 10,
                this.y - Math.sin(this.angle) * 10
            );
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
    }

    isAlive() {
        return this.lifetime > 0;
    }

    getPosition() {
        return { x: this.x, y: this.y, radius: this.radius };
    }
}

// Enemy Ship class - Hostile AI ships that attack the player
class EnemyShip {
    constructor(x = null, y = null, type = 'basic') {
        if ((x === null || y === null) && sphereWorld()) {
            const spot = WORLD_GEO.randomPoint(gameRandom, worldW() / 2, worldH() / 2, 200 * worldScale());
            this.x = spot.x;
            this.y = spot.y;
        } else if (x === null || y === null) {
            const side = Math.floor(gameRandom() * 4);
            switch (side) {
                case 0: this.x = gameRandom() * worldW(); this.y = 0; break;
                case 1: this.x = worldW(); this.y = gameRandom() * worldH(); break;
                case 2: this.x = gameRandom() * worldW(); this.y = worldH(); break;
                case 3: this.x = 0; this.y = gameRandom() * worldH(); break;
            }
        } else {
            this.x = x;
            this.y = y;
        }

        this.type = type;
        this.velocityX = 0;
        this.velocityY = 0;
        this.angle = gameRandom() * Math.PI * 2;
        this.rotationSpeed = 0.08;
        this.thrustPower = 0.12;
        this.friction = 0.98;
        this.maxVelocity = 6;
        this.health = this.getMaxHealth();
        this.maxHealth = this.health;
        this.fireCooldown = 0;
        this.fireRate = this.getFireRate();
        this.bulletSpeed = 7;
        this.attackRange = 300;
        this.detectionRadius = 400;
        this.evasionRadius = 80;
        this.behaviorState = 'pursuit';
        this.target = null;
        this.predictedPosition = null;
        this.coverAsteroid = null;
        this.flankAngle = 0;
        this.size = this.getSize();
        this.radius = this.size * 0.75;
        this.color = this.getColor();
        this.targetAngle = this.angle;
        
        // ENHANCEMENT 1 & 3: Track target score and burst fire state
        this.targetScore = 0;
        this.burstFireActive = false;
        this.burstFireTimer = 0;
        
        // COORDINATION SYSTEM: Enemy attack coordination properties
        this.coordinationRole = null; // 'leader', 'attacker', 'flanker', 'interceptor', null
        this.attackPattern = null; // 'pincer', 'wave', 'flanking', null
        this.formationPosition = null; // {x, y} target position in formation
        this.coordinationGroup = null; // ID of coordination group
        this.patternTimer = 0; // Timer for pattern execution
        this.attackDelay = 0; // Delay before executing attack
        this.coordinationRadius = 300; // Radius for coordination
        this.isCoordinating = false; // Whether this ship is part of coordination
        
        // SPECIALIZED ROLES: Role-based enemy types
        this.specializedRole = null; // 'scout', 'tank', 'sniper', 'interceptor', 'bomber', 'support', null
        this.roleAssigned = false;
        this.chainAttackIndex = -1; // Index in chain attack sequence
        this.chainAttackReady = false; // Whether ready for chain attack
        this.adaptiveFormationOffset = {x: 0, y: 0}; // Offset for adaptive formation
        
        // SHIELD SYSTEM: Add shield to basic and advanced enemy ships (same as blue ships)
        if (this.type === 'basic' || this.type === 'advanced') {
            this.shieldActive = false;
            this.shieldCooldown = 0;
            this.shieldDuration = 0;
            this.shieldRadius = 40; // Same as blue ships
            this.shieldForce = 0.5; // Same as blue ships
            this.shieldCooldownMax = 300; // 5 seconds at 60fps (same as blue ships)
        }
        
        this.stunTimer = 0; // EMP stun counter
        
        // Boss-specific properties for Phase 4
        if (this.type === 'boss') {
            this.bossPhase = 1; // 1, 2, or 3
            this.shieldActive = false;
            this.shieldCooldown = 0;
            this.shieldDuration = 0;
            this.teleportCooldown = 0;
            this.attackPattern = 'normal'; // normal, spread, rapid, circular
            this.attackPatternTimer = 0;
            this.rapidFireBurst = 0;
            this.rapidFireBurstCount = 0;
            this.erraticMovementTimer = 0;
            this.phaseTransitionTimer = 0;
            this.shieldDisabled = false;
            this.turretDisabled = false;
            this.warpDisabled = false;
            const isHeadless = typeof globalThis !== 'undefined' && !!globalThis.__HEADLESS__;
            this.subsystems = isHeadless ? null : [
                { id: 'shield_generator', name: 'Shield Gen', angleOffset: -Math.PI / 2, distance: 34, radius: 11, health: 3, maxHealth: 3, destroyed: false, color: '#00E5FF' },
                { id: 'plasma_battery', name: 'Plasma Turret', angleOffset: Math.PI / 4, distance: 30, radius: 10, health: 3, maxHealth: 3, destroyed: false, color: '#FF6D00' },
                { id: 'warp_core', name: 'Warp Drive', angleOffset: Math.PI * 3 / 4, distance: 30, radius: 10, health: 2, maxHealth: 2, destroyed: false, color: '#D500F9' }
            ];
        }
    }

    getSubsystemWorldPos(sub) {
        const a = (this.angle || 0) + sub.angleOffset;
        return {
            x: this.x + Math.cos(a) * sub.distance,
            y: this.y + Math.sin(a) * sub.distance
        };
    }

    onSubsystemDestroyed(subId) {
        if (subId === 'shield_generator') {
            this.shieldDisabled = true;
            this.shieldActive = false;
            this.shieldDuration = 0;
        } else if (subId === 'plasma_battery') {
            this.turretDisabled = true;
            this.attackPattern = 'normal';
        } else if (subId === 'warp_core') {
            this.warpDisabled = true;
            this.teleportCooldown = 999999;
        }
    }

    getMaxHealth() {
        switch (this.type) {
            case 'basic': return 1;
            case 'advanced': return 2;
            case 'boss': return 5;
            default: return 1;
        }
    }

    getFireRate() {
        switch (this.type) {
            case 'basic': return 30;
            case 'advanced': return 20;
            case 'boss': return 10;
            default: return 30;
        }
    }

    getSize() {
        switch (this.type) {
            case 'basic': return 18;
            case 'advanced': return 22;
            case 'boss': return 35;
            default: return 18;
        }
    }

    getColor() {
        switch (this.type) {
            case 'basic': return GAME_THEME.enemy;
            case 'advanced': return GAME_THEME.purpleLight;
            case 'boss': return GAME_THEME.danger;
            default: return GAME_THEME.enemy;
        }
    }

    // ENHANCEMENT 1: Score target based on multiple factors (smarter prioritization)
    scoreTarget(targetShip) {
        if (!targetShip) return 0;
        
        let score = 0;
        
        // Low health priority (higher score for lower health)
        if (targetShip.health !== undefined && targetShip.maxHealth !== undefined) {
            const healthRatio = targetShip.health / targetShip.maxHealth;
            score += (1.0 - healthRatio) * 100; // 0-100 points (lower health = higher score)
        }
        
        // Alpha ship priority (high priority to disrupt coordination)
        if (targetShip.isAlpha) {
            score += 150; // High priority for alpha ship
        }
        
        // Isolation bonus (no nearby allies)
        // Count nearby allies within 150px
        let nearbyAllies = 0;
        if (typeof aiShips !== 'undefined' && Array.isArray(aiShips)) {
            aiShips.forEach(ally => {
                if (ally && ally !== targetShip) {
                    const dx = ally.x - targetShip.x;
                    const dy = ally.y - targetShip.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < 150) {
                        nearbyAllies++;
                    }
                }
            });
        }
        if (nearbyAllies === 0) {
            score += 50; // Isolated target bonus
        }
        
        // Active threat (check if ship is firing - approximate by checking if it has a target)
        if (targetShip.alphaAttackTarget || targetShip.assignedTarget) {
            score += 75; // Active threat bonus
        }
        
        // Shield penalty (avoid shielded targets)
        if (targetShip.shieldActive) {
            score -= 100; // Avoid shielded targets
        }
        
        return score;
    }
    
    // COORDINATION: Find alpha ship (primary target)
    findAlphaShip(aiShips) {
        if (!aiShips || !Array.isArray(aiShips)) return null;
        
        for (let aiShip of aiShips) {
            if (aiShip && aiShip.isAlpha && alphaAttackEnabled) {
                const dx = aiShip.x - this.x;
                const dy = aiShip.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < this.detectionRadius) {
                    return {
                        ship: aiShip,
                        distance: distance,
                        angle: Math.atan2(dy, dx),
                        isPlayer: false,
                        isAlpha: true,
                        score: 1000 + (1.0 / (distance + 1)) * 500 // High priority for alpha
                    };
                }
            }
        }
        return null;
    }
    
    // ENHANCEMENT 1: Find best target using smart prioritization
    // COORDINATION: Prioritizes alpha ship first, then blue ships
    findTarget(aiShips) {
        let bestTarget = null;
        let bestScore = -Infinity;
        
        // COORDINATION: PRIORITY 1 - Find alpha ship (primary target)
        const alphaTarget = this.findAlphaShip(aiShips);
        if (alphaTarget) {
            bestTarget = alphaTarget;
            bestScore = alphaTarget.score;
        }
        
        // COORDINATION: PRIORITY 2 - If no alpha or alpha out of range, check blue ships
        if (!bestTarget || bestTarget.distance > this.attackRange) {
            // Check player ship
            if (playerShipActive && ship) {
                const dx = ship.x - this.x;
                const dy = ship.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < this.detectionRadius) {
                    // Player ship gets base score (prioritize over AI ships if close)
                    const baseScore = 50;
                    const distanceFactor = 1.0 / (distance + 1); // Closer = better
                    const totalScore = baseScore * distanceFactor;
                    
                    if (totalScore > bestScore) {
                        bestScore = totalScore;
                        bestTarget = { 
                            ship: ship, 
                            distance: distance, 
                            angle: Math.atan2(dy, dx),
                            isPlayer: true,
                            score: totalScore
                        };
                    }
                }
            }
            
            // Check AI ships (blue ships) with smart scoring
            aiShips.forEach(aiShip => {
                if (!aiShip) return;
                
                const dx = aiShip.x - this.x;
                const dy = aiShip.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < this.detectionRadius) {
                    // Calculate smart score
                    const targetScore = this.scoreTarget(aiShip);
                    const distanceFactor = 1.0 / (distance + 1); // Closer = better
                    const totalScore = targetScore * distanceFactor;
                    
                    if (totalScore > bestScore) {
                        bestScore = totalScore;
                        bestTarget = {
                            ship: aiShip,
                            distance: distance,
                            angle: Math.atan2(dy, dx),
                            isPlayer: false,
                            isAlpha: false,
                            score: totalScore
                        };
                    }
                }
            });
        }
        
        return bestTarget;
    }

    // Predict target position for leading shots
    predictTargetPosition(target) {
        if (!target || !target.ship) return null;
        const dx = target.ship.x - this.x;
        const dy = target.ship.y - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance > this.attackRange) return null;
        const timeToReach = distance / this.bulletSpeed;
        const predictedX = target.ship.x + target.ship.velocityX * timeToReach;
        const predictedY = target.ship.y + target.ship.velocityY * timeToReach;
        const wrappedX = ((predictedX % worldW()) + worldW()) % worldW();
        const wrappedY = ((predictedY % worldH()) + worldH()) % worldH();
        return { x: wrappedX, y: wrappedY, angle: Math.atan2(wrappedY - this.y, wrappedX - this.x) };
    }

    shouldFire(aiShips) {
        if (this.fireCooldown > 0) return false;
        const target = this.findTarget(aiShips);
        if (!target || target.distance > this.attackRange) return false;
        return true;
    }

    // ENHANCEMENT 3: Fire at specific angle (helper for burst and spread)
    fireAtAngle(angle) {
        const bulletX = this.x + Math.cos(angle) * this.size;
        const bulletY = this.y + Math.sin(angle) * this.size;
        enemyBullets.push(new EnemyBullet(bulletX, bulletY, angle));
        playFireSound(true);
    }
    
    // SEEKING MISSILES: Fire seeking missile at target
    fireSeekingMissile(target) {
        if (!target || !target.ship) return;
        
        const angle = Math.atan2(target.ship.y - this.y, target.ship.x - this.x);
        const bulletX = this.x + Math.cos(angle) * this.size;
        const bulletY = this.y + Math.sin(angle) * this.size;
        
        // Create seeking missile
        const missile = new EnemyBullet(bulletX, bulletY, angle, true, target.ship);
        missile.speed = 6; // Slightly slower for balance
        missile.turnRate = 0.1; // Good turning ability
        enemyBullets.push(missile);
        playFireSound(true);
    }
    
    // ENHANCEMENT 3: Burst fire pattern (multiple shots in quick succession)
    fireBurst(target, count = 3) {
        if (!target) return;
        
        const predicted = this.predictTargetPosition(target);
        let baseAngle = this.angle;
        if (predicted) {
            baseAngle = predicted.angle;
        } else if (target.angle !== undefined) {
            baseAngle = target.angle;
        }
        
        const spread = Math.PI / 12; // 15 degree total spread
        
        // Fire all shots in the burst (frame-based, not setTimeout)
        this.burstFireActive = true;
        this.burstFireTimer = 0;
        this.burstFireCount = count;
        this.burstFireBaseAngle = baseAngle;
        this.burstFireSpread = spread;
        
        // Fire first shot immediately
        const firstAngle = baseAngle;
        this.fireAtAngle(firstAngle);
        
        // Remaining shots will be fired in updateBurstFire() method
        this.fireCooldown = this.fireRate;
    }
    
    // ENHANCEMENT 3: Update burst fire (called from makeDecision method)
    updateBurstFire() {
        if (!this.burstFireActive) return;
        
        this.burstFireTimer++;
        
        // Fire next shot every 2 frames
        if (this.burstFireTimer % 2 === 0 && this.burstFireTimer / 2 < this.burstFireCount) {
            const shotIndex = Math.floor(this.burstFireTimer / 2);
            if (shotIndex > 0 && shotIndex < this.burstFireCount) {
                const angle = this.burstFireBaseAngle + 
                    (shotIndex - this.burstFireCount/2) * (this.burstFireSpread / (this.burstFireCount - 1));
                this.fireAtAngle(angle);
            }
        }
        
        // End burst after all shots
        if (this.burstFireTimer >= this.burstFireCount * 2) {
            this.burstFireActive = false;
            this.burstFireTimer = 0;
        }
    }
    
    // ENHANCEMENT 3: Predictive spread fire (fire at predicted position with spread)
    firePredictiveSpread(target, spreadCount = 3) {
        if (!target) return;
        
        const predicted = this.predictTargetPosition(target);
        let baseAngle = this.angle;
        if (predicted) {
            baseAngle = predicted.angle;
        } else if (target.angle !== undefined) {
            baseAngle = target.angle;
        }
        
        const spread = Math.PI / 8; // 22.5 degree total spread
        
        for (let i = 0; i < spreadCount; i++) {
            const angle = baseAngle + (i - spreadCount/2) * (spread / (spreadCount - 1));
            this.fireAtAngle(angle);
        }
        
        this.fireCooldown = this.fireRate;
    }

    fire(aiShips) {
        if (this.fireCooldown > 0) return;
        const target = this.findTarget(aiShips);
        if (!target) return;
        
        // Boss enemies use complex attack patterns
        if (this.type === 'boss') {
            this.fireBossPattern(target, aiShips);
            return;
        }
        
        // SPECIALIZED ROLES: Role-based firing behavior
        if (this.specializedRole) {
            switch (this.specializedRole) {
                case 'sniper':
                    // Sniper: Long-range, high-accuracy shots
                    if (target.distance < this.attackRange * 1.5) {
                        const predicted = this.predictTargetPosition(target);
                        if (predicted) {
                            this.fireAtAngle(predicted.angle);
                        } else {
                            this.fireAtAngle(target.angle);
                        }
                        this.fireCooldown = this.fireRate * 1.5; // Slower fire rate
                    }
                    return;
                    
                case 'bomber':
                    // Bomber: Area-effect attacks (spread shots)
                    this.fireBurst(target, 5);
                    this.fireCooldown = this.fireRate * 2; // Much slower
                    return;
                    
                case 'interceptor':
                    // Interceptor: Fast, seeking missiles
                    if (gameRandom() < 0.3) { // 30% chance for seeking missile
                        this.fireSeekingMissile(target);
                    } else {
                        this.fireAtAngle(target.angle);
                    }
                    return;
                    
                case 'scout':
                    // Scout: Fast, frequent shots
                    this.fireAtAngle(target.angle);
                    this.fireCooldown = this.fireRate * 0.7; // Faster fire rate
                    return;
                    
                case 'tank':
                    // Tank: Standard shots, focuses on drawing fire
                    this.fireAtAngle(target.angle);
                    return;
                    
                case 'support':
                    // Support: Occasional seeking missiles to support others
                    if (gameRandom() < 0.2) { // 20% chance
                        this.fireSeekingMissile(target);
                    } else {
                        this.fireAtAngle(target.angle);
                    }
                    return;
            }
        }
        
        // CHAIN ATTACKS: Use chain attack if ready
        if (this.chainAttackReady) {
            if (this.chainAttackIndex % 3 === 0 && this.fireSeekingMissile) {
                // Every 3rd attack in chain is seeking
                this.fireSeekingMissile(target);
            } else if (this.chainAttackIndex % 3 === 1) {
                // Burst fire
                this.fireBurst(target, 3);
            } else {
                // Normal fire
                this.fireAtAngle(target.angle);
            }
            this.chainAttackReady = false;
            return;
        }
        
        // SPECIALIZED ROLES: Role-based firing takes priority
        if (!this.specializedRole && !this.chainAttackReady) {
            // ENHANCEMENT 3: Advanced enemies use burst fire
            if (this.type === 'advanced') {
                // 30% chance for burst fire, 20% for spread, 50% for normal
                const rand = gameRandom();
                if (rand < 0.3) {
                    this.fireBurst(target, 3);
                    return;
                } else if (rand < 0.5) {
                    this.firePredictiveSpread(target, 3);
                    return;
                }
            }
            
            // ENHANCEMENT 3: Basic enemies occasionally use burst (10% chance)
            if (this.type === 'basic' && gameRandom() < 0.1) {
                this.fireBurst(target, 2);
                return;
            }
            
            // Normal firing (default)
            const predicted = this.predictTargetPosition(target);
            let fireAngle = this.angle;
            if (predicted) {
                fireAngle = predicted.angle;
            } else {
                fireAngle = target.angle;
            }
            this.fireAtAngle(fireAngle);
            this.fireCooldown = this.fireRate;
        }
    }
    
    // Boss-specific firing patterns
    fireBossPattern(target, aiShips) {
        const predicted = this.predictTargetPosition(target);
        let baseAngle = this.angle;
        if (predicted) {
            baseAngle = predicted.angle;
        } else if (target) {
            baseAngle = target.angle;
        }
        
        switch (this.attackPattern) {
            case 'spread':
                // Spread shot: 5 bullets in a cone
                const spreadAngle = Math.PI / 6; // 30 degrees total spread
                for (let i = 0; i < 5; i++) {
                    const angle = baseAngle - spreadAngle / 2 + (spreadAngle / 4) * i;
                    const bulletX = this.x + Math.cos(angle) * this.size;
                    const bulletY = this.y + Math.sin(angle) * this.size;
                    enemyBullets.push(new EnemyBullet(bulletX, bulletY, angle));
                }
                this.fireCooldown = this.fireRate * 1.5; // Slightly longer cooldown for spread
                playFireSound(true);
                break;
                
            case 'rapid':
                // Rapid fire: 3 bullets in quick succession
                if (this.rapidFireBurstCount < 3) {
                    const bulletX = this.x + Math.cos(baseAngle) * this.size;
                    const bulletY = this.y + Math.sin(baseAngle) * this.size;
                    enemyBullets.push(new EnemyBullet(bulletX, bulletY, baseAngle));
                    this.rapidFireBurstCount++;
                    this.fireCooldown = 5; // Very short cooldown for rapid fire
                    playFireSound(true);
                } else {
                    this.rapidFireBurstCount = 0;
                    this.fireCooldown = this.fireRate * 2; // Longer cooldown after burst
                    this.attackPatternTimer = 0; // Reset pattern timer
                }
                break;
                
            case 'circular':
                // Circular pattern: 8 bullets in all directions
                for (let i = 0; i < 8; i++) {
                    const angle = (i * Math.PI * 2) / 8;
                    const bulletX = this.x + Math.cos(angle) * this.size;
                    const bulletY = this.y + Math.sin(angle) * this.size;
                    enemyBullets.push(new EnemyBullet(bulletX, bulletY, angle));
                }
                this.fireCooldown = this.fireRate * 2; // Longer cooldown for circular
                playFireSound(true);
                break;
                
            case 'normal':
            default:
                // Normal single shot
                const bulletX = this.x + Math.cos(baseAngle) * this.size;
                const bulletY = this.y + Math.sin(baseAngle) * this.size;
                enemyBullets.push(new EnemyBullet(bulletX, bulletY, baseAngle));
                this.fireCooldown = this.fireRate;
                playFireSound(true);
                break;
        }
    }
    
    // Update boss phase based on health
    updateBossPhase() {
        if (this.type !== 'boss') return;
        
        const healthPercent = this.health / this.maxHealth;
        const oldPhase = this.bossPhase;
        
        if (healthPercent > 0.6) {
            this.bossPhase = 1;
        } else if (healthPercent > 0.3) {
            this.bossPhase = 2;
        } else {
            this.bossPhase = 3;
        }
        
        // Phase transition effects
        if (oldPhase !== this.bossPhase) {
            this.phaseTransitionTimer = 60; // 1 second transition effect
            this.shieldActive = true;
            this.shieldDuration = 30; // Brief invincibility on phase change
            if (typeof onBossPhaseChange === 'function') onBossPhaseChange(this, oldPhase, this.bossPhase);
        }
    }
    
    // Update boss attack pattern based on phase
    updateBossAttackPattern() {
        if (this.type !== 'boss') return;
        
        if (this.turretDisabled) {
            this.attackPattern = 'normal';
            return;
        }
        
        this.attackPatternTimer++;
        
        // Change pattern periodically based on phase
        if (this.attackPatternTimer > 120) { // Change every 2 seconds
            this.attackPatternTimer = 0;
            
            switch (this.bossPhase) {
                case 1:
                    // Phase 1: Normal and occasional spread
                    this.attackPattern = gameRandom() < 0.3 ? 'spread' : 'normal';
                    break;
                case 2:
                    // Phase 2: Spread, rapid fire, and normal
                    const patterns2 = ['spread', 'rapid', 'normal'];
                    this.attackPattern = patterns2[Math.floor(gameRandom() * patterns2.length)];
                    break;
                case 3:
                    // Phase 3: All patterns including circular
                    const patterns3 = ['spread', 'rapid', 'circular', 'normal'];
                    this.attackPattern = patterns3[Math.floor(gameRandom() * patterns3.length)];
                    break;
            }
        }
    }
    
    // SHIELD SYSTEM: Update shield for basic and advanced enemy ships
    updateEnemyShield() {
        if (this.type === 'boss') {
            // Boss uses separate shield system
            this.updateBossShield();
            return;
        }
        
        // Only basic and advanced ships use this shield system
        if (this.type !== 'basic' && this.type !== 'advanced') return;
        
        // Update shield duration
        if (this.shieldDuration > 0) {
            this.shieldDuration--;
            this.shieldActive = true;
        } else {
            this.shieldActive = false;
        }
        
        // Update cooldown
        if (this.shieldCooldown > 0) {
            this.shieldCooldown--;
        }
        
        // Activate shield periodically (same logic as blue ships)
        // Basic enemies: 0.5% chance per frame when cooldown is ready
        // Advanced enemies: 1% chance per frame (more frequent)
        if (this.shieldCooldown <= 0 && !this.shieldActive) {
            const shieldChance = this.type === 'advanced' ? 0.01 : 0.005; // 1% for advanced, 0.5% for basic
            const shieldDuration = 60; // 1 second (same as blue ships)
            
            if (gameRandom() < shieldChance) {
                this.shieldActive = true;
                this.shieldDuration = shieldDuration;
                this.shieldCooldown = this.shieldCooldownMax; // 5 seconds cooldown
            }
        }
    }
    
    // Boss shield system
    updateBossShield() {
        if (this.type !== 'boss') return;
        if (this.shieldDisabled) {
            this.shieldActive = false;
            this.shieldDuration = 0;
            return;
        }
        
        if (this.shieldDuration > 0) {
            this.shieldDuration--;
            this.shieldActive = true;
        } else {
            this.shieldActive = false;
        }
        
        // Activate shield periodically in phase 2 and 3
        if (this.shieldCooldown > 0) {
            this.shieldCooldown--;
        } else if (this.bossPhase >= 2 && !this.shieldActive && gameRandom() < 0.01) {
            this.shieldActive = true;
            this.shieldDuration = 90; // 1.5 seconds of shield
            this.shieldCooldown = 300; // 5 second cooldown
        }
    }
    
    // Boss teleportation ability
    bossTeleport() {
        if (this.type !== 'boss' || this.teleportCooldown > 0 || this.warpDisabled) return false;
        
        // Teleport in phase 3 when health is very low
        if (this.bossPhase === 3 && this.health <= 1 && gameRandom() < 0.02) {
            this.x = gameRandom() * worldW();
            this.y = gameRandom() * worldH();
            this.teleportCooldown = 180; // 3 second cooldown
            return true;
        }
        return false;
    }
    
    // Boss erratic movement in phase 3
    applyErraticMovement() {
        if (this.type !== 'boss' || this.bossPhase !== 3) return;
        
        this.erraticMovementTimer++;
        if (this.erraticMovementTimer > 10) {
            this.erraticMovementTimer = 0;
            // Random direction changes
            if (gameRandom() < 0.3) {
                this.targetAngle += (gameRandom() - 0.5) * Math.PI;
            }
            // Sudden thrust bursts
            if (gameRandom() < 0.2) {
                this.thrust(2.0); // Strong thrust
            }
        }
    }

    detectIncomingBullets() {
        let nearestBullet = null;
        let nearestDistance = Infinity;
        bullets.forEach(bullet => {
            const dx = bullet.x - this.x;
            const dy = bullet.y - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < this.evasionRadius && distance < nearestDistance) {
                const angleToBullet = Math.atan2(dy, dx);
                const bulletAngle = bullet.angle;
                const angleDiff = Math.abs(this.normalizeAngle(angleToBullet - bulletAngle));
                if (angleDiff < Math.PI / 3) {
                    nearestBullet = { bullet, distance, angle: angleToBullet };
                    nearestDistance = distance;
                }
            }
        });
        return nearestBullet;
    }

    evadeBullets() {
        const incoming = this.detectIncomingBullets();
        if (incoming && incoming.distance < this.evasionRadius) {
            const evasionAngle = incoming.angle + Math.PI / 2 + (gameRandom() - 0.5) * Math.PI / 2;
            this.targetAngle = evasionAngle;
            this.behaviorState = 'evade';
            return true;
        }
        return false;
    }

    findCover(asteroids, target) {
        if (!target || !target.ship) return null;
        let bestCover = null;
        let bestCoverValue = 0;
        asteroids.forEach(asteroid => {
            const dxToTarget = target.ship.x - this.x;
            const dyToTarget = target.ship.y - this.y;
            const dxToAsteroid = asteroid.x - this.x;
            const dyToAsteroid = asteroid.y - this.y;
            const dotProduct = dxToTarget * dxToAsteroid + dyToTarget * dyToAsteroid;
            const targetDist = Math.sqrt(dxToTarget * dxToTarget + dyToTarget * dyToTarget);
            const asteroidDist = Math.sqrt(dxToAsteroid * dxToAsteroid + dyToAsteroid * dyToAsteroid);
            if (dotProduct > 0 && asteroidDist < targetDist && asteroidDist < 150) {
                const coverValue = asteroid.size / asteroidDist;
                if (coverValue > bestCoverValue) {
                    bestCoverValue = coverValue;
                    bestCover = asteroid;
                }
            }
        });
        return bestCover;
    }

    useCover(asteroids, target) {
        if (this.type === 'basic') return false;
        const cover = this.findCover(asteroids, target);
        if (cover && target) {
            this.coverAsteroid = cover;
            const dxToTarget = target.ship.x - cover.x;
            const dyToTarget = target.ship.y - cover.y;
            const angleToTarget = Math.atan2(dyToTarget, dxToTarget);
            const coverDistance = cover.size + this.size + 10;
            this.targetAngle = angleToTarget + Math.PI;
            const targetX = cover.x - Math.cos(angleToTarget) * coverDistance;
            const targetY = cover.y - Math.sin(angleToTarget) * coverDistance;
            const dx = targetX - this.x;
            const dy = targetY - this.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > 5) {
                this.behaviorState = 'cover';
                return true;
            }
        }
        return false;
    }

    calculateFlankAngle(target) {
        if (!target || !target.ship || this.type === 'basic') return null;
        const dx = target.ship.x - this.x;
        const dy = target.ship.y - this.y;
        const angleToTarget = Math.atan2(dy, dx);
        const flankOffset = (gameRandom() > 0.5 ? 1 : -1) * Math.PI / 2;
        return angleToTarget + flankOffset;
    }

    normalizeAngle(angle) {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }

    rotate(direction) {
        this.angle += direction * this.rotationSpeed;
    }

    thrust(power = 1) {
        this.velocityX += Math.cos(this.angle) * this.thrustPower * power;
        this.velocityY += Math.sin(this.angle) * this.thrustPower * power;
        const speed = Math.sqrt(this.velocityX * this.velocityX + this.velocityY * this.velocityY);
        if (speed > this.maxVelocity) {
            this.velocityX = (this.velocityX / speed) * this.maxVelocity;
            this.velocityY = (this.velocityY / speed) * this.maxVelocity;
        }
    }

    checkHealth() {
        const healthPercent = this.health / this.maxHealth;
        if (this.type === 'boss') {
            // Boss uses phase system instead
            this.updateBossPhase();
        } else {
            // Basic and advanced enemies use standard health-based behavior
            if (healthPercent < 0.3) {
                this.behaviorState = 'retreat';
            } else if (healthPercent < 0.6) {
                if (this.behaviorState === 'pursuit') {
                    this.behaviorState = 'evade';
                }
            }
        }
    }

    coordinateWithOthers(enemyShips) {
        // COORDINATION: Use formation position if coordinating
        if (this.isCoordinating && this.formationPosition) {
            const dx = this.formationPosition.x - this.x;
            const dy = this.formationPosition.y - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance > 20) {
                // Move toward formation position
                this.targetAngle = Math.atan2(dy, dx);
            }
        }
        
        // Basic avoidance for non-coordinating enemies
        if (this.type === 'basic' || this.isCoordinating) return;
        
        const nearbyEnemies = enemyShips.filter(enemy => {
            if (enemy === this) return false;
            const dx = enemy.x - this.x;
            const dy = enemy.y - this.y;
            return Math.sqrt(dx * dx + dy * dy) < 200;
        });
        if (nearbyEnemies.length > 0) {
            nearbyEnemies.forEach(enemy => {
                const dx = enemy.x - this.x;
                const dy = enemy.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < 50) {
                    const avoidanceAngle = Math.atan2(dy, dx) + Math.PI;
                    this.targetAngle = avoidanceAngle;
                }
            });
        }
    }

    makeDecision(asteroids, enemyShips, aiShips) {
        if (this.stunTimer > 0) {
            this.stunTimer--;
            return; // Stunned by EMP: cannot move, aim, or fire!
        }
        if (this.fireCooldown > 0) this.fireCooldown--;
        if (this.type === 'boss' && this.teleportCooldown > 0) this.teleportCooldown--;
        
        // ENHANCEMENT 3: Update burst fire
        if (this.burstFireActive) {
            this.updateBurstFire();
        }
        
        // SHIELD SYSTEM: Update shield for basic and advanced enemy ships
        if (this.type === 'basic' || this.type === 'advanced') {
            this.updateEnemyShield();
        }
        
        this.checkHealth();
        
        // Boss-specific updates
        if (this.type === 'boss') {
            this.updateBossAttackPattern();
            this.updateBossShield();
            this.applyErraticMovement();
            if (this.bossTeleport()) {
                // Teleported, skip normal movement this frame
                return;
            }
        }
        
        // COORDINATION: Use assigned target from coordination system if coordinating
        let target = null;
        if (this.isCoordinating && this.attackDelay > 0) {
            this.attackDelay--;
            // Still find target but delay attack
            target = this.findTarget(aiShips);
        } else {
            // ENHANCEMENT 1: Find best target using smart prioritization
            target = this.findTarget(aiShips);
        }
        
        if (target) {
            this.target = target; // Store for visual indicators
            this.targetScore = target.score || 0; // Store target score for reference
        }
        
        if (this.evadeBullets()) {
            // Continue evading
        } else if (target && this.useCover(asteroids, target)) {
            // Continue using cover
        } else {
            if (target) {
                if (this.behaviorState === 'retreat' && this.type !== 'boss') {
                    this.targetAngle = target.angle + Math.PI;
                } else {
                    if (target.distance < this.attackRange) {
                        this.behaviorState = 'attack';
                        const predicted = this.predictTargetPosition(target);
                        if (predicted) {
                            this.targetAngle = predicted.angle;
                        } else {
                            this.targetAngle = target.angle;
                        }
                    } else {
                        this.behaviorState = 'pursuit';
                        this.targetAngle = target.angle;
                    }
                    if (this.type === 'advanced' && gameRandom() < 0.1) {
                        const flankAngle = this.calculateFlankAngle(target);
                        if (flankAngle !== null) {
                            this.targetAngle = flankAngle;
                        }
                    }
                }
            } else {
                if (gameRandom() < 0.01) {
                    this.targetAngle = gameRandom() * Math.PI * 2;
                }
            }
        }
        this.coordinateWithOthers(enemyShips);
        let angleDiff = this.normalizeAngle(this.targetAngle - this.angle);
        if (Math.abs(angleDiff) > 0.1) {
            this.rotate(angleDiff > 0 ? 1 : -1);
        }
        
        // Boss has different movement patterns
        if (this.type === 'boss') {
            // Boss movement based on phase
            if (this.bossPhase === 1) {
                // Phase 1: Normal pursuit
                if (gameRandom() < 0.3) {
                    this.thrust();
                }
            } else if (this.bossPhase === 2) {
                // Phase 2: More aggressive
                if (gameRandom() < 0.4) {
                    this.thrust(1.2);
                }
            } else {
                // Phase 3: Erratic movement (handled in applyErraticMovement)
                if (gameRandom() < 0.5) {
                    this.thrust(1.5);
                }
            }
        } else {
            // Normal enemy movement
            if (this.behaviorState !== 'retreat' && gameRandom() < 0.3) {
                this.thrust();
            } else if (this.behaviorState === 'retreat') {
                if (gameRandom() < 0.5) {
                    this.thrust(1.5);
                }
            }
        }
        
        if (this.shouldFire(aiShips)) {
            this.fire(aiShips);
        }
    }

    update() {
        this.velocityX *= this.friction;
        this.velocityY *= this.friction;
        if (sphereWorld()) {
            WORLD_GEO.moveBody(this, 0);
            return;
        }
        this.x += this.velocityX;
        this.y += this.velocityY;
        if (this.x < 0) this.x = worldW();
        if (this.x > worldW()) this.x = 0;
        if (this.y < 0) this.y = worldH();
        if (this.y > worldH()) this.y = 0;
    }

    draw() {
        ctx.save();
        ctx.translate(this.x, this.y);
        
        // SHIELD SYSTEM: Shield visual effect for basic and advanced enemy ships (same as blue ships)
        if ((this.type === 'basic' || this.type === 'advanced') && this.shieldActive) {
            ctx.strokeStyle = GAME_THEME.enemy; // Red shield for enemy ships (different from blue cyan)
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.6;
            ctx.beginPath();
            const shieldRadius = this.shieldRadius || (this.size + 8);
            ctx.arc(0, 0, shieldRadius, 0, Math.PI * 2);
            ctx.stroke();
            
            // Pulsing effect (same as blue ships)
            ctx.globalAlpha = 0.3;
            ctx.beginPath();
            ctx.arc(0, 0, shieldRadius + Math.sin(Date.now() / 100) * 3, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
        
        // Boss shield visual effect
        if (this.type === 'boss' && this.shieldActive) {
            ctx.strokeStyle = GAME_THEME.highlight;
            ctx.lineWidth = 3;
            ctx.globalAlpha = 0.6;
            ctx.beginPath();
            const shieldRadius = this.size + 10 + Math.sin(Date.now() / 50) * 5;
            ctx.arc(0, 0, shieldRadius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
        
        // Phase transition effect
        if (this.type === 'boss' && this.phaseTransitionTimer > 0) {
            ctx.strokeStyle = GAME_THEME.text;
            ctx.lineWidth = 4;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            ctx.arc(0, 0, this.size + 15, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
            this.phaseTransitionTimer--;
        }
        
        // COORDINATION: Visual indicators
        let roleColor = this.color; // Default to ship color
        if (this.isCoordinating) {
            // Role indicator color
            const roleColors = {
                'leader': GAME_THEME.leader,      // Yellow for leader
                'attacker': GAME_THEME.enemy,    // Red for attacker
                'flanker': GAME_THEME.orange,     // Orange for flanker
                'left_flanker': GAME_THEME.orange,
                'right_flanker': GAME_THEME.orange,
                'pincer_left': GAME_THEME.orange,
                'pincer_right': GAME_THEME.orange,
                'direct_attacker': GAME_THEME.enemy,
                'alpha_attacker': GAME_THEME.danger, // Bright red for alpha attacker
                'distraction': GAME_THEME.amber,
                'interceptor': GAME_THEME.orange
            };
            
            roleColor = roleColors[this.coordinationRole] || GAME_THEME.enemy;
        }
        
        // SPECIALIZED ROLES: Visual indicators
        if (this.specializedRole) {
            const specializedRoleColors = {
                'scout': GAME_THEME.formationScout,      // Cyan for scout
                'tank': GAME_THEME.orange,       // Orange for tank
                'sniper': GAME_THEME.magenta,     // Magenta for sniper
                'interceptor': GAME_THEME.formationInterceptor, // Green for interceptor
                'bomber': GAME_THEME.danger,     // Red for bomber
                'support': GAME_THEME.formationSupport      // Yellow for support
            };
            
            const specializedColor = specializedRoleColors[this.specializedRole];
            if (specializedColor) {
                // Draw specialized role indicator
                ctx.strokeStyle = specializedColor;
                ctx.lineWidth = 1.5;
                ctx.globalAlpha = 0.6;
                ctx.beginPath();
                ctx.arc(0, 0, this.size + 3, 0, Math.PI * 2);
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }
            
            // Draw role indicator circle
            ctx.strokeStyle = roleColor;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            ctx.arc(0, 0, this.size + 5, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
            
            // Pattern indicator (pulsing effect)
            if (this.attackPattern) {
                ctx.strokeStyle = roleColor;
                ctx.lineWidth = 1;
                ctx.globalAlpha = 0.4 + Math.sin(Date.now() / 200) * 0.3;
                ctx.beginPath();
                ctx.arc(0, 0, this.size + 8 + Math.sin(Date.now() / 100) * 3, 0, Math.PI * 2);
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }
        }
        
        // Draw formation position indicator
        if (this.formationPosition) {
            ctx.save();
            ctx.translate(this.formationPosition.x - this.x, this.formationPosition.y - this.y);
            ctx.strokeStyle = GAME_THEME.amber;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.5;
            ctx.beginPath();
            ctx.arc(0, 0, 10, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
            ctx.restore();
        }
        
        ctx.rotate(this.angle);
        ctx.strokeStyle = roleColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        if (this.type === 'boss') {
            // Hexagonal shape with phase-based color intensity
            let intensity = 1.0;
            if (this.bossPhase === 2) intensity = 1.2;
            if (this.bossPhase === 3) intensity = 1.5;
            
            ctx.strokeStyle = `rgba(255, ${Math.floor(100 / intensity)}, ${Math.floor(100 / intensity)}, 1)`;
            
            for (let i = 0; i < 6; i++) {
                const angle = (i * Math.PI) / 3;
                const x = Math.cos(angle) * this.size;
                const y = Math.sin(angle) * this.size;
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.closePath();
        } else if (this.type === 'advanced') {
            ctx.moveTo(0, -this.size);
            ctx.lineTo(this.size, 0);
            ctx.lineTo(0, this.size);
            ctx.lineTo(-this.size, 0);
            ctx.closePath();
        } else {
            ctx.moveTo(this.size, 0);
            ctx.lineTo(-this.size / 2, -this.size / 2);
            ctx.lineTo(-this.size / 2, this.size / 2);
            ctx.closePath();
        }
        ctx.stroke();
        if (this.type !== 'basic' && this.health < this.maxHealth) {
            ctx.restore();
            ctx.save();
            ctx.translate(this.x, this.y - this.size - 10);
            const barWidth = this.size * 2;
            const barHeight = 4;
            const healthPercent = this.health / this.maxHealth;
            
            // Background
            ctx.fillStyle = GAME_THEME.barBg;
            ctx.fillRect(-this.size, -3, barWidth, barHeight);
            
            // Health bar color based on health and phase (for boss)
            if (this.type === 'boss') {
                // Boss health bar changes color by phase
                if (this.bossPhase === 1) ctx.fillStyle = GAME_THEME.healthHigh; // Green
                else if (this.bossPhase === 2) ctx.fillStyle = GAME_THEME.healthMid; // Yellow
                else ctx.fillStyle = GAME_THEME.healthLow; // Red
            } else {
                // Advanced enemy uses standard colors
                ctx.fillStyle = healthPercent > 0.5 ? GAME_THEME.healthHigh : healthPercent > 0.25 ? GAME_THEME.healthMid : GAME_THEME.healthLow;
            }
            
            ctx.fillRect(-this.size, -3, barWidth * healthPercent, barHeight);
            
            // Border
            ctx.strokeStyle = GAME_THEME.text;
            ctx.lineWidth = 1;
            ctx.strokeRect(-this.size, -3, barWidth, barHeight);
            
            // Boss phase indicator
            if (this.type === 'boss') {
                ctx.fillStyle = GAME_THEME.text;
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(`Phase ${this.bossPhase}`, barWidth / 2 - this.size, -8);
            }
            
            ctx.restore();
            ctx.save();
            ctx.translate(this.x, this.y);
            ctx.rotate(this.angle);
        }
        if (Math.abs(this.velocityX) > 0.1 || Math.abs(this.velocityY) > 0.1) {
            ctx.strokeStyle = GAME_THEME.orange;
            ctx.beginPath();
            if (this.type === 'boss') {
                ctx.moveTo(-this.size * 0.7, -this.size * 0.3);
                ctx.lineTo(-this.size - 8, 0);
                ctx.lineTo(-this.size * 0.7, this.size * 0.3);
            } else {
                ctx.moveTo(-this.size / 2, -this.size / 4);
                ctx.lineTo(-this.size - 5, 0);
                ctx.lineTo(-this.size / 2, this.size / 4);
            }
            ctx.stroke();
        }
        ctx.restore();

        // Draw Boss Subsystems attached via hardpoint struts
        if (this.type === 'boss' && this.subsystems) {
            this.subsystems.forEach(sub => {
                const pos = this.getSubsystemWorldPos(sub);
                ctx.save();
                // Strut line from boss center to subsystem node
                ctx.strokeStyle = sub.destroyed ? 'rgba(90, 90, 90, 0.4)' : 'rgba(255, 255, 255, 0.5)';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(this.x, this.y);
                ctx.lineTo(pos.x, pos.y);
                ctx.stroke();

                if (sub.destroyed) {
                    ctx.strokeStyle = '#444';
                    ctx.fillStyle = '#222';
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, sub.radius * 0.7, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                } else {
                    if (FXL) FXL.glow(ctx, pos.x, pos.y, sub.radius * 1.8, sub.color, 0.5);
                    ctx.strokeStyle = sub.color;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, sub.radius, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(10, 15, 25, 0.85)';
                    ctx.fill();
                    ctx.stroke();

                    // Mini HP ring
                    const hpR = Math.max(0, sub.health / sub.maxHealth);
                    ctx.strokeStyle = sub.color;
                    ctx.lineWidth = 2.5;
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, sub.radius + 3, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * hpR);
                    ctx.stroke();
                }
                ctx.restore();
            });
        }

        // EMP Stun electric spark arcs
        if (this.stunTimer > 0) {
            ctx.save();
            ctx.strokeStyle = '#00E5FF';
            ctx.lineWidth = 1.5;
            for (let i = 0; i < 4; i++) {
                const a1 = (Date.now() * 0.01 + i * 1.5);
                const r1 = this.size * 0.5;
                const r2 = this.size * 1.1;
                ctx.beginPath();
                ctx.moveTo(this.x + Math.cos(a1) * r1, this.y + Math.sin(a1) * r1);
                ctx.lineTo(this.x + Math.cos(a1 + 0.5) * r2, this.y + Math.sin(a1 + 0.5) * r2);
                ctx.stroke();
            }
            ctx.restore();
        }
    }

    getPosition() {
        return { x: this.x, y: this.y, radius: this.radius };
    }
}
