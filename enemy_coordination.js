// Extracted from game.js as part of a module split.
// Relies on globals defined in game.js/world.js/rng.js/fx.js loaded earlier
// (canvas, ctx, worldW/worldH/sphereWorld, gameRandom, FXL, ACTION_SPACE_SIZE,
// decodeAction, keys, etc.) — same plain-script global scope as before, just
// split across files. Depends on AIShip (defined in ai_ship.js) at runtime.

// Enemy Coordination System - Manages coordinated attacks
class EnemyCoordinationSystem {
    constructor() {
        this.leader = null;
        this.currentPattern = null;
        this.patternStartTime = 0;
        this.coordinationGroups = new Map(); // groupId -> {enemies: [], target: null, pattern: null}
        this.nextGroupId = 0;
        this.updateInterval = 60; // Update every 60 frames (1 second at 60fps)
        this.frameCount = 0;
        
        // MULTI-PATTERN SWITCHING: Dynamic pattern transitions
        this.patternHistory = []; // Track pattern usage
        this.patternEffectiveness = new Map(); // Track pattern effectiveness
        this.patternSwitchTimer = 0;
        this.patternSwitchInterval = 300; // Switch patterns every 5 seconds
        
        // CHAIN ATTACKS: Sequential coordinated attacks
        this.chainAttackActive = false;
        this.chainAttackSequence = [];
        this.chainAttackIndex = 0;
        this.chainAttackTimer = 0;
        
        // ADAPTIVE FORMATIONS: Dynamic formation management
        this.formationType = 'default'; // 'default', 'tight', 'spread', 'diamond', 'line'
        this.formationAdaptationTimer = 0;
        this.battlefieldState = {
            enemyCount: 0,
            targetCount: 0,
            threatLevel: 0,
            averageDistance: 0
        };
    }
    
    // Select leader from nearby enemies
    selectLeader(enemyShips, target) {
        if (!target || enemyShips.length === 0) return null;
        
        let bestLeader = null;
        let bestScore = -Infinity;
        
        enemyShips.forEach(enemy => {
            if (!enemy || enemy.type === 'boss') return; // Boss doesn't coordinate
            
            const dx = enemy.x - target.x;
            const dy = enemy.y - target.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Leader should be close to target and have good health
            const healthScore = enemy.health / enemy.maxHealth;
            const distanceScore = 1.0 / (distance + 1);
            const score = healthScore * 0.6 + distanceScore * 0.4;
            
            if (score > bestScore) {
                bestScore = score;
                bestLeader = enemy;
            }
        });
        
        return bestLeader;
    }
    
    // Find nearby enemies for coordination
    findNearbyEnemies(enemy, enemyShips, radius = 300) {
        return enemyShips.filter(e => {
            if (!e || e === enemy || e.type === 'boss') return false;
            const dx = e.x - enemy.x;
            const dy = e.y - enemy.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            return distance < radius;
        });
    }
    
    // Execute pincer attack pattern
    executePincerAttack(enemies, target) {
        if (!target || enemies.length < 2) return;
        
        const group1 = enemies.slice(0, Math.ceil(enemies.length / 2));
        const group2 = enemies.slice(Math.ceil(enemies.length / 2));
        
        // Calculate flank positions
        const targetAngle = Math.atan2(target.y - target.x, target.x - target.x);
        const flankDistance = 200;
        
        group1.forEach((enemy, index) => {
            enemy.coordinationRole = 'pincer_left';
            enemy.attackPattern = 'pincer';
            const angle = targetAngle - Math.PI / 2; // Left side
            enemy.formationPosition = {
                x: target.x + Math.cos(angle) * flankDistance,
                y: target.y + Math.sin(angle) * flankDistance
            };
            enemy.attackDelay = index * 10; // Stagger attacks
        });
        
        group2.forEach((enemy, index) => {
            enemy.coordinationRole = 'pincer_right';
            enemy.attackPattern = 'pincer';
            const angle = targetAngle + Math.PI / 2; // Right side
            enemy.formationPosition = {
                x: target.x + Math.cos(angle) * flankDistance,
                y: target.y + Math.sin(angle) * flankDistance
            };
            enemy.attackDelay = index * 10;
        });
    }
    
    // Execute wave attack pattern
    executeWaveAttack(enemies, target) {
        if (!target || enemies.length === 0) return;
        
        const waveSize = Math.min(3, enemies.length);
        const waves = [];
        
        for (let i = 0; i < enemies.length; i += waveSize) {
            waves.push(enemies.slice(i, i + waveSize));
        }
        
        waves.forEach((wave, waveIndex) => {
            wave.forEach((enemy, enemyIndex) => {
                if (waveIndex === 0) {
                    // First wave: Engage escorts
                    enemy.coordinationRole = 'distraction';
                    enemy.attackPattern = 'wave';
                } else {
                    // Subsequent waves: Attack target
                    enemy.coordinationRole = 'alpha_attacker';
                    enemy.attackPattern = 'wave';
                    enemy.attackDelay = waveIndex * 60; // 1 second delay per wave
                }
                enemy.formationPosition = {
                    x: target.x + (enemyIndex - waveSize / 2) * 50,
                    y: target.y - waveIndex * 100
                };
            });
        });
    }
    
    // Execute flanking maneuver
    executeFlankingManeuver(enemies, target) {
        if (!target || enemies.length < 2) return;
        
        const directCount = Math.ceil(enemies.length * 0.4);
        const directAttackers = enemies.slice(0, directCount);
        const flankers = enemies.slice(directCount);
        
        // Direct attackers
        directAttackers.forEach(enemy => {
            enemy.coordinationRole = 'direct_attacker';
            enemy.attackPattern = 'flanking';
            enemy.formationPosition = {
                x: target.x,
                y: target.y
            };
        });
        
        // Flankers
        const leftFlankers = flankers.slice(0, Math.ceil(flankers.length / 2));
        const rightFlankers = flankers.slice(Math.ceil(flankers.length / 2));
        
        leftFlankers.forEach((enemy, index) => {
            enemy.coordinationRole = 'left_flanker';
            enemy.attackPattern = 'flanking';
            const angle = Math.atan2(target.y - enemy.y, target.x - enemy.x) - Math.PI / 2;
            enemy.formationPosition = {
                x: target.x + Math.cos(angle) * 150,
                y: target.y + Math.sin(angle) * 150
            };
            enemy.attackDelay = 30;
        });
        
        rightFlankers.forEach((enemy, index) => {
            enemy.coordinationRole = 'right_flanker';
            enemy.attackPattern = 'flanking';
            const angle = Math.atan2(target.y - enemy.y, target.x - enemy.x) + Math.PI / 2;
            enemy.formationPosition = {
                x: target.x + Math.cos(angle) * 150,
                y: target.y + Math.sin(angle) * 150
            };
            enemy.attackDelay = 30;
        });
    }
    
    // SPECIALIZED ROLES: Assign roles to enemies
    assignSpecializedRoles(enemies, target) {
        if (!target || enemies.length === 0) return;
        
        const roles = ['scout', 'tank', 'sniper', 'interceptor', 'bomber', 'support'];
        const roleCounts = {
            scout: Math.ceil(enemies.length * 0.2),
            tank: Math.ceil(enemies.length * 0.15),
            sniper: Math.ceil(enemies.length * 0.15),
            interceptor: Math.ceil(enemies.length * 0.2),
            bomber: Math.ceil(enemies.length * 0.15),
            support: Math.ceil(enemies.length * 0.15)
        };
        
        let roleIndex = 0;
        enemies.forEach((enemy, index) => {
            if (!enemy.roleAssigned) {
                // Assign role based on enemy type and position
                if (enemy.type === 'advanced' && roleIndex < roleCounts.sniper) {
                    enemy.specializedRole = 'sniper';
                } else if (enemy.health === enemy.maxHealth && roleIndex < roleCounts.tank) {
                    enemy.specializedRole = 'tank';
                } else if (index < roleCounts.scout) {
                    enemy.specializedRole = 'scout';
                } else if (index < roleCounts.scout + roleCounts.interceptor) {
                    enemy.specializedRole = 'interceptor';
                } else if (index < roleCounts.scout + roleCounts.interceptor + roleCounts.bomber) {
                    enemy.specializedRole = 'bomber';
                } else {
                    enemy.specializedRole = 'support';
                }
                enemy.roleAssigned = true;
                roleIndex++;
            }
        });
    }
    
    // ADAPTIVE FORMATIONS: Calculate adaptive formation positions
    calculateAdaptiveFormation(enemies, target, formationType) {
        if (!target || enemies.length === 0) return;
        
        const positions = [];
        const centerX = target.x;
        const centerY = target.y;
        
        switch (formationType) {
            case 'tight':
                // Tight formation - close together
                const tightRadius = 100;
                enemies.forEach((enemy, index) => {
                    const angle = (Math.PI * 2 * index) / enemies.length;
                    positions.push({
                        x: centerX + Math.cos(angle) * tightRadius,
                        y: centerY + Math.sin(angle) * tightRadius
                    });
                });
                break;
                
            case 'spread':
                // Spread formation - wider spacing
                const spreadRadius = 250;
                enemies.forEach((enemy, index) => {
                    const angle = (Math.PI * 2 * index) / enemies.length;
                    positions.push({
                        x: centerX + Math.cos(angle) * spreadRadius,
                        y: centerY + Math.sin(angle) * spreadRadius
                    });
                });
                break;
                
            case 'diamond':
                // Diamond formation - 4-point attack
                if (enemies.length >= 4) {
                    const diamondSize = 150;
                    positions.push({x: centerX, y: centerY - diamondSize}); // Top
                    positions.push({x: centerX + diamondSize, y: centerY}); // Right
                    positions.push({x: centerX, y: centerY + diamondSize}); // Bottom
                    positions.push({x: centerX - diamondSize, y: centerY}); // Left
                    // Fill remaining positions
                    for (let i = 4; i < enemies.length; i++) {
                        const angle = (Math.PI * 2 * i) / enemies.length;
                        positions.push({
                            x: centerX + Math.cos(angle) * 200,
                            y: centerY + Math.sin(angle) * 200
                        });
                    }
                }
                break;
                
            case 'line':
                // Line formation - horizontal or vertical line
                const lineSpacing = 80;
                const isHorizontal = gameRandom() > 0.5;
                enemies.forEach((enemy, index) => {
                    if (isHorizontal) {
                        positions.push({
                            x: centerX + (index - enemies.length / 2) * lineSpacing,
                            y: centerY
                        });
                    } else {
                        positions.push({
                            x: centerX,
                            y: centerY + (index - enemies.length / 2) * lineSpacing
                        });
                    }
                });
                break;
                
            default:
                // Default formation - circle
                const defaultRadius = 180;
                enemies.forEach((enemy, index) => {
                    const angle = (Math.PI * 2 * index) / enemies.length;
                    positions.push({
                        x: centerX + Math.cos(angle) * defaultRadius,
                        y: centerY + Math.sin(angle) * defaultRadius
                    });
                });
        }
        
        return positions;
    }
    
    // CHAIN ATTACKS: Execute sequential coordinated attacks
    executeChainAttack(enemies, target) {
        if (!target || enemies.length < 2) return;
        
        this.chainAttackActive = true;
        this.chainAttackSequence = [];
        this.chainAttackIndex = 0;
        this.chainAttackTimer = 0;
        
        // Create chain attack sequence
        // Each enemy attacks in sequence, building on previous attacks
        enemies.forEach((enemy, index) => {
            this.chainAttackSequence.push({
                enemy: enemy,
                delay: index * 20, // 20 frames between each attack
                attackType: index % 3 === 0 ? 'seeking' : (index % 3 === 1 ? 'burst' : 'normal'),
                target: target
            });
        });
    }
    
    // Update chain attack
    updateChainAttack() {
        if (!this.chainAttackActive) return;
        
        this.chainAttackTimer++;
        
        // Execute attacks in sequence
        this.chainAttackSequence.forEach((attack, index) => {
            if (this.chainAttackTimer >= attack.delay && !attack.executed) {
                const enemy = attack.enemy;
                if (enemy && attack.target) {
                    enemy.chainAttackIndex = index;
                    enemy.chainAttackReady = true;
                    
                    // Fire based on attack type
                    if (attack.attackType === 'seeking' && enemy.fireSeekingMissile) {
                        enemy.fireSeekingMissile(attack.target);
                    } else if (attack.attackType === 'burst' && enemy.fireBurst) {
                        enemy.fireBurst(attack.target, 3);
                    } else if (enemy.fire) {
                        enemy.fire(aiShips);
                    }
                    
                    attack.executed = true;
                }
            }
        });
        
        // Check if chain attack is complete
        if (this.chainAttackSequence.every(a => a.executed)) {
            this.chainAttackActive = false;
            this.chainAttackTimer = 0;
        }
    }
    
    // MULTI-PATTERN SWITCHING: Evaluate and switch patterns
    evaluatePatternEffectiveness(enemies, target) {
        if (!target || enemies.length === 0) return;
        
        // Calculate effectiveness based on:
        // - Distance to target
        // - Number of successful hits (approximated by proximity)
        // - Formation integrity
        
        let effectiveness = 0;
        let avgDistance = 0;
        let formationIntegrity = 1.0;
        
        enemies.forEach(enemy => {
            if (enemy && target) {
                const dx = enemy.x - target.x;
                const dy = enemy.y - target.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                avgDistance += distance;
                
                // Check formation integrity
                if (enemy.formationPosition) {
                    const formDx = enemy.x - enemy.formationPosition.x;
                    const formDy = enemy.y - enemy.formationPosition.y;
                    const formDist = Math.sqrt(formDx * formDx + formDy * formDy);
                    if (formDist > 50) formationIntegrity -= 0.1;
                }
            }
        });
        
        avgDistance /= enemies.length;
        effectiveness = (1.0 / (avgDistance + 1)) * 0.6 + formationIntegrity * 0.4;
        
        return effectiveness;
    }
    
    // Select next pattern based on effectiveness
    selectNextPattern(enemies, target) {
        if (!target || enemies.length < 2) return null;
        
        const availablePatterns = [];
        if (enemies.length >= 4) availablePatterns.push('pincer');
        if (enemies.length >= 3) availablePatterns.push('wave');
        if (enemies.length >= 2) availablePatterns.push('flanking');
        
        // Get effectiveness scores
        const patternScores = new Map();
        availablePatterns.forEach(pattern => {
            const score = this.patternEffectiveness.get(pattern) || 0.5;
            patternScores.set(pattern, score);
        });
        
        // Select pattern with highest score, or random if tied
        let bestPattern = null;
        let bestScore = -1;
        patternScores.forEach((score, pattern) => {
            if (score > bestScore) {
                bestScore = score;
                bestPattern = pattern;
            }
        });
        
        return bestPattern || availablePatterns[Math.floor(gameRandom() * availablePatterns.length)];
    }
    
    // Update coordination system
    update(enemyShips, aiShips) {
        this.frameCount++;
        if (this.frameCount < this.updateInterval) {
            // Update chain attacks every frame
            this.updateChainAttack();
            return;
        }
        this.frameCount = 0;
        
        // Find alpha ship target
        let alphaTarget = null;
        if (aiShips && Array.isArray(aiShips)) {
            for (let aiShip of aiShips) {
                if (aiShip && aiShip.isAlpha && alphaAttackEnabled) {
                    alphaTarget = { ship: aiShip, x: aiShip.x, y: aiShip.y };
                    break;
                }
            }
        }
        
        // If no alpha, find best blue ship target
        let target = alphaTarget;
        if (!target && aiShips && Array.isArray(aiShips) && aiShips.length > 0) {
            // Find closest blue ship to average enemy position
            let closestShip = null;
            let closestDist = Infinity;
            const avgX = enemyShips.filter(e => e).reduce((sum, e) => sum + e.x, 0) / Math.max(1, enemyShips.filter(e => e).length);
            const avgY = enemyShips.filter(e => e).reduce((sum, e) => sum + e.y, 0) / Math.max(1, enemyShips.filter(e => e).length);
            
            aiShips.forEach(aiShip => {
                if (!aiShip) return;
                const dx = aiShip.x - avgX;
                const dy = aiShip.y - avgY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < closestDist) {
                    closestDist = dist;
                    closestShip = aiShip;
                }
            });
            if (closestShip) {
                target = { ship: closestShip, x: closestShip.x, y: closestShip.y };
            }
        }
        
        if (!target || enemyShips.length < 2) {
            // Reset coordination if no target or not enough enemies
            enemyShips.forEach(e => {
                if (e) {
                    e.isCoordinating = false;
                    e.coordinationRole = null;
                    e.attackPattern = null;
                }
            });
            return;
        }
        
        // Select leader
        this.leader = this.selectLeader(enemyShips, target);
        
        // Find enemies that can coordinate (not boss, within range)
        const coordinatingEnemies = enemyShips.filter(e => {
            if (!e || e.type === 'boss') return false;
            const dx = e.x - target.x;
            const dy = e.y - target.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            return distance < 500; // Coordination range
        });
        
        if (coordinatingEnemies.length < 2) {
            // Not enough enemies to coordinate
            coordinatingEnemies.forEach(e => {
                if (e) {
                    e.isCoordinating = false;
                    e.coordinationRole = null;
                    e.attackPattern = null;
                }
            });
            return;
        }
        
        // MULTI-PATTERN SWITCHING: Evaluate current pattern and switch if needed
        this.patternSwitchTimer++;
        let pattern = this.currentPattern;
        
        if (this.patternSwitchTimer >= this.patternSwitchInterval || !pattern) {
            // Time to evaluate and potentially switch patterns
            if (this.currentPattern) {
                // Evaluate current pattern effectiveness
                const effectiveness = this.evaluatePatternEffectiveness(coordinatingEnemies, target);
                this.patternEffectiveness.set(this.currentPattern, effectiveness);
                this.patternHistory.push({pattern: this.currentPattern, effectiveness: effectiveness});
                if (this.patternHistory.length > 5) this.patternHistory.shift();
            }
            
            // Select next pattern
            pattern = this.selectNextPattern(coordinatingEnemies, target);
            this.patternSwitchTimer = 0;
        }
        
        // ADAPTIVE FORMATIONS: Adapt formation based on battlefield state
        this.formationAdaptationTimer++;
        if (this.formationAdaptationTimer >= 180) { // Every 3 seconds
            // Analyze battlefield state
            this.battlefieldState.enemyCount = coordinatingEnemies.length;
            this.battlefieldState.targetCount = aiShips ? aiShips.filter(s => s).length : 0;
            
            let totalDistance = 0;
            coordinatingEnemies.forEach(e => {
                if (e && target) {
                    const dx = e.x - target.x;
                    const dy = e.y - target.y;
                    totalDistance += Math.sqrt(dx * dx + dy * dy);
                }
            });
            this.battlefieldState.averageDistance = totalDistance / coordinatingEnemies.length;
            
            // Select formation type based on state
            if (coordinatingEnemies.length >= 4 && this.battlefieldState.averageDistance < 200) {
                this.formationType = 'tight';
            } else if (coordinatingEnemies.length >= 4) {
                this.formationType = 'diamond';
            } else if (coordinatingEnemies.length >= 3) {
                this.formationType = 'spread';
            } else {
                this.formationType = 'default';
            }
            
            this.formationAdaptationTimer = 0;
        }
        
        // Execute selected pattern
        if (pattern === 'pincer') {
            this.executePincerAttack(coordinatingEnemies, target);
        } else if (pattern === 'wave') {
            this.executeWaveAttack(coordinatingEnemies, target);
        } else if (pattern === 'flanking') {
            this.executeFlankingManeuver(coordinatingEnemies, target);
        }
        
        // Apply adaptive formation positions
        const formationPositions = this.calculateAdaptiveFormation(coordinatingEnemies, target, this.formationType);
        coordinatingEnemies.forEach((enemy, index) => {
            if (enemy && formationPositions[index]) {
                // Blend formation position with pattern position
                if (enemy.formationPosition) {
                    enemy.formationPosition.x = enemy.formationPosition.x * 0.7 + formationPositions[index].x * 0.3;
                    enemy.formationPosition.y = enemy.formationPosition.y * 0.7 + formationPositions[index].y * 0.3;
                } else {
                    enemy.formationPosition = formationPositions[index];
                }
                enemy.adaptiveFormationOffset = {
                    x: formationPositions[index].x - target.x,
                    y: formationPositions[index].y - target.y
                };
            }
        });
        
        // SPECIALIZED ROLES: Assign roles
        this.assignSpecializedRoles(coordinatingEnemies, target);
        
        // CHAIN ATTACKS: Occasionally trigger chain attacks
        if (!this.chainAttackActive && gameRandom() < 0.1 && coordinatingEnemies.length >= 3) {
            this.executeChainAttack(coordinatingEnemies, target);
        }
        
        this.currentPattern = pattern;
        coordinatingEnemies.forEach(e => {
            if (e) {
                e.isCoordinating = true;
                if (e === this.leader) {
                    e.coordinationRole = 'leader';
                }
            }
        });
    }
}
