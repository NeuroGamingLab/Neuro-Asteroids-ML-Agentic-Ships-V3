// Extracted from game.js as part of a module split.
// Relies on globals defined in game.js/world.js/rng.js/fx.js loaded earlier
// (canvas, ctx, worldW/worldH/sphereWorld, gameRandom, FXL, ACTION_SPACE_SIZE,
// decodeAction, keys, etc.) — same plain-script global scope as before, just
// split across files.

// Isolated PRNG stream for non-simulation gameplay drops to preserve headless RL determinism
const tacticalRng = (typeof AsteroidsRNG !== 'undefined' && AsteroidsRNG.createStream)
    ? AsteroidsRNG.createStream('tactical-drops')
    : Math.random;

// Ship class
class Ship {
    constructor() {
        this.x = worldW() / 2;
        this.y = worldH() / 2;
        this.angle = 0; // Rotation angle in radians
        this.velocityX = 0;
        this.velocityY = 0;
        this.rotationSpeed = 0.1;
        this.thrustPower = 0.15;
        this.friction = 0.98; // Momentum decay
        this.maxVelocity = 8; // Maximum velocity limit
        this.size = 20;
        this.radius = 15; // For collision detection
        this.shieldActive = false;
        this.shieldRadius = 40; // Shield force field radius
        this.shieldForce = 0.5; // Force strength to repel asteroids
        this.buffRailgun = 0; // Frames of piercing railgun remaining
        this.buffShield = 0;  // Frames of invulnerable overshield remaining
    }

    rotate(direction) {
        // direction: 1 for right, -1 for left
        this.angle += direction * this.rotationSpeed;
        if (this._actionRecord) this._actionRecord.rotation = direction > 0 ? 1 : -1;
    }

    thrust(power = 1) {
        // Apply thrust in the direction the ship is facing
        this.velocityX += Math.cos(this.angle) * this.thrustPower * power;
        this.velocityY += Math.sin(this.angle) * this.thrustPower * power;
        if (this._actionRecord) this._actionRecord.thrust = 1;
        if (FXL) FXL.emitThrust(this.x, this.y, this.angle, this.thrustColor || GAME_THEME.primary, power);
    }

    moveBackward() {
        // Move backward (opposite direction)
        this.velocityX -= Math.cos(this.angle) * this.thrustPower * 0.5;
        this.velocityY -= Math.sin(this.angle) * this.thrustPower * 0.5;
    }

    moveLeft() {
        // Strafe left
        this.velocityX -= Math.cos(this.angle - Math.PI / 2) * this.thrustPower * 0.7;
        this.velocityY -= Math.sin(this.angle - Math.PI / 2) * this.thrustPower * 0.7;
    }

    moveRight() {
        // Strafe right
        this.velocityX -= Math.cos(this.angle + Math.PI / 2) * this.thrustPower * 0.7;
        this.velocityY -= Math.sin(this.angle + Math.PI / 2) * this.thrustPower * 0.7;
    }

    update() {
        // ALPHA SHIP: Handle escape mode BEFORE applying friction
        if (this.isAlpha && alphaAttackEnabled && this.escapeMode) {
            // ESCAPE MODE: Allow movement during escape
            // Skip friction during escape to maintain escape velocity
            if (sphereWorld()) {
                WORLD_GEO.moveBody(this, 0);
                return;
            }
            this.x += this.velocityX;
            this.y += this.velocityY;
            
            // Enable screen wrapping during escape (like normal ships)
            // This prevents the alpha ship from getting trapped in corners
            if (this.x < 0) this.x = worldW();
            if (this.x > worldW()) this.x = 0;
            if (this.y < 0) this.y = worldH();
            if (this.y > worldH()) this.y = 0;
            return; // Skip normal update during escape
        }
        
        // Apply friction (momentum decay) - only for non-escape mode
        this.velocityX *= this.friction;
        this.velocityY *= this.friction;

        // Limit maximum velocity
        const currentSpeed = Math.sqrt(this.velocityX * this.velocityX + this.velocityY * this.velocityY);
        if (currentSpeed > this.maxVelocity) {
            const ratio = this.maxVelocity / currentSpeed;
            this.velocityX *= ratio;
            this.velocityY *= ratio;
        }

        // ALPHA SHIP: Keep alpha ship at center when not in escape mode
        if (this.isAlpha && alphaAttackEnabled) {
            // NORMAL MODE: Keep alpha ship at center of screen
            const centerX = worldW() / 2;
            const centerY = worldH() / 2;
            const distanceToCenter = Math.sqrt(
                (this.x - centerX) * (this.x - centerX) + 
                (this.y - centerY) * (this.y - centerY)
            );
            
            // Smoothly return to center if not already there
            if (distanceToCenter > this.returnToCenterDistance) {
                // Move toward center
                const angleToCenter = Math.atan2(centerY - this.y, centerX - this.x);
                const returnSpeed = 2;
                this.velocityX = Math.cos(angleToCenter) * returnSpeed;
                this.velocityY = Math.sin(angleToCenter) * returnSpeed;
                this.x += this.velocityX;
                this.y += this.velocityY;
            } else {
                // At center, stay there
                this.x = centerX;
                this.y = centerY;
                this.velocityX = 0;
                this.velocityY = 0;
            }
        } else if (sphereWorld()) {
            // Great-circle step; heading is parallel-transported.
            WORLD_GEO.moveBody(this, 0);
        } else {
            // Normal update for non-alpha ships
            // Update position
            this.x += this.velocityX;
            this.y += this.velocityY;

            // Wrap around screen edges
            if (this.x < 0) this.x = worldW();
            if (this.x > worldW()) this.x = 0;
            if (this.y < 0) this.y = worldH();
            if (this.y > worldH()) this.y = 0;
        }

        // Tactical power-up buffs countdown
        if (this.buffRailgun > 0) this.buffRailgun--;
        if (this.buffShield > 0) {
            this.buffShield--;
            this.shieldActive = true;
            this.shieldRadius = 45;
        }
    }

    hyperspace() {
        // Teleport to a random location
        this.x = gameRandom() * worldW();
        this.y = gameRandom() * worldH();
        // Reset velocity on rematerialisation
        this.velocityX = 0;
        this.velocityY = 0;

        // 10% chance of a catastrophic mis-jump
        if (gameRandom() < 0.1) {
            if (FXL) {
                FXL.emitExplosion(this.x, this.y, GAME_THEME.cyan, 0.7);
                FXL.shake(4, 12);
            }
            // Player ship: instant game-over (classic Asteroids behaviour)
            if (this === ship) {
                gameRunning = false;
                alert('Hyperspace mis-jump! Game Over! Final Score: ' + score);
                initGame();
                gameRunning = true;
            }
        }
    }

    draw() {
        ctx.save();
        ctx.translate(this.x, this.y);
        
        // Draw shield force field when active
        if (this.shieldActive) {
            ctx.strokeStyle = GAME_THEME.cyan;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.6;
            ctx.beginPath();
            ctx.arc(0, 0, this.shieldRadius, 0, Math.PI * 2);
            ctx.stroke();
            
            // Pulsing effect
            ctx.globalAlpha = 0.3;
            ctx.beginPath();
            ctx.arc(0, 0, this.shieldRadius + Math.sin(Date.now() / 100) * 3, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
        
        ctx.rotate(this.angle);

        // Draw triangular ship
        ctx.strokeStyle = GAME_THEME.primary;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(this.size, 0); // Nose of the ship
        ctx.lineTo(-this.size / 2, -this.size / 2); // Top back
        ctx.lineTo(-this.size / 2, this.size / 2); // Bottom back
        ctx.closePath();
        ctx.stroke();

        // Draw three gun barrels (glowing green if railgun buff active)
        ctx.strokeStyle = this.buffRailgun > 0 ? '#00FF66' : GAME_THEME.primary;
        ctx.lineWidth = this.buffRailgun > 0 ? 2.5 : 1.5;
        if (this.buffRailgun > 0 && FXL) FXL.glow(ctx, this.size + 4, 0, 10, '#00FF66', 0.6);
        // Center gun barrel
        ctx.beginPath();
        ctx.moveTo(this.size, 0);
        ctx.lineTo(this.size + (this.buffRailgun > 0 ? 12 : 8), 0);
        ctx.stroke();
        // Left gun barrel
        ctx.beginPath();
        ctx.moveTo(this.size - 3, -this.size / 4);
        ctx.lineTo(this.size + (this.buffRailgun > 0 ? 9 : 6), -this.size / 3);
        ctx.stroke();
        // Right gun barrel
        ctx.beginPath();
        ctx.moveTo(this.size - 3, this.size / 4);
        ctx.lineTo(this.size + (this.buffRailgun > 0 ? 9 : 6), this.size / 3);
        ctx.stroke();

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
    }

    getPosition() {
        return { x: this.x, y: this.y, radius: this.radius };
    }
}

// Bullet class
class Bullet {
    // `owner` is the ship that fired this bullet. Without it a kill cannot be
    // attributed to an individual agent, which forces the reward function to
    // split all credit uniformly and lets agents free-ride on their team.
    constructor(x, y, angle, isAIBullet = false, owner = null, piercing = false) {
        this.x = x;
        this.y = y;
        this.angle = angle;
        this.speed = 8;
        this.radius = 3;
        this.lifetime = 60; // Frames before bullet disappears
        this.isAIBullet = isAIBullet; // Distinguish AI bullets from player bullets
        this.owner = owner;
        this.ownerAgentId = owner && owner.agentId !== undefined ? owner.agentId : null;
        this.piercing = piercing || (owner && owner.buffRailgun > 0);
        this.pierceHits = 0;
    }

    update() {
        this.lifetime--;
        if (sphereWorld()) {
            WORLD_GEO.moveAlongHeading(this, this.speed);
            return;
        }
        this.x += Math.cos(this.angle) * this.speed;
        this.y += Math.sin(this.angle) * this.speed;

        // Wrap around screen
        if (this.x < 0) this.x = worldW();
        if (this.x > worldW()) this.x = 0;
        if (this.y < 0) this.y = worldH();
        if (this.y > worldH()) this.y = 0;
    }

    draw() {
        if (this.piercing) {
            ctx.fillStyle = '#00FF66';
            if (FXL) FXL.glow(ctx, this.x, this.y, this.radius * 3.5, '#00FF66', 0.8);
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 1.3, 0, Math.PI * 2);
            ctx.fill();
            // Ion beam trail
            ctx.strokeStyle = 'rgba(0, 255, 102, 0.6)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(this.x, this.y);
            ctx.lineTo(this.x - Math.cos(this.angle) * 12, this.y - Math.sin(this.angle) * 12);
            ctx.stroke();
            return;
        }
        // AI bullets are blue, player bullets are yellow
        ctx.fillStyle = this.isAIBullet ? GAME_THEME.aiBlue : GAME_THEME.playerBullet;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
    }

    isAlive() {
        return this.lifetime > 0;
    }

    getPosition() {
        return { x: this.x, y: this.y, radius: this.radius };
    }
}

// Asteroid class
class Asteroid {
    constructor(x = null, y = null) {
        // Random position if not provided
        if ((x === null || y === null) && sphereWorld()) {
            // A sphere has no edge: spawn uniformly, away from the centre where
            // the alpha is pinned. Draws from the seeded game RNG, so headless
            // sphere episodes stay deterministic.
            const spot = WORLD_GEO.randomPoint(gameRandom, worldW() / 2, worldH() / 2, 150 * worldScale());
            this.x = spot.x;
            this.y = spot.y;
        } else if (x === null || y === null) {
            // Spawn at edge of screen
            const side = Math.floor(gameRandom() * 4);
            switch (side) {
                case 0: // Top
                    this.x = gameRandom() * worldW();
                    this.y = 0;
                    break;
                case 1: // Right
                    this.x = worldW();
                    this.y = gameRandom() * worldH();
                    break;
                case 2: // Bottom
                    this.x = gameRandom() * worldW();
                    this.y = worldH();
                    break;
                case 3: // Left
                    this.x = 0;
                    this.y = gameRandom() * worldH();
                    break;
            }
        } else {
            this.x = x;
            this.y = y;
        }

        // Random velocity in all directions
        const speed = 1 + gameRandom() * 2;
        const angle = gameRandom() * Math.PI * 2;
        this.velocityX = Math.cos(angle) * speed;
        this.velocityY = Math.sin(angle) * speed;

        this.size = 30 + gameRandom() * 30;
        this.radius = this.size / 2;
        this.rotation = 0;
        this.rotationSpeed = (gameRandom() - 0.5) * 0.1;
        this.vertices = this.generateVertices();
        this.isMineral = (typeof globalThis !== 'undefined' && globalThis.__HEADLESS__) ? false : (tacticalRng() < 0.22);
    }

    generateVertices() {
        // Generate irregular polygon vertices
        const vertices = [];
        const numVertices = 8 + Math.floor(gameRandom() * 4);
        for (let i = 0; i < numVertices; i++) {
            const angle = (Math.PI * 2 * i) / numVertices;
            const distance = this.size / 2 + (gameRandom() - 0.5) * 10;
            vertices.push({
                x: Math.cos(angle) * distance,
                y: Math.sin(angle) * distance
            });
        }
        return vertices;
    }

    update() {
        this.rotation += this.rotationSpeed;
        if (sphereWorld()) {
            WORLD_GEO.moveBody(this, 0);
            return;
        }
        this.x += this.velocityX;
        this.y += this.velocityY;

        // Wrap around screen edges
        if (this.x < -this.size) this.x = worldW() + this.size;
        if (this.x > worldW() + this.size) this.x = -this.size;
        if (this.y < -this.size) this.y = worldH() + this.size;
        if (this.y > worldH() + this.size) this.y = -this.size;
    }

    draw() {
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.rotation);

        if (this.isMineral) {
            ctx.strokeStyle = '#FFD700'; // Gold crystalline rim
            ctx.lineWidth = 2.5;
            if (FXL) FXL.glow(ctx, 0, 0, this.radius * 1.4, '#FFD700', 0.5);
        } else {
            ctx.strokeStyle = GAME_THEME.primary;
            ctx.lineWidth = 2;
        }

        ctx.beginPath();
        if (this.vertices && this.vertices.length > 0) {
            ctx.moveTo(this.vertices[0].x, this.vertices[0].y);
            for (let i = 1; i < this.vertices.length; i++) {
                ctx.lineTo(this.vertices[i].x, this.vertices[i].y);
            }
            ctx.closePath();
        } else {
            ctx.arc(0, 0, this.radius, 0, Math.PI * 2);
        }
        ctx.stroke();

        // Shimmering facets on mineral asteroid
        if (this.isMineral && this.vertices && this.vertices.length >= 4) {
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.45)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(this.vertices[0].x, this.vertices[0].y);
            ctx.lineTo(this.vertices[Math.floor(this.vertices.length / 2)].x, this.vertices[Math.floor(this.vertices.length / 2)].y);
            ctx.moveTo(this.vertices[1].x, this.vertices[1].y);
            ctx.lineTo(this.vertices[Math.floor(this.vertices.length * 3 / 4)].x, this.vertices[Math.floor(this.vertices.length * 3 / 4)].y);
            ctx.stroke();
        }

        ctx.restore();
    }

    getPosition() {
        return { x: this.x, y: this.y, radius: this.radius };
    }

    // Break asteroid into fragments with physics-based directions and speeds
    breakApart(collisionNormalX, collisionNormalY, relativeVelocityX, relativeVelocityY) {
        const fragments = [];
        const minSize = 15; // Minimum size before asteroid disappears completely
        
        // Don't break if too small
        if (this.size <= minSize) {
            return fragments; // Return empty array - asteroid will be removed
        }

        // Calculate number of fragments based on size
        const numFragments = 2 + Math.floor(this.size / 20); // 2-4 fragments
        const fragmentSize = this.size * 0.5; // Each fragment is ~50% of original size
        
        // Don't create fragments smaller than minimum
        if (fragmentSize < minSize) {
            return fragments;
        }

        // Calculate collision angle
        const collisionAngle = Math.atan2(collisionNormalY, collisionNormalX);
        
        // Calculate relative speed magnitude
        const relativeSpeed = Math.sqrt(relativeVelocityX * relativeVelocityX + relativeVelocityY * relativeVelocityY);
        
        // Base fragment speed (proportional to relative speed and size ratio)
        const baseSpeed = relativeSpeed * 0.6 + (this.size / 30) * 0.5; // Scale with collision energy
        
        // Mass ratio (for momentum conservation)
        const originalMass = this.size * this.size; // Mass proportional to area
        const fragmentMass = fragmentSize * fragmentSize;
        const massRatio = fragmentMass / originalMass;

        // Create fragments
        for (let i = 0; i < numFragments; i++) {
            // Calculate scatter angle around collision normal
            // Fragments spread in a cone around the collision direction
            const scatterAngle = (gameRandom() - 0.5) * Math.PI * 0.8; // ±72 degrees spread
            const fragmentAngle = collisionAngle + scatterAngle;
            
            // Add some tangential component for more realistic spread
            const tangentialAngle = fragmentAngle + Math.PI / 2;
            const tangentialComponent = (gameRandom() - 0.5) * 0.3;
            
            // Calculate fragment velocity
            // Base velocity from collision + random variation
            const speedVariation = 0.7 + gameRandom() * 0.6; // 70-130% of base speed
            const fragmentSpeed = baseSpeed * speedVariation * massRatio;
            
            // Combine normal and tangential components
            const normalVelX = Math.cos(fragmentAngle) * fragmentSpeed;
            const normalVelY = Math.sin(fragmentAngle) * fragmentSpeed;
            const tangentialVelX = Math.cos(tangentialAngle) * tangentialComponent * fragmentSpeed;
            const tangentialVelY = Math.sin(tangentialAngle) * tangentialComponent * fragmentSpeed;
            
            // Fragment inherits some of original velocity
            const inheritedVelX = this.velocityX * 0.3;
            const inheritedVelY = this.velocityY * 0.3;
            
            // Final fragment velocity
            const fragmentVelX = normalVelX + tangentialVelX + inheritedVelX;
            const fragmentVelY = normalVelY + tangentialVelY + inheritedVelY;
            
            // Fragment position (slightly offset from collision point)
            const offsetDistance = this.radius * 0.3;
            const offsetAngle = fragmentAngle + Math.PI; // Opposite to movement direction
            const fragmentX = this.x + Math.cos(offsetAngle) * offsetDistance;
            const fragmentY = this.y + Math.sin(offsetAngle) * offsetDistance;
            
            // Create new asteroid fragment
            const fragment = new Asteroid(fragmentX, fragmentY);
            fragment.size = fragmentSize * (0.8 + gameRandom() * 0.4); // 80-120% variation
            fragment.radius = fragment.size / 2;
            fragment.velocityX = fragmentVelX;
            fragment.velocityY = fragmentVelY;
            fragment.isMineral = this.isMineral; // Fragments inherit mineral type
            fragment.vertices = fragment.generateVertices();
            
            // Add some rotation based on collision
            fragment.rotationSpeed = (gameRandom() - 0.5) * 0.2;
            
            fragments.push(fragment);
        }
        
        return fragments;
    }
}

// Tactical PowerUp class
class PowerUp {
    constructor(x, y, type = null) {
        this.x = x;
        this.y = y;
        const types = ['railgun', 'overshield', 'emp', 'fuel'];
        this.type = type || types[Math.floor(tacticalRng() * types.length)];
        this.radius = 12;
        this.lifetime = 900; // 15 seconds at 60fps
        this.blinkStart = 180; // Start blinking last 3 seconds
        const angle = tacticalRng() * Math.PI * 2;
        const spd = 0.4 + tacticalRng() * 0.6;
        this.velocityX = Math.cos(angle) * spd;
        this.velocityY = Math.sin(angle) * spd;
        this.pulse = tacticalRng() * Math.PI * 2;
        
        switch (this.type) {
            case 'railgun':
                this.color = '#00FF66'; // Neon green
                this.label = '⚡ RAILGUN';
                break;
            case 'overshield':
                this.color = '#00E5FF'; // Cyan
                this.label = '🛡 SHIELD';
                break;
            case 'emp':
                this.color = '#FFD600'; // Amber gold
                this.label = '💥 EMP';
                break;
            case 'fuel':
                this.color = '#FF0077'; // Magenta
                this.label = '⛽ FUEL';
                break;
        }
    }

    update() {
        this.lifetime--;
        this.pulse += 0.08;
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
        if (this.lifetime <= 0) return;
        if (this.lifetime < this.blinkStart && Math.floor(this.lifetime / 8) % 2 === 0) return;
        ctx.save();
        const glowR = this.radius + Math.sin(this.pulse) * 3;
        if (FXL) FXL.glow(ctx, this.x, this.y, glowR * 1.8, this.color, 0.6);
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        // Rotating diamond glyph
        const a = this.pulse * 0.4;
        for (let i = 0; i < 4; i++) {
            const th = a + (i * Math.PI / 2);
            const px = this.x + Math.cos(th) * this.radius;
            const py = this.y + Math.sin(th) * this.radius;
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.stroke();
        ctx.fillStyle = this.color;
        ctx.font = 'bold 9px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this.type[0].toUpperCase(), this.x, this.y);
        ctx.restore();
    }
}
