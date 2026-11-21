// Unified canonical feature extraction for ML, MARL, and NN training
// Replaces fragmented 13-dim (Phase 5) vs 21-dim (MARL) representations

const CANONICAL_FEATURE_DIM = 30;
const CANONICAL_FEATURE_NAMES = [
    'health', 'phase', 'shieldActive', 'velocity', 'shieldCooldownNorm',
    'isAlpha', 'roleNorm', 'formationError',
    'asteroidCount', 'enemyCount', 'allyCount', 'bossCount',
    'nearestAsteroidDistance', 'nearestAsteroidAngle', 'nearestAsteroidSize',
    'nearestEnemyDistance', 'nearestEnemyAngle', 'nearestEnemyHealth',
    'nearestBossDistance', 'nearestBossHealth',
    'nearestAllyDistance', 'nearestAllyAngle', 'nearestAllyHealth',
    'threatLevel', 'flockDensity', 'bulletThreat',
    'aggregateAsteroidThreat', 'aggregateEnemyThreat',
    'curriculumStageNorm', 'formationCohesion'
];

// Agent-relative features used only by the RL policy. Appended after the
// canonical 30 so that vector[0..29] keeps its existing meaning.
const RL_EXTRA_FEATURE_NAMES = [
    'headingSin', 'headingCos',
    'aimSignedAsteroid', 'aimAlignedAsteroid',
    'aimSignedEnemy', 'aimAlignedEnemy',
    'shootCooldownNorm', 'canFire'
];
const RL_FEATURE_NAMES = CANONICAL_FEATURE_NAMES.concat(RL_EXTRA_FEATURE_NAMES);
const RL_FEATURE_DIM = RL_FEATURE_NAMES.length;

// Fuel-rescue features, appended only when extractRL() is called with
// { fuelFeatures: true }. Kept out of the base layout so checkpoints trained
// before the fuel mechanic existed still load and still see the inputs they
// were trained on.
const RL_FUEL_FEATURE_NAMES = [
    'alphaFuelNorm',    // fuel fraction of the swarm's alpha (own fuel if this ship is the alpha)
    'helpPingActive',   // 1 while the alpha is broadcasting HELP
    'tankerDistance',   // distance to the green tanker, 1 when there is none
    'tankerBearing',    // bearing to the tanker relative to own heading, 0.5 = dead ahead
    'fuelLinked'        // 1 while this ship is docked or transferring (cannot manoeuvre)
];
const RL_FEATURE_NAMES_FUEL = RL_FEATURE_NAMES.concat(RL_FUEL_FEATURE_NAMES);
const RL_FEATURE_DIM_FUEL = RL_FEATURE_NAMES_FUEL.length;

// Multi-entity token slots (Phase 4): 10 tokens x 6 features = 60 dims appended
// after RL_FEATURE_NAMES_FUEL (total 103 dims) when { entityTokens: true }.
// Token slots: 3 asteroids, 3 enemies/bosses, 2 incoming bullets, 2 allies.
// Every token begins with a presence indicator (1 = present, 0 = empty pad)
// so EntityAttentionNet can apply exact masked permutation-invariant attention.
const ENTITY_TOKEN_SLOTS = 10;
const ENTITY_TOKEN_DIM = 6;
const RL_ENTITY_TOKEN_NAMES = (function () {
    const labels = [
        'entAst0', 'entAst1', 'entAst2',
        'entEne0', 'entEne1', 'entEne2',
        'entBul0', 'entBul1',
        'entAly0', 'entAly1'
    ];
    const fields = ['Present', 'Dist', 'Bearing', 'RelVx', 'RelVy', 'Attr'];
    const out = [];
    for (const lbl of labels) for (const f of fields) out.push(lbl + f);
    return out;
})();
const RL_FEATURE_NAMES_ENTITY = RL_FEATURE_NAMES_FUEL.concat(RL_ENTITY_TOKEN_NAMES);
const RL_FEATURE_DIM_ENTITY = RL_FEATURE_NAMES_ENTITY.length;

// Global team-state vector for MAPPO Centralized Critic training (12 dims).
// Used only during training (and optional critic inspection); actors remain
// strictly decentralized over local observations.
const GLOBAL_STATE_NAMES = [
    'teamAliveFrac', 'teamMeanHealth', 'teamMinHealth', 'alphaAlive',
    'alphaFuelGlobal', 'alphaHelpGlobal', 'tankerAlive',
    'globalAsteroidNorm', 'globalEnemyNorm', 'globalBossNorm',
    'globalBulletNorm', 'globalWaveNorm'
];
const GLOBAL_STATE_DIM = GLOBAL_STATE_NAMES.length;

class MLFeatureExtractor {
    static clamp01(value) {
        return Math.max(0, Math.min(1, value));
    }

    static normalize(value, min, max) {
        if (max <= min) return 0;
        return MLFeatureExtractor.clamp01((value - min) / (max - min));
    }

    // Tangent-plane offset from (ax, ay) to (bx, by). On the flat torus this is
    // plain (bx - ax, by - ay); on the 3D sphere it delegates to WORLD.delta() so
    // distances and bearings stay exact across the antimeridian and near the poles.
    static relDelta(ax, ay, bx, by) {
        if (typeof WORLD !== 'undefined' && WORLD && WORLD.isSphere()) {
            return WORLD.delta(ax, ay, bx, by);
        }
        return { dx: bx - ax, dy: by - ay };
    }

    static findNearestEntity(agent, entities, maxDistance = Infinity) {
        let nearest = null;
        let nearestDistance = Infinity;

        entities.forEach(entity => {
            if (!entity) return;
            const { dx, dy } = MLFeatureExtractor.relDelta(agent.x, agent.y, entity.x, entity.y);
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < nearestDistance && distance <= maxDistance) {
                nearestDistance = distance;
                nearest = {
                    entity,
                    distance,
                    angle: Math.atan2(dy, dx),
                    health: entity.health || entity.maxHealth || 1,
                    size: entity.radius || entity.size || 20
                };
            }
        });

        return nearest;
    }

    static attentionThreat(agent, entities, kind = 'asteroid', topK = 3) {
        if (!entities || entities.length === 0) return 0;

        const scores = entities.map(entity => {
            if (!entity) return 0;
            const { dx, dy } = MLFeatureExtractor.relDelta(agent.x, agent.y, entity.x, entity.y);
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance <= 0) return 1;

            let weight = 1 / (distance + 1);
            if (kind === 'enemy' && entity.aiType === 'boss') weight *= 2;
            if (kind === 'asteroid' && entity.radius) weight *= (entity.radius / 20);
            return weight;
        }).sort((a, b) => b - a).slice(0, topK);

        const sum = scores.reduce((acc, s) => acc + s, 0);
        return MLFeatureExtractor.clamp01(sum / topK);
    }

    static roleToNorm(role) {
        const roles = { scout: 0.2, tank: 0.4, support: 0.6, dps: 0.8, interceptor: 1.0 };
        return roles[role] || 0.5;
    }

    static extract(ship, asteroids, aiShips, playerShip, enemyShips, enemyBullets = [], options = {}) {
        // Normalise positions by the logical world, which on a scaled sphere is
        // larger than the canvas (see world.js). Falls back to the canvas.
        const worldSize = (typeof WORLD !== 'undefined' && WORLD && WORLD.isSphere()) ? WORLD.size() : null;
        const canvasW = worldSize ? worldSize.w : ((typeof canvas !== 'undefined' && canvas) ? canvas.width : 1200);
        const canvasH = worldSize ? worldSize.h : ((typeof canvas !== 'undefined' && canvas) ? canvas.height : 600);

        const bosses = (enemyShips || []).filter(e => e && (e.type === 'boss' || e.aiType === 'boss'));
        const regularEnemies = (enemyShips || []).filter(e => e && e.type !== 'boss' && e.aiType !== 'boss');

        const nearestAsteroid = MLFeatureExtractor.findNearestEntity(ship, asteroids || [], 400);
        const nearestEnemy = MLFeatureExtractor.findNearestEntity(ship, regularEnemies, 500);
        const nearestBoss = MLFeatureExtractor.findNearestEntity(ship, bosses, 600);

        let nearestAlly = null;
        let nearbyAllies = 0;
        const flockRadius = ship.flockRadius || 150;
        (aiShips || []).forEach(other => {
            if (!other || other === ship) return;
            const { dx, dy } = MLFeatureExtractor.relDelta(ship.x, ship.y, other.x, other.y);
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < flockRadius) nearbyAllies++;
            if (!nearestAlly || distance < nearestAlly.distance) {
                nearestAlly = { distance, angle: Math.atan2(dy, dx), health: other.health || 3 };
            }
        });

        const threatLevel = typeof ship.calculateThreatLevel === 'function'
            ? ship.calculateThreatLevel(asteroids || [], enemyShips || [])
            : 0;

        const bulletThreat = typeof ship.predictBulletCollision === 'function'
            ? (ship.predictBulletCollision(enemyBullets || []) ? 1 : 0)
            : 0;

        const maxAllies = Math.max(1, (aiShips || []).length - 1);
        const flockDensity = MLFeatureExtractor.clamp01(nearbyAllies / maxAllies);

        let formationError = 0;
        if (typeof ship.calculateFormationPositionAdvanced === 'function') {
            const target = ship.calculateFormationPositionAdvanced(aiShips || []);
            if (target) {
                const { dx, dy } = MLFeatureExtractor.relDelta(ship.x, ship.y, target.x, target.y);
                formationError = MLFeatureExtractor.clamp01(Math.sqrt(dx * dx + dy * dy) / 200);
            }
        }

        const curriculumStage = options.curriculumStage || (typeof window !== 'undefined' ? window.marlCurriculumStage : 0) || 0;
        const velocity = Math.sqrt((ship.velocityX || 0) ** 2 + (ship.velocityY || 0) ** 2);

        const features = {
            health: MLFeatureExtractor.clamp01((ship.health || 0) / (ship.maxHealth || 3)),
            phase: MLFeatureExtractor.clamp01((ship.aiPhase || 1) / 3),
            shieldActive: ship.shieldActive ? 1 : 0,
            velocity: MLFeatureExtractor.clamp01(velocity / 5),
            shieldCooldownNorm: MLFeatureExtractor.clamp01((ship.shieldCooldown || 0) / 300),
            isAlpha: ship.isAlpha ? 1 : 0,
            roleNorm: MLFeatureExtractor.roleToNorm(ship.role),
            formationError,
            asteroidCount: MLFeatureExtractor.clamp01((asteroids || []).length / 10),
            enemyCount: MLFeatureExtractor.clamp01(regularEnemies.length / 10),
            allyCount: MLFeatureExtractor.clamp01(nearbyAllies / maxAllies),
            bossCount: MLFeatureExtractor.clamp01(bosses.length / 3),
            nearestAsteroidDistance: nearestAsteroid ? MLFeatureExtractor.clamp01(nearestAsteroid.distance / 200) : 1,
            nearestAsteroidAngle: nearestAsteroid ? MLFeatureExtractor.normalize(nearestAsteroid.angle, -Math.PI, Math.PI) : 0,
            nearestAsteroidSize: nearestAsteroid ? MLFeatureExtractor.clamp01((nearestAsteroid.size || 20) / 50) : 0,
            nearestEnemyDistance: nearestEnemy ? MLFeatureExtractor.clamp01(nearestEnemy.distance / 400) : 1,
            nearestEnemyAngle: nearestEnemy ? MLFeatureExtractor.normalize(nearestEnemy.angle, -Math.PI, Math.PI) : 0,
            nearestEnemyHealth: nearestEnemy ? MLFeatureExtractor.clamp01(nearestEnemy.health / 5) : 0,
            nearestBossDistance: nearestBoss ? MLFeatureExtractor.clamp01(nearestBoss.distance / 600) : 1,
            nearestBossHealth: nearestBoss ? MLFeatureExtractor.clamp01(nearestBoss.health / 5) : 0,
            nearestAllyDistance: nearestAlly ? MLFeatureExtractor.clamp01(nearestAlly.distance / 300) : 1,
            nearestAllyAngle: nearestAlly ? MLFeatureExtractor.normalize(nearestAlly.angle, -Math.PI, Math.PI) : 0,
            nearestAllyHealth: nearestAlly ? MLFeatureExtractor.clamp01(nearestAlly.health / 3) : 1,
            threatLevel: MLFeatureExtractor.clamp01(threatLevel),
            flockDensity,
            bulletThreat,
            aggregateAsteroidThreat: MLFeatureExtractor.attentionThreat(ship, asteroids || [], 'asteroid'),
            aggregateEnemyThreat: MLFeatureExtractor.attentionThreat(ship, enemyShips || [], 'enemy'),
            curriculumStageNorm: MLFeatureExtractor.clamp01(curriculumStage / 3),
            formationCohesion: MLFeatureExtractor.clamp01(1 - formationError)
        };

        features._vector = CANONICAL_FEATURE_NAMES.map(name => features[name]);
        features._canvas = { width: canvasW, height: canvasH };
        return features;
    }

    // Smallest signed difference between two angles, in [-PI, PI].
    static angleDelta(from, to) {
        let d = to - from;
        while (d > Math.PI) d -= 2 * Math.PI;
        while (d < -Math.PI) d += 2 * Math.PI;
        return d;
    }

    // Extra features for reinforcement learning.
    //
    // The canonical 30 describe the world but not the agent's relationship to it:
    // they carry the world-frame bearing to the nearest asteroid but not the
    // ship's own heading, so aim alignment is not recoverable from the
    // observation, and they omit the weapon cooldown, so "can I shoot right now"
    // is unknowable. A policy cannot learn to aim and fire without these, which is
    // why they are added here rather than shaping the reward to compensate.
    //
    // The canonical 30 are left untouched so the existing supervised pipeline and
    // any previously collected data stay valid.
    static extractRL(ship, asteroids, aiShips, playerShip, enemyShips, enemyBullets = [], options = {}) {
        const features = MLFeatureExtractor.extract(
            ship, asteroids, aiShips, playerShip, enemyShips, enemyBullets, options
        );

        const bosses = (enemyShips || []).filter(e => e && (e.type === 'boss' || e.aiType === 'boss'));
        const regularEnemies = (enemyShips || []).filter(e => e && e.type !== 'boss' && e.aiType !== 'boss');
        const nearestAsteroid = MLFeatureExtractor.findNearestEntity(ship, asteroids || [], 400);
        const nearestEnemy = MLFeatureExtractor.findNearestEntity(ship, regularEnemies.concat(bosses), 600);

        const heading = ship.angle || 0;
        const asteroidDelta = nearestAsteroid
            ? MLFeatureExtractor.angleDelta(heading, nearestAsteroid.angle) : Math.PI;
        const enemyDelta = nearestEnemy
            ? MLFeatureExtractor.angleDelta(heading, nearestEnemy.angle) : Math.PI;

        const shootCooldown = ship.shootCooldown || 0;
        const rapidFireCooldown = ship.rapidFireCooldown || 15;

        const extras = {
            headingSin: (Math.sin(heading) + 1) / 2,
            headingCos: (Math.cos(heading) + 1) / 2,
            // Signed: 0.5 means aimed at the target, and which side it is on tells
            // the policy which way to rotate.
            aimSignedAsteroid: (asteroidDelta / Math.PI + 1) / 2,
            aimAlignedAsteroid: MLFeatureExtractor.clamp01(1 - Math.abs(asteroidDelta) / Math.PI),
            aimSignedEnemy: (enemyDelta / Math.PI + 1) / 2,
            aimAlignedEnemy: MLFeatureExtractor.clamp01(1 - Math.abs(enemyDelta) / Math.PI),
            shootCooldownNorm: MLFeatureExtractor.clamp01(shootCooldown / Math.max(1, rapidFireCooldown)),
            canFire: shootCooldown <= 0 ? 1 : 0
        };

        Object.assign(features, extras);

        if (options.fuelFeatures || options.entityTokens) {
            let alpha = null;
            let tanker = null;
            (aiShips || []).forEach(other => {
                if (!other) return;
                if (other.isAlpha) alpha = other;
                if (other.role === 'fuel') tanker = other;
            });
            if (ship.isAlpha) alpha = ship;

            let tankerDistance = 1;
            let tankerBearing = 0.5;
            if (tanker && tanker !== ship) {
                const { dx, dy } = MLFeatureExtractor.relDelta(ship.x, ship.y, tanker.x, tanker.y);
                tankerDistance = MLFeatureExtractor.clamp01(Math.sqrt(dx * dx + dy * dy) / 600);
                const delta = MLFeatureExtractor.angleDelta(heading, Math.atan2(dy, dx));
                tankerBearing = (delta / Math.PI + 1) / 2;
            }

            Object.assign(features, {
                alphaFuelNorm: alpha && alpha.maxFuel > 0
                    ? MLFeatureExtractor.clamp01((alpha.fuel || 0) / alpha.maxFuel) : 1,
                helpPingActive: alpha && alpha.helpPingActive ? 1 : 0,
                tankerDistance,
                tankerBearing,
                fuelLinked: (ship.fuelDocked || ship.fuelTransferActive) ? 1 : 0
            });

            if (options.entityTokens) {
                const tokenVec = MLFeatureExtractor.extractEntityTokens(
                    ship, asteroids, aiShips, enemyShips, enemyBullets, heading
                );
                for (let idx = 0; idx < RL_ENTITY_TOKEN_NAMES.length; idx++) {
                    features[RL_ENTITY_TOKEN_NAMES[idx]] = tokenVec[idx];
                }
                features._entities = tokenVec._entities;
                features._vector = RL_FEATURE_NAMES_ENTITY.map(name => features[name]);
                return features;
            }

            features._vector = RL_FEATURE_NAMES_FUEL.map(name => features[name]);
            return features;
        }

        features._vector = RL_FEATURE_NAMES.map(name => features[name]);
        return features;
    }

    // Build fixed-layout 10 x 6 entity tokens: [present, distNorm, bearingNorm,
    // relVxNorm, relVyNorm, attrNorm] for top-3 asteroids, top-3 enemies/bosses,
    // top-2 enemy bullets, and top-2 allies. Also stores `_entities` refs for
    // 3D attention-beam visualization.
    static extractEntityTokens(ship, asteroids, aiShips, enemyShips, enemyBullets, heading) {
        const out = new Array(ENTITY_TOKEN_SLOTS * ENTITY_TOKEN_DIM).fill(0);
        const refs = new Array(ENTITY_TOKEN_SLOTS).fill(null);

        function nearestK(list, k, maxDist, exclude) {
            if (!list || !list.length) return [];
            const scored = [];
            for (let i = 0; i < list.length; i++) {
                const e = list[i];
                if (!e || e === exclude) continue;
                const { dx, dy } = MLFeatureExtractor.relDelta(ship.x, ship.y, e.x, e.y);
                const dist = Math.hypot(dx, dy);
                if (dist <= maxDist) scored.push({ e, dx, dy, dist });
            }
            scored.sort((a, b) => a.dist - b.dist);
            return scored.slice(0, k);
        }

        function writeSlot(slotIdx, item, maxDist, attrVal) {
            if (!item) return;
            const off = slotIdx * ENTITY_TOKEN_DIM;
            const bearing = MLFeatureExtractor.angleDelta(heading, Math.atan2(item.dy, item.dx));
            const rvx = ((item.e.velocityX || 0) - (ship.velocityX || 0));
            const rvy = ((item.e.velocityY || 0) - (ship.velocityY || 0));
            out[off] = 1;
            out[off + 1] = MLFeatureExtractor.clamp01(item.dist / maxDist);
            out[off + 2] = (bearing / Math.PI + 1) / 2;
            out[off + 3] = MLFeatureExtractor.clamp01((rvx / 10 + 1) / 2);
            out[off + 4] = MLFeatureExtractor.clamp01((rvy / 10 + 1) / 2);
            out[off + 5] = MLFeatureExtractor.clamp01(attrVal);
            refs[slotIdx] = item.e;
        }

        const topAst = nearestK(asteroids, 3, 500, null);
        for (let i = 0; i < topAst.length; i++) {
            writeSlot(i, topAst[i], 500, (topAst[i].e.radius || 20) / 50);
        }
        const topEne = nearestK(enemyShips, 3, 700, null);
        for (let i = 0; i < topEne.length; i++) {
            const isBoss = topEne[i].e.type === 'boss' || topEne[i].e.aiType === 'boss';
            writeSlot(3 + i, topEne[i], 700, isBoss ? 1.0 : Math.min(0.8, (topEne[i].e.health || 1) / 5));
        }
        const topBul = nearestK(enemyBullets, 2, 400, null);
        for (let i = 0; i < topBul.length; i++) {
            writeSlot(6 + i, topBul[i], 400, 1.0);
        }
        const topAly = nearestK(aiShips, 2, 600, ship);
        for (let i = 0; i < topAly.length; i++) {
            writeSlot(8 + i, topAly[i], 600, topAly[i].e.isAlpha ? 1.0 : (topAly[i].e.health || 3) / 6);
        }

        out._entities = refs;
        return out;
    }

    // Global team-state vector for MAPPO Centralized Critic (12 dims).
    static extractGlobalState(aiShips, asteroids, enemyShips, enemyBullets, options = {}) {
        const ships = (aiShips || []).filter(Boolean);
        const maxAgents = Math.max(1, options.numAgents || ships.length || 3);
        let hpSum = 0;
        let hpMin = 1;
        let alpha = null;
        let tankerAlive = 0;
        for (const s of ships) {
            const frac = MLFeatureExtractor.clamp01((s.health || 0) / Math.max(1, s.maxHealth || 3));
            hpSum += frac;
            if (frac < hpMin) hpMin = frac;
            if (s.isAlpha) alpha = s;
            if (s.role === 'fuel') tankerAlive = 1;
        }
        const bosses = (enemyShips || []).filter(e => e && (e.type === 'boss' || e.aiType === 'boss')).length;
        const regEnemies = (enemyShips || []).filter(Boolean).length - bosses;
        const waveNum = (typeof waveState !== 'undefined' && waveState) ? (waveState.wave || 0) : 0;
        return [
            MLFeatureExtractor.clamp01(ships.length / maxAgents),
            ships.length ? hpSum / ships.length : 0,
            ships.length ? hpMin : 0,
            alpha ? 1 : 0,
            alpha && alpha.maxFuel > 0 ? MLFeatureExtractor.clamp01((alpha.fuel || 0) / alpha.maxFuel) : 1,
            alpha && alpha.helpPingActive ? 1 : 0,
            tankerAlive,
            MLFeatureExtractor.clamp01((asteroids || []).length / 15),
            MLFeatureExtractor.clamp01(regEnemies / 10),
            MLFeatureExtractor.clamp01(bosses / 4),
            MLFeatureExtractor.clamp01((enemyBullets || []).length / 30),
            MLFeatureExtractor.clamp01(waveNum / 10)
        ];
    }

    static toLegacy13(features) {
        return {
            health: features.health,
            phase: features.phase,
            shieldActive: features.shieldActive,
            velocity: features.velocity,
            asteroidCount: features.asteroidCount,
            nearestAsteroidDistance: features.nearestAsteroidDistance,
            enemyCount: features.enemyCount,
            nearestEnemyDistance: features.nearestEnemyDistance,
            allyCount: features.allyCount,
            threatLevel: features.threatLevel,
            flockDensity: features.flockDensity,
            hasNearbyEnemies: features.nearestEnemyDistance < 0.5 ? 1 : 0
        };
    }

    static toMARL21(features) {
        return features._vector.slice(0, 21);
    }

    static vectorFromObject(features) {
        if (features._vector) return features._vector;
        return CANONICAL_FEATURE_NAMES.map(name => features[name] ?? 0);
    }
}

if (typeof globalThis !== 'undefined') {
    globalThis.MLFeatureExtractor = MLFeatureExtractor;
    globalThis.CANONICAL_FEATURE_DIM = CANONICAL_FEATURE_DIM;
    globalThis.CANONICAL_FEATURE_NAMES = CANONICAL_FEATURE_NAMES;
    globalThis.RL_FEATURE_NAMES = RL_FEATURE_NAMES;
    globalThis.RL_FEATURE_DIM = RL_FEATURE_DIM;
    globalThis.RL_FEATURE_NAMES_FUEL = RL_FEATURE_NAMES_FUEL;
    globalThis.RL_FEATURE_DIM_FUEL = RL_FEATURE_DIM_FUEL;
    globalThis.ENTITY_TOKEN_SLOTS = ENTITY_TOKEN_SLOTS;
    globalThis.ENTITY_TOKEN_DIM = ENTITY_TOKEN_DIM;
    globalThis.RL_ENTITY_TOKEN_NAMES = RL_ENTITY_TOKEN_NAMES;
    globalThis.RL_FEATURE_NAMES_ENTITY = RL_FEATURE_NAMES_ENTITY;
    globalThis.RL_FEATURE_DIM_ENTITY = RL_FEATURE_DIM_ENTITY;
    globalThis.GLOBAL_STATE_NAMES = GLOBAL_STATE_NAMES;
    globalThis.GLOBAL_STATE_DIM = GLOBAL_STATE_DIM;
}
if (typeof window !== 'undefined') {
    window.MLFeatureExtractor = MLFeatureExtractor;
    window.CANONICAL_FEATURE_DIM = CANONICAL_FEATURE_DIM;
    window.CANONICAL_FEATURE_NAMES = CANONICAL_FEATURE_NAMES;
    window.RL_FEATURE_NAMES = RL_FEATURE_NAMES;
    window.RL_FEATURE_DIM = RL_FEATURE_DIM;
    window.RL_FEATURE_NAMES_FUEL = RL_FEATURE_NAMES_FUEL;
    window.RL_FEATURE_DIM_FUEL = RL_FEATURE_DIM_FUEL;
    window.RL_FEATURE_NAMES_ENTITY = RL_FEATURE_NAMES_ENTITY;
    window.RL_FEATURE_DIM_ENTITY = RL_FEATURE_DIM_ENTITY;
    window.GLOBAL_STATE_DIM = GLOBAL_STATE_DIM;
    window.marlCurriculumStage = 3;
    window.setMARLCurriculumStage = function(stage) {
        window.marlCurriculumStage = Math.max(0, Math.min(3, stage));
        console.log(`[ML] Curriculum stage: ${window.marlCurriculumStage}`);
    };
}
