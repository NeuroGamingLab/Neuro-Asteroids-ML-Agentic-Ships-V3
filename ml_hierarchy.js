// Hierarchical ML: strategic layer (slow) + tactical layer (fast)

class MLHierarchyController {
    constructor() {
        this.enabled = false;
        this.strategicFrame = 0;
        this.strategicInterval = 30; // ~2 Hz at 60 FPS
        this.strategicState = {
            formationBias: 'balanced',
            targetMode: 'focus',
            aggression: 0.5,
            defense: 0.5,
            escortPriority: 0.3
        };
    }

    setEnabled(enabled) {
        this.enabled = enabled;
    }

    decideStrategic(features, ship, aiShips) {
        const state = { ...this.strategicState };

        if (features.bossCount > 0) {
            state.formationBias = 'defensive';
            state.targetMode = 'focus';
            state.aggression = 0.7;
            state.defense = 0.8;
        } else if (features.enemyCount > 0.5) {
            state.formationBias = 'offensive';
            state.targetMode = features.allyCount > 0.5 ? 'split' : 'focus';
            state.aggression = 0.75;
            state.defense = 0.6;
        } else if (features.threatLevel > 0.6) {
            state.formationBias = 'evasive';
            state.aggression = 0.3;
            state.defense = 0.9;
        } else {
            state.formationBias = 'balanced';
            state.aggression = 0.5;
            state.defense = 0.5;
        }

        if (features.health < 0.4) {
            state.defense = Math.min(1, state.defense + 0.3);
            state.aggression = Math.max(0.2, state.aggression - 0.2);
        }

        if (ship.isAlpha && features.allyCount > 0.3) {
            state.escortPriority = 0.5;
        }

        return state;
    }

    applyStrategicToShip(ship, strategic) {
        if (!ship || !strategic) return;

        if (typeof multiTargetMode !== 'undefined' && ship.isAlpha) {
            if (strategic.targetMode === 'split') {
                ship.multiTargetMode = 'split';
            } else if (strategic.targetMode === 'focus') {
                ship.multiTargetMode = 'focus';
            }
        }

        if (strategic.formationBias === 'defensive') {
            ship.defensiveFlockWeight = Math.max(ship.defensiveFlockWeight || 0.4, 0.7);
        } else if (strategic.formationBias === 'offensive') {
            ship.defensiveFlockWeight = Math.min(ship.defensiveFlockWeight || 0.4, 0.3);
        }

        ship._strategicState = strategic;
    }

    update(frame, shipsFeatures) {
        if (!this.enabled || !shipsFeatures || shipsFeatures.length === 0) return;

        this.strategicFrame++;
        if (this.strategicFrame < this.strategicInterval) return;
        this.strategicFrame = 0;

        shipsFeatures.forEach(({ ship, features, aiShips }) => {
            if (!ship || ship.isAlpha) {
                const strategic = this.decideStrategic(features, ship, aiShips);
                this.strategicState = strategic;
            }
            this.applyStrategicToShip(ship, this.strategicState);
        });
    }

    getState() {
        return this.strategicState;
    }
}

let mlHierarchyController = null;

function initializeMLHierarchy() {
    if (!mlHierarchyController) {
        mlHierarchyController = new MLHierarchyController();
    }
    return mlHierarchyController;
}

function setMLHierarchyEnabled(enabled) {
    const controller = initializeMLHierarchy();
    controller.setEnabled(enabled);
}

if (typeof window !== 'undefined') {
    window.MLHierarchyController = MLHierarchyController;
    window.initializeMLHierarchy = initializeMLHierarchy;
    window.setMLHierarchyEnabled = setMLHierarchyEnabled;
}
