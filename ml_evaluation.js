// ML evaluation tracker for A/B comparison and performance monitoring

class MLEvaluationTracker {
    constructor() {
        this.enabled = false;
        this.mlMode = false;
        this.frameCount = 0;
        this.sessionStart = Date.now();
        this.metrics = {
            survivalFrames: 0,
            totalKills: 0,
            friendlyFireEvents: 0,
            formationErrors: [],
            inferenceMs: [],
            mlDecisions: 0,
            ruleDecisions: 0,
            deaths: 0,
            scoreGained: 0,
            maxScore: 0
        };
        this.previousScore = 0;
        this.previousHealth = new Map();
    }

    setEnabled(enabled) {
        this.enabled = enabled;
        if (enabled) this.reset();
    }

    setMLMode(mlMode) {
        this.mlMode = mlMode;
    }

    reset() {
        this.frameCount = 0;
        this.sessionStart = Date.now();
        this.metrics = {
            survivalFrames: 0,
            totalKills: 0,
            friendlyFireEvents: 0,
            formationErrors: [],
            inferenceMs: [],
            mlDecisions: 0,
            ruleDecisions: 0,
            deaths: 0,
            scoreGained: 0,
            maxScore: 0
        };
        this.previousScore = 0;
        this.previousHealth.clear();
    }

    recordFrame(context) {
        if (!this.enabled) return;

        const {
            aiShips = [],
            score = 0,
            inferenceMs = 0,
            usedML = false,
            formationErrors = []
        } = context;

        this.frameCount++;
        this.metrics.survivalFrames++;

        if (score > this.previousScore) {
            this.metrics.scoreGained += score - this.previousScore;
            this.metrics.totalKills += Math.floor((score - this.previousScore) / 10);
        }
        this.metrics.maxScore = Math.max(this.metrics.maxScore, score);
        this.previousScore = score;

        if (usedML) this.metrics.mlDecisions++;
        else this.metrics.ruleDecisions++;

        if (inferenceMs > 0) {
            this.metrics.inferenceMs.push(inferenceMs);
            if (this.metrics.inferenceMs.length > 100) {
                this.metrics.inferenceMs.shift();
            }
        }

        aiShips.forEach((ship, i) => {
            if (!ship) return;
            const prev = this.previousHealth.get(i) ?? ship.health;
            if (ship.health < prev) {
                if (ship.health <= 0) this.metrics.deaths++;
            }
            this.previousHealth.set(i, ship.health);
        });

        formationErrors.forEach(err => this.metrics.formationErrors.push(err));
        if (this.metrics.formationErrors.length > 200) {
            this.metrics.formationErrors = this.metrics.formationErrors.slice(-100);
        }
    }

    getSummary() {
        const avgInference = this.metrics.inferenceMs.length > 0
            ? this.metrics.inferenceMs.reduce((a, b) => a + b, 0) / this.metrics.inferenceMs.length
            : 0;

        const avgFormationError = this.metrics.formationErrors.length > 0
            ? this.metrics.formationErrors.reduce((a, b) => a + b, 0) / this.metrics.formationErrors.length
            : 0;

        const sessionMinutes = (Date.now() - this.sessionStart) / 60000;
        const totalDecisions = this.metrics.mlDecisions + this.metrics.ruleDecisions;

        return {
            sessionMinutes: sessionMinutes.toFixed(1),
            survivalFrames: this.metrics.survivalFrames,
            deaths: this.metrics.deaths,
            scoreGained: this.metrics.scoreGained,
            maxScore: this.metrics.maxScore,
            avgInferenceMs: avgInference.toFixed(2),
            avgFormationError: avgFormationError.toFixed(3),
            mlDecisionRate: totalDecisions > 0
                ? ((this.metrics.mlDecisions / totalDecisions) * 100).toFixed(1) + '%'
                : '0%',
            mode: this.mlMode ? 'ML' : 'Rule-based'
        };
    }
}

let mlEvaluationTracker = null;

function initializeMLEvaluation() {
    if (!mlEvaluationTracker) {
        mlEvaluationTracker = new MLEvaluationTracker();
    }
    return mlEvaluationTracker;
}

function setMLEvaluationEnabled(enabled, mlMode = false) {
    const tracker = initializeMLEvaluation();
    tracker.setEnabled(enabled);
    tracker.setMLMode(mlMode);
}

function getMLEvaluationSummary() {
    return initializeMLEvaluation().getSummary();
}

if (typeof window !== 'undefined') {
    window.MLEvaluationTracker = MLEvaluationTracker;
    window.initializeMLEvaluation = initializeMLEvaluation;
    window.setMLEvaluationEnabled = setMLEvaluationEnabled;
    window.getMLEvaluationSummary = getMLEvaluationSummary;
}
