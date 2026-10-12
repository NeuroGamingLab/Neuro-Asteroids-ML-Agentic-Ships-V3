// Policy visualisation overlays: a critic value heatmap and per-ship
// action-probability wheels. Load after policy_runner.js and before game.js
// (game.js calls drawValueHeatmap() / drawActionWheels() from draw()).
//
// Both overlays read from the PolicyRunner and never write game state.
//
// Value heatmap. The critic V(s) is a function of one ship's observation, and
// the observation depends on where that ship is. Evaluating the critic with the
// focus ship hypothetically moved to each cell of a coarse grid gives a map of
// where the policy believes that ship would be better or worse off - a "danger
// map" in the policy's own terms rather than the designer's. It is computed one
// grid row per frame so the cost is bounded (24 feature extractions + critic
// forwards per frame) and the whole map refreshes every 16 frames.

(function (root) {
    'use strict';

    const GRID_COLS = 24;
    const GRID_ROWS = 16;

    let heatmapEnabled = false;
    let wheelsEnabled = false;

    const values = new Float32Array(GRID_COLS * GRID_ROWS);
    let haveValues = false;
    let nextRow = 0;
    let vMin = 0;
    let vMax = 1;

    function runner() {
        return (typeof root.getPolicyRunner === 'function') ? root.getPolicyRunner() : null;
    }

    // The ship whose viewpoint the heatmap takes: the alpha when there is one,
    // otherwise the first live policy-driven ship.
    function focusShip() {
        if (typeof aiShips === 'undefined') return null;
        let first = null;
        for (let i = 0; i < aiShips.length; i++) {
            const s = aiShips[i];
            if (!s || s.role === 'fuel') continue;
            if (s.isAlpha) return s;
            if (!first) first = s;
        }
        return first;
    }

    function updateHeatmapRow(r, shipRef) {
        const w = canvas.width;
        const h = canvas.height;
        const cellW = w / GRID_COLS;
        const cellH = h / GRID_ROWS;

        // Move the ship hypothetically; the extractor only reads position, so
        // restoring x/y afterwards leaves the game untouched. try/finally in
        // case an extractor throws part-way through.
        const ox = shipRef.x;
        const oy = shipRef.y;
        try {
            const row = nextRow;
            const y = (row + 0.5) * cellH;
            for (let c = 0; c < GRID_COLS; c++) {
                shipRef.x = (c + 0.5) * cellW;
                shipRef.y = y;
                const features = MLFeatureExtractor.extractRL(
                    shipRef, asteroids, aiShips, ship, enemyShips, enemyBullets,
                    Object.assign({ curriculumStage: r.curriculumStage }, r.featureOptions || {})
                );
                const v = r.valueOf(features._vector);
                values[row * GRID_COLS + c] = (v === null || v === undefined) ? 0 : v;
            }
        } finally {
            shipRef.x = ox;
            shipRef.y = oy;
        }

        nextRow = (nextRow + 1) % GRID_ROWS;
        if (nextRow === 0) {
            haveValues = true;
            let lo = Infinity;
            let hi = -Infinity;
            for (let i = 0; i < values.length; i++) {
                if (values[i] < lo) lo = values[i];
                if (values[i] > hi) hi = values[i];
            }
            // Smooth the colour range so the map does not flicker between refreshes.
            vMin = haveValues ? vMin * 0.7 + lo * 0.3 : lo;
            vMax = haveValues ? vMax * 0.7 + hi * 0.3 : hi;
            if (vMax - vMin < 1e-3) vMax = vMin + 1e-3;
        }
    }

    // Low value -> magenta/red, high value -> cyan. Matches the theme.
    function heatColor(t) {
        const clamped = Math.max(0, Math.min(1, t));
        const rC = Math.round(255 * (1 - clamped));
        const gC = Math.round(210 * clamped);
        const bC = 255;
        return `rgb(${rC},${gC},${bC})`;
    }

    function drawValueHeatmap(ctx) {
        if (!heatmapEnabled) return;
        const r = runner();
        if (!r || !r.enabled || !r.critic) return;
        const shipRef = focusShip();
        if (!shipRef) return;

        updateHeatmapRow(r, shipRef);
        if (!haveValues) return;

        const cellW = canvas.width / GRID_COLS;
        const cellH = canvas.height / GRID_ROWS;
        const prevAlpha = ctx.globalAlpha;
        for (let row = 0; row < GRID_ROWS; row++) {
            for (let c = 0; c < GRID_COLS; c++) {
                const t = (values[row * GRID_COLS + c] - vMin) / (vMax - vMin);
                ctx.fillStyle = heatColor(t);
                ctx.globalAlpha = 0.10 + 0.14 * (1 - t);   // danger is drawn stronger than safety
                ctx.fillRect(c * cellW, row * cellH, cellW + 0.5, cellH + 0.5);
            }
        }
        ctx.globalAlpha = 0.8;
        ctx.fillStyle = GAME_THEME.overlayMuted;
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(
            `critic V(s) for ${shipRef.isAlpha ? 'alpha' : 'agent ' + shipRef.agentId}` +
            `  [${vMin.toFixed(1)} .. ${vMax.toFixed(1)}]  magenta=low  cyan=high`,
            10, 14
        );
        ctx.globalAlpha = prevAlpha;
    }

    // Action wheel. The 48 joint actions factor into rotation x thrust x fire x
    // shield; the wheel shows the marginal probability of each factor level as
    // an arc whose opacity is the probability, so a decisive policy shows one
    // bright arc per ring and a hesitant one shows several dim ones.
    //
    // Layout (index -> {rotation, thrust, fire, shield}) mirrors decodeAction():
    //   rotation = floor(i / 16) - 1, thrust = (i >> 3) & 1,
    //   fire = (i >> 1) & 3, shield = i & 1.
    function marginals(probs) {
        const rot = [0, 0, 0];
        const thrust = [0, 0];
        const fire = [0, 0, 0, 0];
        const shield = [0, 0];
        for (let i = 0; i < probs.length; i++) {
            const p = probs[i];
            rot[Math.floor(i / 16)] += p;
            const rem = i % 16;
            thrust[(rem >> 3) & 1] += p;
            fire[(rem >> 1) & 3] += p;
            shield[rem & 1] += p;
        }
        return { rot, thrust, fire, shield };
    }

    function arc(ctx, x, y, radius, from, to, color, alpha, width) {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.globalAlpha = Math.max(0.05, Math.min(1, alpha));
        ctx.beginPath();
        ctx.arc(x, y, radius, from, to);
        ctx.stroke();
    }

    function drawActionWheels(ctx) {
        if (!wheelsEnabled) return;
        const r = runner();
        if (!r || !r.enabled || !r.lastDecisions || !r.lastDecisions.length) return;

        ctx.save();
        for (let i = 0; i < aiShips.length; i++) {
            const s = aiShips[i];
            if (!s) continue;
            const d = r.lastDecisions[s.agentId];
            if (!d || !d.probs) continue;
            const m = marginals(d.probs);
            // On the sphere the wheel sits at the ship's projected position and
            // shrinks with distance; worldToScreen returns null for the far side.
            const sp = (typeof worldToScreen === 'function') ? worldToScreen(s.x, s.y) : { x: s.x, y: s.y, scale: 1 };
            if (!sp) continue;
            const x = sp.x;
            const y = sp.y;
            const base = ((s.size || 15) + 14) * (sp.scale || 1);

            // Ring 1 (inner): rotation. Left arc = rotate left, right arc = rotate
            // right, top arc = hold heading.
            arc(ctx, x, y, base, Math.PI * 0.75, Math.PI * 1.25, GAME_THEME.primary, m.rot[0], 3);
            arc(ctx, x, y, base, Math.PI * 1.35, Math.PI * 1.65, GAME_THEME.muted, m.rot[1], 3);
            arc(ctx, x, y, base, Math.PI * 1.75, Math.PI * 2.25, GAME_THEME.primary, m.rot[2], 3);

            // Ring 2: thrust (bottom arc) and shield (small top arc).
            arc(ctx, x, y, base + 5, Math.PI * 0.3, Math.PI * 0.7, GAME_THEME.orange, m.thrust[1], 3);
            arc(ctx, x, y, base + 5, Math.PI * 1.45, Math.PI * 1.55, GAME_THEME.cyan, m.shield[1], 3);

            // Ring 3 (outer): fire mode, four quadrants: hold / forward / asteroid / enemy.
            const fireColors = [GAME_THEME.barDim, GAME_THEME.text, GAME_THEME.amber, GAME_THEME.enemy];
            for (let k = 0; k < 4; k++) {
                const from = -Math.PI / 2 + k * (Math.PI / 2) + 0.08;
                const to = from + Math.PI / 2 - 0.16;
                arc(ctx, x, y, base + 10, from, to, fireColors[k], m.fire[k], 2);
            }
        }
        ctx.restore();
    }

    root.drawValueHeatmap = drawValueHeatmap;
    root.drawActionWheels = drawActionWheels;
    root.setValueHeatmap = function (on) {
        heatmapEnabled = !!on;
        if (!heatmapEnabled) { haveValues = false; nextRow = 0; }
    };
    root.setActionWheels = function (on) { wheelsEnabled = !!on; };
    root.getPolicyVizState = function () {
        return { heatmap: heatmapEnabled, wheels: wheelsEnabled, refreshed: haveValues };
    };
})(typeof globalThis !== 'undefined' ? globalThis : this);
