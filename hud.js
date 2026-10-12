// In-canvas telemetry HUD: sparklines for the quantities that tell you whether
// the swarm is doing well, drawn in screen space in the top-right corner. No
// charting library and no round trip to Streamlit; sampled every 5 frames and
// kept for the last 20 seconds.
//
// Load before game.js (game.js calls drawHUD() at the end of draw()).

(function (root) {
    'use strict';

    const SAMPLE_EVERY = 5;
    const HISTORY = 240;            // 240 samples x 5 frames = 1200 frames = 20 s
    const PANEL_W = 230;
    const ROW_H = 30;

    let enabled = false;
    let frame = 0;
    let lastScore = 0;
    let lastScoreFrame = 0;
    let scoreRate = 0;

    const series = {};
    function track(name, label, color, min, max, fmt) {
        series[name] = {
            label, color, min, max, fmt,
            data: new Float32Array(HISTORY),
            head: 0,
            count: 0,
            last: 0
        };
    }
    function push(name, value) {
        const s = series[name];
        const v = Number.isFinite(value) ? value : 0;
        s.data[s.head] = v;
        s.head = (s.head + 1) % HISTORY;
        if (s.count < HISTORY) s.count++;
        s.last = v;
    }

    track('scoreRate', 'score / s', GAME_THEME.score, 0, null, v => v.toFixed(0));
    track('alive', 'AI ships alive', GAME_THEME.aiBlue, 0, null, v => v.toFixed(0));
    track('fuel', 'alpha fuel %', GAME_THEME.fuelBar, 0, 100, v => v.toFixed(0));
    track('threat', 'alpha threat', GAME_THEME.danger, 0, 1, v => (v * 100).toFixed(0) + '%');
    track('value', 'policy V(s)', GAME_THEME.valueText, null, null, v => v.toFixed(1));
    track('entropy', 'policy entropy', GAME_THEME.accent, 0, Math.log(48), v => v.toFixed(2));

    function sample() {
        frame++;
        // Score rate over a sliding one-second window.
        if (frame - lastScoreFrame >= 60) {
            scoreRate = (score - lastScore) * 60 / (frame - lastScoreFrame);
            lastScore = score;
            lastScoreFrame = frame;
        }
        if (frame % SAMPLE_EVERY !== 0) return;

        push('scoreRate', scoreRate);

        let alive = 0;
        let alpha = null;
        for (let i = 0; i < aiShips.length; i++) {
            const s = aiShips[i];
            if (!s) continue;
            alive++;
            if (s.isAlpha) alpha = s;
        }
        push('alive', alive);
        push('fuel', alpha && alpha.maxFuel ? 100 * alpha.fuel / alpha.maxFuel : NaN);
        push('threat', alpha ? (alpha.threatLevel || 0) : 0);

        const runner = (typeof root.getPolicyRunner === 'function') ? root.getPolicyRunner() : null;
        if (runner && runner.enabled && runner.lastDecisions && runner.lastDecisions.length) {
            let vSum = 0; let hSum = 0; let n = 0;
            for (const d of runner.lastDecisions) {
                if (!d) continue;
                if (d.value !== null && d.value !== undefined) vSum += d.value;
                hSum += d.entropy || 0;
                n++;
            }
            push('value', n ? vSum / n : 0);
            push('entropy', n ? hSum / n : 0);
        } else {
            push('value', 0);
            push('entropy', 0);
        }
    }

    function sparkline(ctx, s, x, y, w, h) {
        if (s.count < 2) return;
        let lo = s.min === null ? Infinity : s.min;
        let hi = s.max === null ? -Infinity : s.max;
        if (s.min === null || s.max === null) {
            for (let k = 0; k < s.count; k++) {
                const v = s.data[(s.head - 1 - k + HISTORY * 2) % HISTORY];
                if (s.min === null && v < lo) lo = v;
                if (s.max === null && v > hi) hi = v;
            }
            if (!Number.isFinite(lo)) lo = 0;
            if (!Number.isFinite(hi)) hi = 1;
            if (hi - lo < 1e-6) hi = lo + 1;
        }
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        for (let k = 0; k < s.count; k++) {
            const idx = (s.head - s.count + k + HISTORY) % HISTORY;
            const v = s.data[idx];
            const px = x + (k / (HISTORY - 1)) * w;
            const py = y + h - ((v - lo) / (hi - lo)) * h;
            if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
    }

    function drawHUD(ctx) {
        if (!enabled || !ctx) return;
        sample();

        const names = Object.keys(series);
        const x = canvas.width - PANEL_W - 10;
        const y = 10;
        const h = names.length * ROW_H + 26;

        ctx.save();
        ctx.globalAlpha = 0.72;
        ctx.fillStyle = GAME_THEME.barBg;
        ctx.fillRect(x, y, PANEL_W, h);
        ctx.globalAlpha = 0.9;
        ctx.strokeStyle = GAME_THEME.primary;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, PANEL_W - 1, h - 1);

        ctx.fillStyle = GAME_THEME.primary;
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('TELEMETRY 20s', x + 8, y + 14);

        // Right-aligned status: where the policy runs ("wkr" when a Web Worker
        // owns inference, otherwise main-thread time) and live particle count.
        const runner = (typeof root.getPolicyRunner === 'function') ? root.getPolicyRunner() : null;
        const fx = (typeof root.FX !== 'undefined') ? root.FX.stats() : null;
        ctx.fillStyle = GAME_THEME.overlayMuted;
        ctx.font = '9px monospace';
        ctx.textAlign = 'right';
        const status = [];
        if (runner && runner.enabled) {
            status.push(runner.worker && runner.workerReady
                ? 'wkr'
                : `${runner.inferenceMs.toFixed(1)}ms`);
        }
        if (fx) status.push(`${fx.particles}p`);
        ctx.fillText(status.join(' '), x + PANEL_W - 8, y + 14);

        for (let i = 0; i < names.length; i++) {
            const s = series[names[i]];
            const rowY = y + 22 + i * ROW_H;
            ctx.globalAlpha = 0.85;
            ctx.fillStyle = GAME_THEME.muted;
            ctx.font = '9px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(s.label, x + 8, rowY + 9);
            ctx.textAlign = 'right';
            ctx.fillStyle = s.color;
            ctx.fillText(s.count ? s.fmt(s.last) : '-', x + PANEL_W - 8, rowY + 9);
            sparkline(ctx, s, x + 8, rowY + 12, PANEL_W - 16, ROW_H - 16);
        }
        ctx.restore();

        // Draw the kill-feed on top of the HUD panel
        drawKillFeed(ctx);
    }

    root.drawHUD = drawHUD;
    root.setHudEnabled = function (on) { enabled = !!on; };
    root.getHudSeries = function () {
        const out = {};
        for (const name of Object.keys(series)) out[name] = series[name].last;
        return out;
    };

    // ── Kill-feed ring buffer ─────────────────────────────────────────────
    // Push events from game.js via killFeedPush(msg, color).
    // Entries fade out over FEED_TTL frames then are dropped.

    const FEED_MAX  = 8;    // max lines shown simultaneously
    const FEED_TTL  = 240;  // ~4 s at 60 fps
    const FEED_LINE = 16;   // px per row
    const feedEntries = []; // { msg, color, ttl }

    root.killFeedPush = function (msg, color) {
        feedEntries.push({ msg: msg || '', color: color || '#ffffff', ttl: FEED_TTL });
        // Drop oldest entry when over the cap
        while (feedEntries.length > FEED_MAX) feedEntries.shift();
    };

    function drawKillFeed(ctx) {
        // Age all entries; remove expired ones
        for (let i = feedEntries.length - 1; i >= 0; i--) {
            feedEntries[i].ttl--;
            if (feedEntries[i].ttl <= 0) feedEntries.splice(i, 1);
        }
        if (feedEntries.length === 0) return;

        ctx.save();
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'left';

        // Anchor bottom-left; newest entry at the bottom, older ones scroll up
        const baseY = canvas.height - 14;
        const xPos  = 10;

        feedEntries.forEach((entry, i) => {
            // alpha fades in during last second of life
            const alpha = Math.min(1, entry.ttl / 60);
            const yPos  = baseY - i * FEED_LINE;

            // Dark backing pill for readability
            const textW = ctx.measureText(entry.msg).width;
            ctx.globalAlpha = alpha * 0.50;
            ctx.fillStyle = '#000000';
            ctx.fillRect(xPos - 3, yPos - 11, textW + 10, 14);

            // Coloured event text
            ctx.globalAlpha = alpha;
            ctx.fillStyle = entry.color;
            ctx.fillText(entry.msg, xPos, yPos);
        });

        ctx.restore();
    }

    // ── Bidirectional Telemetry System & Flight Recorder ────────────────────
    const flightRecorder = [];
    const MAX_FLIGHT_RECORDS = 180; // 180 samples x 10 frames = 1800 frames = 30s history
    let telemetryBroadcastInterval = 10; // Every 10 frames (~6Hz)

    function captureTelemetrySnapshot() {
        const activeAlpha = typeof aiShips !== 'undefined' ? aiShips.find(s => s && s.isAlpha) : null;
        let bossData = null;
        if (typeof enemyShips !== 'undefined') {
            const boss = enemyShips.find(e => e && e.type === 'boss');
            if (boss && boss.subsystems) {
                bossData = boss.subsystems.map(sub => ({
                    id: sub.id,
                    name: sub.name,
                    health: sub.health,
                    maxHealth: sub.maxHealth,
                    destroyed: sub.destroyed,
                    color: sub.color
                }));
            }
        }

        const fleetData = (typeof aiShips !== 'undefined') ? aiShips.map(s => {
            if (!s) return null;
            return {
                agentId: s.agentId,
                role: s.role,
                isAlpha: !!s.isAlpha,
                fuel: Math.round(s.fuel || 0),
                maxFuel: s.maxFuel || 100,
                health: s.health,
                buffRailgun: s.buffRailgun || 0,
                buffShield: s.buffShield || 0,
                speed: Math.hypot(s.velocityX || 0, s.velocityY || 0).toFixed(1)
            };
        }).filter(Boolean) : [];

        let marlData = null;
        if (typeof window !== 'undefined' && window.marlSystem && typeof window.marlSystem.getMARLTelemetry === 'function') {
            const obs = (typeof aiShips !== 'undefined') ? aiShips.map(s => s && typeof s.getObservation === 'function' ? s.getObservation() : null) : [];
            marlData = window.marlSystem.getMARLTelemetry(aiShips, obs);
        }

        const pUps = (typeof powerUps !== 'undefined') ? powerUps.map(p => ({
            type: p.type,
            x: Math.round(p.x),
            y: Math.round(p.y),
            lifetime: p.lifetime
        })) : [];

        return {
            timestamp: Date.now(),
            frame: frame,
            score: typeof score !== 'undefined' ? score : 0,
            scoreRate: Math.round(scoreRate),
            alphaFuel: activeAlpha && activeAlpha.maxFuel ? Math.round((activeAlpha.fuel / activeAlpha.maxFuel) * 100) : 100,
            fleetCount: fleetData.length,
            fleet: fleetData,
            bossSubsystems: bossData,
            powerUps: pUps,
            marl: marlData,
            recentEvents: feedEntries.slice(-5).map(e => e.msg)
        };
    }

    function dispatchTacticalCommand(command, payload) {
        if (typeof root.__HEADLESS__ !== 'undefined' && root.__HEADLESS__) return false;
        const targetPos = (typeof aiShips !== 'undefined') ? (aiShips.find(s => s && s.isAlpha) || aiShips[0] || { x: worldW() / 2, y: worldH() / 2 }) : { x: 600, y: 300 };

        switch (command) {
            case 'AIRDROP_RAILGUN':
            case 'AIRDROP_SHIELD':
            case 'AIRDROP_EMP':
            case 'AIRDROP_FUEL': {
                const typeMap = {
                    'AIRDROP_RAILGUN': 'railgun',
                    'AIRDROP_SHIELD': 'overshield',
                    'AIRDROP_EMP': 'emp',
                    'AIRDROP_FUEL': 'fuel'
                };
                if (typeof PowerUp !== 'undefined' && typeof powerUps !== 'undefined') {
                    const p = new PowerUp(targetPos.x, targetPos.y, typeMap[command]);
                    powerUps.push(p);
                    if (typeof root.killFeedPush === 'function') {
                        root.killFeedPush(`🛰 MISSION CONTROL: ${p.label} AIRDROP DISPATCHED!`, p.color);
                    }
                }
                return true;
            }
            case 'DETONATE_EMP': {
                if (typeof triggerEMPBlast === 'function') {
                    triggerEMPBlast(targetPos.x, targetPos.y);
                    if (typeof root.killFeedPush === 'function') {
                        root.killFeedPush('💥 MISSION CONTROL: SATELLITE EMP DISCHARGED!', '#FFD600');
                    }
                }
                return true;
            }
            case 'EMERGENCY_REFUEL': {
                if (typeof aiShips !== 'undefined') {
                    aiShips.forEach(s => {
                        if (s && s.fuel !== undefined) {
                            s.fuel = s.maxFuel || 100;
                            s.helpPingActive = false;
                        }
                    });
                    if (typeof root.killFeedPush === 'function') {
                        root.killFeedPush('⛽ MISSION CONTROL: ORBITAL FLEET REFUEL DELIVERED!', '#FF0077');
                    }
                }
                return true;
            }
            case 'OVERCHARGE_SHIELDS': {
                if (typeof aiShips !== 'undefined') {
                    aiShips.forEach(s => {
                        if (s) {
                            s.buffShield = 600;
                            s.shieldActive = true;
                            s.shieldRadius = 45;
                        }
                    });
                    if (typeof root.killFeedPush === 'function') {
                        root.killFeedPush('🛡 MISSION CONTROL: FLEET OVERSHIELDS SYNCHRONIZED!', '#00E5FF');
                    }
                }
                return true;
            }
            default:
                console.warn('[Telemetry] Unknown command:', command);
                return false;
        }
    }

    function exportFlightRecorderJSON() {
        const payload = {
            exportTime: new Date().toISOString(),
            gameScore: typeof score !== 'undefined' ? score : 0,
            recordCount: flightRecorder.length,
            records: flightRecorder
        };
        const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2));
        if (typeof document !== 'undefined') {
            const dlAnchor = document.createElement('a');
            dlAnchor.setAttribute('href', dataStr);
            dlAnchor.setAttribute('download', `asteroids_telemetry_${Date.now()}.json`);
            document.body.appendChild(dlAnchor);
            dlAnchor.click();
            dlAnchor.remove();
        }
        return payload;
    }

    // Broadcast telemetry to parent window / Streamlit iframe
    function broadcastTelemetry() {
        if (typeof window === 'undefined' || !window.parent) return;
        const snapshot = captureTelemetrySnapshot();
        flightRecorder.push(snapshot);
        if (flightRecorder.length > MAX_FLIGHT_RECORDS) flightRecorder.shift();

        try {
            window.parent.postMessage({
                type: 'ASTEROIDS_TELEMETRY',
                payload: snapshot
            }, '*');
        } catch (e) {
            // Sandboxed environment fallback
        }
    }

    // Set up message listener for incoming commands from Streamlit
    if (typeof window !== 'undefined' && !root.__HEADLESS__) {
        window.addEventListener('message', function (event) {
            if (!event.data || typeof event.data !== 'object') return;
            if (event.data.type === 'COMMAND_INTERVENE') {
                dispatchTacticalCommand(event.data.command, event.data.payload);
            } else if (event.data.type === 'REQUEST_TELEMETRY_EXPORT') {
                exportFlightRecorderJSON();
            }
        });

        // Keyboard shortcut 'T' to toggle HUD
        window.addEventListener('keydown', function (e) {
            if (e.key === 't' || e.key === 'T') {
                enabled = !enabled;
                if (typeof root.killFeedPush === 'function') {
                    root.killFeedPush(enabled ? '📡 TELEMETRY HUD ENABLED' : '📡 TELEMETRY HUD DISABLED', '#00D2FF');
                }
            }
        });
    }

    // Periodic telemetry sampler hooked into drawHUD
    const origDrawHUD = drawHUD;
    root.drawHUD = function (ctx) {
        origDrawHUD(ctx);
        if (frame % telemetryBroadcastInterval === 0 && !root.__HEADLESS__) {
            broadcastTelemetry();
        }
    };

    root.AsteroidsTelemetry = {
        captureSnapshot: captureTelemetrySnapshot,
        dispatchCommand: dispatchTacticalCommand,
        exportJSON: exportFlightRecorderJSON,
        getFlightRecords: () => flightRecorder,
        setBroadcastInterval: (interval) => {
            telemetryBroadcastInterval = Math.max(1, Math.min(120, parseInt(interval) || 10));
        }
    };

})(typeof globalThis !== 'undefined' ? globalThis : this);

