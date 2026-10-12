// Visual effects layer: cached glow sprites, a pooled particle system, screen
// shake and hit flash. Load after theme.js and rng.js, before game.js.
//
// Design constraints:
//   - Purely cosmetic. Nothing here reads or writes game state, and game.js only
//     ever *emits* into it, so the headless trainer can disable it wholesale
//     without changing an episode. It draws from its own RNG stream, never from
//     gameRandom(), so a particle burst cannot perturb a seeded replay.
//   - No per-frame allocation on the hot path. Particles live in a fixed pool of
//     typed arrays; emission overwrites the oldest slot when the pool is full.
//   - No ctx.shadowBlur. A Canvas 2D shadow is re-rasterised for every stroke
//     and is the single most expensive thing this renderer used to do. Glows are
//     radial-gradient sprites rendered once per (colour, radius) and blitted
//     with additive compositing.

(function (root) {
    'use strict';

    const HEADLESS = !!root.__HEADLESS__;
    const MAX_PARTICLES = 2000;

    const rng = (root.AsteroidsRNG && root.AsteroidsRNG.createStream)
        ? root.AsteroidsRNG.createStream('fx-visual-only')
        : Math.random;

    let enabled = !HEADLESS;

    // ------------------------------------------------------------------
    // Glow sprites
    // ------------------------------------------------------------------
    const glowCache = new Map();

    function makeCanvas(w, h) {
        if (typeof OffscreenCanvas !== 'undefined') {
            try { return new OffscreenCanvas(w, h); } catch (e) { /* fall through */ }
        }
        if (typeof document !== 'undefined' && document.createElement) {
            const c = document.createElement('canvas');
            c.width = w;
            c.height = h;
            return c;
        }
        return null;
    }

    // Returns a sprite whose visible radius is `radius` (the canvas is 2x that
    // with the gradient fading to transparent at the edge).
    function glowSprite(color, radius) {
        const r = Math.max(2, Math.round(radius));
        const key = color + '|' + r;
        let sprite = glowCache.get(key);
        if (sprite) return sprite;
        const size = r * 2;
        const c = makeCanvas(size, size);
        if (!c) return null;
        const g = c.getContext('2d');
        const grad = g.createRadialGradient(r, r, 0, r, r, r);
        grad.addColorStop(0, color);
        grad.addColorStop(0.35, color);
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        g.fillStyle = grad;
        g.fillRect(0, 0, size, size);
        sprite = { canvas: c, size, r };
        glowCache.set(key, sprite);
        return sprite;
    }

    // Draw an additive glow centred on (x, y) in the current transform.
    function glow(ctx, x, y, radius, color, alpha) {
        if (!enabled || !ctx) return;
        const sprite = glowSprite(color, radius);
        if (!sprite) return;
        const prevOp = ctx.globalCompositeOperation;
        const prevAlpha = ctx.globalAlpha;
        ctx.globalCompositeOperation = 'lighter';
        ctx.globalAlpha = alpha === undefined ? 0.6 : alpha;
        ctx.drawImage(sprite.canvas, x - sprite.r, y - sprite.r);
        ctx.globalAlpha = prevAlpha;
        ctx.globalCompositeOperation = prevOp;
    }

    // ------------------------------------------------------------------
    // Particle pool
    // ------------------------------------------------------------------
    const px = new Float32Array(MAX_PARTICLES);
    const py = new Float32Array(MAX_PARTICLES);
    const pvx = new Float32Array(MAX_PARTICLES);
    const pvy = new Float32Array(MAX_PARTICLES);
    const plife = new Float32Array(MAX_PARTICLES);   // frames remaining
    const pmax = new Float32Array(MAX_PARTICLES);    // initial life, for fade
    const psize = new Float32Array(MAX_PARTICLES);
    const pdrag = new Float32Array(MAX_PARTICLES);
    const pkind = new Uint8Array(MAX_PARTICLES);     // 0 dot, 1 spark (line), 2 debris (polygon)
    const pcolor = new Array(MAX_PARTICLES).fill('#ffffff');
    let cursor = 0;
    let liveCount = 0;

    function emit(x, y, vx, vy, life, size, color, kind, drag) {
        if (!enabled) return;
        const i = cursor;
        cursor = (cursor + 1) % MAX_PARTICLES;
        if (plife[i] <= 0) liveCount++;
        px[i] = x; py[i] = y; pvx[i] = vx; pvy[i] = vy;
        plife[i] = life; pmax[i] = life; psize[i] = size;
        pcolor[i] = color; pkind[i] = kind || 0;
        pdrag[i] = drag === undefined ? 0.97 : drag;
    }

    function rand(min, max) { return min + rng() * (max - min); }

    // Thruster exhaust behind a ship. `angle` is the ship heading.
    function emitThrust(x, y, angle, color, power) {
        if (!enabled) return;
        const p = power || 1;
        const n = p > 1.2 ? 3 : 2;
        for (let k = 0; k < n; k++) {
            const spread = rand(-0.35, 0.35);
            const speed = rand(1.5, 3.5) * p;
            const a = angle + Math.PI + spread;
            emit(
                x + Math.cos(angle + Math.PI) * 12, y + Math.sin(angle + Math.PI) * 12,
                Math.cos(a) * speed, Math.sin(a) * speed,
                rand(10, 22), rand(1.5, 3), color, 0, 0.92
            );
        }
    }

    // Rock fragments when an asteroid is destroyed or splits.
    function emitDebris(x, y, radius, color) {
        if (!enabled) return;
        const n = Math.min(40, Math.max(8, Math.round(radius * 0.8)));
        for (let k = 0; k < n; k++) {
            const a = rand(0, Math.PI * 2);
            const speed = rand(0.5, 3.0) * Math.max(0.5, radius / 25);
            emit(x, y, Math.cos(a) * speed, Math.sin(a) * speed,
                rand(25, 55), rand(1, 3.5), color || '#9aa3b5', 2, 0.985);
        }
        for (let k = 0; k < Math.round(n / 2); k++) {
            const a = rand(0, Math.PI * 2);
            const speed = rand(2, 6);
            emit(x, y, Math.cos(a) * speed, Math.sin(a) * speed,
                rand(8, 16), rand(1, 2), '#ffffff', 1, 0.9);
        }
    }

    // Ship / enemy explosion. `scale` 1 = fighter, 3 = boss.
    function emitExplosion(x, y, color, scale) {
        if (!enabled) return;
        const s = scale || 1;
        const n = Math.round(30 * s);
        for (let k = 0; k < n; k++) {
            const a = rand(0, Math.PI * 2);
            const speed = rand(1, 5) * Math.sqrt(s);
            emit(x, y, Math.cos(a) * speed, Math.sin(a) * speed,
                rand(20, 45) * Math.sqrt(s), rand(1.5, 4) * Math.sqrt(s), color, 0, 0.95);
        }
        for (let k = 0; k < Math.round(12 * s); k++) {
            const a = rand(0, Math.PI * 2);
            const speed = rand(4, 9) * Math.sqrt(s);
            emit(x, y, Math.cos(a) * speed, Math.sin(a) * speed,
                rand(6, 14), rand(1, 2), '#ffffff', 1, 0.9);
        }
        flash(Math.min(0.5, 0.15 * s), color);
    }

    // Plasma stream between a docked tanker and the Alpha.
    function emitDockStream(x1, y1, x2, y2, color) {
        if (!enabled) return;
        const dx = x2 - x1;
        const dy = y2 - y1;
        for (let k = 0; k < 2; k++) {
            const t = rng();
            const jitter = rand(-3, 3);
            const nx = -dy;
            const ny = dx;
            const len = Math.sqrt(nx * nx + ny * ny) || 1;
            emit(
                x1 + dx * t + (nx / len) * jitter, y1 + dy * t + (ny / len) * jitter,
                dx * 0.02 + rand(-0.3, 0.3), dy * 0.02 + rand(-0.3, 0.3),
                rand(8, 18), rand(1, 2.5), color, 0, 0.9
            );
        }
    }

    // Expanding ring of sparks for the HELP ping.
    function emitRipple(x, y, color) {
        if (!enabled) return;
        const n = 24;
        for (let k = 0; k < n; k++) {
            const a = (k / n) * Math.PI * 2;
            emit(x, y, Math.cos(a) * 2.2, Math.sin(a) * 2.2, 30, 1.5, color, 0, 1.0);
        }
    }

    // Small flash at a gun muzzle.
    function emitMuzzle(x, y, angle, color) {
        if (!enabled) return;
        for (let k = 0; k < 3; k++) {
            const a = angle + rand(-0.25, 0.25);
            const speed = rand(2, 4);
            emit(x, y, Math.cos(a) * speed, Math.sin(a) * speed, rand(4, 8), 1.5, color, 1, 0.85);
        }
    }

    function updateParticles() {
        if (!enabled) return;
        for (let i = 0; i < MAX_PARTICLES; i++) {
            if (plife[i] <= 0) continue;
            plife[i] -= 1;
            if (plife[i] <= 0) { liveCount--; continue; }
            px[i] += pvx[i];
            py[i] += pvy[i];
            pvx[i] *= pdrag[i];
            pvy[i] *= pdrag[i];
        }
    }

    // Visit live particles (used by sphere_view.js to draw them in 3D; the pool
    // stays chart-space so emitters need no changes).
    function forEachParticle(fn) {
        if (!enabled || liveCount === 0) return;
        for (let i = 0; i < MAX_PARTICLES; i++) {
            if (plife[i] <= 0) continue;
            fn(px[i], py[i], plife[i] / pmax[i], pcolor[i], psize[i], pkind[i]);
        }
    }

    function drawParticles(ctx) {
        if (!enabled || !ctx || liveCount === 0) return;
        const prevOp = ctx.globalCompositeOperation;
        const prevAlpha = ctx.globalAlpha;
        ctx.globalCompositeOperation = 'lighter';
        ctx.lineWidth = 1;
        for (let i = 0; i < MAX_PARTICLES; i++) {
            if (plife[i] <= 0) continue;
            const t = plife[i] / pmax[i];
            ctx.globalAlpha = Math.max(0, Math.min(1, t));
            const kind = pkind[i];
            if (kind === 1) {
                ctx.strokeStyle = pcolor[i];
                ctx.beginPath();
                ctx.moveTo(px[i], py[i]);
                ctx.lineTo(px[i] - pvx[i] * 2, py[i] - pvy[i] * 2);
                ctx.stroke();
            } else if (kind === 2) {
                ctx.fillStyle = pcolor[i];
                const s = psize[i] * (0.5 + 0.5 * t);
                ctx.fillRect(px[i] - s / 2, py[i] - s / 2, s, s);
            } else {
                ctx.fillStyle = pcolor[i];
                const s = psize[i] * t;
                ctx.beginPath();
                ctx.arc(px[i], py[i], Math.max(0.3, s), 0, Math.PI * 2);
                ctx.fill();
            }
        }
        ctx.globalAlpha = prevAlpha;
        ctx.globalCompositeOperation = prevOp;
    }

    // ------------------------------------------------------------------
    // Screen shake and hit flash
    // ------------------------------------------------------------------
    let shakeMag = 0;
    let shakeFrames = 0;
    let shakeX = 0;
    let shakeY = 0;
    let flashAlpha = 0;
    let flashColor = '#ffffff';

    function shake(magnitude, frames) {
        if (!enabled) return;
        shakeMag = Math.max(shakeMag, magnitude);
        shakeFrames = Math.max(shakeFrames, frames);
    }

    function flash(alpha, color) {
        if (!enabled) return;
        if (alpha > flashAlpha) {
            flashAlpha = alpha;
            flashColor = color || '#ffffff';
        }
    }

    function updateShake() {
        if (shakeFrames > 0) {
            shakeFrames--;
            const decay = shakeFrames > 0 ? shakeMag : 0;
            shakeX = rand(-decay, decay);
            shakeY = rand(-decay, decay);
            shakeMag *= 0.9;
            if (shakeFrames === 0) { shakeMag = 0; shakeX = 0; shakeY = 0; }
        }
        if (flashAlpha > 0) flashAlpha = Math.max(0, flashAlpha - 0.04);
    }

    // Call at the top of draw(), after clearing. Applies the shake transform.
    function beginFrame(ctx) {
        if (!enabled || !ctx) return;
        ctx.save();
        if (shakeX !== 0 || shakeY !== 0) ctx.translate(shakeX, shakeY);
    }

    // Call at the end of draw(). Restores the transform and paints the flash.
    function endFrame(ctx, width, height) {
        if (!enabled || !ctx) return;
        ctx.restore();
        if (flashAlpha > 0) {
            const prevAlpha = ctx.globalAlpha;
            ctx.globalAlpha = flashAlpha;
            ctx.fillStyle = flashColor;
            ctx.fillRect(0, 0, width, height);
            ctx.globalAlpha = prevAlpha;
        }
    }

    // ------------------------------------------------------------------
    // Parallax starfield (three depth layers, deterministic layout)
    // ------------------------------------------------------------------
    const STAR_LAYERS = [
        { count: 70, speed: 0.02, size: 1, alpha: 0.35 },
        { count: 40, speed: 0.05, size: 1, alpha: 0.6 },
        { count: 18, speed: 0.10, size: 2, alpha: 0.9 }
    ];
    let starFrame = 0;

    function drawStarfield(ctx, width, height) {
        if (!ctx) return;
        starFrame++;
        const prevAlpha = ctx.globalAlpha;
        for (let L = 0; L < STAR_LAYERS.length; L++) {
            const layer = STAR_LAYERS[L];
            ctx.globalAlpha = layer.alpha;
            ctx.fillStyle = '#ffffff';
            const drift = starFrame * layer.speed;
            for (let i = 0; i < layer.count; i++) {
                // Fixed pseudo-random layout from index; drifts slowly rightward.
                const bx = ((i * 7919 + L * 104729) % 10007) / 10007 * width;
                const by = ((i * 7907 + L * 1301) % 10009) / 10009 * height;
                const x = (bx + drift) % width;
                const tw = L === 2 ? 0.7 + 0.3 * Math.sin(starFrame * 0.05 + i) : 1;
                ctx.globalAlpha = layer.alpha * tw;
                ctx.fillRect(x, by, layer.size, layer.size);
            }
        }
        ctx.globalAlpha = prevAlpha;
    }

    // ------------------------------------------------------------------
    // Frame tick (call once per update())
    // ------------------------------------------------------------------
    function update() {
        if (!enabled) return;
        updateParticles();
        updateShake();
    }

    function setEnabled(on) {
        enabled = !!on && !HEADLESS;
        if (!enabled) {
            plife.fill(0);
            liveCount = 0;
            shakeMag = 0; shakeFrames = 0; shakeX = 0; shakeY = 0; flashAlpha = 0;
        }
    }

    function stats() {
        return { enabled, particles: liveCount, glowSprites: glowCache.size };
    }

    root.FX = {
        glow,
        emitThrust,
        emitDebris,
        emitExplosion,
        emitDockStream,
        emitRipple,
        emitMuzzle,
        shake,
        flash,
        update,
        beginFrame,
        endFrame,
        drawParticles,
        forEachParticle,
        drawStarfield,
        setEnabled,
        stats,
        MAX_PARTICLES
    };
})(typeof globalThis !== 'undefined' ? globalThis : this);
