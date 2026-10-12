// world.js - world geometry shared by the browser game and the headless simulator.
//
// The game state keeps every object in chart coordinates (x, y, angle, velocity)
// sized by the canvas. This module decides what that chart *means*:
//
//   torus  (default) - the classic flat rectangle whose edges wrap. Behaviour is
//                      byte-identical to the original game; nothing here runs on
//                      the hot path in this mode.
//   sphere           - the chart is an equirectangular map of a sphere:
//                      x is longitude (0..W -> -180..180 deg), y is latitude
//                      (0 -> north pole, H -> south pole). One chart unit along y
//                      is one unit of arc, so R = H / PI. Objects move along great
//                      circles (their heading is parallel-transported), x wraps at
//                      the antimeridian and crossing a pole reflects y and shifts x
//                      by half a turn. Distances are arc lengths.
//
// game.js branches into WORLD only when the sphere is active, so the headless
// training sim and the browser share this one implementation and cannot drift.
// The 3D view (sphere_view.js) uses toWorld3()/frameAt() to place the same state
// on a Three.js sphere.
(function (root) {
    'use strict';

    const TWO_PI = Math.PI * 2;
    let mode = 'torus';
    // Size provider; game.js binds this to the live canvas so a resize is picked
    // up automatically. The default matches sim/browser_shim.js.
    let sizeFn = function () { return { w: 1200, h: 600 }; };
    // World scale (sphere only): the logical chart is `scale` x the canvas, so
    // the globe can be bigger than the screen. R scales with it; ship sizes,
    // speeds and formation offsets do not, which is what buys room. At scale 1
    // a 1200x600 canvas gives R = 191 and 36% *less* surface than the flat map
    // (4piR^2 vs W*H); scale 2 gives 4x that area. Ignored on the torus, where
    // the canvas is the viewport and must equal the world.
    let scale = 1;

    function size() {
        const s = sizeFn();
        if (mode !== 'sphere' || scale === 1) return s;
        return { w: s.w * scale, h: s.h * scale };
    }
    function radius(h) { return h / Math.PI; }
    function lat(y, h) { return (0.5 - y / h) * Math.PI; }
    function lon(x, w) { return (x / w - 0.5) * TWO_PI; }

    // Unit position vector and local east/north tangent basis at chart (x, y).
    // Right-handed for Three.js (y up): seen from outside with north up, east is
    // to the right, so the sphere reads like the flat map.
    function frameAt(x, y, w, h) {
        const phi = lat(y, h);
        const lam = lon(x, w);
        const cp = Math.cos(phi), sp = Math.sin(phi);
        const cl = Math.cos(lam), sl = Math.sin(lam);
        return {
            p: [cp * cl, sp, -cp * sl],
            e: [-sl, 0, -cl],
            n: [-sp * cl, cp, sp * sl]
        };
    }

    function fromUnit(p, w, h) {
        const py = Math.max(-1, Math.min(1, p[1]));
        const phi = Math.asin(py);
        const lam = Math.atan2(-p[2], p[0]);
        let x = (lam / TWO_PI + 0.5) * w;
        if (x >= w) x -= w;
        if (x < 0) x += w;
        const y = Math.max(0, Math.min(h, (0.5 - phi / Math.PI) * h));
        return { x: x, y: y };
    }

    function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    function cross(a, b) {
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    }

    function wrapAngle(a) {
        while (a > Math.PI) a -= TWO_PI;
        while (a < -Math.PI) a += TWO_PI;
        return a;
    }

    // Wrap an object back into the chart. `margin` is the torus overhang used by
    // asteroids (they leave fully before re-entering). On the sphere there is no
    // edge: x is periodic and y reflects through the pole.
    function wrap(obj, margin) {
        const s = size();
        margin = margin || 0;
        if (mode === 'sphere') {
            if (obj.y < 0 || obj.y > s.h) {
                obj.y = obj.y < 0 ? -obj.y : 2 * s.h - obj.y;
                obj.x += s.w / 2;
                if (typeof obj.angle === 'number') obj.angle += Math.PI;
                if (typeof obj.velocityX === 'number') { obj.velocityX = -obj.velocityX; obj.velocityY = -obj.velocityY; }
            }
            obj.x = ((obj.x % s.w) + s.w) % s.w;
            return;
        }
        if (obj.x < -margin) obj.x = s.w + margin;
        if (obj.x > s.w + margin) obj.x = -margin;
        if (obj.y < -margin) obj.y = s.h + margin;
        if (obj.y > s.h + margin) obj.y = -margin;
    }

    // Advance obj by the chart-frame velocity (vx, vy) [chart units per frame,
    // canvas convention: +y is down/south] and return the transported velocity and
    // the heading change. On the torus this is a plain translation. On the sphere
    // it is one exact great-circle step: rotate the position about the axis
    // perpendicular to the motion, parallel-transport velocity and heading, and
    // express both in the new local frame.
    function step(obj, vx, vy) {
        const s = size();
        if (mode !== 'sphere') {
            obj.x += vx;
            obj.y += vy;
            return { vx: vx, vy: vy, dAngle: 0 };
        }
        const speed = Math.sqrt(vx * vx + vy * vy);
        if (speed < 1e-9) return { vx: vx, vy: vy, dAngle: 0 };

        // Keep off the exact pole where east is undefined.
        if (obj.y <= 1e-6) obj.y = 1e-6;
        if (obj.y >= s.h - 1e-6) obj.y = s.h - 1e-6;

        const R = radius(s.h);
        const f = frameAt(obj.x, obj.y, s.w, s.h);
        const P = f.p, E = f.e, N = f.n;
        // Tangent direction of travel (canvas +y is south = -N).
        const t = [
            (vx * E[0] - vy * N[0]) / speed,
            (vx * E[1] - vy * N[1]) / speed,
            (vx * E[2] - vy * N[2]) / speed
        ];
        const k = cross(P, t); // rotation axis, unit and tangent
        const a = speed / R;
        const ca = Math.cos(a), sa = Math.sin(a);
        const kxp = cross(k, P);
        const P2 = [P[0] * ca + kxp[0] * sa, P[1] * ca + kxp[1] * sa, P[2] * ca + kxp[2] * sa];
        const t2 = [t[0] * ca - P[0] * sa, t[1] * ca - P[1] * sa, t[2] * ca - P[2] * sa];

        const heading = typeof obj.angle === 'number' ? obj.angle : 0;
        // Heading as a tangent vector, decomposed along (t, k) which transport
        // to (t2, k).
        const hd = [
            Math.cos(heading) * E[0] - Math.sin(heading) * N[0],
            Math.cos(heading) * E[1] - Math.sin(heading) * N[1],
            Math.cos(heading) * E[2] - Math.sin(heading) * N[2]
        ];
        const ht = dot(hd, t), hk = dot(hd, k);
        const hd2 = [ht * t2[0] + hk * k[0], ht * t2[1] + hk * k[1], ht * t2[2] + hk * k[2]];

        const c2 = fromUnit(P2, s.w, s.h);
        obj.x = c2.x;
        obj.y = c2.y;
        const f2 = frameAt(obj.x, obj.y, s.w, s.h);
        const nvx = speed * dot(t2, f2.e);
        const nvy = -speed * dot(t2, f2.n);
        const heading2 = Math.atan2(-dot(hd2, f2.n), dot(hd2, f2.e));
        return { vx: nvx, vy: nvy, dAngle: wrapAngle(heading2 - heading) };
    }

    // Move a body that carries velocityX/velocityY (ships, asteroids, enemies).
    function moveBody(obj, margin) {
        const r = step(obj, obj.velocityX, obj.velocityY);
        obj.velocityX = r.vx;
        obj.velocityY = r.vy;
        if (typeof obj.angle === 'number') obj.angle += r.dAngle;
        wrap(obj, margin);
    }

    // Move a projectile that travels along its heading at `speed`.
    function moveAlongHeading(obj, speed) {
        const r = step(obj, Math.cos(obj.angle) * speed, Math.sin(obj.angle) * speed);
        obj.angle += r.dAngle;
        wrap(obj, 0);
    }

    // Surface distance between two chart points. Torus mode returns the plain
    // Euclidean distance the game has always used (no seam wrapping), so existing
    // behaviour is unchanged.
    function distance(x1, y1, x2, y2) {
        if (mode !== 'sphere') {
            const dx = x1 - x2, dy = y1 - y2;
            return Math.sqrt(dx * dx + dy * dy);
        }
        const s = size();
        const a = frameAt(x1, y1, s.w, s.h).p;
        const b = frameAt(x2, y2, s.w, s.h).p;
        const d = Math.max(-1, Math.min(1, dot(a, b)));
        return radius(s.h) * Math.acos(d);
    }

    // Offset of (tx, ty) as seen from (fx, fy) in the local canvas-convention
    // frame at the origin: dx along local east, dy along local south, scaled so
    // sqrt(dx^2 + dy^2) is the arc distance. Torus mode returns the raw difference.
    function delta(fx, fy, tx, ty) {
        if (mode !== 'sphere') return { dx: tx - fx, dy: ty - fy };
        const s = size();
        const f = frameAt(fx, fy, s.w, s.h);
        const b = frameAt(tx, ty, s.w, s.h).p;
        const c = Math.max(-1, Math.min(1, dot(f.p, b)));
        const arc = radius(s.h) * Math.acos(c);
        // Tangent component of b at f.p.
        const bt = [b[0] - c * f.p[0], b[1] - c * f.p[1], b[2] - c * f.p[2]];
        const m = Math.sqrt(dot(bt, bt));
        if (m < 1e-12) return { dx: 0, dy: 0 };
        return { dx: arc * dot(bt, f.e) / m, dy: -arc * dot(bt, f.n) / m };
    }

    // Uniformly distributed random chart point, optionally at least `minDist`
    // (arc units) from (ax, ay). Used for spawning on the sphere, where "the edge
    // of the screen" would funnel everything through the poles.
    function randomPoint(rand, ax, ay, minDist) {
        const s = size();
        for (let i = 0; i < 32; i++) {
            const u = rand(), v = rand();
            const phi = Math.asin(2 * v - 1);
            const pt = { x: u * s.w, y: (0.5 - phi / Math.PI) * s.h };
            if (minDist === undefined || distance(pt.x, pt.y, ax, ay) >= minDist) return pt;
        }
        return { x: rand() * s.w, y: rand() * s.h };
    }

    // 3D position (Three.js axes, world radius R) of a local canvas-frame offset
    // (cx, cy) attached to chart point (x, y), lifted `lift` units off the surface.
    function toWorld3(x, y, cx, cy, lift) {
        const s = size();
        const R = radius(s.h) + (lift || 0);
        const f = frameAt(x, y, s.w, s.h);
        cx = cx || 0; cy = cy || 0;
        return [
            f.p[0] * R + cx * f.e[0] - cy * f.n[0],
            f.p[1] * R + cx * f.e[1] - cy * f.n[1],
            f.p[2] * R + cx * f.e[2] - cy * f.n[2]
        ];
    }

    const WORLD = {
        setMode(m) { mode = m === 'sphere' ? 'sphere' : 'torus'; },
        getMode() { return mode; },
        isSphere() { return mode === 'sphere'; },
        configure(fn) { if (typeof fn === 'function') sizeFn = fn; },
        setScale(k) { const v = Number(k); scale = (isFinite(v) && v > 0) ? Math.max(0.5, Math.min(4, v)) : 1; },
        getScale() { return scale; },
        // Scale actually in effect (1 on the torus).
        effectiveScale() { return mode === 'sphere' ? scale : 1; },
        size: size,
        radius() { return radius(size().h); },
        lat(y) { return lat(y, size().h); },
        lon(x) { return lon(x, size().w); },
        frameAt(x, y) { const s = size(); return frameAt(x, y, s.w, s.h); },
        fromUnit(p) { const s = size(); return fromUnit(p, s.w, s.h); },
        wrap: wrap,
        step: step,
        moveBody: moveBody,
        moveAlongHeading: moveAlongHeading,
        distance: distance,
        delta: delta,
        randomPoint: randomPoint,
        toWorld3: toWorld3,
        wrapAngle: wrapAngle
    };

    root.WORLD = WORLD;
    if (typeof module !== 'undefined' && module.exports) module.exports = WORLD;
})(typeof globalThis !== 'undefined' ? globalThis : this);
