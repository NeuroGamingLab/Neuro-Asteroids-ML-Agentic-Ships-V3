// sphere_view.js - Three.js renderer for the spherical world.
//
// The simulation (game.js) is untouched: objects live in chart coordinates and
// world.js turns those into great-circle motion. This module only *draws* that
// state on a sphere. Every object is rebuilt each frame into one dynamic
// LineSegments buffer (vector-display look, one draw call) plus two Points
// clouds (bullets, fx particles), so per-frame cost is a few thousand vertex
// writes rather than scene-graph churn.
//
// Layering: the game's own <canvas> has an opaque 2D context, so a WebGL canvas
// is stacked over it and a transparent overlay canvas over that. game.js points
// `ctx` at the overlay while the sphere is active, so the HUD, policy labels and
// minimap keep drawing with unchanged code.
//
// Requires THREE (r128 CDN build) and WORLD (world.js). Without either,
// init() fails soft and game.js stays on its 2D path.
(function (root) {
    'use strict';

    const MAX_SEGMENTS = 24000;
    const MAX_POINTS = 4096;
    const THEME = root.GAME_THEME || {};

    let ready = false;
    let host = null, glCanvas = null, overlayCanvas = null, overlayCtx = null;
    let renderer = null, scene = null, camera = null, keyLight = null;
    let lines = null, linePos = null, lineCol = null, lineCount = 0;
    let points = null, pointPos = null, pointCol = null, pointCount = 0;
    let gridGroup = null, stars = null, globe = null, halo = null;
    // Real, lit 3D meshes for AI/player ship hulls and boss hulls (basic/
    // advanced enemies, asteroids and bullets stay in the flat, unlit
    // vector-line buffers above -- see updateShipHullMesh()/
    // updateBossHullMesh() for why these two get the special treatment).
    // Keyed by the ship/boss object itself so a mesh survives across frames
    // without needing a stable id; reconciled once per render() against the
    // live list so a destroyed one's mesh/geometry/material actually get
    // disposed instead of accumulating over a long session.
    const shipMeshPool = new Map();
    let W = 0, H = 0, R = 0;
    let lastLeft = -1, lastTop = -1;

    // Camera state. `camDir` is the unit vector the camera sits on; smoothed
    // toward the follow target so the sphere appears to roll under the fleet.
    let followMode = 'alpha';      // 'alpha' | 'fleet' | 'orbit'
    // Requested camera distance in radii *of the unscaled globe*. The
    // effective distance shrinks as the world scale grows so that, at a given
    // setting, a ship keeps the same on-screen size: (zoom-1)*R is the height
    // of the camera above the surface, and pixel scale = focal / that.
    //
    // 4.2 (not the visually-nicer-sounding "2.6") is deliberate: at the
    // default world scale (2x) and the 46-degree vertical FOV, the globe's
    // full angular diameter needs effectiveZoom() >= ~2.56 to fit inside the
    // frame at all -- effectiveZoom = 1 + (zoomReq-1)/scale, so with scale=2
    // that means zoomReq needs to clear ~4.1 just to stop the globe
    // overflowing the view. 2.6 (effectiveZoom 1.8) rendered a permanent
    // close-up of the surface with no horizon ever visible, regardless of
    // canvas size -- this is an angular relationship, not a pixel-resolution
    // one, so no amount of "make the canvas taller" fixes it.
    let zoomReq = 4.2;
    let zoom = 4.2;                // effective camera distance in radii (derived)
    function effectiveZoom() {
        const k = root.WORLD.effectiveScale ? root.WORLD.effectiveScale() : 1;
        return Math.max(1.3, Math.min(4.5, 1 + (zoomReq - 1) / k));
    }
    let camDir = [1, 0, 0];
    let camUp = [0, 1, 0];
    let orbitAngle = 0;
    let hiddenCount = 0;
    let frame = 0;

    const colorCache = new Map();
    const tmpColor = () => new root.THREE.Color();
    let scratchColor = null;
    function rgb(css) {
        let c = colorCache.get(css);
        if (c) return c;
        if (!scratchColor) scratchColor = tmpColor();
        try { scratchColor.setStyle(css); } catch (e) { scratchColor.setRGB(1, 1, 1); }
        c = [scratchColor.r, scratchColor.g, scratchColor.b];
        colorCache.set(css, c);
        return c;
    }
    function dim(c, k) { return [c[0] * k, c[1] * k, c[2] * k]; }

    // ------------------------------------------------------------------
    // Setup
    // ------------------------------------------------------------------
    function init(hostCanvas) {
        if (ready) return true;
        if (typeof root.THREE === 'undefined' || typeof root.WORLD === 'undefined' || !hostCanvas) return false;
        const THREE = root.THREE;
        host = hostCanvas;
        W = host.width; H = host.height;
        R = root.WORLD.radius();

        const parent = host.parentNode;
        if (parent && parent.style) parent.style.position = parent.style.position || 'relative';

        glCanvas = document.createElement('canvas');
        glCanvas.id = 'sphereCanvas';
        overlayCanvas = document.createElement('canvas');
        overlayCanvas.id = 'sphereOverlay';
        for (const c of [glCanvas, overlayCanvas]) {
            c.width = W; c.height = H;
            c.style.position = 'absolute';
            c.style.pointerEvents = 'none';
            c.style.zIndex = c === glCanvas ? '5' : '6';
            c.style.boxSizing = 'border-box';
        }
        if (parent) { parent.insertBefore(glCanvas, host.nextSibling); parent.insertBefore(overlayCanvas, glCanvas.nextSibling); }
        overlayCtx = overlayCanvas.getContext('2d');

        try {
            // preserveDrawingBuffer lets the page (and users) read the frame back
            // with toDataURL for screenshots; the cost is one buffer copy per frame.
            renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: true, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: true });
        } catch (e) {
            console.error('[Sphere] WebGL unavailable', e);
            return false;
        }
        renderer.setPixelRatio(Math.min(root.devicePixelRatio || 1, 2));
        renderer.setSize(W, H, false);
        renderer.setClearColor(new THREE.Color(THEME.bg || '#000000'), 1);

        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(46, W / H, 1, R * 80);
        scene.add(camera);

        scene.add(new THREE.AmbientLight(0x6070a0, 0.55));
        keyLight = new THREE.DirectionalLight(0xffffff, 0.7);
        keyLight.position.set(R * 1.6, R * 1.4, R * 0.8); // camera space: upper-right
        camera.add(keyLight);

        buildGlobe();
        buildGrid();
        buildStars();
        buildDynamicBuffers();
        placeCanvases();
        ready = true;
        console.log('[Sphere] view ready R=' + R.toFixed(1) + ' (' + W + 'x' + H + ')');
        return true;
    }

    function buildGlobe() {
        const THREE = root.THREE;
        const geo = new THREE.SphereGeometry(R * 0.996, 96, 48);
        // Lambert: a soft day/night gradient from the key light, no specular
        // hot-spot competing with the vector objects.
        const mat = new THREE.MeshLambertMaterial({ color: 0x0c1530, emissive: 0x04060e });
        globe = new THREE.Mesh(geo, mat);
        scene.add(globe);

        // Thin atmospheric rim: a back-facing shell drawn additively.
        const haloGeo = new THREE.SphereGeometry(R * 1.045, 64, 32);
        const haloMat = new THREE.MeshBasicMaterial({
            color: 0x2050c0, transparent: true, opacity: 0.16, side: THREE.BackSide,
            blending: THREE.AdditiveBlending, depthWrite: false
        });
        halo = new THREE.Mesh(haloGeo, haloMat);
        scene.add(halo);
    }

    function buildGrid() {
        const THREE = root.THREE;
        const WORLD = root.WORLD;
        const pos = [];
        const col = [];
        const faint = rgb('#1d4470');
        const bright = rgb('#2f6fae');
        const lift = 0.8;
        const push = (a, b, c) => { pos.push(a[0], a[1], a[2], b[0], b[1], b[2]); col.push(c[0], c[1], c[2], c[0], c[1], c[2]); };
        // Chart coordinates run over the logical world, not the canvas.
        const ws = WORLD.size();
        const CW = ws.w, CH = ws.h;
        // Latitude circles every 30 deg (skip poles).
        for (let latDeg = -60; latDeg <= 60; latDeg += 30) {
            const y = (0.5 - (latDeg * Math.PI / 180) / Math.PI) * CH;
            const c = latDeg === 0 ? bright : faint;
            const n = 96;
            for (let i = 0; i < n; i++) {
                const a = WORLD.toWorld3((i / n) * CW, y, 0, 0, lift);
                const b = WORLD.toWorld3(((i + 1) / n) * CW, y, 0, 0, lift);
                push(a, b, c);
            }
        }
        // Meridians every 30 deg.
        for (let lonDeg = -180; lonDeg < 180; lonDeg += 30) {
            const x = (lonDeg / 360 + 0.5) * CW;
            const c = lonDeg === 0 ? bright : faint;
            const n = 48;
            for (let i = 0; i < n; i++) {
                const a = WORLD.toWorld3(x, (i / n) * CH, 0, 0, lift);
                const b = WORLD.toWorld3(x, ((i + 1) / n) * CH, 0, 0, lift);
                push(a, b, c);
            }
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
        gridGroup = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }));
        scene.add(gridGroup);
    }

    function buildStars() {
        const THREE = root.THREE;
        const n = 1800;
        const pos = new Float32Array(n * 3);
        const col = new Float32Array(n * 3);
        // Fixed-seed LCG so the sky is the same every load.
        let s = 1234567;
        const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
        for (let i = 0; i < n; i++) {
            const u = rnd() * 2 - 1, t = rnd() * Math.PI * 2;
            const rr = Math.sqrt(1 - u * u);
            const d = R * (30 + rnd() * 20);
            pos[i * 3] = rr * Math.cos(t) * d;
            pos[i * 3 + 1] = u * d;
            pos[i * 3 + 2] = rr * Math.sin(t) * d;
            const b = 0.35 + rnd() * 0.65;
            const tint = rnd();
            col[i * 3] = b * (tint < 0.15 ? 1.0 : 0.85);
            col[i * 3 + 1] = b * 0.9;
            col[i * 3 + 2] = b * (tint > 0.7 ? 1.0 : 0.95);
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
        stars = new THREE.Points(geo, new THREE.PointsMaterial({ size: 1.6, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false }));
        scene.add(stars);
    }

    function buildDynamicBuffers() {
        const THREE = root.THREE;
        linePos = new Float32Array(MAX_SEGMENTS * 6);
        lineCol = new Float32Array(MAX_SEGMENTS * 6);
        const lg = new THREE.BufferGeometry();
        lg.setAttribute('position', new THREE.BufferAttribute(linePos, 3).setUsage(THREE.DynamicDrawUsage));
        lg.setAttribute('color', new THREE.BufferAttribute(lineCol, 3).setUsage(THREE.DynamicDrawUsage));
        lg.setDrawRange(0, 0);
        lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 1.0 }));
        lines.frustumCulled = false;
        scene.add(lines);

        pointPos = new Float32Array(MAX_POINTS * 3);
        pointCol = new Float32Array(MAX_POINTS * 3);
        const pg = new THREE.BufferGeometry();
        pg.setAttribute('position', new THREE.BufferAttribute(pointPos, 3).setUsage(THREE.DynamicDrawUsage));
        pg.setAttribute('color', new THREE.BufferAttribute(pointCol, 3).setUsage(THREE.DynamicDrawUsage));
        pg.setDrawRange(0, 0);
        points = new THREE.Points(pg, new THREE.PointsMaterial({
            size: 3.2, sizeAttenuation: false, vertexColors: true, transparent: true,
            blending: THREE.AdditiveBlending, depthWrite: false
        }));
        points.frustumCulled = false;
        scene.add(points);
    }

    function placeCanvases() {
        if (!host || !glCanvas) return;
        const left = host.offsetLeft, top = host.offsetTop;
        if (left === lastLeft && top === lastTop && glCanvas.width === host.width && glCanvas.height === host.height) return;
        lastLeft = left; lastTop = top;
        const cs = root.getComputedStyle ? root.getComputedStyle(host) : null;
        for (const c of [glCanvas, overlayCanvas]) {
            c.style.left = left + 'px';
            c.style.top = top + 'px';
            c.style.width = host.offsetWidth + 'px';
            c.style.height = host.offsetHeight + 'px';
            if (cs) { c.style.borderRadius = cs.borderRadius; }
        }
        if (cs) { glCanvas.style.border = cs.border; }
        if (host.width !== W || host.height !== H) {
            W = host.width; H = host.height;
            glCanvas.width = W; glCanvas.height = H;
            overlayCanvas.width = W; overlayCanvas.height = H;
            renderer.setSize(W, H, false);
            camera.aspect = W / H;
            camera.updateProjectionMatrix();
        }
        syncWorld();
    }

    // The globe radius follows WORLD.size() (canvas x world scale). When it
    // changes -- resize, or setWorldScale() from the GUI -- rebuild everything
    // that was sized from R and keep the camera's far plane / light in step.
    function syncWorld() {
        const r = root.WORLD.radius();
        if (r === R) return;
        R = r;
        scene.remove(globe); scene.remove(halo); scene.remove(gridGroup); scene.remove(stars);
        buildGlobe(); buildGrid(); buildStars();
        camera.far = R * 80;
        camera.updateProjectionMatrix();
        keyLight.position.set(R * 1.6, R * 1.4, R * 0.8);
        console.log('[Sphere] world radius -> ' + R.toFixed(1) + ' (scale ' + root.WORLD.getScale() + ')');
    }

    function setVisible(on) {
        if (!glCanvas) return;
        glCanvas.style.display = on ? 'block' : 'none';
        overlayCanvas.style.display = on ? 'block' : 'none';
    }

    // ------------------------------------------------------------------
    // Primitive emitters into the dynamic buffers
    // ------------------------------------------------------------------
    function seg(a, b, c) {
        if (lineCount >= MAX_SEGMENTS) return;
        const i = lineCount * 6;
        linePos[i] = a[0]; linePos[i + 1] = a[1]; linePos[i + 2] = a[2];
        linePos[i + 3] = b[0]; linePos[i + 4] = b[1]; linePos[i + 5] = b[2];
        lineCol[i] = c[0]; lineCol[i + 1] = c[1]; lineCol[i + 2] = c[2];
        lineCol[i + 3] = c[0]; lineCol[i + 4] = c[1]; lineCol[i + 5] = c[2];
        lineCount++;
    }
    function dotp(p, c) {
        if (pointCount >= MAX_POINTS) return;
        const i = pointCount * 3;
        pointPos[i] = p[0]; pointPos[i + 1] = p[1]; pointPos[i + 2] = p[2];
        pointCol[i] = c[0]; pointCol[i + 1] = c[1]; pointCol[i + 2] = c[2];
        pointCount++;
    }
    // Polygon given in the object's local canvas frame (+u = heading).
    function poly(x, y, pts, angle, lift, c, closed) {
        const WORLD = root.WORLD;
        const ca = Math.cos(angle), sa = Math.sin(angle);
        let prev = null, first = null;
        for (let i = 0; i < pts.length; i++) {
            const u = pts[i][0], v = pts[i][1];
            const p = WORLD.toWorld3(x, y, u * ca - v * sa, u * sa + v * ca, lift);
            if (prev) seg(prev, p, c);
            else first = p;
            prev = p;
        }
        if (closed && first && prev) seg(prev, first, c);
    }
    function circle(x, y, r, lift, c, n) {
        n = n || 28;
        const pts = [];
        for (let i = 0; i < n; i++) { const t = (i / n) * Math.PI * 2; pts.push([Math.cos(t) * r, Math.sin(t) * r]); }
        poly(x, y, pts, 0, lift, c, true);
    }
    // Great-circle arc between two chart points.
    function arc(x1, y1, x2, y2, lift, c, n, dashed) {
        const WORLD = root.WORLD;
        const a = WORLD.frameAt(x1, y1).p, b = WORLD.frameAt(x2, y2).p;
        const d = Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
        const om = Math.acos(d);
        if (om < 1e-6) return;
        const so = Math.sin(om);
        const rr = R + (lift || 0);
        n = n || Math.max(4, Math.ceil(om * R / 12));
        let prev = null;
        for (let i = 0; i <= n; i++) {
            const t = i / n;
            const k1 = Math.sin((1 - t) * om) / so, k2 = Math.sin(t * om) / so;
            const p = [(k1 * a[0] + k2 * b[0]) * rr, (k1 * a[1] + k2 * b[1]) * rr, (k1 * a[2] + k2 * b[2]) * rr];
            if (prev && (!dashed || (i % 2 === 1))) seg(prev, p, c);
            prev = p;
        }
    }

    // ------------------------------------------------------------------
    // Object drawing
    // ------------------------------------------------------------------
    const SHIP_TRI = (s) => [[s, 0], [-s / 2, -s / 2], [-s / 2, s / 2]];
    const ENEMY_DIAMOND = (s) => [[s, 0], [0, -s * 0.6], [-s * 0.7, 0], [0, s * 0.6]];
    const BOSS_HEX = (s) => { const p = []; for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; p.push([Math.cos(t) * s, Math.sin(t) * s]); } return p; };
    const SHIP_HULL_LIFT = 2; // matches the lift the old flat outline drew at

    // Ship hull as a small solid: same nose/back-corner footprint as SHIP_TRI,
    // plus a raised "spine" apex, giving four triangular faces that catch
    // keyLight differently as the ship turns -- the actual "3D shading" ask.
    // Non-indexed (each face owns 3 unique array slots even where positions
    // coincide) so computeVertexNormals() below gives flat per-face normals
    // without extra bookkeeping.
    function shipHullTriangles(s) {
        const nose = [s, 0, 0];
        const backL = [-s / 2, -s / 2, 0];
        const backR = [-s / 2, s / 2, 0];
        const apex = [-s / 6, 0, s * 0.7];
        return [
            [nose, backR, backL],
            [nose, apex, backR],
            [nose, backL, apex],
            [backL, backR, apex]
        ];
    }

    // Boss hull: a hexagonal bipyramid over the same footprint as BOSS_HEX --
    // six side faces up to a raised center apex, six more down to a matching
    // apex below, so it reads as a solid faceted "gem" from any angle
    // (bosses are big enough, and get circled/viewed from behind often
    // enough, that a single-sided top-only shape would show as flat or
    // inside-out too often).
    function bossHullTriangles(s) {
        const n = 6;
        const rim = [];
        for (let i = 0; i < n; i++) {
            const t = (i / n) * Math.PI * 2;
            rim.push([Math.cos(t) * s, Math.sin(t) * s, 0]);
        }
        const top = [0, 0, s * 0.6];
        const bottom = [0, 0, -s * 0.6];
        const tris = [];
        for (let i = 0; i < n; i++) {
            const a = rim[i], b = rim[(i + 1) % n];
            tris.push([a, b, top]);
            tris.push([b, a, bottom]);
        }
        return tris;
    }

    // 3D Asteroid Polyhedron: 8-sided faceted rock with randomized surface variations
    function asteroidHullTriangles(r) {
        const top = [0, 0, r * 0.75];
        const btm = [0, 0, -r * 0.75];
        const n = 8;
        const mid = [];
        for (let i = 0; i < n; i++) {
            const th = (i / n) * Math.PI * 2;
            const pert = 1 + (((i * 7) % 5) - 2) * 0.12;
            mid.push([Math.cos(th) * r * pert, Math.sin(th) * r * pert, (((i * 3) % 3) - 1) * r * 0.12]);
        }
        const tris = [];
        for (let i = 0; i < n; i++) {
            const a = mid[i], b = mid[(i + 1) % n];
            tris.push([a, b, top]);
            tris.push([b, a, btm]);
        }
        return tris;
    }

    // 3D Basic Enemy Drone: Aerodynamic delta dart with dorsal spine
    function basicEnemyTriangles(s) {
        const nose = [s * 1.2, 0, 0];
        const wingL = [-s * 0.6, -s * 0.65, s * 0.1];
        const wingR = [-s * 0.6, s * 0.65, s * 0.1];
        const spine = [-s * 0.1, 0, s * 0.45];
        const keel = [-s * 0.3, 0, -s * 0.2];
        return [
            [nose, wingR, spine],
            [nose, spine, wingL],
            [nose, wingL, keel],
            [nose, keel, wingR],
            [wingL, spine, wingR],
            [wingL, wingR, keel]
        ];
    }

    // 3D Advanced Enemy Cruiser: Stealth faceted interceptor
    function advancedEnemyTriangles(s) {
        const nose = [s * 1.1, 0, 0];
        const wingL = [0, -s * 0.8, s * 0.15];
        const wingR = [0, s * 0.8, s * 0.15];
        const tail = [-s * 0.7, 0, 0];
        const apex = [-s * 0.15, 0, s * 0.6];
        const ventral = [-s * 0.2, 0, -s * 0.35];
        return [
            [nose, apex, wingL],
            [nose, wingR, apex],
            [wingL, apex, tail],
            [apex, wingR, tail],
            [nose, wingL, ventral],
            [nose, ventral, wingR],
            [wingL, tail, ventral],
            [wingR, ventral, tail]
        ];
    }

    // 3D Power-Up Gem: Rotating double-pyramid octahedron
    function powerUpTriangles(s) {
        const top = [0, 0, s * 1.2];
        const btm = [0, 0, -s * 1.2];
        const p0 = [s, 0, 0], p1 = [0, s, 0], p2 = [-s, 0, 0], p3 = [0, -s, 0];
        return [
            [p0, p1, top], [p1, p2, top], [p2, p3, top], [p3, p0, top],
            [p1, p0, btm], [p2, p1, btm], [p3, p2, btm], [p0, p3, btm]
        ];
    }

    // 3D Boss Subsystem: Satellite faceted node
    function subsystemTriangles(s) {
        const top = [0, 0, s * 0.9];
        const btm = [0, 0, -s * 0.9];
        const p0 = [s, 0, 0], p1 = [0, s, 0], p2 = [-s, 0, 0], p3 = [0, -s, 0];
        return [
            [p0, p1, top], [p1, p2, top], [p2, p3, top], [p3, p0, top],
            [p1, p0, btm], [p2, p1, btm], [p3, p2, btm], [p0, p3, btm]
        ];
    }

    // Generic per-frame vertex writer for hull shapes: rotates local (u,v) footprint
    // and projects each vertex (including height h) into 3D world space.
    function writeHullVertices(entry, x, y, angle, hullTriangles, lift = SHIP_HULL_LIFT) {
        const WORLD = root.WORLD;
        const ca = Math.cos(angle), sa = Math.sin(angle);
        const pos = entry.geometry.attributes.position.array;
        let i = 0;
        for (const tri of hullTriangles) {
            for (const [u, v, h] of tri) {
                const cx = u * ca - v * sa, cy = u * sa + v * ca;
                const p = WORLD.toWorld3(x, y, cx, cy, lift + h);
                pos[i++] = p[0]; pos[i++] = p[1]; pos[i++] = p[2];
            }
        }
        entry.geometry.attributes.position.needsUpdate = true;
        entry.geometry.computeVertexNormals();
    }

    function createHullMeshEntry(vertexCount) {
        const THREE = root.THREE;
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3));
        const material = new THREE.MeshLambertMaterial({
            // Winding direction isn't hand-verified against the camera; DoubleSide
            // sidesteps the risk of a backwards face going invisible or
            // inside-out-looking -- Three flips the effective normal for
            // back-facing fragments, so shading stays correct either way.
            side: THREE.DoubleSide
        });
        const mesh = new THREE.Mesh(geometry, material);
        // The position buffer is rewritten in place every frame (the ship
        // moves), but a BufferGeometry's boundingSphere -- what Three uses for
        // frustum culling -- is only computed once, on first render, and never
        // refreshed on its own. Left at defaults, the mesh would render fine
        // right after creation and then silently vanish (culled) once the ship
        // had moved far enough from wherever that first bounding sphere was.
        // Recomputing it every frame is the "correct" fix but pointless here:
        // there are at most a couple dozen of these, so skipping culling
        // entirely costs nothing.
        mesh.frustumCulled = false;
        scene.add(mesh);
        return { mesh, geometry, material };
    }

    function updateShipHullMesh(ship, s, angle, colorRgb) {
        let entry = shipMeshPool.get(ship);
        if (!entry) {
            entry = createHullMeshEntry(shipHullTriangles(s).length * 3);
            shipMeshPool.set(ship, entry);
        }
        entry.material.color.setRGB(colorRgb[0], colorRgb[1], colorRgb[2]);
        writeHullVertices(entry, ship.x, ship.y, angle, shipHullTriangles(s));
    }

    function updateBossHullMesh(boss, s, angle, colorRgb) {
        let entry = shipMeshPool.get(boss);
        if (!entry) {
            entry = createHullMeshEntry(bossHullTriangles(s).length * 3);
            shipMeshPool.set(boss, entry);
        }
        entry.material.color.setRGB(colorRgb[0], colorRgb[1], colorRgb[2]);
        writeHullVertices(entry, boss.x, boss.y, angle, bossHullTriangles(s));
    }

    function updateAsteroidMesh(a, r, angle, colorRgb, isMineral) {
        let entry = shipMeshPool.get(a);
        if (!entry) {
            entry = createHullMeshEntry(asteroidHullTriangles(r).length * 3);
            shipMeshPool.set(a, entry);
        }
        entry.material.color.setRGB(colorRgb[0], colorRgb[1], colorRgb[2]);
        writeHullVertices(entry, a.x, a.y, angle, asteroidHullTriangles(r), r * 0.3);
    }

    function updateBasicEnemyMesh(e, s, angle, colorRgb) {
        let entry = shipMeshPool.get(e);
        if (!entry) {
            entry = createHullMeshEntry(basicEnemyTriangles(s).length * 3);
            shipMeshPool.set(e, entry);
        }
        entry.material.color.setRGB(colorRgb[0], colorRgb[1], colorRgb[2]);
        writeHullVertices(entry, e.x, e.y, angle, basicEnemyTriangles(s));
    }

    function updateAdvancedEnemyMesh(e, s, angle, colorRgb) {
        let entry = shipMeshPool.get(e);
        if (!entry) {
            entry = createHullMeshEntry(advancedEnemyTriangles(s).length * 3);
            shipMeshPool.set(e, entry);
        }
        entry.material.color.setRGB(colorRgb[0], colorRgb[1], colorRgb[2]);
        writeHullVertices(entry, e.x, e.y, angle, advancedEnemyTriangles(s));
    }

    function updatePowerUpMesh(p, s, pulse, colorRgb) {
        let entry = shipMeshPool.get(p);
        if (!entry) {
            entry = createHullMeshEntry(powerUpTriangles(s).length * 3);
            shipMeshPool.set(p, entry);
        }
        entry.material.color.setRGB(colorRgb[0], colorRgb[1], colorRgb[2]);
        writeHullVertices(entry, p.x, p.y, pulse, powerUpTriangles(s), s * 0.8);
    }

    function updateSubsystemMesh(sub, x, y, s, colorRgb, destroyed) {
        let entry = shipMeshPool.get(sub);
        if (!entry) {
            entry = createHullMeshEntry(subsystemTriangles(s).length * 3);
            shipMeshPool.set(sub, entry);
        }
        if (destroyed) {
            entry.material.color.setRGB(0.2, 0.2, 0.2);
        } else {
            entry.material.color.setRGB(colorRgb[0], colorRgb[1], colorRgb[2]);
        }
        writeHullVertices(entry, x, y, 0, subsystemTriangles(s), s * 0.5);
    }

    // Called once per render() with every ship/boss still alive this frame,
    // so a destroyed one's mesh/geometry/material get disposed instead of
    // sitting in the pool (keyed by object reference) forever.
    function reconcileShipMeshPool(liveShips) {
        for (const [shipObj, entry] of shipMeshPool) {
            if (liveShips.has(shipObj)) continue;
            scene.remove(entry.mesh);
            entry.geometry.dispose();
            entry.material.dispose();
            shipMeshPool.delete(shipObj);
        }
    }

    function drawShip(sh, colorCss, isPlayer) {
        const s = sh.size || 20;
        const c = rgb(colorCss);
        updateShipHullMesh(sh, s, sh.angle, c);
        poly(sh.x, sh.y, SHIP_TRI(s), sh.angle, 2, c, true);
        if (sh.shieldActive) {
            circle(sh.x, sh.y, (sh.shieldRadius || s * 1.6) + Math.sin(frame / 6) * 2, 2, rgb(THEME.cyan || '#00D2FF'));
        }
        if (sh.isAlpha && !isPlayer) {
            circle(sh.x, sh.y, s * 1.35, 2, dim(c, 0.45), 24);
        }
        if (sh.helpPingActive) {
            const pulse = (frame % 40) / 40;
            circle(sh.x, sh.y, s * 1.5 + pulse * s * 2.5, 2.5, dim(rgb(THEME.helpPing || '#FF00FF'), 1 - pulse), 32);
        }
        if (sh.role === 'fuel' && sh.fuelTransferActive === false && sh.fuelDocked === false) {
            // Tanker marker: small ring so it reads as a support craft.
            circle(sh.x, sh.y, s * 0.35, 2, c, 10);
        }
        // Health ticks for AI ships.
        if (!isPlayer && sh.maxHealth > 0 && sh.health < sh.maxHealth) {
            const hp = Math.max(0, sh.health / sh.maxHealth);
            const barC = rgb(hp > 0.66 ? (THEME.healthHigh || '#00D2FF') : hp > 0.33 ? (THEME.healthMid || '#FF00FF') : (THEME.healthLow || '#FF3366'));
            const w = s * 1.6;
            const a = root.WORLD.toWorld3(sh.x, sh.y, -w / 2, -s - 6, 2.5);
            const b = root.WORLD.toWorld3(sh.x, sh.y, -w / 2 + w * hp, -s - 6, 2.5);
            seg(a, b, barC);
        }
    }

    function drawAsteroid(a) {
        const isMin = !!a.isMineral;
        const c = isMin ? rgb('#FFD700') : rgb('#6FA8C8');
        updateAsteroidMesh(a, a.radius || 15, a.rotation || 0, c, isMin);
        const pts = a.vertices ? a.vertices.map(v => [v.x, v.y]) : [[a.radius, 0], [0, a.radius], [-a.radius, 0], [0, -a.radius]];
        poly(a.x, a.y, pts, a.rotation || 0, 1.5, c, true);
    }

    function drawEnemy(e) {
        const s = e.size || 15;
        const css = e.color || THEME.enemy || '#FF4466';
        const c = rgb(css);
        if (e.type === 'boss') {
            updateBossHullMesh(e, s, e.angle || 0, c);
            poly(e.x, e.y, BOSS_HEX(s), e.angle || 0, 2, c, true);
            poly(e.x, e.y, BOSS_HEX(s * 0.55), -(e.angle || 0) * 2, 2.5, dim(c, 0.7), true);
            if (e.subsystems) {
                e.subsystems.forEach(sub => {
                    const pos = e.getSubsystemWorldPos(sub);
                    const subC = sub.destroyed ? rgb('#444444') : rgb(sub.color || '#00E5FF');
                    updateSubsystemMesh(sub, pos.x, pos.y, sub.radius * 0.8, subC, sub.destroyed);
                    circle(pos.x, pos.y, sub.radius, 1.5, subC, 12);
                    seg(root.WORLD.toWorld3(e.x, e.y, 0, 0, 2), root.WORLD.toWorld3(pos.x, pos.y, 0, 0, 2), dim(subC, 0.4));
                });
            }
        } else if (e.type === 'advanced') {
            updateAdvancedEnemyMesh(e, s, e.angle || 0, c);
            poly(e.x, e.y, ENEMY_DIAMOND(s), e.angle || 0, 2, c, true);
        } else {
            updateBasicEnemyMesh(e, s, e.angle || 0, c);
            poly(e.x, e.y, ENEMY_DIAMOND(s), e.angle || 0, 2, c, true);
        }
        if (e.shieldActive) circle(e.x, e.y, s * 1.5, 2, dim(c, 0.8));
        if (e.maxHealth > 0 && e.health < e.maxHealth) {
            const hp = Math.max(0, e.health / e.maxHealth);
            const w = s * 1.6;
            seg(root.WORLD.toWorld3(e.x, e.y, -w / 2, -s - 6, 2.5), root.WORLD.toWorld3(e.x, e.y, -w / 2 + w * hp, -s - 6, 2.5), c);
        }
    }

    function drawPowerUp(p) {
        if (!p || p.lifetime <= 0) return;
        const c = rgb(p.color || '#00FF66');
        updatePowerUpMesh(p, p.radius || 10, p.pulse || 0, c);
        circle(p.x, p.y, (p.radius || 10) * 1.25, 1.5, c, 12);
    }

    function drawBullet(b, css) {
        const c = rgb(css);
        const WORLD = root.WORLD;
        const ca = Math.cos(b.angle || 0), sa = Math.sin(b.angle || 0);
        seg(WORLD.toWorld3(b.x, b.y, -ca * 5, -sa * 5, 2), WORLD.toWorld3(b.x, b.y, ca * 3, sa * 3, 2), c);
        dotp(WORLD.toWorld3(b.x, b.y, 0, 0, 2.5), c);
    }

    // ------------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------------
    function normalize(v) { const m = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / m, v[1] / m, v[2] / m]; }
    function lerp3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

    function followTarget(state) {
        const WORLD = root.WORLD;
        if (followMode === 'orbit') {
            orbitAngle += 0.0025;
            const tilt = 0.42;
            return { dir: [Math.cos(tilt) * Math.cos(orbitAngle), Math.sin(tilt), -Math.cos(tilt) * Math.sin(orbitAngle)], up: [0, 1, 0] };
        }
        let focus = null;
        if (followMode === 'fleet' && state.aiShips && state.aiShips.length) {
            const acc = [0, 0, 0];
            let n = 0;
            for (const s of state.aiShips) {
                if (!s) continue;
                const p = WORLD.frameAt(s.x, s.y).p;
                acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2]; n++;
            }
            if (n && Math.hypot(acc[0], acc[1], acc[2]) > 1e-3) focus = normalize(acc);
        }
        if (!focus) {
            let lead = null;
            if (state.aiShips) for (const s of state.aiShips) { if (s && s.isAlpha) { lead = s; break; } }
            if (!lead && state.aiShips && state.aiShips.length) lead = state.aiShips.find(s => s) || null;
            if (!lead && state.ship) lead = state.ship;
            focus = lead ? WORLD.frameAt(lead.x, lead.y).p : [1, 0, 0];
        }
        // Re-derive the frame at the focus so "up" on screen is local north.
        const chart = WORLD.fromUnit(focus);
        const fr = WORLD.frameAt(chart.x, chart.y);
        return { dir: fr.p, up: fr.n };
    }

    function updateCamera(state) {
        const tgt = followTarget(state);
        const k = followMode === 'orbit' ? 1 : 0.08;
        camDir = normalize(lerp3(camDir, tgt.dir, k));
        let up = lerp3(camUp, tgt.up, k);
        const d = up[0] * camDir[0] + up[1] * camDir[1] + up[2] * camDir[2];
        up = normalize([up[0] - camDir[0] * d, up[1] - camDir[1] * d, up[2] - camDir[2] * d]);
        if (!isFinite(up[0]) || Math.hypot(up[0], up[1], up[2]) < 1e-6) up = [0, 1, 0];
        camUp = up;
        zoom = effectiveZoom();
        const dist = R * zoom;
        camera.position.set(camDir[0] * dist, camDir[1] * dist, camDir[2] * dist);
        camera.up.set(camUp[0], camUp[1], camUp[2]);
        camera.lookAt(0, 0, 0);
        camera.updateMatrixWorld(true);
    }

    // ------------------------------------------------------------------
    // Frame
    // ------------------------------------------------------------------
    function render(state) {
        if (!ready) return;
        frame++;
        placeCanvases();
        lineCount = 0; pointCount = 0; hiddenCount = 0;
        const WORLD = root.WORLD;

        const liveEntities = new Set(state.aiShips || []);
        if (state.ship) liveEntities.add(state.ship);
        (state.enemyShips || []).forEach(e => {
            if (!e) return;
            liveEntities.add(e);
            if (e.subsystems) e.subsystems.forEach(sub => liveEntities.add(sub));
        });
        (state.asteroids || []).forEach(a => { if (a) liveEntities.add(a); });
        (state.powerUps || []).forEach(p => { if (p) liveEntities.add(p); });
        reconcileShipMeshPool(liveEntities);

        updateCamera(state);
        const cx = camera.position.x, cy = camera.position.y, cz = camera.position.z;
        // Everything on the far hemisphere is occluded by the globe anyway; skip
        // building its geometry so the buffers stay small.
        const isFront = (o) => {
            const p = WORLD.frameAt(o.x, o.y).p;
            const v = p[0] * (cx - p[0] * R) + p[1] * (cy - p[1] * R) + p[2] * (cz - p[2] * R);
            return v > -R * 0.15;
        };
        const visible = (o) => { if (isFront(o)) return true; hiddenCount++; return false; };

        const alphaAttack = !!state.alphaAttackEnabled;
        const fuelOn = !!state.fuelRescueEnabled && alphaAttack;

        (state.asteroids || []).forEach(a => { if (a && visible(a)) drawAsteroid(a); });

        const goldCss = 'rgb(255,215,0)';
        const aiCss = THEME.aiBlue || '#0078FF';
        const fuelCss = THEME.fuelShip || '#00FF88';
        let alpha = null;
        (state.aiShips || []).forEach(s => {
            if (!s) return;
            if (s.isAlpha && alphaAttack) alpha = s;
        });

        // Formation spokes and fuel tethers (arcs on the surface).
        if (alpha) {
            const spoke = dim(rgb(goldCss), 0.35);
            (state.aiShips || []).forEach(s => {
                if (!s || s === alpha || s.role === 'fuel') return;
                if (isFront(s) || isFront(alpha)) arc(alpha.x, alpha.y, s.x, s.y, 1.2, spoke, undefined, true);
            });
        }
        if (fuelOn) {
            const tether = rgb(THEME.fuelTransfer || '#00FF88');
            (state.aiShips || []).forEach(s => {
                if (!s || s.role !== 'fuel' || (!s.fuelTransferActive && !s.fuelDocked) || !s.fuelTransferPartner) return;
                const partner = s.fuelTransferPartner;
                if (!(state.aiShips || []).includes(partner)) return;
                arc(s.x, s.y, partner.x, partner.y, 2.2, s.fuelDocked ? tether : dim(tether, 0.7), undefined, !s.fuelDocked);
            });
        }

        (state.aiShips || []).forEach(s => {
            if (!s || !visible(s)) return;
            let css = aiCss;
            if (s.role === 'fuel' && fuelOn) css = fuelCss;
            else if (s.isAlpha && alphaAttack) css = goldCss;
            drawShip(s, css, false);
        });

        if (state.ship) {
            // Inactive player ship is drawn dimmed, like the 2D path's 0.3 alpha.
            const css = state.playerShipActive ? (THEME.primary || '#00D2FF') : '#2a4a5a';
            if (visible(state.ship)) drawShip(state.ship, css, true);
        }

        (state.enemyShips || []).forEach(e => { if (e && visible(e)) drawEnemy(e); });
        (state.powerUps || []).forEach(p => { if (p && visible(p)) drawPowerUp(p); });
        (state.bullets || []).forEach(b => {
            const bColor = b.piercing ? '#00FF66' : (b.isAIBullet ? (THEME.aiBlue || '#0078FF') : (THEME.playerBullet || '#00D2FF'));
            if (b && visible(b)) drawBullet(b, bColor);
        });
        (state.enemyBullets || []).forEach(b => { if (b && visible(b)) drawBullet(b, THEME.enemy || '#FF4466'); });

        // FX particles from the 2D pool, re-projected onto the sphere.
        if (state.fx && typeof state.fx.forEachParticle === 'function') {
            state.fx.forEachParticle((x, y, t, css, size) => {
                if (pointCount >= MAX_POINTS) return;
                const c = rgb(css);
                const k = Math.max(0, Math.min(1, t)) * Math.min(1, 0.4 + size * 0.25);
                dotp(WORLD.toWorld3(x, y, 0, 0, 3), [c[0] * k, c[1] * k, c[2] * k]);
            });
        }

        const lg = lines.geometry;
        lg.attributes.position.needsUpdate = true;
        lg.attributes.color.needsUpdate = true;
        lg.setDrawRange(0, lineCount * 2);
        const pg = points.geometry;
        pg.attributes.position.needsUpdate = true;
        pg.attributes.color.needsUpdate = true;
        pg.setDrawRange(0, pointCount);

        renderer.render(scene, camera);
    }

    // Chart -> overlay pixel coordinates; null if behind the horizon or off
    // screen. `scale` is pixels per chart unit at that depth.
    const projVec = { v: null };
    function project(x, y) {
        if (!ready) return null;
        const THREE = root.THREE;
        const WORLD = root.WORLD;
        const p = WORLD.toWorld3(x, y, 0, 0, 2);
        const n = WORLD.frameAt(x, y).p;
        const dx = camera.position.x - p[0], dy = camera.position.y - p[1], dz = camera.position.z - p[2];
        if (n[0] * dx + n[1] * dy + n[2] * dz <= 0) return null;
        if (!projVec.v) projVec.v = new THREE.Vector3();
        const v = projVec.v.set(p[0], p[1], p[2]).project(camera);
        if (v.z > 1 || v.x < -1.2 || v.x > 1.2 || v.y < -1.2 || v.y > 1.2) return null;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const focal = (H / 2) / Math.tan((camera.fov * Math.PI / 180) / 2);
        return { x: (v.x + 1) / 2 * W, y: (1 - v.y) / 2 * H, scale: focal / dist };
    }

    // ------------------------------------------------------------------
    // 2D overlay: minimap + caption
    // ------------------------------------------------------------------
    let lastState = null;
    function drawOverlay(ctx) {
        if (!ready || !ctx) return;
        const WORLD = root.WORLD;
        const st = lastState;
        // Bottom-right: the telemetry HUD owns the top-right corner.
        const mw = 168, mh = 84, mx = W - mw - 12, my = H - mh - 30;
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(mx, my, mw, mh);
        ctx.strokeStyle = 'rgba(120,160,200,0.55)';
        ctx.lineWidth = 1;
        ctx.strokeRect(mx + 0.5, my + 0.5, mw - 1, mh - 1);
        ctx.strokeStyle = 'rgba(60,110,160,0.5)';
        ctx.beginPath(); ctx.moveTo(mx, my + mh / 2); ctx.lineTo(mx + mw, my + mh / 2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(mx + mw / 2, my); ctx.lineTo(mx + mw / 2, my + mh); ctx.stroke();

        // Minimap is an equirectangular chart of the *world*, which may be
        // WORLD.getScale() times larger than the canvas.
        const ws = WORLD.size();
        const toMap = (x, y) => [mx + (x / ws.w) * mw, my + (y / ws.h) * mh];
        const dot = (x, y, r, css) => { const m = toMap(x, y); ctx.fillStyle = css; ctx.beginPath(); ctx.arc(m[0], m[1], r, 0, Math.PI * 2); ctx.fill(); };

        if (st) {
            (st.asteroids || []).forEach(a => a && dot(a.x, a.y, 1.2, 'rgba(111,168,200,0.8)'));
            (st.enemyShips || []).forEach(e => e && dot(e.x, e.y, e.type === 'boss' ? 2.6 : 1.8, THEME.enemy || '#FF4466'));
            (st.aiShips || []).forEach(s => {
                if (!s) return;
                const css = (s.role === 'fuel' && st.fuelRescueEnabled && st.alphaAttackEnabled) ? (THEME.fuelShip || '#00FF88')
                    : (s.isAlpha && st.alphaAttackEnabled) ? 'rgb(255,215,0)' : (THEME.aiBlue || '#0078FF');
                dot(s.x, s.y, s.isAlpha ? 2.6 : 2, css);
            });
            if (st.ship && st.playerShipActive) dot(st.ship.x, st.ship.y, 2, THEME.text || '#FFFFFF');
        }

        // Visible-cap outline around the camera axis.
        const rho = Math.acos(Math.min(1, 1 / zoom));
        const cc = WORLD.fromUnit(camDir);
        const fr = WORLD.frameAt(cc.x, cc.y);
        ctx.strokeStyle = 'rgba(255,255,255,0.55)';
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        let prev = null;
        const n = 64;
        for (let i = 0; i <= n; i++) {
            const t = (i / n) * Math.PI * 2;
            const p = [
                Math.cos(rho) * fr.p[0] + Math.sin(rho) * (Math.cos(t) * fr.e[0] + Math.sin(t) * fr.n[0]),
                Math.cos(rho) * fr.p[1] + Math.sin(rho) * (Math.cos(t) * fr.e[1] + Math.sin(t) * fr.n[1]),
                Math.cos(rho) * fr.p[2] + Math.sin(rho) * (Math.cos(t) * fr.e[2] + Math.sin(t) * fr.n[2])
            ];
            const c = WORLD.fromUnit(p);
            const m = toMap(c.x, c.y);
            if (prev && Math.abs(m[0] - prev[0]) < mw / 2) ctx.lineTo(m[0], m[1]);
            else ctx.moveTo(m[0], m[1]);
            prev = m;
        }
        ctx.stroke();
        ctx.setLineDash([]);
        const cm = toMap(cc.x, cc.y);
        ctx.strokeStyle = 'rgba(255,255,255,0.8)';
        ctx.beginPath(); ctx.moveTo(cm[0] - 4, cm[1]); ctx.lineTo(cm[0] + 4, cm[1]); ctx.moveTo(cm[0], cm[1] - 4); ctx.lineTo(cm[0], cm[1] + 4); ctx.stroke();

        ctx.font = '10px monospace';
        ctx.fillStyle = 'rgba(209,209,209,0.85)';
        ctx.textAlign = 'right';
        const modeLabel = followMode === 'orbit' ? 'orbit' : followMode === 'fleet' ? 'follow fleet' : 'follow alpha';
        const sc = WORLD.getScale();
        const scaleLabel = sc !== 1 ? `  x${sc}` : '';
        ctx.fillText(`SPHERE R=${R.toFixed(0)}${scaleLabel}  cam: ${modeLabel} ${zoom.toFixed(1)}R  far side: ${hiddenCount}`, mx + mw, my + mh + 12);
        ctx.fillText('Designed by Dang.', mx + mw, my + mh + 24);
        ctx.restore();
    }

    const api = {
        init: init,
        ready: () => ready,
        render: (state) => { lastState = state; render(state); },
        drawOverlay: drawOverlay,
        project: project,
        overlayContext: () => overlayCtx,
        setVisible: setVisible,
        setCamera: (mode) => { followMode = (mode === 'orbit' || mode === 'fleet') ? mode : 'alpha'; },
        getCamera: () => followMode,
        setZoom: (z) => { const v = Number(z); if (isFinite(v)) zoomReq = Math.max(1.3, Math.min(4.5, v)); zoom = effectiveZoom(); },
        getZoom: () => zoom,
        getRequestedZoom: () => zoomReq,
        stats: () => ({ segments: lineCount, points: pointCount, hidden: hiddenCount }),
        // Composite the WebGL frame and the 2D overlay into one PNG data URL.
        snapshot: () => {
            if (!ready) return null;
            const c = document.createElement('canvas');
            c.width = W; c.height = H;
            const g = c.getContext('2d');
            g.drawImage(glCanvas, 0, 0, W, H);
            g.drawImage(overlayCanvas, 0, 0, W, H);
            return c.toDataURL('image/png');
        }
    };
    root.SphereView = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
