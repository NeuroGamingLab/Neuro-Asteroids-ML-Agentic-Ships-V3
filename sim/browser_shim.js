// Minimal browser host shim so the *real* game.js can run outside a browser.
//
// The point of running the real game.js rather than a Python/NumPy reimplementation
// is that a reimplementation drifts. Any physics detail that differs between the
// training sim and the shipped game produces a policy that scores well in training
// and falls apart in the browser, and that class of bug is very hard to see.
//
// game.js only touches a small host surface: document.getElementById,
// addEventListener, a canvas + 2D context, AudioContext, requestAnimationFrame,
// performance.now and alert. All of it is stubbed here.

(function (root) {
    'use strict';

    const NOOP = function () {};

    // JavaScriptCore has `print` but no `console`. Training runs millions of
    // steps, and game.js logs on several of its paths, so console output is
    // off by default and opt-in via setHeadlessVerbose(true).
    let verbose = false;
    function emit(prefix) {
        return function () {
            if (!verbose) return;
            const parts = [];
            for (let i = 0; i < arguments.length; i++) {
                const a = arguments[i];
                parts.push(typeof a === 'string' ? a : safeStringify(a));
            }
            if (typeof print === 'function') print(prefix + parts.join(' '));
        };
    }
    function safeStringify(value) {
        try { return JSON.stringify(value); } catch (e) { return String(value); }
    }
    if (typeof root.console === 'undefined') {
        root.console = {
            log: emit(''),
            warn: emit('[warn] '),
            error: emit('[error] '),
            info: emit(''),
            debug: emit(''),
            trace: NOOP,
            table: NOOP,
            group: NOOP,
            groupEnd: NOOP,
            time: NOOP,
            timeEnd: NOOP
        };
    }
    root.setHeadlessVerbose = function (on) { verbose = !!on; };

    function createStubContext() {
        const ctx = {
            // Style properties are plain data slots; game.js assigns to them freely.
            fillStyle: '#000',
            strokeStyle: '#000',
            lineWidth: 1,
            font: '10px sans-serif',
            textAlign: 'left',
            globalAlpha: 1,
            shadowBlur: 0,
            shadowColor: '#000',
            imageSmoothingEnabled: true,
            imageSmoothingQuality: 'high'
        };
        const methods = [
            'arc', 'arcTo', 'beginPath', 'bezierCurveTo', 'clearRect', 'clip', 'closePath',
            'drawImage', 'ellipse', 'fill', 'fillRect', 'fillText', 'lineTo', 'moveTo',
            'putImageData', 'quadraticCurveTo', 'rect', 'restore', 'rotate', 'save',
            'scale', 'setLineDash', 'setTransform', 'stroke', 'strokeRect', 'strokeText',
            'transform', 'translate'
        ];
        methods.forEach(m => { ctx[m] = NOOP; });
        ctx.measureText = () => ({ width: 0 });
        ctx.createLinearGradient = () => ({ addColorStop: NOOP });
        ctx.createRadialGradient = () => ({ addColorStop: NOOP });
        ctx.getImageData = () => ({ data: [] });
        ctx.createImageData = () => ({ data: [] });
        ctx.isPointInPath = () => false;
        return ctx;
    }

    function createStubElement(id) {
        return {
            id,
            textContent: '',
            innerHTML: '',
            style: {},
            width: 0,
            height: 0,
            addEventListener: NOOP,
            removeEventListener: NOOP,
            appendChild: NOOP,
            removeChild: NOOP,
            click: NOOP,
            focus: NOOP,
            blur: NOOP,
            setAttribute: NOOP,
            removeAttribute: NOOP,
            getAttribute: () => null,
            hasAttribute: () => false,
            classList: { add: NOOP, remove: NOOP, toggle: NOOP, contains: () => false },
            requestFullscreen: NOOP,
            webkitRequestFullscreen: NOOP,
            mozRequestFullScreen: NOOP,
            msRequestFullscreen: NOOP,
            getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 })
        };
    }

    const canvasStub = createStubElement('gameCanvas');
    // Canvas dimensions are the world bounds: game.js reads canvas.width/height in
    // 76 places for wrapping and spawning. Serving them from the stub means the
    // headless sim needs no changes to any of those call sites.
    canvasStub.width = 1200;
    canvasStub.height = 600;
    canvasStub._ctx = createStubContext();
    canvasStub.getContext = function () { return canvasStub._ctx; };

    const elements = { gameCanvas: canvasStub };

    const documentStub = {
        getElementById(id) {
            if (!elements[id]) elements[id] = createStubElement(id);
            return elements[id];
        },
        createElement(tag) { return createStubElement(tag); },
        addEventListener: NOOP,
        removeEventListener: NOOP,
        body: createStubElement('body'),
        fullscreenElement: null,
        webkitFullscreenElement: null,
        mozFullScreenElement: null,
        msFullscreenElement: null,
        exitFullscreen: NOOP,
        webkitExitFullscreen: NOOP,
        mozCancelFullScreen: NOOP,
        msExitFullscreen: NOOP
    };

    root.document = documentStub;

    // No AudioContext: game.js already guards every sound call on a null context,
    // so leaving these undefined makes playFireSound() an early return.
    root.window = root;
    root.innerWidth = 1200;
    root.innerHeight = 600;
    root.addEventListener = NOOP;
    root.removeEventListener = NOOP;
    root.alert = NOOP;

    // A no-op rAF stops game.js's self-perpetuating loop at load time. The
    // headless driver calls update() itself, at whatever rate it likes.
    root.requestAnimationFrame = function () { return 0; };
    root.cancelAnimationFrame = NOOP;

    if (typeof root.setTimeout !== 'function') {
        root.setTimeout = function (fn) { if (typeof fn === 'function') fn(); return 0; };
        root.clearTimeout = NOOP;
        root.setInterval = function () { return 0; };
        root.clearInterval = NOOP;
    }

    if (typeof root.performance === 'undefined') {
        root.performance = { now: () => Date.now() };
    }

    root.__HEADLESS__ = true;

    root.setWorldSize = function (width, height) {
        canvasStub.width = width;
        canvasStub.height = height;
    };
})(typeof globalThis !== 'undefined' ? globalThis : this);
