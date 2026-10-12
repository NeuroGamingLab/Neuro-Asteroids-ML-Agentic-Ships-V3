// Deterministic seeded PRNG shared by the game, the headless simulator and the
// training code. Every stochastic decision in the game must draw from here:
// a single unseeded Math.random() call anywhere in the update path makes an
// episode unreplayable and invalidates seed-matched A/B evaluation.

(function (root) {
    'use strict';

    // mulberry32: 32-bit state, passes gjrand/practrand at this scale, and is
    // exactly reproducible across JavaScriptCore and browser engines because it
    // only uses Math.imul and unsigned shifts (no floating point accumulation).
    function mulberry32(seed) {
        let a = seed | 0;
        return function next() {
            a = (a + 0x6d2b79f5) | 0;
            let t = Math.imul(a ^ (a >>> 15), 1 | a);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    // Hash an arbitrary string/number into a well-distributed 32-bit seed so
    // that sequential seeds (1, 2, 3...) produce uncorrelated streams.
    function hashSeed(value) {
        const str = String(value);
        let h = 2166136261 >>> 0;
        for (let i = 0; i < str.length; i++) {
            h ^= str.charCodeAt(i);
            h = Math.imul(h, 16777619);
        }
        return h >>> 0;
    }

    function createStream(seed) {
        return mulberry32(hashSeed(seed));
    }

    const DEFAULT_SEED = 20260903;

    let worldSeed = DEFAULT_SEED;
    let worldStream = createStream(worldSeed);

    // Primary game stream. game.js calls this in place of Math.random().
    function gameRandom() {
        return worldStream();
    }

    // Reseed the world stream. Called by initGame() and by the headless env on
    // reset() so a given seed always replays the identical episode.
    function setGameSeed(seed) {
        worldSeed = seed === undefined || seed === null ? DEFAULT_SEED : seed;
        worldStream = createStream(worldSeed);
        return worldSeed;
    }

    function getGameSeed() {
        return worldSeed;
    }

    // Derived helpers so callers stop hand-rolling `min + rand()*(max-min)`.
    function randRange(min, max) {
        return min + gameRandom() * (max - min);
    }

    function randInt(minInclusive, maxExclusive) {
        return Math.floor(randRange(minInclusive, maxExclusive));
    }

    function randChoice(array) {
        return array[randInt(0, array.length)];
    }

    // Box-Muller, used by the policy network initialiser.
    function randNormal(mean, stdDev, stream) {
        const draw = stream || gameRandom;
        let u = 0;
        let v = 0;
        while (u === 0) u = draw();
        while (v === 0) v = draw();
        const n = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        return (mean === undefined ? 0 : mean) + (stdDev === undefined ? 1 : stdDev) * n;
    }

    const api = {
        mulberry32,
        hashSeed,
        createStream,
        gameRandom,
        setGameSeed,
        getGameSeed,
        randRange,
        randInt,
        randChoice,
        randNormal,
        DEFAULT_SEED
    };

    root.AsteroidsRNG = api;

    // Expose the hot-path helpers as bare globals: game.js is a plain script
    // with no module system, so `gameRandom()` has to resolve lexically.
    root.gameRandom = gameRandom;
    root.setGameSeed = setGameSeed;
    root.getGameSeed = getGameSeed;
    root.randRange = randRange;
    root.randInt = randInt;
    root.randChoice = randChoice;
    root.randNormal = randNormal;

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    }
})(typeof globalThis !== 'undefined' ? globalThis : this);
