// Cross-platform JS runtime shim for headless training, benchmarks, and tests.
//
// On macOS the pipeline runs directly on JavaScriptCore (`jsc`), which provides
// `load()`, `read()`, `print()`, `quit()`, and top-level `arguments`.
// On Linux/Windows (or CI on ubuntu-latest), Node.js can run the exact same
// scripts by preloading this shim:
//
//   node -r ./sim/runtime_shim.js training/train.js -- --stage 0 --iterations 10
//   node -r ./sim/runtime_shim.js training/test_nn.js
//
// Under jsc this file is a no-op.

(function (root) {
    'use strict';

    if (typeof process === 'undefined' || !process.versions || !process.versions.node) {
        return;
    }

    const fs = require('fs');
    const path = require('path');
    const vm = require('vm');

    const loadedRoot = process.cwd();

    if (typeof root.read !== 'function') {
        root.read = function read(filePath) {
            return fs.readFileSync(path.resolve(loadedRoot, filePath), 'utf8');
        };
    }

    if (typeof root.load !== 'function') {
        root.load = function load(filePath) {
            const abs = path.resolve(loadedRoot, filePath);
            const code = fs.readFileSync(abs, 'utf8');
            vm.runInThisContext(code, { filename: abs });
        };
    }

    if (typeof root.print !== 'function') {
        root.print = function print(...args) {
            process.stdout.write(args.map(String).join(' ') + '\n');
        };
    }

    if (typeof root.quit !== 'function') {
        root.quit = function quit(code) {
            process.exit(code || 0);
        };
    }

    // Mirror jsc's `arguments` array (everything after `--`, or script argv).
    if (typeof root.arguments === 'undefined') {
        const raw = process.argv.slice(2);
        const dashIdx = raw.indexOf('--');
        root.arguments = dashIdx >= 0 ? raw.slice(dashIdx + 1) : raw.slice(1);
    }
})(typeof globalThis !== 'undefined' ? globalThis : this);
