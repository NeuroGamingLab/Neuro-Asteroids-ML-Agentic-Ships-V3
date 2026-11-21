#!/bin/bash
# Runs every test in the RL pipeline.
#
# Uses macOS JavaScriptCore (`jsc`) by default, or falls back to Node.js via
# sim/runtime_shim.js on Linux/Windows (override with JS_RUNTIME=node|jsc).

set -u
cd "$(dirname "$0")"

JSC="/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
RUNTIME="${JS_RUNTIME:-auto}"
if [ "$RUNTIME" = "auto" ]; then
    if [ -x "$JSC" ]; then
        RUNTIME="jsc"
    elif command -v node >/dev/null 2>&1; then
        RUNTIME="node"
    else
        echo "Neither JavaScriptCore ($JSC) nor node was found."
        exit 1
    fi
fi

run_js() {
    local file="$1"
    if [ "$RUNTIME" = "node" ]; then
        node -r ./sim/runtime_shim.js "$file"
    else
        "$JSC" "$file"
    fi
}

syntax_check() {
    local file="$1"
    if [ "$RUNTIME" = "node" ]; then
        node --check "$file"
    else
        "$JSC" -e "new Function(read('$file'))"
    fi
}

failed=0

run() {
    local name="$1" file="$2"
    printf '\n=== %s ===\n' "$name"
    if [ ! -f "$file" ]; then
        echo "  missing: $file"
        failed=1
        return
    fi
    if run_js "$file"; then
        :
    else
        failed=1
    fi
}

run "neural network (gradients, Adam, serialisation)" training/test_nn.js
run "environment (determinism, rewards, credit)"      training/test_env.js
run "browser inference path"                          training/test_policy_runner.js
run "waves and boss phases"                           training/test_waves.js

# Parse-check everything else that ships to the browser, so a syntax error in a
# file with no test still fails the suite rather than reaching the page.
printf '\n=== syntax (%s) ===\n' "$RUNTIME"
for f in theme.js game.js entities.js ai_ship.js enemy_ship.js enemy_coordination.js \
         rng.js fx.js hud.js policy_viz.js policy_worker.js \
         world.js sphere_view.js ml_features.js policy_runner.js marl_system.js \
         marl_environment.js marl_integration.js marl_data_collector.js \
         ml_model_adapter.js ml_model_loader.js ml_hierarchy.js ml_evaluation.js \
         sim/env.js sim/browser_shim.js sim/runtime_shim.js training/nn.js training/ppo.js \
         training/bc.js training/evaluate.js training/train.js training/benchmark.js \
         training/self_play.js training/test_waves.js; do
    [ -f "$f" ] || continue
    if syntax_check "$f" >/dev/null 2>&1; then
        echo "  ok    $f"
    else
        echo "  FAIL  $f"
        syntax_check "$f" || true
        failed=1
    fi
done

printf '\n'
if [ "$failed" -eq 0 ]; then
    echo "ALL TESTS PASSED"
else
    echo "TESTS FAILED"
fi
exit "$failed"
