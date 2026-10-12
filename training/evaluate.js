// Evaluation harness: fixed held-out seeds, identical conditions for every
// policy, and confidence intervals.
//
// The previous ml_evaluation.js accumulated counters over one continuous browser
// session, which cannot answer the only question that matters - is the learned
// policy better than the hand-written one? Here every policy plays the same seed
// set under the same reward function, so the comparison is paired and meaningful.

(function (root) {
    'use strict';

    // Held-out seeds. These are derived from a fixed base and must never be used
    // for training rollouts, otherwise "evaluation" just measures memorisation.
    function evalSeeds(count, base) {
        const seeds = [];
        const start = base === undefined ? 900000 : base;
        for (let i = 0; i < count; i++) seeds.push(start + i * 7919); // stride by a prime
        return seeds;
    }

    function mean(values) {
        if (!values.length) return 0;
        let s = 0;
        for (const v of values) s += v;
        return s / values.length;
    }

    function stdDev(values) {
        if (values.length < 2) return 0;
        const m = mean(values);
        let acc = 0;
        for (const v of values) acc += (v - m) ** 2;
        return Math.sqrt(acc / (values.length - 1));
    }

    // Normal-approximation 95% interval. With 100+ episodes this is fine and
    // avoids needing a t-table.
    function ci95(values) {
        if (values.length < 2) return 0;
        return 1.96 * stdDev(values) / Math.sqrt(values.length);
    }

    // Paired comparison: because both policies see the same seeds, the per-seed
    // difference removes seed variance and is far more sensitive than comparing
    // two independent means.
    function pairedComparison(a, b) {
        const n = Math.min(a.length, b.length);
        const diffs = [];
        for (let i = 0; i < n; i++) diffs.push(a[i] - b[i]);
        const m = mean(diffs);
        const se = stdDev(diffs) / Math.sqrt(Math.max(1, n));
        return {
            meanDiff: m,
            ci95: 1.96 * se,
            // |t| > 1.96 is the usual 5% threshold under a normal approximation.
            tStat: se > 0 ? m / se : 0,
            significant: se > 0 ? Math.abs(m / se) > 1.96 : false,
            n
        };
    }

    // policyFn(observations, env) -> array of action indices, or null to let the
    // built-in rule-based controller drive.
    function runEpisodes(env, policyFn, seeds) {
        const returns = [];
        const scores = [];
        const lengths = [];
        const survivors = [];
        const refuels = [];
        const dockDeaths = [];

        for (const seed of seeds) {
            let obs = env.reset(seed);
            let res = null;
            for (let t = 0; t < env.maxSteps; t++) {
                const actions = policyFn ? policyFn(obs, env) : null;
                res = env.step(actions);
                obs = res.observations;
                if (res.done) break;
            }
            if (!res) continue;
            // Report task reward, excluding shaping. Shaping exists to help
            // learning; scoring a policy on how well it harvested its own shaping
            // bonus would make the comparison meaningless.
            returns.push(mean(res.info.episodeTaskReturn || res.info.episodeReturn));
            scores.push(res.info.score);
            lengths.push(res.info.step);
            survivors.push(res.info.aliveCount);
            if (res.info.fuel) {
                refuels.push(res.info.fuel.refuels);
                dockDeaths.push(res.info.fuel.dockDeaths);
            }
        }

        const summary = {
            episodes: returns.length,
            meanReturn: mean(returns),
            ciReturn: ci95(returns),
            meanScore: mean(scores),
            ciScore: ci95(scores),
            meanLength: mean(lengths),
            ciLength: ci95(lengths),
            meanSurvivors: mean(survivors)
        };
        if (refuels.length) {
            summary.meanRefuels = mean(refuels);
            summary.meanDockDeaths = mean(dockDeaths);
        }

        return { returns, scores, lengths, survivors, summary };
    }

    function randomPolicy(rng, actionSize) {
        return function (observations) {
            const actions = [];
            for (let i = 0; i < observations.length; i++) {
                actions.push(Math.floor(rng() * actionSize));
            }
            return actions;
        };
    }

    // Stochastic evaluation gets its own RNG, re-seeded here rather than shared
    // with training. Without this, evaluating the same policy on the same seeds
    // twice returns different numbers, and a difference between two iterations
    // cannot be separated from sampling luck.
    function trainedPolicy(trainer, deterministic, seed) {
        const rng = AsteroidsRNG.createStream(seed || 'eval-policy-sampling');
        return function (observations, env) {
            const actions = [];
            for (let i = 0; i < observations.length; i++) {
                if (!env.alive[i]) { actions.push(null); continue; }
                actions.push(trainer.act(observations[i], deterministic, rng).action);
            }
            return actions;
        };
    }

    function formatSummary(label, result) {
        const s = result.summary;
        return `${label.padEnd(22)} return ${s.meanReturn.toFixed(2)} +/- ${s.ciReturn.toFixed(2)}` +
            `   score ${s.meanScore.toFixed(0)} +/- ${s.ciScore.toFixed(0)}` +
            `   length ${s.meanLength.toFixed(0)}` +
            `   survivors ${s.meanSurvivors.toFixed(2)}` +
            (s.meanRefuels !== undefined
                ? `   refuels ${s.meanRefuels.toFixed(2)}   dockDeaths ${s.meanDockDeaths.toFixed(2)}`
                : '') +
            `   n=${s.episodes}`;
    }

    root.Evaluation = {
        evalSeeds,
        runEpisodes,
        randomPolicy,
        trainedPolicy,
        pairedComparison,
        formatSummary,
        mean,
        stdDev,
        ci95
    };
})(typeof globalThis !== 'undefined' ? globalThis : this);
