// Tests for escalating waves and boss phase mechanics.
//
// Both are off by default in the headless build, because the benchmark baseline
// and every published training result were measured without them. The first test
// pins that. The rest turn them on through the env options and check the schedule.

load('sim/browser_shim.js');
load('rng.js');
load('theme.js');
load('world.js');
load('ml_features.js');
load('entities.js');
load('ai_ship.js');
load('enemy_ship.js');
load('enemy_coordination.js');
load('game.js');
load('sim/env.js');

globalThis.draw = function () {};

let failures = 0;
function check(name, condition, detail) {
    if (condition) print(`  PASS  ${name}${detail ? '  (' + detail + ')' : ''}`);
    else { print(`  FAIL  ${name}${detail ? '  (' + detail + ')' : ''}`); failures++; }
}

function types() { return enemyShips.filter(e => e).map(e => e.type); }
function count(type) { return types().filter(t => t === type).length; }
function frames(n) { for (let i = 0; i < n; i++) update(); }
// Kill everything hostile the way the game does after a hit: remove it from the array.
function wipeEnemies() { enemyShips = []; }

print('Defaults');
(function defaults() {
    const env = new AsteroidsEnv({ numAgents: 3, maxSteps: 50, curriculumStage: 3 });
    env.reset(11);
    check('waves are off in the headless env by default', wavesEnabled === false);
    check('boss phase mechanics are off by default', bossPhasePlusEnabled === false);
    check('static fleet spawns immediately without waves', enemyShips.length === 3, 'ships=' + enemyShips.length);
})();

print('Wave schedule');
(function schedule() {
    const env = new AsteroidsEnv({ numAgents: 2, maxSteps: 5000, curriculumStage: 3,
        enemies: 10, bosses: 2, waves: true });
    env.reset(21);
    check('wave mode starts with an empty field', enemyShips.length === 0, 'ships=' + enemyShips.length);
    check('starts in intermission before wave 1', waveState.mode === 'intermission' && waveState.wave === 0);

    frames(WAVE_FIRST_INTERMISSION_FRAMES + 2);
    check('wave 1 begins after the intermission', waveState.wave === 1 && waveState.mode === 'active', 'wave=' + waveState.wave);
    const w1 = count('basic') + count('advanced');
    check('wave 1 sends a fraction of the configured enemies', w1 >= 1 && w1 < 10, 'enemies=' + w1);
    check('no boss on wave 1', count('boss') === 0);

    wipeEnemies();
    const scoreBefore = score;
    frames(1);
    check('clearing the field ends the wave', waveState.mode === 'intermission' && waveState.cleared === 1);
    check('clear bonus is paid (100 x wave)', score - scoreBefore >= 100, 'delta=' + (score - scoreBefore));

    frames(WAVE_INTERMISSION_FRAMES + 2);
    check('wave 2 follows', waveState.wave === 2 && waveState.mode === 'active');
    const w2 = count('basic') + count('advanced');
    check('wave 2 is bigger than wave 1', w2 > w1, `${w1} -> ${w2}`);

    wipeEnemies(); frames(1); frames(WAVE_INTERMISSION_FRAMES + 2);
    check('wave 3 is a boss wave', waveState.wave === 3 && count('boss') === 1, 'bosses=' + count('boss'));

    // Never exceed the sliders.
    for (let n = 4; n <= 14; n++) {
        wipeEnemies(); frames(1); frames(WAVE_INTERMISSION_FRAMES + 2);
        if (count('basic') + count('advanced') > 10 || count('boss') > 2) {
            check('wave ' + n + ' respects the configured ceilings', false); break;
        }
    }
    check('enemy and boss counts never exceed the configured ceilings', waveState.wave >= 14);
})();

print('Wave pressure');
(function pressure() {
    const early = waveSpec(1), late = waveSpec(12);
    check('later waves fire faster', late.firePressure > early.firePressure);
    check('pressure is capped', waveSpec(500).firePressure <= 2.0 && waveSpec(500).speedPressure <= 1.5);
    const env = new AsteroidsEnv({ numAgents: 1, maxSteps: 5000, curriculumStage: 3, enemies: 6, bosses: 0, waves: true });
    env.reset(5);
    numEnemyShips = 6;
    // Jump the schedule to a late wave and compare an enemy against a fresh one.
    waveState.wave = 11; waveState.timer = 1;
    frames(2);
    const spawned = enemyShips.filter(e => e)[0];
    const base = new EnemyShip(0, 0, spawned.type);
    check('late-wave enemies fire faster than a fresh one', spawned.fireRate < base.fireRate,
        `${spawned.fireRate} vs ${base.fireRate}`);
    check('late-wave enemies are faster', spawned.maxVelocity > base.maxVelocity);
})();

print('Determinism and restart');
(function determinism() {
    function run(seed) {
        const env = new AsteroidsEnv({ numAgents: 2, maxSteps: 3000, curriculumStage: 3, enemies: 5, bosses: 1, waves: true });
        env.reset(seed);
        const log = [];
        for (let t = 0; t < 1200; t++) {
            update();
            if (t % 100 === 0) log.push([waveState.wave, waveState.mode, enemyShips.length, score]);
        }
        return JSON.stringify(log);
    }
    check('same seed gives the same wave sequence', run(77) === run(77));
    const env = new AsteroidsEnv({ numAgents: 1, maxSteps: 3000, curriculumStage: 3, enemies: 5, bosses: 1, waves: true });
    env.reset(3);
    frames(WAVE_FIRST_INTERMISSION_FRAMES + 2);
    check('a wave is running before the reset', waveState.wave === 1);
    initGame();
    check('initGame restarts the wave sequence', waveState.wave === 0 && waveState.mode === 'intermission' && enemyShips.length === 0);
})();

print('setWavesEnabled');
(function toggle() {
    const env = new AsteroidsEnv({ numAgents: 1, maxSteps: 3000, curriculumStage: 3, enemies: 4, bosses: 1 });
    env.reset(9);
    check('static fleet before enabling waves', enemyShips.length === 5);
    setWavesEnabled(true);
    check('enabling waves clears the static fleet', enemyShips.length === 0 && waveState.wave === 0);
    const w = waveState;
    setWavesEnabled(true);
    check('re-sending the same setting is a no-op', waveState === w);
    setWavesEnabled(false);
    check('disabling waves restores the static fleet', enemyShips.length === 5, 'ships=' + enemyShips.length);
})();

print('Boss phases');
(function bossPhases() {
    function bossRun(enabled) {
        const env = new AsteroidsEnv({ numAgents: 2, maxSteps: 3000, curriculumStage: 2, enemies: 0, bosses: 1, bossPhases: enabled });
        env.reset(31);
        const boss = enemyShips.find(e => e && e.type === 'boss');
        return { env, boss };
    }
    const off = bossRun(false);
    off.boss.health = Math.floor(off.boss.maxHealth * 0.5);
    off.boss.checkHealth();
    check('phase 2 is reached by damage', off.boss.bossPhase === 2);
    check('no escorts without the mechanic', enemyShips.length === 1);

    const on = bossRun(true);
    const fireRate0 = on.boss.fireRate, vel0 = on.boss.maxVelocity;
    on.boss.health = Math.floor(on.boss.maxHealth * 0.5);
    on.boss.checkHealth();
    check('phase 2 summons two escorts', enemyShips.length === 3, 'ships=' + enemyShips.length);
    on.boss.health = Math.floor(on.boss.maxHealth * 0.5);
    on.boss.checkHealth();
    check('escorts are summoned only once', enemyShips.length === 3);

    on.boss.health = 1;
    on.boss.checkHealth();
    check('phase 3 enrages: faster fire', on.boss.fireRate < fireRate0, `${fireRate0} -> ${on.boss.fireRate}`);
    check('phase 3 enrages: faster movement', on.boss.maxVelocity > vel0);
    const rate = on.boss.fireRate;
    on.boss.checkHealth();
    check('enrage applies once', on.boss.fireRate === rate);
})();

print(failures === 0 ? '\nALL WAVE TESTS PASSED' : `\n${failures} WAVE TEST(S) FAILED`);
if (failures) throw new Error(failures + ' wave test(s) failed');
