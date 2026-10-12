#!/usr/bin/env python3
"""Check the HTML that app.py builds for the Streamlit iframe, without Streamlit.

app.py runs its whole UI at import time, so it cannot be imported in CI without a
Streamlit runtime. The page template, though, is a plain function. This test pulls
get_game_html() and its two helpers out of the source with `ast`, calls it with a
spread of settings, and parse-checks every inline <script> with JavaScriptCore.

What it catches: an unescaped `{` or `}` in the f-string (a syntax error in the
generated JavaScript that only shows up as a blank game), a setter that the sidebar
sends but game.js no longer defines, and a user-supplied string that closes the
script block early.

Run from the repository root:  python3 tests/test_app_template.py
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSC = ("/System/Library/Frameworks/JavaScriptCore.framework/"
       "Versions/A/Helpers/jsc")
WANTED_FUNCS = {"js_string", "read_source", "get_game_html", "load_policy_file"}
WANTED_NAMES = {"TFJS_TAG", "TFJS_WASM_TAG", "THREE_JS_TAG", "POLICY_DIR", "MAX_POLICY_BYTES"}


def load_template_namespace():
    with open(os.path.join(ROOT, "app.py")) as f:
        tree = ast.parse(f.read())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCS:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in WANTED_NAMES for t in node.targets):
            keep.append(node)
    ns = {"json": json, "os": os}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "app.py", "exec"), ns)
    missing = (WANTED_FUNCS | WANTED_NAMES) - set(ns)
    assert not missing, f"could not extract from app.py: {sorted(missing)}"
    return ns


class ScriptCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inline, self.external, self._in = [], [], False
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            src = dict(attrs).get("src")
            if src:
                self.external.append(src)
            else:
                self._in, self._buf = True, []

    def handle_endtag(self, tag):
        if tag == "script" and self._in:
            self.inline.append("".join(self._buf))
            self._in = False

    def handle_data(self, data):
        if self._in:
            self._buf.append(data)


def syntax_ok(source):
    # jsc cannot read stdin, so the script goes through a temp file.
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(source)
        path = handle.name
    try:
        proc = subprocess.run(
            [JSC, "-e", "try { new Function(read(arguments[0])); print('OK'); } "
                        "catch (e) { print('SYNTAX ' + e); }", "--", path],
            capture_output=True, text=True)
    finally:
        os.unlink(path)
    return proc.stdout.strip() == "OK", proc.stdout.strip() + proc.stderr.strip()


CASES = {
    "defaults (torus)": {},
    "sphere + camera": dict(world_mode="sphere", sphere_camera="boss", sphere_zoom=3.0, world_scale=2.0),
    "ML + MARL + trained policy": dict(ml_enabled=True, marl_enabled=True, nn_models_enabled=True,
                                       trained_policy_enabled=True),
    "hostile strings": dict(formation_type='"</script><script>alert(1)//', multi_target_mode="'; alert(1); '",
                            escort_mode="\\"),
    "everything off": dict(fx_enabled=False, hud_enabled=False, sfx_enabled=False, music_enabled=False,
                           kill_cam_enabled=False, fuel_rescue_enabled=False, player_ship_active=True),
}
BASE = dict(num_ai_ships=10, num_enemy_ships=10, num_boss_ships=10, canvas_width=1200,
            canvas_height=600, player_ship_active=False, ml_enabled=False)


def main():
    if not os.path.exists(JSC):
        print("JavaScriptCore not found; skipping template test")
        return 0
    os.chdir(ROOT)
    ns = load_template_namespace()
    # models/ is not versioned, so give the trained-policy case a stand-in
    # checkpoint inside a temporary POLICY_DIR (the real function still validates it).
    policy_dir = tempfile.mkdtemp()
    fake = os.path.join(policy_dir, "policy_stage0.json")
    with open(fake, "w") as handle:
        json.dump({"format": "asteroids-ppo-policy", "obsDim": 38}, handle)
    ns["POLICY_DIR"] = policy_dir
    CASES["ML + MARL + trained policy"]["trained_policy_path"] = fake
    failures = 0
    for name, overrides in CASES.items():
        html = ns["get_game_html"](**{**BASE, **overrides})
        parser = ScriptCollector()
        parser.feed(html)
        problems = []

        if not parser.inline:
            problems.append("no inline <script> found")
        for i, body in enumerate(parser.inline):
            ok, detail = syntax_ok(body)
            if not ok:
                problems.append(f"inline script {i}: {detail[:200]}")
        if name == "hostile strings":
            if re.search(r"alert\(1\)", "".join(parser.inline)) and "\\u003c/script" not in html:
                problems.append("hostile string appears unescaped in a script block")
            if any("alert(1)" in s and "\\u003c" not in s for s in parser.inline):
                problems.append("hostile string was not escaped")
        if name == "defaults (torus)":
            joined = "".join(parser.inline)
            for needed in ("setSoundEnabled", "setMusicEnabled", "setKillCamEnabled"):
                if needed not in joined:
                    problems.append(f"missing {needed} in the injected settings")

        status = "FAIL" if problems else "PASS"
        print(f"  {status}  {name}  ({len(parser.inline)} inline scripts, {len(html) // 1024} KB)")
        for p in problems:
            print(f"        - {p}")
        failures += bool(problems)

    print("ALL TEMPLATE CHECKS PASSED" if not failures else f"{failures} TEMPLATE CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
