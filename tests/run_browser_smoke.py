#!/usr/bin/env python3
"""Open the real game in headless Chrome and run tests/browser_smoke.html against it.

No Playwright or Selenium: the harness page checks itself and writes a JSON verdict
into the DOM, and this script reads it back from `chrome --dump-dom`. Chrome ships
on the GitHub runners and on most developer machines.

It also fails if the page logs an uncaught exception, because a broken script can
still leave a page that "loads".

Usage:
    python3 tests/run_browser_smoke.py            # find Chrome automatically
    CHROME=/path/to/chrome python3 tests/run_browser_smoke.py
    python3 tests/run_browser_smoke.py --extra    # also run extra feature checks
"""

import argparse
import functools
import http.server
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = [
    os.environ.get("CHROME", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome",
]


def find_chrome():
    for candidate in CANDIDATES:
        if candidate and (os.path.exists(candidate) or shutil.which(candidate)):
            return candidate
    return None


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="store_true", help="run the feature checks in tests/browser_extra.js")
    ap.add_argument("--budget", type=int, default=20000, help="virtual-time budget in ms")
    args = ap.parse_args()

    chrome = find_chrome()
    if not chrome:
        print("Chrome not found; set CHROME=/path/to/chrome")
        return 2

    handler = functools.partial(QuietHandler, directory=ROOT)
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://127.0.0.1:{port}/tests/browser_smoke.html" + ("?extra=1" if args.extra else "")
    cmd = [chrome, "--headless=new", "--disable-gpu", "--use-gl=angle", "--use-angle=swiftshader",
           "--enable-unsafe-swiftshader", "--no-sandbox", "--mute-audio",
           f"--virtual-time-budget={args.budget}", "--enable-logging=stderr", "--v=0",
           "--dump-dom", url]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    server.shutdown()

    match = re.search(r'<pre id="smoke-result">(.*?)</pre>', proc.stdout, re.S)
    if not match or match.group(1).strip() == "pending":
        print("the smoke page never reported a result")
        print(proc.stderr[-1500:])
        return 1
    verdict = json.loads(match.group(1).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))

    # Console lines from Chrome look like: INFO:CONSOLE:123] "message", source: url
    uncaught = [line for line in proc.stderr.splitlines()
                if "CONSOLE" in line and re.search(r'Uncaught|ReferenceError|TypeError|SyntaxError', line)]

    for r in verdict["results"]:
        print(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['name']}" + (f"  ({r['detail'][:160]})" if not r["ok"] and r["detail"] else ""))
    for line in uncaught[:10]:
        print("  FAIL  console error:", line.split("CONSOLE", 1)[-1][:200])

    failed = verdict["failed"] + len(uncaught)
    print("BROWSER SMOKE PASSED" if not failed else f"BROWSER SMOKE FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
