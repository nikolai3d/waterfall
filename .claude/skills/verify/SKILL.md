---
name: verify
description: Verify waterfall changes by driving the WebGL2 app in headless Chrome and screenshotting it
---

# Verifying waterfall

No test suite; the only end-to-end check is running the app and looking at it.

## Serve + screenshot

```sh
python3 -m http.server 8123   # in the repo root; ES modules need http
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --no-first-run --window-size=900,650 \
  --screenshot=out.png --virtual-time-budget=4000 \
  "http://localhost:8123/?warm=800&dbg=1"
```

- `?warm=N` pre-runs N substeps synchronously before the first frame — use it
  to reach a steady state; do NOT rely on `--virtual-time-budget` to simulate
  time (it only advances ~7 rAF frames per virtual second).
- On the real GPU (no swiftshader flags) warm=800 takes seconds. Add
  `--use-angle=swiftshader --enable-unsafe-swiftshader` only if a GPU-less
  environment forces it; then keep warm small (`?p=128&l=800&warm=600`).
- The HUD stats line (bottom-left) and `?dbg=1` overlay are the observable
  state; page console.log does not reach the Chrome log. Stats refresh only
  every 500ms of rAF time, so under virtual time they can lag several UI
  actions behind — trust chips/URL for immediate state, stats only after a
  generous tail delay.
- `?r=points` exercises the legacy impostor render path.
- macOS has no `timeout` command.

## Driving the UI (clicks) headlessly

`--screenshot` can't click. Write a temporary same-origin harness page in the
repo root (e.g. `_verify_ui.html`, delete after) that iframes `/?...`, waits
for `load` (module script incl. warmup runs before it), then dispatches
`.click()` on `#panel button[data-g=...]` / `[data-p=...]` chips and
keyboard events on the iframe window, logging iframe `location.search`,
`#stats` text, `#err` visibility, and `webglcontextlost` into an outer
`<pre>` so the screenshot doubles as the report. Use generous virtual-time
budget (15000+) and sleeps between actions.
