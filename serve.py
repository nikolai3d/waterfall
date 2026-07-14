#!/usr/bin/env python3
# Dev server: threaded (parallel browser connections) + no-store (a plain
# `python3 -m http.server` lets Chrome heuristically cache the JS modules,
# so a tab can silently keep testing stale shader code after edits).
import http.server
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


print(f"serving on http://localhost:{port}")
http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
