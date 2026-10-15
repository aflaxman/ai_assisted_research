"""Wrap the artifact page into a standalone HTML document.

`yf_replication.html` is authored for Claude's Artifact host, which supplies the
document skeleton (doctype, charset, viewport, a small reset) at publish time.
Served raw -- for example by GitHub Pages -- that file would render in quirks
mode with no viewport meta. This script adds the equivalent skeleton and writes
`index.html`, so one source serves both targets.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "yf_replication.html"
OUT = ROOT / "index.html"

# Mirrors the skeleton the Artifact host injects, so the standalone page and the
# published artifact render identically.
SKELETON_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
:root{
  color-scheme:light;
  padding-top:env(safe-area-inset-top,0px);
  padding-bottom:env(safe-area-inset-bottom,0px);
}
body{margin:0;font:14px system-ui,sans-serif}
img{max-width:100%}
[hidden]{display:none!important}
</style>
"""


def main() -> None:
    src = SRC.read_text()
    if src.lstrip().lower().startswith("<!doctype"):
        raise SystemExit(f"{SRC.name} already carries a skeleton; nothing to wrap")
    marker = "</style>\n\n<div class=\"wrap\">"
    if marker not in src:
        raise SystemExit("could not find the head/body boundary in " + SRC.name)
    head, body = src.split(marker, 1)
    OUT.write_text(SKELETON_HEAD + head + "</style>\n</head>\n<body>\n<div class=\"wrap\">"
                   + body + "\n</body>\n</html>\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
