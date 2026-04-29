"""
Render the dashboard HTML from the Jinja template.

Reads `dashboard/template.html.j2`, substitutes metadata (build date,
data manifest), writes `dashboard/index.html`.

Usage:
    cd "Step3. April_v1.1"
    python scripts/build_dashboard.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STEP_DIR = SCRIPT_DIR.parent
DASH_DIR = STEP_DIR / "dashboard"
TEMPLATE = DASH_DIR / "template.html.j2"
INDEX = DASH_DIR / "index.html"
MANIFEST = DASH_DIR / "data" / "_manifest.json"


def main() -> int:
    if not TEMPLATE.exists():
        print(f"[ERROR] missing template {TEMPLATE}", file=sys.stderr)
        return 1

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        print("[ERROR] jinja2 not installed — pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    env = Environment(
        loader=FileSystemLoader(DASH_DIR),
        autoescape=select_autoescape(["html", "j2"]),
    )
    tmpl = env.get_template(TEMPLATE.name)

    context = {
        "build_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    if MANIFEST.exists():
        try:
            context["manifest"] = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    html = tmpl.render(**context)
    INDEX.write_text(html, encoding="utf-8")
    size_kb = INDEX.stat().st_size / 1024
    print(f"[INFO] wrote {INDEX.relative_to(STEP_DIR)} ({size_kb:,.1f} KB)")

    if not (DASH_DIR / "data").exists():
        print("[WARN] dashboard/data/ is missing — run export_dashboard_data.py first "
              "so the HTML has JSON to fetch", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
