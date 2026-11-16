#!/usr/bin/env python3
"""Build every shipped page and artifact from source, in one go.

    .venv/bin/python scripts/build_site.py             # everything
    .venv/bin/python scripts/build_site.py --no-report # skip the (slower) report

Sources live in web/. Each page is assembled into a single self-contained HTML
file (fonts, styles and scripts inlined, no network requests) and written to
the repo root, where GitHub Pages serves it, plus a copy in handoff/.

Order of work:
  1. solve + fit the deployed function        (src/deploy.py)
  2. assemble the simulator                   (web/sim.html + sim.js + alphasim_core.js)
  3. write the JSON / Java / Python artifacts, and stamp the function into the simulator
  4. compute the report's figures and assemble it   (scripts/build_report.py)
  5. assemble the landing page
"""
import argparse
import base64
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_coefficients import deployment_from_configs, write_artifacts  # noqa: E402
from src.config import load_game, load_robot  # noqa: E402

WEB = REPO_ROOT / "web"
FONT_FACES = (
    ("Barlow", 400, "normal", "barlow-400"),
    ("Barlow", 400, "italic", "barlow-400i"),
    ("Barlow", 600, "normal", "barlow-600"),
    ("Barlow Semi Condensed", 500, "normal", "barlow-sc-500"),
    ("Barlow Semi Condensed", 600, "normal", "barlow-sc-600"),
)
INCLUDE = re.compile(r"/\*@include ([\w./-]+)\*/")


def font_css() -> str:
    """@font-face rules with the subsetted Barlow files inlined (SIL OFL 1.1,
    see web/fonts/OFL.txt)."""
    rules = []
    for family, weight, style, stem in FONT_FACES:
        data = base64.b64encode((WEB / "fonts" / f"{stem}.woff2").read_bytes()).decode()
        rules.append(
            f'@font-face{{font-family:"{family}";font-weight:{weight};font-style:{style};'
            f'font-display:swap;src:url(data:font/woff2;base64,{data}) format("woff2")}}')
    return "/* Barlow, (c) The Barlow Project Authors, SIL Open Font License 1.1 */\n" + "\n".join(rules)


def assemble(source: str, extra: dict | None = None) -> str:
    """Inline /*@include file*/ and /*@fonts*/ markers; fill {{name}} slots."""
    text = (WEB / source).read_text()
    text = text.replace("/*@fonts*/", font_css())
    text = INCLUDE.sub(lambda m: (WEB / m.group(1)).read_text(), text)
    for key, value in (extra or {}).items():
        text = text.replace("{{" + key + "}}", value)
    return text


def write_page(name: str, html: str, handoff: bool = True) -> None:
    (REPO_ROOT / name).write_text(html)
    if handoff:
        (REPO_ROOT / "handoff" / name).write_text(html)
    print(f"built {name}  ({len(html) / 1024:.0f} kB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-report", action="store_true", help="skip the research report")
    ap.add_argument("--game-config", default=str(REPO_ROOT / "config" / "game_2026.yaml"))
    ap.add_argument("--robot-config", default=str(REPO_ROOT / "config" / "robot.yaml"))
    args = ap.parse_args()

    game, robot = load_game(args.game_config), load_robot(args.robot_config)
    dep = deployment_from_configs(game, robot)

    write_page("shooter_sim.html", assemble("sim.html"))
    for path in write_artifacts(dep):
        print(f"wrote {path.relative_to(REPO_ROOT)}")

    if not args.no_report and (WEB / "report.html").exists():
        from build_report import build_report
        write_page("research_report.html", build_report(game, robot, dep, assemble))
    if (WEB / "index.html").exists():
        from build_report import landing_slots
        write_page("index.html", assemble("index.html", landing_slots(game, robot, dep)),
                   handoff=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
