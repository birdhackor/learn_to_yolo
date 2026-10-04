"""Shared settings for the website's program-generated charts.

Text stays as SVG <text>, so each reader's browser draws the Traditional Chinese labels with its own
fonts; no CJK font is needed on the machine that draws the chart. The saved file carries no timestamp,
so the same data always gives the same bytes.
"""
from __future__ import annotations

from pathlib import Path
import warnings

FONT_STACK = "'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', 'Heiti TC', 'DejaVu Sans', sans-serif"


def use_svg_text():
    """Call before creating figures."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "learn-to-yolo",
                                "font.family": "DejaVu Sans", "axes.unicode_minus": False})
    # Layout uses DejaVu metrics; the browser supplies the CJK glyphs.
    warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")


def save_svg(fig, path: str | Path, title: str, label: str) -> str:
    """Save fig as an accessible SVG with the CJK font stack and no timestamp; return the text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Date": None})
    svg = path.read_text()
    svg = svg.replace("font-family: 'DejaVu Sans'", f"font-family: {FONT_STACK}")
    svg = svg.replace("<svg ", f'<svg role="img" aria-label="{label}" ', 1)
    svg = svg.replace('version="1.1">', f'version="1.1"><title>{title}</title>', 1)
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    path.write_text(svg)
    return svg
