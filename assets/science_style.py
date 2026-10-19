"""
science_style.py — shared matplotlib settings for the Science figures (Figs. 2-4).

Science figure requirements this enforces:
  - sans-serif type (Helvetica; Arial is the metric-compatible substitute
    installed on Windows), nothing smaller than 5 pt at printed size
  - figures drawn at their final printed width (1, 2 or 3 columns =
    5.7, 12.1 or 18.4 cm), saved at 300 dpi, no title inside the image
  - bold uppercase 10 pt panel labels in the upper left of each panel
  - no minor ticks, grid lines or screen shading
  - colorblind-safe palette (Okabe-Ito); never red with green, never two
    similar hues for different groups
"""

import matplotlib as mpl

CM = 1 / 2.54
WIDTH_1COL = 5.7 * CM
WIDTH_2COL = 12.1 * CM
WIDTH_3COL = 18.4 * CM
DPI = 300

# Okabe & Ito (2008) colorblind-safe palette
BLUE = "#0072B2"
VERMILLION = "#D55E00"
ORANGE = "#E69F00"
SKY = "#56B4E9"
BLACK = "#000000"
GRAY = "#7F7F7F"
LIGHT_GRAY = "#BFBFBF"

MIN_PT = 5


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(fig, ax, letter, dx=-0.02, dy=0.0):
    """Bold uppercase 10 pt label at the upper-left corner of ax (figure coords)."""
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, letter, fontsize=10, fontweight="bold",
             ha="right", va="bottom")
