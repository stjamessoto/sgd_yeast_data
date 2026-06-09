"""
generate_figure1.py — generate assets/subnetwork_motifs.png (Figure 1 from Scruse et al.)

Run once:  python assets/generate_figure1.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Node colour scheme matching the paper figure
BLUE  = "#1565C0"
GREEN = "#2E7D32"
GRAY  = "#757575"
RED   = "#C62228"

# Each motif: (top_colour, bottom_left_colour, bottom_right_colour, label)
MOTIFS = [
    (BLUE,  GREEN, GRAY,  "A"),
    (GRAY,  BLUE,  RED,   "B"),
    (RED,   BLUE,  GREEN, "C"),
    (GREEN, RED,   GRAY,  "D"),
]

NODE_RADIUS = 0.07
LINE_WIDTH  = 3.0
NODE_COORDS = dict(top=(0.5, 0.80), bot_l=(0.15, 0.22), bot_r=(0.85, 0.22))

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
fig.patch.set_facecolor("white")

for ax, (top_c, bl_c, br_c, label) in zip(axes.flat, MOTIFS):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    top = NODE_COORDS["top"]
    bl  = NODE_COORDS["bot_l"]
    br  = NODE_COORDS["bot_r"]

    # Edges
    for (x0, y0), (x1, y1) in [(top, bl), (top, br)]:
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=LINE_WIDTH, zorder=1)

    # Nodes
    for (x, y), colour in [(top, top_c), (bl, bl_c), (br, br_c)]:
        circle = mpatches.Circle((x, y), NODE_RADIUS, color=colour, zorder=2)
        ax.add_patch(circle)

    # Label
    ax.text(0.50, 0.51, label, fontsize=36, fontweight="bold",
            ha="center", va="center", color="black", zorder=3)

plt.tight_layout(pad=1.5)
out = Path(__file__).parent / "subnetwork_motifs.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
