"""Read the cell values off the paper's Figure 3 heat maps.

Usage: python digitize_fig3.py FIG3.png OUT.tsv

FIG3.png is the full-size figure from the journal site,
https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-021-91338-5/MediaObjects/41598_2021_91338_Fig3_HTML.png
(1498 x 1245 pixels).  Each panel is an 11 x 11 grid of flat-coloured cells;
a cell's colour is matched to the nearest colour on the figure's own colour
bar, which runs linearly from 0 (row 1106) to 2 (row 80).  The bar has 513
pixels per unit, so a value is read to about 0.005, and to about 0.02 near 1.0
where the scale passes through white and neighbouring colours are close.
"""
import sys

import numpy as np
import pandas as pd
from matplotlib import image as mpimg

#: coverage label -> (left, right, top, bottom) of the heat map interior, pixels
PANELS = {
    "25/25": (148, 706, 70, 513),
    "50/50": (804, 1363, 70, 513),
    "75/75": (148, 706, 659, 1102),
    "100/100": (804, 1363, 659, 1102),
}
BAR_X = (1395, 1430)
BAR_ROWS = (80, 1106)   # value 2 at the top row, 0 at the bottom row
LEVELS = np.round(np.arange(0.0, 1.01, 0.1), 1)


def read_panels(png, panels, bar_x, bar_rows, vmax=2.0):
    """Decode 11 x 11 heat maps against the figure's own colour bar."""
    im = mpimg.imread(png)[:, :, :3] * 255.0
    top, bottom = bar_rows
    bar = im[top:bottom + 1, bar_x[0]:bar_x[1]].mean(axis=1)
    bar_value = vmax * (bottom - np.arange(top, bottom + 1)) / (bottom - top)
    rows = []
    for panel, (x0, x1, y0, y1) in panels.items():
        cw, ch = (x1 - x0) / 11.0, (y1 - y0) / 11.0
        for i, eps_t in enumerate(LEVELS):          # rows, bottom to top
            for j, eps_e in enumerate(LEVELS):      # columns, left to right
                xc = x0 + (j + 0.5) * cw
                yc = y1 - (i + 0.5) * ch
                patch = im[int(yc - ch / 4):int(yc + ch / 4), int(xc - cw / 4):int(xc + cw / 4)]
                colour = np.median(patch.reshape(-1, 3), axis=0)
                dist = ((bar - colour) ** 2).sum(axis=1)
                rows.append({"coverage": panel, "eps_T": eps_t, "eps_E": eps_e,
                             "mean_R0": bar_value[dist.argmin()],
                             "colour_error": np.sqrt(dist.min())})
    return pd.DataFrame(rows)


def main(png, out):
    read_panels(png, PANELS, BAR_X, BAR_ROWS).to_csv(out, sep="\t", index=False, float_format="%.3f")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
