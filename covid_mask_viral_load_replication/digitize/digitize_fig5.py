"""Read the box plots off the paper's Figure 5.

Usage: python digitize_fig5.py FIG5.png OUT.tsv

FIG5.png is the full-size figure from the journal site,
https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-021-91338-5/MediaObjects/41598_2021_91338_Fig5_HTML.png
(1773 x 712 pixels).  Both panels share a log10 axis from 4 (row 621) to 8
(row 66), 138.75 pixels per log10, so quartiles are read to about 0.01.
Boxes are found by their fill colour; the median is the dark line inside the
fill and the whiskers are the dark vertical line through the box centre.
"""
import sys

import numpy as np
import pandas as pd
from matplotlib import image as mpimg

COLOURS = {
    "unmasked_to_unmasked": (68, 1, 84),
    "unmasked_to_masked": (49, 104, 142),
    "masked_to_unmasked": (53, 183, 121),
    "masked_to_masked": (253, 231, 37),
}
PANELS = {"transmitter": (165, 885), "exposure": (1045, 1765)}   # x range of each plot area
ROW_AT_4, ROW_AT_8 = 621.0, 66.0
EFFICACY = [0.0, 0.3, 0.6, 0.9]


def to_log10(row):
    return 4.0 + 4.0 * (ROW_AT_4 - row) / (ROW_AT_4 - ROW_AT_8)


def runs(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    parts = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
    return [(int(p[0]), int(p[-1])) for p in parts]


def main(png, out):
    im = mpimg.imread(png)[:, :, :3] * 255.0
    # The legend of panel A is drawn inside the plot area, below log10 = 5.4
    # and right of the first group of boxes.  Blank it before reading.
    im[428:619, 492:889] = 255.0
    area = slice(70, 618)
    rows = []
    for panel, (xa, xb) in PANELS.items():
        for name, rgb in COLOURS.items():
            hit = np.abs(im[area, xa:xb] - np.array(rgb)).sum(axis=2) < 60
            # columns holding a box of this colour
            boxes = [r for r in runs(hit.sum(axis=0) > 3) if r[1] - r[0] > 15]
            assert len(boxes) == 4, (panel, name, boxes)
            for eps, (c0, c1) in zip(EFFICACY, boxes):
                xc = xa + (c0 + c1) // 2
                fill = np.flatnonzero(hit[:, c0 + 4:c1 - 3].any(axis=1)) + 70
                top, bottom = fill.min(), fill.max()
                inside = im[top:bottom + 1, xa + c0 + 6:xa + c1 - 5].mean(axis=1)
                dark = np.flatnonzero(inside.max(axis=1) < 90) + top
                dark = dark[(dark > top + 2) & (dark < bottom - 2)]
                median = dark.mean() if len(dark) else np.nan
                column = im[area, xc - 1:xc + 2].mean(axis=1)
                dark_col = np.flatnonzero(column.max(axis=1) < 110) + 70
                rows.append({
                    "panel": panel, "efficacy": eps, "pair": name,
                    "whisker_low": to_log10(dark_col.max()), "q1": to_log10(bottom + 1.5),
                    "median": to_log10(median), "q3": to_log10(top - 1.5),
                    "whisker_high": to_log10(dark_col.min()),
                })
    df = pd.DataFrame(rows).sort_values(["panel", "efficacy"], kind="stable")
    df.to_csv(out, sep="\t", index=False, float_format="%.3f")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
