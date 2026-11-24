"""Read the bar heights off the paper's Figure 4.

Usage: python digitize_fig4.py FIG4.png OUT.tsv

FIG4.png is the full-size figure from the journal site,
https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-021-91338-5/MediaObjects/41598_2021_91338_Fig4_HTML.png
(1502 x 1249 pixels).  The axes are 240 pixels per unit of R0, so a height is
read to about 0.005.
"""
import sys

import numpy as np
import pandas as pd
from matplotlib import image as mpimg

COLOURS = {  # legend colours, RGB
    "unmasked_to_unmasked": (68, 1, 84),
    "unmasked_to_masked": (49, 104, 142),
    "masked_to_unmasked": (53, 183, 121),
    "masked_to_masked": (253, 231, 37),
}
PANELS = {  # coverage label -> (row of y=0 axis line, left edge of first bar)
    "25/25": (549.5, 146), "50/50": (549.5, 888),
    "75/75": (1164.5, 146), "90/90": (1164.5, 888),
}
BAR_WIDTH, BAR_PITCH, PX_PER_UNIT = 115, 154, 240.0
EFFICACY = [0.0, 0.3, 0.6, 0.9]


def main(png, out):
    im = mpimg.imread(png)[:, :, :3] * 255.0
    rows = []
    for panel, (y0, x0) in PANELS.items():
        for j, eps in enumerate(EFFICACY):
            xc = int(x0 + j * BAR_PITCH + BAR_WIDTH / 2)
            # Pixels of this bar from the axis upward, bottom first.
            column = im[int(y0 - 485):int(y0), xc - 20:xc + 20].mean(axis=1)[::-1]
            hits = {name: np.abs(column - np.array(rgb)).sum(axis=1) < 60
                    for name, rgb in COLOURS.items()}
            in_bar = np.any(list(hits.values()), axis=0)
            # The bar ends at the first long run of background; this keeps the
            # legend swatches drawn above the bars of panel D out of the count.
            gap = np.convolve(~in_bar, np.ones(6), mode="valid") == 6
            top = int(np.argmax(gap)) if gap.any() else len(in_bar)
            for name, hit in hits.items():
                rows.append({"coverage": panel, "efficacy": eps, "pair": name,
                             "R0_contribution": hit[:top].sum() / PX_PER_UNIT})
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False, float_format="%.4f")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
