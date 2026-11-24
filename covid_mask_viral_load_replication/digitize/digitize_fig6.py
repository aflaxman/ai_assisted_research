"""Read two King County series off the paper's Figure 6.

Usage: python digitize_fig6.py FIG6.png OUT.tsv

FIG6.png is the full-size figure from the journal site,
https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-021-91338-5/MediaObjects/41598_2021_91338_Fig6_HTML.png
(1502 x 1490 pixels, visibly upscaled from a smaller original, so every curve
is blurred over 4 to 6 pixels).  One day is 6.22 pixels.  Two curves are read,
daily from 1 March to 30 August 2020:

  mask_always   panel B, the black smoothed line: fraction who report always
                wearing a mask in public (SCAN survey from 7 May; before that,
                the authors' logit-linear interpolation from 1% on 1 March)
  Re            panel C, the thin red line: R_e estimated from King County
                cases, hospital admissions and deaths (Thakkar & Famulare)
  Re_low/high   panel C, the edges of the red 95% band

The image is too blurred, and panel C too crowded, for thresholding alone.
Each curve is therefore followed from guesses read by eye (the ANCHORS
below): on each day the guess is moved to the nearest ridge of the right
colour within a tolerance, and days where no ridge is found (the red line
runs under the black ones for stretches of July) are filled by interpolation
and flagged.  Calibration checks against the text: the red line ends at 0.80
on 30 August (text: 0.8) inside a band of 0.46 to 1.15 (text: 0.5 to 1.2).
I would trust R_e to about 0.03 and the mask fraction to about 0.01.
"""
import sys

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from scipy.interpolate import PchipInterpolator

DAYS = pd.date_range("2020-03-01", "2020-08-30")
PX_PER_DAY = 6.22
B_X0, B_Y0, B_PX = 153.0, 871.0, 490.0        # panel B: x of 1 March, row of 0, pixels per unit
C_X0, C_Y0, C_PX = 144.0, 1448.5, 145.0       # panel C

ANCHORS_RE = {
    "03-01": 2.52, "03-05": 2.05, "03-09": 1.65, "03-13": 1.40, "03-18": 1.36, "03-23": 1.33,
    "03-27": 1.12, "03-31": 0.88, "04-05": 0.73, "04-12": 0.70, "04-20": 0.71, "04-28": 0.74,
    "05-05": 0.80, "05-10": 0.83, "05-15": 0.76, "05-19": 0.72, "05-23": 0.90, "05-26": 0.96,
    "05-30": 0.85, "06-02": 0.80, "06-06": 0.95, "06-10": 1.25, "06-14": 1.48, "06-18": 1.54,
    "06-22": 1.42, "06-26": 1.20, "06-30": 1.10, "07-05": 1.06, "07-12": 1.02, "07-20": 1.00,
    "07-27": 0.97, "07-31": 0.94, "08-04": 1.00, "08-09": 0.97, "08-14": 0.90, "08-19": 0.86,
    "08-24": 0.72, "08-30": 0.80,
}
ANCHORS_MASK = {
    "03-01": 0.01, "03-15": 0.03, "04-01": 0.11, "04-10": 0.21, "04-20": 0.36, "05-01": 0.50,
    "05-10": 0.60, "05-20": 0.66, "06-01": 0.73, "06-15": 0.76, "07-01": 0.80, "07-10": 0.86,
    "07-20": 0.87, "08-01": 0.88, "08-15": 0.88, "08-30": 0.88,
}


def guess(anchors):
    days = [(pd.Timestamp(f"2020-{k}") - DAYS[0]).days for k in anchors]
    return PchipInterpolator(days, list(anchors.values()))(np.arange(len(DAYS)))


def follow(score, x0, y0, px, guesses, tol):
    """Move each day's guess to the weighted centre of the ridge near it."""
    out = np.full(len(guesses), np.nan)
    for i, value in enumerate(guesses):
        x = int(round(x0 + i * PX_PER_DAY))
        yc = y0 - value * px
        lo, hi = int(yc - tol), int(yc + tol) + 1
        s = score[lo:hi, x - 1:x + 2].mean(axis=1)
        if s.max() <= 0:
            continue
        w = np.clip(s - 0.5 * s.max(), 0.0, None)
        out[i] = (y0 - (lo + (np.arange(len(s)) * w).sum() / w.sum())) / px
    return out


def main(png, out):
    im = mpimg.imread(png)[:, :, :3] * 255.0
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    lum = im.mean(axis=2)

    # Panel B: the smoothed line is black; score darkness.
    mask_score = np.clip(120.0 - lum, 0.0, None)
    mask = follow(mask_score, B_X0, B_Y0, B_PX, guess(ANCHORS_MASK), tol=14)

    # Panel C: the red line is redder and darker than its band, and not grey.
    red_score = np.where((r > 140) & (r - g > 80) & (g < 115), 115.0 - g, 0.0)
    re = follow(red_score, C_X0, C_Y0, C_PX, guess(ANCHORS_RE), tol=9)

    # Edges of the red band: first and last reddish pixel in the column.
    low, high = np.full(len(DAYS), np.nan), np.full(len(DAYS), np.nan)
    rows = np.arange(985, 1446)
    for i in range(len(DAYS)):
        x = int(round(C_X0 + i * PX_PER_DAY))
        reddish = (r[985:1446, x] - g[985:1446, x] > 40) & (r[985:1446, x] > 150)
        if reddish.any():
            high[i] = (C_Y0 - rows[reddish].min()) / C_PX
            low[i] = (C_Y0 - rows[reddish].max()) / C_PX

    df = pd.DataFrame({"date": DAYS.strftime("%Y-%m-%d"), "mask_always": mask, "Re": re,
                       "Re_low": low, "Re_high": high})
    for col in ("mask_always", "Re"):
        df[f"{col}_interpolated"] = df[col].isna().astype(int)
        df[col] = df[col].interpolate(limit_direction="both")
    df.to_csv(out, sep="\t", index=False, float_format="%.3f")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
