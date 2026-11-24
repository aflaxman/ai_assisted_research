"""Which formula do the paper's pairwise transmission-risk figures plot?

Usage: python decode_pairwise_figures.py FIG2.png SUPP_PAGE1.png SUPP_PAGE2.png OUT.tsv POINTS.tsv

OUT.tsv gets one row per figure with the fitted exponents; POINTS.tsv gets the
values read off main Figure 2 on a coarse grid, for plotting.

The Methods give one formula for the risk of transmission to one exposure
contact, with P(V) = V^a / (lambda^a + V^a), a = 0.8, lambda = 1e7:

    risk = P( V (1 - eps_T) (1 - eps_E) )^2 .

Four published figures plot that risk as a colour or a curve.  This script
reads each one back to numbers and fits the family

    risk = (1 - eps_E)^a (1 - eps_T)^b * P( V (1 - eps_T)^c (1 - eps_E)^d )^2 ,

in which the stated formula is (a, b, c, d) = (0, 0, 1, 1).  With equal
efficacy on both sides only a + b and c + d are identified.

FIG2.png is the main-text Figure 2 from the journal site (1502 x 1174).
SUPP_PAGE1.png and SUPP_PAGE2.png are pages 1 and 2 of the supplementary
figures (Supplementary figures 1 and 2) rendered at 220 dpi; see
``get_figures.sh``.  Colours are matched to each figure's own colour bar,
calibrated on its tick marks.
"""
import sys

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from scipy.optimize import least_squares

ALPHA, LAMBDA = 0.8, 1.0e7


def P(v):
    return v**ALPHA / (LAMBDA**ALPHA + v**ALPHA)


def family(theta, v, eps_t, eps_e):
    a, b, c, d = theta
    return (1 - eps_e)**a * (1 - eps_t)**b * P(v * (1 - eps_t)**c * (1 - eps_e)**d)**2


def decoder(im, rows, cols, row_at_one, row_at_zero):
    """Nearest-colour lookup against a vertical colour bar."""
    r0, r1 = rows
    bar = im[r0:r1, cols[0]:cols[1]].mean(axis=1)
    value = (row_at_zero - np.arange(r0, r1)) / (row_at_zero - row_at_one)

    def decode(pixels):
        dist = ((bar[None] - pixels[:, None])**2).sum(axis=-1)
        return value[dist.argmin(axis=1)]
    return decode


def rms(pred, obs):
    return float(np.sqrt(np.mean((pred - obs)**2)))


def main(fig2_png, supp1_png, supp2_png, out, points_out):
    rows, points = [], []

    # ---- main Figure 2: x = log10 V from 5 to 10, y = efficacy from 0 to 1 ----
    im = mpimg.imread(fig2_png)[:, :, :3] * 255.0
    lv, ep = np.meshgrid(np.linspace(5.1, 9.9, 40), np.linspace(0.03, 0.97, 30))
    v = 10.0**lv
    panels = {  # name: (x0, x1, y_top, y_bottom), colour bar rows, which masks
        "Fig 2B, transmitter masked": ((790, 1330, 77, 510), (70, 513), "T"),
        "Fig 2C, exposed contact masked": ((120, 662, 650, 1083), (650, 1093), "E"),
        "Fig 2D, both masked, same efficacy": ((790, 1330, 650, 1083), (650, 1093), "both"),
    }
    for name, ((x0, x1, y0, y1), (b0, b1), who) in panels.items():
        decode = decoder(im, (b0, b1), (1375, 1395), b0, b1 - 1)
        xs = np.round(x0 + (lv - 5) / 5 * (x1 - x0)).astype(int)
        ys = np.round(y1 - ep * (y1 - y0)).astype(int)
        obs = decode(im[ys.ravel(), xs.ravel()]).reshape(lv.shape)
        for logv in np.arange(6.0, 10.01, 0.5):
            for e in np.round(np.arange(0.05, 0.96, 0.05), 2):
                x = int(round(x0 + (min(logv, 9.95) - 5) / 5 * (x1 - x0)))
                y = int(round(y1 - e * (y1 - y0)))
                points.append({"panel": name.split(",")[0], "masked": who, "log10_viral_load": logv,
                               "efficacy": e, "risk": float(decode(im[y:y + 1, x])[0])})
        et = ep if who in ("T", "both") else 0.0 * ep
        ee = ep if who in ("E", "both") else 0.0 * ep
        # with equal or single efficacy, fit a multiplier power and a dose power
        fit = least_squares(lambda th: ((1 - ep)**th[0] * P(v * (1 - ep)**th[1])**2 - obs).ravel(), [0.5, 1.5])
        stated = family((0, 0, 1, 1), v, et, ee)
        rows.append({"figure": name, "points": obs.size,
                     "fitted_power_on_risk": f"{round(fit.x[0], 2) + 0.0:.2f}", "fitted_power_in_dose": f"{round(fit.x[1], 2) + 0.0:.2f}",
                     "stated_power_on_risk": "0", "stated_power_in_dose": "2" if who == "both" else "1",
                     "rms_fitted": rms((1 - ep)**fit.x[0] * P(v * (1 - ep)**fit.x[1])**2, obs),
                     "rms_stated_formula": rms(stated, obs)})

    # ---- Supplementary figure 1: curves of risk against efficacy, V = 1e9 ----
    im = mpimg.imread(supp1_png)[:, :, :3] * 255.0
    eps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    for name, (y_top, y_bottom), power in (("Supp Fig 1A, transmitter masked, V = 1e9", (264, 589), 1.0),
                                           ("Supp Fig 1C, both masked, V = 1e9", (708, 1033), 2.0)):
        obs = np.full(len(eps), np.nan)
        for i, e in enumerate(eps):
            x = int(round(318 + e * (806 - 318)))
            col = im[y_top - 5:y_bottom + 3, x - 1:x + 2].mean(axis=1)
            pink = np.flatnonzero((col[:, 0] > 215) & (col[:, 0] - col[:, 1] > 60) & (col[:, 2] > 150))
            if len(pink) == 0:                         # curve hidden behind a grid line or label
                continue
            top = pink[pink <= pink[0] + 6]            # the uppermost curve is V = 1e9
            obs[i] = (y_bottom - (top.mean() + y_top - 5)) / (y_bottom - y_top)
        ok = ~np.isnan(obs)
        e, obs = eps[ok], obs[ok]
        fit = least_squares(lambda th: (1 - e)**th[0] * P(1e9 * (1 - e)**th[1])**2 - obs, [0.5, 1.5])
        rows.append({"figure": name, "points": obs.size,
                     "fitted_power_on_risk": f"{round(fit.x[0], 2) + 0.0:.2f}", "fitted_power_in_dose": f"{round(fit.x[1], 2) + 0.0:.2f}",
                     "stated_power_on_risk": "0", "stated_power_in_dose": f"{power:.0f}",
                     "rms_fitted": rms((1 - e)**fit.x[0] * P(1e9 * (1 - e)**fit.x[1])**2, obs),
                     "rms_stated_formula": rms(P(1e9 * (1 - e)**power)**2, obs)})

    # ---- Supplementary figure 2: x = eps_T, y = eps_E, four viral loads ----
    im = mpimg.imread(supp2_png)[:, :, :3] * 255.0
    decode = decoder(im, (250, 697), (1528, 1548), 276.0, 694.0)     # ticks: 100% at row 276, 0% at 694
    g = np.linspace(0.04, 0.96, 24)
    et, ee = np.meshgrid(g, g)
    frames = {10**6.5: (359, 873, 285, 688), 1e7: (969, 1483, 285, 688),
              10**7.5: (359, 873, 817, 1220), 1e8: (969, 1483, 817, 1220)}
    data = []
    for v0, (x0, x1, y0, y1) in frames.items():
        xs = np.round(x0 + et * (x1 - x0)).astype(int)
        ys = np.round(y1 - ee * (y1 - y0)).astype(int)
        data.append((v0, decode(im[ys.ravel(), xs.ravel()])))
    resid = lambda th: np.concatenate([family(th, v0, et, ee).ravel() - obs for v0, obs in data])
    fit = least_squares(resid, [0.5, 0.5, 0.5, 0.5])
    a, b, c, d = fit.x
    rows.append({"figure": "Supp Fig 2, both masked, efficacies varied separately",
                 "points": sum(len(obs) for _, obs in data),
                 "fitted_power_on_risk": f"(1-eps_E)^{a:.2f} (1-eps_T)^{b:.2f}",
                 "fitted_power_in_dose": f"(1-eps_T)^{c:.2f} (1-eps_E)^{d:.2f}",
                 "stated_power_on_risk": "none", "stated_power_in_dose": "(1-eps_T)^1 (1-eps_E)^1",
                 "rms_fitted": rms(resid(fit.x), 0.0), "rms_stated_formula": rms(resid((0, 0, 1, 1)), 0.0)})

    table = pd.DataFrame(rows)
    table.to_csv(out, sep="\t", index=False, float_format="%.3f")
    pd.DataFrame(points).to_csv(points_out, sep="\t", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:6])
