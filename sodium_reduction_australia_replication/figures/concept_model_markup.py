"""Redraw the hand-sketched concept model and mark what it still needs.

Solid blue is what the notebook sketch already has. Dashed orange is a
suggested addition. The numbered badges match the key at the foot of the
figure and the 2026-11-27 entry in NOTES.md.

    python figures/concept_model_markup.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

INK, INK_SECONDARY, SURFACE = "#191919", "#666666", "#ffffff"
SKETCH, ADDED = "#257de0", "#f47230"  # validated pair; dashes carry the same distinction
DASH = (0, (4.5, 2.6))
OUTPUT = Path(__file__).with_name("concept_model_markup.png")

fig = plt.figure(figsize=(9, 14.85), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(-5, 160)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)


def box(x, y, text, added=False, w=18.0, h=9.0, size=12.5):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=2.2",
        linewidth=2.2, edgecolor=ADDED if added else SKETCH, facecolor=SURFACE,
        linestyle=DASH if added else "-", zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK,
            linespacing=1.15, zorder=4)


def arrow(*points, added=False):
    """A straight arrow through two points, or a cubic Bezier through four."""
    codes = [MplPath.MOVETO] + ([MplPath.LINETO] if len(points) == 2 else [MplPath.CURVE4] * 3)
    ax.add_patch(FancyArrowPatch(
        path=MplPath(points, codes), arrowstyle="-|>,head_length=0.55,head_width=0.26",
        mutation_scale=17, linewidth=2.0, color=ADDED if added else SKETCH,
        linestyle=DASH if added else "-", shrinkA=0, shrinkB=0, zorder=2))


def badge(x, y, number, r=2.05, size=11.0):
    ax.add_patch(Circle((x, y), r, facecolor=ADDED, edgecolor=SURFACE, linewidth=1.6, zorder=6))
    ax.text(x, y - 0.05, str(number), ha="center", va="center", fontsize=size,
            fontweight="bold", color=SURFACE, zorder=7)


# ---- title and legend -------------------------------------------------------
ax.text(3, 156.3, "Concept model: the sketch, with what it still needs",
        fontsize=15.5, fontweight="bold", color=INK, va="center")
ax.plot([3.5, 9.5], [151.6, 151.6], color=SKETCH, lw=2.4, solid_capstyle="round")
ax.text(11, 151.6, "in the sketch", fontsize=11.5, color=INK_SECONDARY, va="center")
ax.plot([32, 38], [151.6, 151.6], color=ADDED, lw=2.4, linestyle=DASH)
ax.text(39.5, 151.6, "suggested addition", fontsize=11.5, color=INK_SECONDARY, va="center")
badge(68.5, 151.6, "n", r=1.8)
ax.text(71.5, 151.6, "see the key below", fontsize=11.5, color=INK_SECONDARY, va="center")

# ---- nodes ------------------------------------------------------------------
A, B, C, D = 22.0, 45.0, 67.5, 89.5
INTERVENTION, INTAKE, ACTIVITY, BMI, MEDIATOR = 143.0, 122.0, 105.0, 88.0, 65.0

box(A, INTERVENTION, "GLP-1s")
box(B, INTERVENTION, "Sodium policy", added=True)
box(C, INTERVENTION, "Potassium-\nenriched salt", added=True, w=20, h=10)
box(D, INTERVENTION, "Hypertension\ncare: contact,\ndiagnosis", added=True, h=12.5, size=11.5)

box(A, INTAKE, "Caloric intake")
box(B, INTAKE, "Sodium intake")
box(C, INTAKE, "Potassium intake", w=20)
box(D, INTAKE, "Drug treatment:\nintensity,\nadherence", added=True, h=12.5, size=11.5)

box(A, ACTIVITY, "Physical activity")
box(A, BMI, "BMI")
box(A, MEDIATOR, "FPG", w=16)
box(B, MEDIATOR, "LDL-C", w=16)
box(68.0, MEDIATOR, "SBP", w=18)

# outcomes: one frame, as in the sketch, with the three original boxes inside
ax.add_patch(FancyBboxPatch((3, 27), 94, 24, boxstyle="round,pad=0,rounding_size=2.5",
                            linewidth=2.2, edgecolor=SKETCH, facecolor=SURFACE, zorder=1))
box(14.5, 44.3, "IHD", w=15, h=7.4)
box(33.0, 44.3, "HF", w=15, h=7.4)
box(51.5, 44.3, "Stroke", w=15, h=7.4)
box(80.0, 44.3, "Type 2 diabetes, CKD", added=True, w=29, h=7.4, size=11.5)
box(18.5, 33.6, "HHD, AF, PAD,\naortic aneurysm", added=True, w=27, h=9.4, size=11.5)
box(50.0, 33.6, "Stomach cancer,\nosteoarthritis, cancers", added=True, w=31, h=9.4, size=11.5)
box(82.0, 33.6, "Harms: hyperkalaemia,\ndrug side effects", added=True, w=27, h=9.4, size=11.5)

# ---- arrows in the sketch ---------------------------------------------------
arrow((A, 138.5), (A, 126.5))                                      # GLP-1s to caloric intake
arrow((13, 141.3), (7.5, 132), (7.5, 100), (13, 89.3))             # GLP-1s to BMI
arrow((13, 144.2), (2.6, 134), (2.6, 82), (14, 66.5))              # GLP-1s to FPG
arrow((29, 117.5), (36.6, 112), (36.6, 97), (31, 90.3))            # caloric intake to BMI
arrow((31, 122.6), (36, 122.6))                                    # caloric intake to sodium
arrow((31, 119.6), (40, 107.5), (52, 107.5), (57.5, 119.4))        # caloric intake to potassium
arrow((A, 100.5), (A, 92.5))                                       # physical activity to BMI
arrow((A, 83.5), (A, 69.5))                                        # BMI to FPG
arrow((29, 83.5), (41, 69.5))                                      # BMI to LDL-C
arrow((31, 87.4), (61, 69.5))                                      # BMI to SBP
arrow((27, 83.5), (32, 76), (33.5, 64), (33.5, 51))                # BMI to outcomes
arrow((A, 60.5), (A, 51))                                          # FPG to outcomes
arrow((B, 60.5), (B, 51))                                          # LDL-C to outcomes
arrow((68, 60.5), (68, 51))                                        # SBP to outcomes
arrow((47, 117.5), (64, 69.5))                                     # sodium to SBP
arrow((67.5, 117.5), (67.9, 69.5))                                 # potassium to SBP

# ---- suggested arrows -------------------------------------------------------
arrow((B, 138.5), (B, 126.5), added=True)                          # sodium policy to sodium
arrow((C, 138.0), (C, 126.5), added=True)                          # potassium salt to potassium
arrow((59.5, 138.2), (52.5, 126.5), added=True)                    # potassium salt to sodium
arrow((D, 136.75), (D, 128.25), added=True)                        # care to drug treatment
arrow((85.5, 115.75), (72.5, 69.5), added=True)                    # drug treatment to SBP
arrow((13, 146.4), (-1.2, 140), (-1.2, 70), (6.5, 51), added=True)  # GLP-1s to outcomes, direct

# ---- numbered needs ---------------------------------------------------------
badge(80.5, 132.5, 1)
badge(79.5, 65.0, 2)
badge(31.0, 127.5, 3)
badge(2.7, 104.0, 4)
badge(33.4, 57.0, 5)
badge(56.3, 65.0, 6)
badge(62.0, 44.3, 7)
badge(33.5, 143.0, 8)
badge(31.0, 100.0, 9)

KEY = [
    "The comparator: hypertension care, drug treatment, and who is treated now",
    "A stated rule for how effects combine at SBP",
    "Trial effects on SBP, FPG and LDL-C as inputs, with calories derived from weight",
    "A direct GLP-1 effect on outcomes, calibrated to the outcome trials",
    "The direct BMI arrow is only the share that does not go through mediators",
    "Correlated exposures, so that eligibility selects the right people",
    "Split stroke by subtype, build HF as a cause, add outcomes and harms",
    "Uptake, persistence and a business-as-usual trend for every intervention",
    "Physical activity: give it direct arrows, or drop it",
]
for i, line in enumerate(KEY):
    y = 22.0 - 3.05 * i
    badge(5.3, y, i + 1, r=1.4, size=9.5)
    ax.text(8.6, y, line, fontsize=11.3, color=INK, va="center")

fig.savefig(OUTPUT, facecolor=SURFACE)
print(f"wrote {OUTPUT}")
