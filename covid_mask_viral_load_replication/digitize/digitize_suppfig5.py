"""Read the cell values off the paper's Supplementary figure 5.

Usage: python digitize_suppfig5.py PAGE5.png OUT.tsv

The supplementary figures are embedded in a Word file as EMF drawings
(41598_2021_91338_MOESM1_ESM.docx on the journal site).  PAGE5.png is page 5
of that file converted to PDF with LibreOffice and rendered at 220 dpi:

    soffice --headless --convert-to pdf 41598_2021_91338_MOESM1_ESM.docx
    pdftoppm -r 220 -png -f 5 -l 5 41598_2021_91338_MOESM1_ESM.pdf page

The four panels are the targeted-masking heat maps (everyone with more than
10 exposure contacts in a day masked), read against the colour bar as in
``digitize_fig3.py``.
"""
import sys

from digitize_fig3 import read_panels

PANELS = {
    "25/25": (358, 895, 280, 705),
    "50/50": (988, 1525, 280, 705),
    "75/75": (358, 895, 846, 1271),
    "100/100": (988, 1525, 846, 1271),
}

if __name__ == "__main__":
    read_panels(sys.argv[1], PANELS, bar_x=(1565, 1585), bar_rows=(292, 1272)).to_csv(
        sys.argv[2], sep="\t", index=False, float_format="%.3f")
