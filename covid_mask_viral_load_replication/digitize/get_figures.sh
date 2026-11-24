#!/usr/bin/env bash
# Fetch the published figures and read the numbers off them.
#
# The article is open access (CC BY 4.0): Goyal et al. (2021), Scientific
# Reports 11:11838, https://doi.org/10.1038/s41598-021-91338-5.  The figures
# are not kept in this repository; this script downloads them into
# digitize/figures/ and rewrites the tables in data/ and results/ that are
# derived from them.  It needs curl, LibreOffice (soffice) and pdftoppm, and
# the Python environment of the parent directory.
#
#     cd digitize && ./get_figures.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p figures
base="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-021-91338-5/MediaObjects/41598_2021_91338"
for i in 2 3 4 5 6; do
    curl -sSL -o "figures/fig$i.png" "${base}_Fig${i}_HTML.png"
done
curl -sSL -o figures/supplementary_figures.docx \
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-021-91338-5/MediaObjects/41598_2021_91338_MOESM1_ESM.docx"
soffice --headless --convert-to pdf --outdir figures figures/supplementary_figures.docx >/dev/null
pdftoppm -r 220 -png figures/supplementary_figures.pdf figures/supp

py="uv run --project .. python"
$py digitize_fig3.py figures/fig3.png ../data/paper_fig3_digitized.tsv
$py digitize_fig4.py figures/fig4.png ../data/paper_fig4_digitized.tsv
$py digitize_fig5.py figures/fig5.png ../data/paper_fig5_digitized.tsv
$py digitize_fig6.py figures/fig6.png ../data/paper_fig6_digitized.tsv
$py digitize_suppfig5.py figures/supp-5.png ../data/paper_suppfig5_digitized.tsv
$py decode_pairwise_figures.py figures/fig2.png figures/supp-1.png figures/supp-2.png \
    ../results/pairwise_figure_fits.tsv ../data/paper_fig2_digitized.tsv
