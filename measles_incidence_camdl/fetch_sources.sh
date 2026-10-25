#!/usr/bin/env bash
# Fetch the two source datasets into sources/ (both are committed to the
# repo already; this script documents where they came from).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p sources

# He, Ionides & King (2010) UK 20-town registry data (London, Liverpool)
curl -sL -o sources/twentycities.rda \
    https://kingaa.github.io/pomp/vignettes/twentycities.rda

# Dalziel et al. (2016) 40-city US biweekly data, via the epimdr2 R package
# (Bjornstad's book companion package; dataset `dalziel`)
ver=$(curl -s https://cran.r-project.org/web/packages/epimdr2/index.html |
    grep -o 'epimdr2_[0-9.-]*\.tar\.gz' | head -1)
curl -sL -o sources/epimdr2.tar.gz "https://cran.r-project.org/src/contrib/$ver"
tar xzf sources/epimdr2.tar.gz -C sources epimdr2/data/dalziel.rda
mv sources/epimdr2/data/dalziel.rda sources/
rm -r sources/epimdr2 sources/epimdr2.tar.gz
