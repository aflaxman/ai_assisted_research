#!/usr/bin/env python3
"""Fetch every third-party input at a pinned version into data/raw/.

data/raw/ is git-ignored because two source repositories carry no license.
Run once after cloning:

    uv run python fetch_data.py

Writes data/raw/MANIFEST.json with URL, byte count and SHA-256 of each file.
Nothing here interprets the data; parsing lives in tbmodels/data.py.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
import json
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (tb-model-structures fetch script)"}

# Pinned to the commits inspected on 2026-10-12.
REPOS = {
    "richards": ("https://github.com/alexandrasrichards/natural_history_tb_disease",
                 "cf0ed162da87c6edcdf9573161293a1692b9ed9e"),
    "horton": ("https://github.com/ERC-TBornotTB/Inf-Dis",
               "8566eea87f70e3c759ae7bca9715f50693ef91a7"),
    "epidemery": ("https://github.com/epidemery/subclinical_transmission",
                  "6d000e5b738075528a1fcc7f73746508bfd90e32"),
}

# Harvard Dataverse doi:10.7910/DVN/STRA2P (CC0): Menzies 2018 replication data.
DATAVERSE_FILES = {
    "LTBI_Review_Data_Clean_10-18-2017.csv": 3135016,
    "LTBI_Citation_Data_Clean_10-18-2017.csv": 3135017,
    "Ferebee_1970_data_rev3.tab": 3135020,
    "Sutherland_1968_data_rev.tab": 3135019,
    "Reactivation_model_results_10-18-2017.tab": 3135015,
    "ltbi_model_review_descriptive_stats.r": 3135009,
    "ltbi_model_review_estimate_results.r": 3135011,
    "ltbi_model_review_plot_results.r": 3135010,
}

FILES = {
    "horton_si/supplementary.zip":
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10666121/supplementaryFiles",
    "tiemersma_2011/journal.pone.0017601.xml":
        "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0017601&type=manuscript",
    "menzies_2018/nihms971881f1.jpg":
        "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/b30e/6070419/607cd256225c/nihms971881f1.jpg",
    "act3/Marks_Epidemiology_TBMAC_2019.pdf":
        "https://tb-mac.org/wp-content/uploads/2019/11/Marks_Epidemiology.pdf",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    delay = 2
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} for {url}: {e}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2


def clone_pinned(name: str, url: str, commit: str) -> Path:
    dest = RAW / "repos" / name
    if (dest / ".git").exists():
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        if head == commit:
            return dest
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
    return dest


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    for name, (url, commit) in REPOS.items():
        print(f"repo {name} @ {commit[:7]}")
        p = clone_pinned(name, url, commit)
        manifest[f"repos/{name}"] = {"url": url, "commit": commit}

    traces_zip = RAW / "repos" / "horton" / "02_Cohort" / "param_traces.zip"
    if traces_zip.exists():
        with zipfile.ZipFile(traces_zip) as z:
            z.extractall(RAW / "horton_traces")

    for rel, fid in DATAVERSE_FILES.items():
        dest = RAW / "menzies_dataverse" / rel
        url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}?format=original"
        if not dest.exists():
            print(f"dataverse {rel}")
            download(url, dest)
        manifest[f"menzies_dataverse/{rel}"] = {"url": url, "bytes": dest.stat().st_size,
                                                "sha256": sha256(dest), "license": "CC0 1.0"}

    for rel, url in FILES.items():
        dest = RAW / rel
        if not dest.exists():
            print(f"file {rel}")
            try:
                download(url, dest)
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED (left missing): {e}", file=sys.stderr)
                manifest[rel] = {"url": url, "status": "missing", "error": str(e)}
                continue
        manifest[rel] = {"url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest)}

    si_zip = RAW / "horton_si" / "supplementary.zip"
    if si_zip.exists():
        with zipfile.ZipFile(si_zip) as z:
            z.extractall(RAW / "horton_si")
        sapp = RAW / "horton_si" / "pnas.2221186120.sapp.pdf"
        if sapp.exists() and shutil.which("pdftotext"):
            subprocess.run(["pdftotext", "-layout", str(sapp), str(sapp.with_suffix(".txt"))], check=False)

    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote MANIFEST.json with {len(manifest)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
