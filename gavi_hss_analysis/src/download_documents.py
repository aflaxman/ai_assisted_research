"""
Download Gavi HSS documents and build corpus inventory.

Respects robots.txt and rate-limits downloads.
Produces a corpus inventory CSV with metadata and SHA256 hashes.
"""

import csv
import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path

import requests

from document_sources import DOCUMENT_SOURCES, ADDITIONAL_SOURCES

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
INVENTORY_PATH = Path(__file__).parent.parent / "outputs" / "corpus_inventory.csv"
SOURCE_LOG_PATH = Path(__file__).parent.parent / "outputs" / "source_log.json"

HEADERS = {
    "User-Agent": "GaviHSSResearch/1.0 (academic research; rate-limited)"
}
RATE_LIMIT_SECONDS = 3  # seconds between downloads


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def make_doc_id(country: str, year, idx: int) -> str:
    iso_slug = country.lower().replace(" ", "_").replace("'", "")
    year_str = str(year) if year else "unk"
    return f"hss_{iso_slug}_{year_str}_{idx:02d}"


def download_document(url: str, dest: Path, max_retries: int = 3) -> dict:
    """Download a document, return status dict."""
    result = {"url": url, "local_path": str(dest), "status": "pending"}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
            if resp.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(resp.content)
                result["status"] = "ok"
                result["size_bytes"] = len(resp.content)
                result["content_type"] = resp.headers.get("Content-Type", "")
                return result
            else:
                result["status"] = f"http_{resp.status_code}"
                result["status_code"] = resp.status_code
        except requests.RequestException as e:
            result["status"] = f"error: {str(e)[:100]}"

        if attempt < max_retries - 1:
            wait = 2 ** (attempt + 1)
            print(f"  Retry {attempt + 1} in {wait}s...")
            time.sleep(wait)

    return result


def build_inventory():
    """Download all documents and produce corpus inventory CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_sources = DOCUMENT_SOURCES + ADDITIONAL_SOURCES
    inventory = []
    source_log = []
    today = date.today().isoformat()

    for idx, src in enumerate(all_sources):
        doc_id = make_doc_id(src["country"], src.get("year"), idx)
        ext = src["format"]
        filename = f"{doc_id}.{ext}"
        dest = RAW_DIR / filename

        print(f"[{idx + 1}/{len(all_sources)}] Downloading {src['country']} ({ext.upper()})...")

        if dest.exists():
            print(f"  Already exists: {dest}")
            dl_result = {"status": "cached", "local_path": str(dest)}
        else:
            dl_result = download_document(src["url"], dest)
            time.sleep(RATE_LIMIT_SECONDS)  # rate limit

        # Compute hash if file exists
        file_hash = ""
        file_size = 0
        if dest.exists():
            file_hash = sha256_file(dest)
            file_size = dest.stat().st_size

        row = {
            "doc_id": doc_id,
            "country": src["country"],
            "iso3": src.get("iso3", ""),
            "who_region": "",  # Will be filled from country_selection
            "year": src.get("year", ""),
            "doc_type": src.get("doc_type", "approved_proposal"),
            "support_type": "HSS",
            "title": f"Proposal for HSS support - {src['country']}",
            "source_url": src["url"],
            "gavi_country_docs_url": src.get("gavi_country_docs_url", ""),
            "download_date": today,
            "local_path": str(dest.relative_to(dest.parent.parent.parent)),
            "sha256_hash": file_hash,
            "pages": "",  # Will be filled during extraction
            "file_size_bytes": file_size,
            "format": ext,
            "text_extraction_status": "",  # Will be filled during extraction
            "download_status": dl_result["status"],
            "notes": "",
        }

        inventory.append(row)
        source_log.append({
            "doc_id": doc_id,
            "url": src["url"],
            "download_date": today,
            "status": dl_result["status"],
        })

        print(f"  Status: {dl_result['status']}, Hash: {file_hash[:16]}...")

    # Write inventory CSV
    if inventory:
        fieldnames = list(inventory[0].keys())
        with open(INVENTORY_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(inventory)
        print(f"\nInventory written to {INVENTORY_PATH} ({len(inventory)} rows)")

    # Write source log
    with open(SOURCE_LOG_PATH, "w") as f:
        json.dump(source_log, f, indent=2)
    print(f"Source log written to {SOURCE_LOG_PATH}")

    # Summary
    statuses = {}
    for row in inventory:
        s = row["download_status"]
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nDownload summary: {statuses}")

    return inventory


if __name__ == "__main__":
    build_inventory()
