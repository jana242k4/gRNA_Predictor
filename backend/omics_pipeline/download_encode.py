"""
Download ENCODE RNA-seq and ATAC-seq data for target cell types.

For each cell type in config.TARGET_CELL_TYPES:
  1. RNA-seq  — queries ENCODE API for gene quantification TSV files
               (polyA RNA-seq, GRCh38, released experiments)
               Writes: data/omics/encode/rnaseq/{cell_type}.tsv
               Merged:  data/omics/encode/rnaseq_tpm.csv
                        Columns: gene_id, gene_name, cell_type, tpm

  2. ATAC-seq — queries ENCODE API for optimal IDR-thresholded peak BEDs
               (GRCh38, released experiments)
               Writes: data/omics/encode/atac/{cell_type}.narrowPeak.gz
               Index:   data/omics/encode/atac_index.csv
                        Columns: cell_type, chr, start, end, score

Run from backend/:
    python -m omics_pipeline.download_encode
"""
import csv
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import (
    ENCODE_BASE,
    ENCODE_ATAC_PARAMS,
    ENCODE_DIR,
    ENCODE_RNA_PARAMS,
    TARGET_CELL_TYPES,
)

RNA_DIR  = ENCODE_DIR / "rnaseq"
ATAC_DIR = ENCODE_DIR / "atac"
RNA_DIR.mkdir(parents=True, exist_ok=True)
ATAC_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 1 << 20  # 1 MB


# ── ENCODE REST API helpers ───────────────────────────────────────────────────

def _encode_get(url: str, retries: int = 3) -> dict | list | None:
    """GET an ENCODE API endpoint, return parsed JSON or None on failure."""
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "OmicsCRISPR/1.0"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # ENCODE returns 404 when search yields no results — treat as empty
            if e.code == 404:
                return {"@graph": []}
            print(f"    HTTP {e.code}" + (" - retrying" if attempt < retries - 1 else ""))
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    Error: {e}" + (" - retrying" if attempt < retries - 1 else ""))
            time.sleep(2 ** attempt)
    return None


def _search_encode(params: dict, biosample: str) -> list[dict]:
    """Search ENCODE for files matching given params + biosample term."""
    p = dict(params)
    p["biosample_ontology.term_name"] = biosample
    url = ENCODE_BASE + "/search/?" + urllib.parse.urlencode(p)
    result = _encode_get(url)
    if not result:
        return []
    return result.get("@graph", [])


def _download_file(url: str, dest: Path, label: str) -> bool:
    """Stream download to dest. Returns True on success."""
    if dest.exists():
        print(f"    [skip] {dest.name} already downloaded")
        return True
    full_url = ENCODE_BASE + url if url.startswith("/") else url
    print(f"    Downloading {label}...")
    try:
        req = urllib.request.Request(
            full_url,
            headers={"User-Agent": "OmicsCRISPR/1.0"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r      {pct:5.1f}% ({downloaded >> 10} KB)", end="", flush=True)
            print(f"\r      Done. {downloaded >> 10} KB                   ")
        return True
    except Exception as e:
        print(f"    ERROR downloading {label}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ── RNA-seq ───────────────────────────────────────────────────────────────────

def _pick_best_rna_file(files: list[dict]) -> dict | None:
    """Pick the single best RNA-seq gene quantification file."""
    # Prefer files from experiments with the highest number of replicates
    # and most recent lab. For simplicity, take the first released TSV.
    for f in files:
        href = f.get("href") or f.get("download_href", "")
        if href and f.get("status") == "released":
            return f
    return files[0] if files else None


def _parse_rsem_tsv(path: Path, cell_type: str) -> list[dict]:
    """
    Parse ENCODE RSEM gene quantification TSV.
    Columns: gene_id, transcript_id(s), length, effective_length, expected_count, TPM, FPKM
    Returns list of {gene_id, cell_type, tpm}
    """
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene_id = row.get("gene_id", "").strip()
            tpm_str = row.get("TPM", row.get("tpm", "0")).strip()
            try:
                tpm = float(tpm_str)
            except ValueError:
                continue
            rows.append({
                "gene_id":   gene_id,
                "cell_type": cell_type,
                "tpm":       round(tpm, 4),
            })
    return rows


def download_rnaseq() -> Path:
    """Download RNA-seq for all target cell types. Returns merged TPM CSV path."""
    out_csv = ENCODE_DIR / "rnaseq_tpm.csv"
    if out_csv.exists():
        print(f"  [skip] rnaseq_tpm.csv already built")
        return out_csv

    print("\n  RNA-seq downloads:")
    all_rows: list[dict] = []
    found_types: list[str] = []

    for cell_key, biosample in TARGET_CELL_TYPES.items():
        print(f"\n  [{cell_key}] biosample: '{biosample}'")
        files = _search_encode(ENCODE_RNA_PARAMS, biosample)

        if not files:
            # Fallback: try a broader search without polyA restriction
            broad = dict(ENCODE_RNA_PARAMS)
            broad.pop("assay_title", None)
            broad["assay_title"] = "RNA-seq"
            files = _search_encode(broad, biosample)

        if not files:
            print(f"    WARNING: no RNA-seq files found for {biosample}")
            continue

        best = _pick_best_rna_file(files)
        if not best:
            continue

        href     = best.get("href", "")
        acc      = best.get("accession", "unknown")
        dest     = RNA_DIR / f"{cell_key}.tsv.gz" if "gz" in href else RNA_DIR / f"{cell_key}.tsv"

        ok = _download_file(href, dest, f"{cell_key} RNA-seq ({acc})")
        if not ok:
            continue

        rows = _parse_rsem_tsv(dest, cell_key)
        print(f"    Parsed {len(rows):,} gene TPM values for {cell_key}")
        all_rows.extend(rows)
        found_types.append(cell_key)

    if not all_rows:
        print("  WARNING: No RNA-seq data downloaded. Skipping merge.")
        return out_csv

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gene_id", "cell_type", "tpm"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n  RNA-seq merged: {len(all_rows):,} records across {len(found_types)} cell types")
    print(f"  Written -> {out_csv.name}")
    return out_csv


# ── ATAC-seq ──────────────────────────────────────────────────────────────────

def _pick_best_atac_file(files: list[dict]) -> dict | None:
    """Pick the best ATAC-seq peak file (prefer optimal IDR peaks, GRCh38)."""
    for f in files:
        href = f.get("href") or f.get("download_href", "")
        if href and f.get("status") == "released":
            return f
    return files[0] if files else None


def _parse_narrowpeak(path: Path, cell_type: str, max_peaks: int = 200_000) -> list[dict]:
    """
    Parse narrowPeak BED file (ENCODE ATAC-seq output).
    Columns: chr, start, end, name, score, strand, signalValue, pValue, qValue, peak
    Returns list of {cell_type, chr, start, end, score, signal}
    """
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            try:
                rows.append({
                    "cell_type": cell_type,
                    "chr":       parts[0],
                    "start":     int(parts[1]),
                    "end":       int(parts[2]),
                    "score":     int(parts[4]),
                    "signal":    float(parts[6]) if len(parts) > 6 else 0.0,
                })
            except (ValueError, IndexError):
                continue
            if len(rows) >= max_peaks:
                break
    return rows


def download_atac() -> Path:
    """Download ATAC-seq peaks for all target cell types. Returns index CSV path."""
    index_csv = ENCODE_DIR / "atac_index.csv"
    if index_csv.exists():
        print(f"  [skip] atac_index.csv already built")
        return index_csv

    print("\n  ATAC-seq downloads:")
    all_peaks: list[dict] = []
    found_types: list[str] = []

    for cell_key, biosample in TARGET_CELL_TYPES.items():
        print(f"\n  [{cell_key}] biosample: '{biosample}'")
        files = _search_encode(ENCODE_ATAC_PARAMS, biosample)

        if not files:
            # Fallback: broader ATAC-seq search
            broad = dict(ENCODE_ATAC_PARAMS)
            broad["output_type"] = "peaks"
            files = _search_encode(broad, biosample)

        if not files:
            print(f"    WARNING: no ATAC-seq files found for {biosample}")
            continue

        best = _pick_best_atac_file(files)
        if not best:
            continue

        href = best.get("href", "")
        acc  = best.get("accession", "unknown")
        ext  = ".narrowPeak.gz" if ".narrowPeak.gz" in href else ".bed.gz"
        dest = ATAC_DIR / f"{cell_key}{ext}"

        ok = _download_file(href, dest, f"{cell_key} ATAC-seq peaks ({acc})")
        if not ok:
            continue

        peaks = _parse_narrowpeak(dest, cell_key)
        print(f"    Parsed {len(peaks):,} peaks for {cell_key}")
        all_peaks.extend(peaks)
        found_types.append(cell_key)

    if not all_peaks:
        print("  WARNING: No ATAC-seq data downloaded. Skipping index.")
        return index_csv

    # Sort by chr + start for efficient range queries later
    all_peaks.sort(key=lambda r: (r["chr"], r["start"]))

    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cell_type", "chr", "start", "end", "score", "signal"])
        w.writeheader()
        w.writerows(all_peaks)

    print(f"\n  ATAC-seq index: {len(all_peaks):,} peaks across {len(found_types)} cell types")
    print(f"  Written -> {index_csv.name}")
    return index_csv


# ── Master ────────────────────────────────────────────────────────────────────

def download_encode() -> None:
    print(f"\n{'='*60}")
    print(f"  ENCODE — RNA-seq + ATAC-seq for {len(TARGET_CELL_TYPES)} cell types")
    print(f"{'='*60}")
    print(f"  Target cell types: {', '.join(TARGET_CELL_TYPES)}")

    download_rnaseq()
    download_atac()

    print("\nENCODE download complete.")
    print(f"Outputs in: {ENCODE_DIR}")


if __name__ == "__main__":
    download_encode()
