"""
Download and process DepMap Avana CRISPR screen data.

Downloads from DepMap 22Q2 public figshare release (article 19700056):
  - Achilles_guide_map.csv      guide sequences + gene targets + genomic coords
  - Achilles_guide_efficacy.csv pre-aggregated mean LFC per guide (2 MB, already computed)
  - sample_info.csv             cell line metadata (lineage, tissue)

Outputs written to data/omics/depmap/:
  - guide_map.csv       guide_id, sequence, gene, chr, start, strand
  - guide_efficacy.csv  guide_id, gene, mean_lfc, sd_lfc (from efficacy file)
  - cell_metadata.csv   cell_line_id, cell_line_name, lineage, primary_disease

Run from backend/:
    python -m omics_pipeline.download_depmap
"""
import csv
import gzip
import json
import re
import urllib.request
from pathlib import Path
from typing import Iterator

from .config import (
    DEPMAP_DIR,
    DEPMAP_FIGSHARE_API,
    DEPMAP_FILES,
    DEPMAP_RELEASE,
)

CHUNK = 1 << 20  # 1 MB streaming chunk


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fetch_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "OmicsCRISPR/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _stream_download(url: str, dest: Path, label: str) -> None:
    """Stream a large file to disk with progress reporting."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  Downloading {label} -> {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "OmicsCRISPR/1.0"})
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
                    print(f"\r    {pct:5.1f}%  ({downloaded >> 20} / {total >> 20} MB)", end="", flush=True)
        print(f"\r    Done. {downloaded >> 10} KB written.           ")


def _open_csv(path: Path) -> Iterator[dict]:
    """Open plain or gzip CSV and yield dicts."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _get_figshare_urls() -> dict[str, str]:
    """Query figshare API and return {filename: download_url} for target files."""
    print(f"\nQuerying figshare for DepMap {DEPMAP_RELEASE} file list...")
    files = _fetch_json(DEPMAP_FIGSHARE_API)
    wanted = set(DEPMAP_FILES.values())
    found: dict[str, str] = {}
    for entry in files:
        name = entry.get("name", "")
        if name in wanted:
            found[name] = entry["download_url"]
    missing = wanted - set(found)
    if missing:
        print(f"  WARNING: not found in figshare: {missing}")
    return found


# ── Processing ────────────────────────────────────────────────────────────────

def _process_guide_map(raw_path: Path) -> Path:
    """
    Parse Achilles_guide_map.csv -> clean guide_map.csv.

    22Q2 columns: sgrna, genome_alignment, gene, n_alignments
      - sgrna          : 20bp sequence (IS the guide ID in this release)
      - genome_alignment: "chr10_110964620_+" (underscore-separated)
      - gene           : "SHOC2 (8036)"  (name + Entrez ID)
      - n_alignments   : int
    Output columns: guide_id, sequence, gene, entrez_id, chr, start, strand
    """
    out = DEPMAP_DIR / "guide_map.csv"
    if out.exists():
        print(f"  [skip] {out.name} already processed")
        return out

    print("  Processing guide map...")
    # Format: "chr10_110964620_+"  (chrom _ pos _ strand)
    coord_re  = re.compile(r"^(chr[\w]+)_(\d+)_([+-])$")
    # Extract gene name and Entrez ID: "SHOC2 (8036)" -> ("SHOC2", "8036")
    gene_re   = re.compile(r"^(.+?)\s*\((\d+)\)$")
    rows = []
    for row in _open_csv(raw_path):
        seq    = row.get("sgrna", "").strip().upper()   # sgrna IS the sequence
        gene_r = row.get("gene", "").strip()
        aln    = row.get("genome_alignment", "").strip()
        n_aln  = int(row.get("n_alignments", 0) or 0)

        if not (seq and len(seq) == 20):
            continue
        if n_aln != 1:  # unique genome alignments only
            continue
        if not set(seq) <= {"A", "C", "G", "T"}:
            continue

        gm = gene_re.match(gene_r)
        gene      = gm.group(1).strip() if gm else gene_r
        entrez_id = gm.group(2)         if gm else ""

        chrom, start, strand = "", "", ""
        m = coord_re.match(aln)
        if m:
            chrom, start, strand = m.group(1), m.group(2), m.group(3)

        rows.append({
            "guide_id":  seq,
            "sequence":  seq,
            "gene":      gene,
            "entrez_id": entrez_id,
            "chr":       chrom,
            "start":     start,
            "strand":    strand,
        })

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["guide_id", "sequence", "gene", "entrez_id", "chr", "start", "strand"])
        w.writeheader()
        w.writerows(rows)

    print(f"  Wrote {len(rows):,} unique-aligning guides -> {out.name}")
    return out


def _process_guide_efficacy(eff_path: Path, guide_map_path: Path) -> Path:
    """
    Parse Achilles_guide_efficacy.csv -> guide_efficacy.csv.

    Achilles_guide_efficacy.csv (DepMap 22Q2) columns:
        guide, efficacy  (or: sgrna, mean_LFC — column names vary by release)
    We detect whichever column layout is present.

    Output: guide_id, gene, efficacy
    """
    out = DEPMAP_DIR / "guide_efficacy.csv"
    if out.exists():
        print(f"  [skip] {out.name} already processed")
        return out

    print("  Loading guide -> gene map for efficacy annotation...")
    guide_to_gene: dict[str, str] = {}
    for row in _open_csv(guide_map_path):
        guide_to_gene[row["guide_id"]] = row["gene"]

    print("  Processing guide efficacy file...")
    rows_out = []

    for row in _open_csv(eff_path):
        # 22Q2: unnamed first column (key="") holds the guide sequence
        guide_id = (row.get("sgrna") or row.get("guide") or
                    row.get("guide_id") or row.get("") or "").strip()
        eff_val  = (row.get("efficacy") or row.get("mean_LFC") or
                    row.get("mean_lfc") or row.get("LFC") or "").strip()
        if not guide_id or not eff_val:
            continue
        try:
            efficacy = float(eff_val)
        except ValueError:
            continue

        gene = guide_to_gene.get(guide_id, "")
        rows_out.append({
            "guide_id": guide_id,
            "gene":     gene,
            "efficacy": round(efficacy, 5),
        })

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["guide_id", "gene", "efficacy"])
        w.writeheader()
        w.writerows(rows_out)

    print(f"  Wrote {len(rows_out):,} guide efficacy records -> {out.name}")
    return out


def _process_cell_metadata(meta_path: Path) -> Path:
    """
    Parse sample_info.csv -> cell_metadata.csv.
    Output: cell_line_id, cell_line_name, lineage, primary_disease, oncotree_code
    """
    out = DEPMAP_DIR / "cell_metadata.csv"
    if out.exists():
        print(f"  [skip] {out.name} already processed")
        return out

    print("  Processing cell line metadata...")
    rows_out = []
    for row in _open_csv(meta_path):
        rows_out.append({
            "cell_line_id":    row.get("DepMap_ID", row.get("ModelID", "")).strip(),
            "cell_line_name":  row.get("cell_line_name", row.get("CellLineName", "")).strip(),
            "lineage":         row.get("lineage", row.get("OncotreeLineage", "")).strip(),
            "primary_disease": row.get("primary_disease", row.get("OncotreePrimaryDisease", "")).strip(),
            "oncotree_code":   row.get("Subtype", row.get("OncotreeCode", "")).strip(),
        })

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cell_line_id", "cell_line_name", "lineage",
                                           "primary_disease", "oncotree_code"])
        w.writeheader()
        w.writerows(rows_out)

    print(f"  Wrote {len(rows_out):,} cell line records -> {out.name}")
    return out


# ── Master download function ───────────────────────────────────────────────────

def download_depmap() -> None:
    print(f"\n{'='*60}")
    print(f"  DepMap {DEPMAP_RELEASE} -- CRISPR screen data")
    print(f"{'='*60}")

    # Step 1: query figshare for download URLs
    try:
        url_map = _get_figshare_urls()
    except Exception as e:
        print(f"  ERROR querying figshare API: {e}")
        return

    # Step 2: download raw files
    raw_paths: dict[str, Path] = {}
    for key, filename in DEPMAP_FILES.items():
        dest = DEPMAP_DIR / filename
        raw_paths[key] = dest
        if filename in url_map:
            _stream_download(url_map[filename], dest, filename)
        elif dest.exists():
            print(f"  [skip] {filename} already present")
        else:
            print(f"  WARNING: {filename} not found in figshare {DEPMAP_RELEASE}")

    # Step 3: process raw -> clean outputs
    print("\nProcessing raw files...")

    guide_map_out = DEPMAP_DIR / "guide_map.csv"
    if raw_paths.get("guide_map", Path("x")).exists():
        guide_map_out = _process_guide_map(raw_paths["guide_map"])
    else:
        print("  SKIP guide_map -- raw file missing")

    if raw_paths.get("guide_efficacy", Path("x")).exists() and guide_map_out.exists():
        _process_guide_efficacy(raw_paths["guide_efficacy"], guide_map_out)
    else:
        print("  SKIP guide_efficacy -- raw file(s) missing")

    if raw_paths.get("cell_info", Path("x")).exists():
        _process_cell_metadata(raw_paths["cell_info"])
    else:
        print("  SKIP cell metadata -- raw file missing")

    print("\nDepMap download complete.")
    print(f"Outputs in: {DEPMAP_DIR}")


if __name__ == "__main__":
    download_depmap()
