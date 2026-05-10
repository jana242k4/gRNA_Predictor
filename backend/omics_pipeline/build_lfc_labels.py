"""
OmicsCRISPR Phase 3 (v3) -- Cell-Type-Specific Gene Effect Labels

Downloads Achilles_gene_effect.csv (CERES-corrected gene essentiality, 336 MB)
from DepMap 22Q2 figshare and maps each cell line to one of our 5 immune / cancer
cell types using ontology codes from sample_info.csv.

Mapping (DepMap cancer lines as proxies for primary ENCODE cell types):
  B_cell     <- B-ALL (28), B-CLL (4), B-cell unspecified (3), Hairy Cell (4)
  T_cell_CD4 <- T-ALL (23)  [proxy for both CD4 and CD8]
  T_cell_CD8 <- T-ALL (23)  [same proxy]
  NK_cell    <- NK lymphoblastic (2)
  monocyte   <- AML / CML myeloid lines (40+)
  K562       <- K562 specifically (CML cell line used as ENCODE control)

Output: data/omics/depmap/cell_type_gene_effect.csv
  Columns: gene_name, cell_type, mean_effect, n_cell_lines

Run from backend/:
    python -m omics_pipeline.build_lfc_labels
"""
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND))

from .config import (
    CELL_TYPE_GENE_EFFECT_CSV,
    DEPMAP_DIR,
    DEPMAP_FIGSHARE_API,
    DEPMAP_FILES,
)

CHUNK = 1 << 20  # 1 MB stream chunk

# ── Cell type assignment from DepMap oncotree codes ───────────────────────────

# Maps oncotree_code (or primary_disease if oncotree blank) -> our cell_type label
ONCOTREE_MAP: dict[str, str] = {
    # B-cell malignancies
    "Acute Lymphoblastic Leukemia (ALL), B-cell":        "B_cell",
    "Chronic Lymphoblastic Leukemia (CLL), B-cell":      "B_cell",
    "B-cell, unspecified":                               "B_cell",
    "Hairy Cell":                                        "B_cell",
    # T-cell malignancies (proxy for both CD4 and CD8)
    "Acute Lymphoblastic Leukemia (ALL), T-cell":        "T_cell_CD4",
    # NK malignancies
    "Natural Killer Cell Lymphoblastic Leukemia/Lymphoma": "NK_cell",
    # Myeloid / monocytic (proxy for monocyte)
    "Acute Myelogenous Leukemia (AML)":                  "monocyte",
    "Acute Myelogenous Leukemia (AML), M2 (Myeloblastic)": "monocyte",
    "Acute Myelogenous Leukemia (AML), M3 (Promyelocytic)": "monocyte",
    "Acute Myelogenous Leukemia (AML), M4 (Myelomonocytic)": "monocyte",
    "Acute Myelogenous Leukemia (AML), M5 (Eosinophilic/Monocytic)": "monocyte",
    "Acute Myelogenous Leukemia (AML), M5 (Monocytic)":  "monocyte",
    "Acute Myelogenous Leukemia (AML), M6 (Erythroleukemia)": "monocyte",
    "Acute Myelogenous Leukemia (AML), M7 (Megakaryoblastic)": "monocyte",
    "Chronic Myelogenous Leukemia (CML)":                "monocyte",
    "Chronic Myelogenous Leukemia (CML), blast crisis":  "monocyte",
}

GENE_NAME_RE = re.compile(r"^(.+?)\s+\(\d+\)$")  # "EGFR (1956)" -> "EGFR"


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fetch_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "OmicsCRISPR/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _stream_download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded ({dest.stat().st_size >> 20} MB)")
        return
    print(f"  Downloading {label} -> {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "OmicsCRISPR/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
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
                    print(f"\r    {pct:5.1f}%  ({downloaded >> 20} / {total >> 20} MB)",
                          end="", flush=True)
        print(f"\r    Done. {downloaded >> 20} MB written.           ")


# ── Cell line -> cell type mapping ────────────────────────────────────────────

def _build_cell_type_map() -> dict[str, str]:
    """
    Returns {depmap_id: cell_type} for all blood-lineage cell lines.
    K562 identified by cell line name (specific CML line used in ENCODE).
    T_cell_CD8 is assigned the same lines as T_cell_CD4 (cancer proxy).
    """
    meta_path = DEPMAP_DIR / "cell_metadata.csv"
    ct_map: dict[str, str] = {}
    with open(meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name   = row["cell_line_name"].upper()
            onco   = row["oncotree_code"].strip()
            lin    = row["lineage"].strip()
            did    = row["cell_line_id"].strip()

            # K562 by name — stored as "K-562" in sample_info, strip hyphens to match
            if "K562" in name.replace("-", ""):
                ct_map[did] = "K562"
                continue

            # Blood lineage only
            if lin != "blood":
                continue

            ct = ONCOTREE_MAP.get(onco)
            if ct:
                ct_map[did] = ct
                # T-ALL lines double as CD8 proxy
                if ct == "T_cell_CD4":
                    ct_map[did + "__CD8"] = "T_cell_CD8"

    return ct_map


# ── Parse gene_effect.csv (chunk-based to stay within RAM) ───────────────────

def _compute_cell_type_effects(ge_path: Path, ct_map: dict[str, str]) -> dict:
    """
    Parse Achilles_gene_effect.csv in chunks.
    Format: rows=cell_lines (DepMap IDs), cols=genes ("EGFR (1956)").
    Returns {(gene_name, cell_type): [sum, count]}.
    """
    import pandas as pd

    print("  Parsing gene effect matrix (chunked)...")
    t0 = time.time()

    # First pass: read header to map column indices -> gene names
    with open(ge_path, encoding="utf-8") as f:
        header_line = f.readline()
    raw_cols = [c.strip().strip('"') for c in header_line.split(",")]
    # col 0 is the row index (DepMap_ID), rest are genes
    col_genes: list[str] = []
    for raw in raw_cols[1:]:
        m = GENE_NAME_RE.match(raw)
        col_genes.append(m.group(1) if m else raw)

    n_genes = len(col_genes)
    print(f"    {n_genes:,} genes in matrix")

    # Accumulate: for each cell_type, sum array + count
    ct_labels = list({ct for ct in ct_map.values()})
    sum_arrays:   dict[str, np.ndarray] = {ct: np.zeros(n_genes, dtype=np.float64)
                                            for ct in ct_labels}
    count_arrays: dict[str, np.ndarray] = {ct: np.zeros(n_genes, dtype=np.int32)
                                            for ct in ct_labels}

    n_lines_used = {ct: 0 for ct in ct_labels}

    # Second pass: read in chunks
    chunk_size = 200
    reader = pd.read_csv(ge_path, index_col=0, chunksize=chunk_size,
                         low_memory=False)
    total_rows = 0
    for chunk in reader:
        for did in chunk.index:
            did_str = str(did).strip()
            ct = ct_map.get(did_str)
            if ct is None:
                continue
            vals = chunk.loc[did].to_numpy(dtype=np.float64, na_value=np.nan)
            mask = ~np.isnan(vals)
            sum_arrays[ct][mask]   += vals[mask]
            count_arrays[ct][mask] += 1

            # Also assign T_cell_CD8 for T_ALL lines
            if ct == "T_cell_CD4" and (did_str + "__CD8") in ct_map:
                sum_arrays["T_cell_CD8"][mask]   += vals[mask]
                count_arrays["T_cell_CD8"][mask] += 1

        total_rows += len(chunk)
        if total_rows % 200 == 0:
            print(f"\r    Processed {total_rows:,} cell lines...", end="", flush=True)

    print(f"\r    Processed {total_rows:,} cell lines in {time.time()-t0:.0f}s")
    for ct in ct_labels:
        n_lines_used[ct] = int(count_arrays[ct].max())
        print(f"    {ct}: ~{n_lines_used[ct]} cell lines contributing")

    return {
        "col_genes":    col_genes,
        "sum_arrays":   sum_arrays,
        "count_arrays": count_arrays,
    }


# ── Write output ──────────────────────────────────────────────────────────────

def _write_cell_type_gene_effect(data: dict) -> None:
    col_genes    = data["col_genes"]
    sum_arrays   = data["sum_arrays"]
    count_arrays = data["count_arrays"]

    rows = []
    for ct, sums in sum_arrays.items():
        counts = count_arrays[ct]
        for i, gene in enumerate(col_genes):
            n = counts[i]
            if n == 0:
                continue
            rows.append({
                "gene_name":   gene,
                "cell_type":   ct,
                "mean_effect": round(sums[i] / n, 5),
                "n_cell_lines": int(n),
            })

    with open(CELL_TYPE_GENE_EFFECT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gene_name", "cell_type",
                                          "mean_effect", "n_cell_lines"])
        w.writeheader()
        w.writerows(rows)

    n_entries = len(rows)
    n_genes   = len({r["gene_name"] for r in rows})
    n_cts     = len({r["cell_type"] for r in rows})
    print(f"\n  Wrote {n_entries:,} (gene, cell_type) entries  "
          f"({n_genes:,} genes x {n_cts} cell types)")
    print(f"  -> {CELL_TYPE_GENE_EFFECT_CSV.name}")


# ── Master function ───────────────────────────────────────────────────────────

def build_lfc_labels() -> None:
    print(f"\n{'='*60}")
    print("  DepMap Gene Effect -> Cell-Type-Specific Labels")
    print(f"{'='*60}")

    if CELL_TYPE_GENE_EFFECT_CSV.exists():
        n = sum(1 for _ in open(CELL_TYPE_GENE_EFFECT_CSV)) - 1
        print(f"  [skip] {CELL_TYPE_GENE_EFFECT_CSV.name} already built ({n:,} rows)")
        return

    # ── Download gene_effect.csv ──
    ge_filename = DEPMAP_FILES["gene_effect"]
    ge_path     = DEPMAP_DIR / ge_filename

    if not ge_path.exists():
        print("\nQuerying figshare for download URL...")
        files = _fetch_json(DEPMAP_FIGSHARE_API)
        url = next((f["download_url"] for f in files if f["name"] == ge_filename), None)
        if not url:
            print(f"  ERROR: {ge_filename} not found in figshare {DEPMAP_FIGSHARE_API}")
            return
        _stream_download(url, ge_path, ge_filename)
    else:
        print(f"  {ge_filename} already present ({ge_path.stat().st_size >> 20} MB)")

    # ── Build cell type map ──
    print("\nBuilding cell line -> cell type mapping...")
    ct_map = _build_cell_type_map()
    summary = {}
    for did, ct in ct_map.items():
        if not did.endswith("__CD8"):
            summary[ct] = summary.get(ct, 0) + 1
    for ct, n in sorted(summary.items()):
        print(f"  {ct}: {n} cell lines")

    # ── Process gene effect matrix ──
    print("\nComputing per-(gene, cell_type) mean gene effect...")
    data = _compute_cell_type_effects(ge_path, ct_map)

    # ── Write output ──
    print("\nWriting output...")
    _write_cell_type_gene_effect(data)
    print("\nDone.")


if __name__ == "__main__":
    build_lfc_labels()
