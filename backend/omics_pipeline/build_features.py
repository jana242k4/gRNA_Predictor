"""
OmicsCRISPR Phase 2 -- Multi-Omics Feature Engineering

Integrates Phase 1 outputs into a unified training matrix.

For each guide in DepMap (68,140 with genomic coordinates):
  - Sequence features   : 450-dim from existing feature_engineering.py
  - Per cell-type omics :
      rna_tpm_log1p     log1p(TPM) for target gene (Entrez join)
      atac_signal       max ATAC signal in +-500bp window (0-1 normalised)
      atac_n_peaks      count of ATAC peaks in window
      splice_dist_log   log1p(bp to nearest splice donor/acceptor)
      cell_type_idx     integer index for cell type embedding
  - Label               : efficacy (0-1, pre-computed mean LFC from DepMap)

Outputs written to data/omics/features/:
  guide_metadata.csv    N_guides rows -- guide_id, gene, entrez_id, chr, start,
                        strand, efficacy, guide_idx
  cell_features.csv     N_guides * N_cell_types rows -- omics features per
                        (guide, cell_type) pair
  seq_features.npz      numpy float32 array, shape (N_guides, 450)
  feature_summary.json  stats + feature dimension counts

Run from backend/:
    python -m omics_pipeline.build_features

Expected runtime: 10-20 min for 68k guides * 5 cell types.
Expected disk:    ~50-60 MB total.
"""
import bisect
import csv
import gzip
import json
import math
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND))          # so 'app.*' imports work

from .config import (
    CELL_TYPE_GENE_EFFECT_CSV,
    DEPMAP_DIR,
    ENCODE_DIR,
    FEATURES_DIR,
    GENCODE_GTF_GZ,
    SPLICE_PKL_OUT,
    TARGET_CELL_TYPES,
)

GUIDE_METADATA_CSV = FEATURES_DIR / "guide_metadata.csv"
CELL_FEATURES_CSV  = FEATURES_DIR / "cell_features.csv"
SEQ_FEATURES_NPZ   = FEATURES_DIR / "seq_features.npz"
FEATURE_SUMMARY    = FEATURES_DIR / "feature_summary.json"

# ── Constants ─────────────────────────────────────────────────────────────────
ATAC_WINDOW_BP  = 500          # +-500bp around guide cut site
ENTREZ_RE       = re.compile(r"\((\d+)\)$")  # extracts "8036" from "SHOC2 (8036)"
ENSG_RE         = re.compile(r"(ENSG\d+)")   # strips version suffix: ENSG00000108691.9 -> ENSG00000108691

# Cell types with at least ATAC data -- these become rows in cell_features.csv
ATAC_CELL_TYPES = ["T_cell_CD4", "T_cell_CD8", "NK_cell", "B_cell", "K562"]
RNA_CELL_TYPES  = ["T_cell_CD4", "T_cell_CD8", "B_cell", "K562"]   # NK has no RNA-seq
CELL_TYPE_IDX   = {ct: i for i, ct in enumerate(TARGET_CELL_TYPES)}


# ── Data loading ──────────────────────────────────────────────────────────────

def _build_ensembl_map() -> dict[str, str]:
    """
    Parse GENCODE v44 GTF (gene lines only) to build {ensembl_base_id: gene_name}.
    e.g. "ENSG00000108691" -> "SHOC2"
    Used to convert ENCODE RSEM gene_id (Ensembl) to gene symbol for DepMap join.
    """
    print("  Building Ensembl -> gene_name map from GENCODE v44 GTF...")
    if not GENCODE_GTF_GZ.exists():
        print("    WARNING: GENCODE GTF not found -- RNA-seq features will be 0")
        return {}

    ensg_re    = re.compile(r'gene_id "([^"]+)"')
    name_re    = re.compile(r'gene_name "([^"]+)"')
    mapping: dict[str, str] = {}

    with gzip.open(GENCODE_GTF_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            attrs = cols[8]
            m_id   = ensg_re.search(attrs)
            m_name = name_re.search(attrs)
            if m_id and m_name:
                # Strip version suffix: "ENSG00000108691.9" -> "ENSG00000108691"
                base_id = m_id.group(1).split(".")[0]
                mapping[base_id] = m_name.group(1)

    print(f"    {len(mapping):,} Ensembl -> gene_name mappings loaded")
    return mapping


def _load_guides() -> list[dict]:
    """
    Merge guide_map.csv + guide_efficacy.csv.
    Returns guides that have: valid sequence + genomic coords + efficacy score.
    """
    print("  Loading guide map + efficacy...")
    guide_map_path  = DEPMAP_DIR / "guide_map.csv"
    guide_eff_path  = DEPMAP_DIR / "guide_efficacy.csv"

    guides: dict[str, dict] = {}
    with open(guide_map_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq = row.get("sequence", "").strip()
            chr_ = row.get("chr", "").strip()
            start_s = row.get("start", "").strip()
            if not (seq and chr_ and start_s):
                continue
            # entrez_id column added in guide_map.csv; fallback: regex on gene name
            entrez_id = row.get("entrez_id", "").strip()
            if not entrez_id:
                gm = ENTREZ_RE.search(row.get("gene", ""))
                entrez_id = gm.group(1) if gm else ""
            guides[row["guide_id"]] = {
                "guide_id":  row["guide_id"],
                "sequence":  seq,
                "gene":      row.get("gene", ""),
                "entrez_id": entrez_id,
                "chr":       chr_,
                "start":     int(start_s),
                "strand":    row.get("strand", "+"),
                "efficacy":  None,
            }

    n_matched = 0
    with open(guide_eff_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = row.get("guide_id", "").strip()
            if gid in guides:
                try:
                    guides[gid]["efficacy"] = float(row["efficacy"])
                    n_matched += 1
                except (ValueError, KeyError):
                    pass

    valid = [v for v in guides.values() if v["efficacy"] is not None]
    print(f"    {len(valid):,} guides with coords + efficacy  "
          f"({n_matched:,} efficacy matches)")
    return valid


def _load_rna_tpm(ensembl_map: dict[str, str]) -> dict[tuple, float]:
    """
    Load rnaseq_tpm.csv -> {(gene_name, cell_type): tpm}.
    ENCODE RSEM files use Ensembl gene_ids; ensembl_map converts them to
    gene symbols (e.g. "ENSG00000108691" -> "SHOC2") for joining to DepMap.
    """
    print("  Loading RNA-seq TPM index...")
    rna: dict[tuple, float] = {}
    rna_csv = ENCODE_DIR / "rnaseq_tpm.csv"
    if not rna_csv.exists():
        print("    WARNING: rnaseq_tpm.csv not found -- RNA-seq features will be 0")
        return rna

    n_mapped = 0
    with open(rna_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_id   = row.get("gene_id", "").strip()
            cell_type = row.get("cell_type", "").strip()
            try:
                tpm = float(row["tpm"])
            except (ValueError, KeyError):
                continue

            # Convert ENSG ID to gene symbol; skip non-Ensembl IDs
            m = ENSG_RE.match(raw_id)
            if not m:
                continue
            gene_name = ensembl_map.get(m.group(1), "")
            if not gene_name:
                continue

            key = (gene_name, cell_type)
            # Keep maximum TPM when multiple isoforms/transcripts present
            if tpm > rna.get(key, 0.0):
                rna[key] = tpm
            n_mapped += 1

    print(f"    {n_mapped:,} ENSG records mapped -> {len(rna):,} (gene_name, cell_type) TPM entries")
    return rna


def _load_atac_index() -> dict[tuple, list]:
    """
    Load atac_index.csv into {(cell_type, chr): [(start, end, signal), ...]}
    sorted by start. Signals are max-normalised per cell type (0-1).
    """
    print("  Loading ATAC-seq peak index...")
    raw: dict[tuple, list] = defaultdict(list)
    atac_csv = ENCODE_DIR / "atac_index.csv"
    if not atac_csv.exists():
        print("    WARNING: atac_index.csv not found -- ATAC features will be 0")
        return {}

    max_sig: dict[str, float] = defaultdict(float)
    n_total = 0
    with open(atac_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ct  = row["cell_type"]
                sig = float(row["signal"])
                raw[(ct, row["chr"])].append((
                    int(row["start"]), int(row["end"]), sig
                ))
                max_sig[ct] = max(max_sig[ct], sig)
                n_total += 1
            except (ValueError, KeyError):
                continue

    atac: dict[tuple, list] = {}
    for key, peaks in raw.items():
        ct   = key[0]
        norm = max_sig[ct] or 1.0
        # Sort by start, normalise signal
        atac[key] = sorted([(s, e, sig / norm) for s, e, sig in peaks])

    n_buckets = len(atac)
    print(f"    {n_total:,} peaks -> {n_buckets} (cell_type, chr) buckets")
    return atac


def _load_gene_effect() -> dict[tuple, float]:
    """
    Load cell_type_gene_effect.csv -> {(gene_name, cell_type): mean_effect}.
    Returns empty dict if file not built yet (graceful fallback).
    """
    print("  Loading cell-type gene effect index...")
    ge: dict[tuple, float] = {}
    if not CELL_TYPE_GENE_EFFECT_CSV.exists():
        print("    WARNING: cell_type_gene_effect.csv not found -- "
              "using global efficacy as label (run build_lfc_labels first)")
        return ge
    with open(CELL_TYPE_GENE_EFFECT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ge[(row["gene_name"], row["cell_type"])] = float(row["mean_effect"])
            except (ValueError, KeyError):
                pass
    print(f"    {len(ge):,} (gene, cell_type) effect entries loaded")
    return ge


def _load_splice_db() -> dict[tuple, list]:
    """Load GENCODE splice site pkl -> {(chr, strand): sorted int list}."""
    print("  Loading splice site DB...")
    if not SPLICE_PKL_OUT.exists():
        print("    WARNING: splice_sites_grch38.pkl not found")
        return {}
    with open(SPLICE_PKL_OUT, "rb") as f:
        db = pickle.load(f)
    total = sum(len(v) for v in db.values())
    print(f"    {total:,} splice site positions in {len(db)} buckets")
    return db


def _load_seq_extractor():
    """Import extract_features from existing app, fallback to built-in."""
    try:
        from app.services.feature_engineering import extract_features
        # Verify it returns 450-dim
        test = extract_features("ATGCATGCATGCATGCATGC", thirty_mer="")
        if len(test) == 450:
            print("  Sequence features: using app.services.feature_engineering (450-dim)")
            return extract_features, len(test)
    except Exception:
        pass
    print("  Sequence features: using built-in fallback (97-dim)")
    return _basic_seq_features, 97


# ── Feature queries ───────────────────────────────────────────────────────────

def _query_atac(atac: dict, chr_: str, cut: int, cell_type: str,
                window: int = ATAC_WINDOW_BP) -> tuple[float, int]:
    """
    Return (max_signal, n_peaks) for ATAC peaks in [cut-window, cut+window].
    Uses bisect on sorted start positions for O(log n + k) lookup.
    """
    key = (cell_type, chr_)
    peaks = atac.get(key)
    if not peaks:
        return 0.0, 0
    lo, hi = cut - window, cut + window
    starts = [p[0] for p in peaks]
    # First peak index where start >= lo
    i = bisect.bisect_left(starts, lo)
    # Also look back a few (peaks starting before lo that extend into window)
    i = max(0, i - 20)
    max_sig, n = 0.0, 0
    for start, end, sig in peaks[i:]:
        if start > hi:
            break
        if end >= lo:          # overlaps the window
            max_sig = max(max_sig, sig)
            n += 1
    return max_sig, n


def _query_splice(db: dict, chr_: str, pos: int, strand: str) -> float:
    """Return log1p(distance bp) to nearest splice site. 0 if DB missing."""
    positions = db.get((chr_, strand))
    if not positions:
        return 0.0
    idx = bisect.bisect_left(positions, pos)
    candidates = []
    if idx < len(positions):
        candidates.append(abs(positions[idx] - pos))
    if idx > 0:
        candidates.append(abs(positions[idx - 1] - pos))
    return math.log1p(min(candidates)) if candidates else 0.0


def _query_rna(rna: dict, gene_name: str, cell_type: str) -> float:
    """Return log1p(TPM) for (gene_name, cell_type). 0 if not found."""
    tpm = rna.get((gene_name, cell_type), 0.0)
    return math.log1p(tpm)


def _basic_seq_features(sequence: str, thirty_mer: str = "") -> np.ndarray:
    """97-dim fallback: 80 pos-onehot + 1 GC + 16 dinucs."""
    BASES = "ACGT"
    feats: list[float] = []
    for base in sequence:
        for b in BASES:
            feats.append(1.0 if base == b else 0.0)
    feats.append(sum(1 for b in sequence if b in "GC") / max(len(sequence), 1))
    dinucs = [a + b for a in BASES for b in BASES]
    n_di = max(len(sequence) - 1, 1)
    counts = {d: 0 for d in dinucs}
    for i in range(len(sequence) - 1):
        d = sequence[i:i+2]
        if d in counts:
            counts[d] += 1
    for d in dinucs:
        feats.append(counts[d] / n_di)
    return np.array(feats, dtype=np.float32)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_features() -> None:
    print(f"\n{'='*60}")
    print("  OmicsCRISPR Phase 2 -- Feature Engineering")
    print(f"{'='*60}")

    if (GUIDE_METADATA_CSV.exists() and CELL_FEATURES_CSV.exists()
            and SEQ_FEATURES_NPZ.exists()):
        print("  [skip] All feature outputs already exist. Delete to rebuild.")
        _print_summary()
        return

    t0_total = time.time()

    # ── Load data ──
    print("\nLoading Phase 1 data...")
    guides      = _load_guides()
    ensembl_map = _build_ensembl_map()
    rna         = _load_rna_tpm(ensembl_map)
    atac        = _load_atac_index()
    splice_db   = _load_splice_db()
    gene_effect = _load_gene_effect()
    extract_fn, seq_dim = _load_seq_extractor()

    # Pre-compute z-score stats for combined label construction
    # combined_label = 0.4 * z(guide_eff) + 0.6 * z(gene_effect_ct)
    ge_has_data   = len(gene_effect) > 0
    guide_effs    = [g["efficacy"] for g in guides]
    ge_mean_g     = float(np.mean(guide_effs))
    ge_std_g      = float(np.std(guide_effs)) + 1e-8
    if ge_has_data:
        ge_vals   = list(gene_effect.values())
        ge_mean_e = float(np.mean(ge_vals))
        ge_std_e  = float(np.std(ge_vals)) + 1e-8
    else:
        ge_mean_e = ge_std_e = 0.0

    n_guides = len(guides)
    n_cts    = len(ATAC_CELL_TYPES)
    print(f"\n  Guides to process : {n_guides:,}")
    print(f"  Cell types        : {n_cts}  ({', '.join(ATAC_CELL_TYPES)})")
    print(f"  Sequence feat dim : {seq_dim}")
    print(f"  Output rows (est) : {n_guides * n_cts:,}")

    # ── Compute features ──
    print("\nComputing features...")
    seq_matrix:    np.ndarray = np.zeros((n_guides, seq_dim), dtype=np.float32)
    meta_rows:     list[dict] = []
    cell_rows:     list[dict] = []

    t0 = time.time()
    for guide_idx, guide in enumerate(guides):
        if guide_idx % 5000 == 0 and guide_idx > 0:
            elapsed = time.time() - t0
            rate    = guide_idx / elapsed
            eta     = (n_guides - guide_idx) / rate
            print(f"\r  {guide_idx:>6,}/{n_guides:,}  "
                  f"({guide_idx/n_guides*100:.1f}%)  "
                  f"ETA {eta/60:.1f} min", end="", flush=True)

        seq       = guide["sequence"]
        chr_      = guide["chr"]
        start     = guide["start"]
        strand    = guide["strand"]
        gene_name = guide["gene"]   # e.g. "SHOC2" — used for RNA-seq join

        # Cas9 cut site: 17bp from guide start (between pos 17-18)
        cut_site = start + 17 if strand == "+" else start + 3

        # Sequence features (once per guide)
        seq_feats = extract_fn(seq, thirty_mer="")
        seq_matrix[guide_idx] = seq_feats[:seq_dim]

        # Guide metadata row
        meta_rows.append({
            "guide_id":  guide["guide_id"],
            "gene":      gene_name,
            "entrez_id": guide["entrez_id"],
            "chr":       chr_,
            "start":     start,
            "strand":    strand,
            "efficacy":  guide["efficacy"],
            "guide_idx": guide_idx,
        })

        # Per cell-type omics features
        splice_log  = _query_splice(splice_db, chr_, cut_site, strand)
        guide_eff_z = (guide["efficacy"] - ge_mean_g) / ge_std_g

        for ct in ATAC_CELL_TYPES:
            atac_sig, atac_n = _query_atac(atac, chr_, cut_site, ct)
            rna_log          = _query_rna(rna, gene_name, ct) if ct in RNA_CELL_TYPES else 0.0

            # Gene effect for this (gene, cell_type) pair
            ge_ct = gene_effect.get((gene_name, ct))
            if ge_ct is None and ct == "T_cell_CD8":
                ge_ct = gene_effect.get((gene_name, "T_cell_CD4"))  # same DepMap proxy

            # Combined label: blend guide efficiency + cell-type gene essentiality
            if ge_has_data and ge_ct is not None:
                ge_z    = (ge_ct - ge_mean_e) / ge_std_e
                label   = round(0.4 * guide_eff_z + 0.6 * ge_z, 5)
            else:
                label   = round(guide_eff_z, 5)

            cell_rows.append({
                "guide_id":        guide["guide_id"],
                "guide_idx":       guide_idx,
                "cell_type":       ct,
                "rna_tpm_log1p":   round(rna_log,         5),
                "atac_signal":     round(atac_sig,         5),
                "atac_n_peaks":    atac_n,
                "splice_dist_log": round(splice_log,       5),
                "gene_effect":     round(ge_ct, 5) if ge_ct is not None else 0.0,
                "cell_type_idx":   CELL_TYPE_IDX.get(ct, -1),
                "label":           label,
            })

    print(f"\r  {n_guides:,}/{n_guides:,}  (100.0%)  Done in "
          f"{(time.time()-t0)/60:.1f} min")

    # ── Write outputs ──
    print("\nWriting outputs...")

    print(f"  guide_metadata.csv  ({len(meta_rows):,} rows)...")
    with open(GUIDE_METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "guide_id", "gene", "entrez_id", "chr", "start",
            "strand", "efficacy", "guide_idx",
        ])
        w.writeheader()
        w.writerows(meta_rows)

    print(f"  cell_features.csv   ({len(cell_rows):,} rows)...")
    with open(CELL_FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "guide_id", "guide_idx", "cell_type", "rna_tpm_log1p",
            "atac_signal", "atac_n_peaks", "splice_dist_log",
            "gene_effect", "cell_type_idx", "label",
        ])
        w.writeheader()
        w.writerows(cell_rows)

    print(f"  seq_features.npz    (shape {seq_matrix.shape})...")
    np.savez_compressed(SEQ_FEATURES_NPZ, seq_features=seq_matrix)

    # ── Summary ──
    n_atac_nonzero = sum(1 for r in cell_rows if r["atac_signal"] > 0)
    n_rna_nonzero  = sum(1 for r in cell_rows if r["rna_tpm_log1p"] > 0)
    n_splice_known = sum(1 for r in cell_rows if r["splice_dist_log"] > 0)
    n_ge_nonzero   = sum(1 for r in cell_rows if r["gene_effect"] != 0.0)

    summary = {
        "n_guides":              n_guides,
        "n_cell_types":          n_cts,
        "n_cell_feature_rows":   len(cell_rows),
        "seq_feature_dim":       seq_dim,
        "atac_cell_types":       ATAC_CELL_TYPES,
        "rna_cell_types":        RNA_CELL_TYPES,
        "has_gene_effect_labels": ge_has_data,
        "atac_coverage_pct":     round(n_atac_nonzero / max(len(cell_rows), 1) * 100, 2),
        "rna_coverage_pct":      round(n_rna_nonzero  / max(len(cell_rows), 1) * 100, 2),
        "splice_coverage_pct":   round(n_splice_known / max(len(cell_rows), 1) * 100, 2),
        "gene_effect_coverage_pct": round(n_ge_nonzero / max(len(cell_rows), 1) * 100, 2),
        "total_time_min":        round((time.time() - t0_total) / 60, 2),
    }
    with open(FEATURE_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 2 complete in {summary['total_time_min']} min")
    _print_summary()


def _print_summary() -> None:
    if not FEATURE_SUMMARY.exists():
        return
    with open(FEATURE_SUMMARY, encoding="utf-8") as f:
        s = json.load(f)
    print("\n--- Feature Matrix Summary ---")
    print(f"  Guides                : {s['n_guides']:,}")
    print(f"  Cell types            : {s['n_cell_types']}  ({', '.join(s['atac_cell_types'])})")
    print(f"  Total training rows   : {s['n_cell_feature_rows']:,}")
    print(f"  Sequence feature dim  : {s['seq_feature_dim']}")
    print(f"  ATAC coverage         : {s['atac_coverage_pct']}% of (guide, cell_type) pairs")
    print(f"  RNA-seq coverage      : {s['rna_coverage_pct']}%")
    print(f"  Splice dist coverage  : {s['splice_coverage_pct']}%")
    ge_cov = s.get("gene_effect_coverage_pct", 0)
    print(f"  Gene effect coverage  : {ge_cov}%  "
          f"({'combined labels active' if s.get('has_gene_effect_labels') else 'global label fallback'})")


def load_training_data() -> dict:
    """
    Utility: load all feature outputs and return ready-to-use arrays.

    Returns:
        {
          "seq_features":   np.ndarray (N_guides, seq_dim),
          "cell_features":  np.ndarray (N_rows, 6),   # rna, atac_sig, atac_n, splice, gene_eff, ct_idx
          "labels":         np.ndarray (N_rows,),      # combined label (z-scored)
          "guide_idx":      np.ndarray (N_rows,),      # maps cell row -> seq row
          "metadata":       list[dict],                # guide-level metadata
        }
    """
    seq = np.load(SEQ_FEATURES_NPZ)["seq_features"]

    meta: list[dict] = []
    with open(GUIDE_METADATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta.append(row)

    guide_idxs, cell_feats, labels = [], [], []
    with open(CELL_FEATURES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gidx = int(row["guide_idx"])
            guide_idxs.append(gidx)
            cell_feats.append([
                float(row["rna_tpm_log1p"]),
                float(row["atac_signal"]),
                float(row["atac_n_peaks"]),
                float(row["splice_dist_log"]),
                float(row["gene_effect"]),
                float(row["cell_type_idx"]),
            ])
            labels.append(float(row["label"]))

    return {
        "seq_features":  seq,
        "cell_features": np.array(cell_feats, dtype=np.float32),
        "labels":        np.array(labels,     dtype=np.float32),
        "guide_idx":     np.array(guide_idxs, dtype=np.int32),
        "metadata":      meta,
    }


if __name__ == "__main__":
    build_features()
