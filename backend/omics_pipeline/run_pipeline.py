"""
OmicsCRISPR -Phase 1 Data Pipeline Master Runner

Runs all data download and processing steps in order:

  Step 1 -DepMap Avana CRISPR screen data
    Guide sequences, LFC scores, cell line expression + metadata
    Source: DepMap 23Q4 figshare (https://figshare.com/articles/24667389)

  Step 2 -ENCODE RNA-seq + ATAC-seq
    Gene quantification TSVs and peak BEDs for 7 target cell types
    Source: ENCODE portal REST API (encodeproject.org)

  Step 3 -GENCODE v44 splice site database
    Exon boundary donor/acceptor positions across hg38
    Source: EBI FTP (ftp.ebi.ac.uk/pub/databases/gencode)

Usage (from backend/ directory):
    # Full pipeline
    python -m omics_pipeline.run_pipeline

    # Individual steps
    python -m omics_pipeline.download_depmap
    python -m omics_pipeline.download_encode
    python -m omics_pipeline.build_splice_db

    # Check what's been downloaded
    python -m omics_pipeline.run_pipeline --status

Expected disk usage:
  DepMap (guide map + LFC + expression): ~500 MB
  ENCODE (RNA-seq TSVs + ATAC peaks):    ~200 MB
  GENCODE GTF (gzipped):                 ~1.5 GB
  Processed outputs:                     ~100 MB
  Total:                                 ~2.3 GB
"""
import argparse
import sys
import time
from pathlib import Path

from .config import (
    DEPMAP_DIR,
    DEPMAP_FILES,
    DEPMAP_RELEASE,
    ENCODE_DIR,
    GENCODE_GTF_GZ,
    OMICS_DIR,
    SPLICE_BED_OUT,
    SPLICE_PKL_OUT,
    TARGET_CELL_TYPES,
)


# ── Status check ──────────────────────────────────────────────────────────────

def _fmt_size(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    size = path.stat().st_size
    if size > 1 << 30:
        return f"{size / (1<<30):.1f} GB"
    if size > 1 << 20:
        return f"{size >> 20} MB"
    return f"{size >> 10} KB"


def print_status() -> None:
    print("\nOmicsCRISPR Phase 1 - Data Status")
    print("=" * 60)

    print(f"\n[DepMap {DEPMAP_RELEASE}]")
    for key, filename in DEPMAP_FILES.items():
        p = DEPMAP_DIR / filename
        print(f"  {filename:<55} {_fmt_size(p)}")

    processed = [
        DEPMAP_DIR / "guide_map.csv",
        DEPMAP_DIR / "guide_efficacy.csv",
        DEPMAP_DIR / "cell_metadata.csv",
    ]
    print("  Processed outputs:")
    for p in processed:
        print(f"    {p.name:<50} {_fmt_size(p)}")

    print(f"\n[ENCODE - {len(TARGET_CELL_TYPES)} cell types]")
    rna_csv  = ENCODE_DIR / "rnaseq_tpm.csv"
    atac_csv = ENCODE_DIR / "atac_index.csv"
    rna_dir  = ENCODE_DIR / "rnaseq"
    atac_dir = ENCODE_DIR / "atac"
    print(f"  RNA-seq TSVs downloaded: {len(list(rna_dir.glob('*.tsv*')))}/{len(TARGET_CELL_TYPES)}")
    print(f"  ATAC peaks downloaded:   {len(list(atac_dir.glob('*.gz')))}/{len(TARGET_CELL_TYPES)}")
    print(f"  rnaseq_tpm.csv:          {_fmt_size(rna_csv)}")
    print(f"  atac_index.csv:          {_fmt_size(atac_csv)}")

    print("\n[GENCODE v44 Splice DB]")
    print(f"  GTF (gzipped):           {_fmt_size(GENCODE_GTF_GZ)}")
    print(f"  splice_sites.bed:        {_fmt_size(SPLICE_BED_OUT)}")
    print(f"  splice_sites.pkl:        {_fmt_size(SPLICE_PKL_OUT)}")

    print(f"\nAll outputs in: {OMICS_DIR}")

    # Completeness summary
    steps_done = [
        (DEPMAP_DIR / "guide_map.csv").exists(),
        (DEPMAP_DIR / "guide_efficacy.csv").exists(),
        rna_csv.exists(),
        atac_csv.exists(),
        SPLICE_PKL_OUT.exists(),
    ]
    n_done = sum(steps_done)
    print(f"\nPipeline completeness: {n_done}/5 key outputs present")
    if n_done == 5:
        print("  Phase 1 COMPLETE - ready for Phase 2 feature engineering")
    else:
        print("  Run: python -m omics_pipeline.run_pipeline  to complete")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(steps: list[str] | None = None) -> None:
    all_steps = ["depmap", "encode", "splice", "lfc_labels", "features", "train", "phase4"]
    if steps:
        to_run = [s for s in all_steps if s in steps]
    else:
        to_run = all_steps

    phase = "1+2+3+4" if "phase4" in to_run else ("1+2+3" if "train" in to_run else ("1+2" if "features" in to_run else "1"))
    print("\n" + "=" * 60)
    print(f"  OmicsCRISPR Phase {phase} - Multi-Omics Data Pipeline")
    print("=" * 60)
    print(f"  Steps to run: {', '.join(to_run)}")
    print(f"  Output root:  {OMICS_DIR}")

    t_total = time.time()

    if "depmap" in to_run:
        t0 = time.time()
        try:
            from .download_depmap import download_depmap
            download_depmap()
            print(f"\n  [DepMap] Done in {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"\n  [DepMap] FAILED: {e}")
            print("  Continuing with remaining steps...")

    if "encode" in to_run:
        t0 = time.time()
        try:
            from .download_encode import download_encode
            download_encode()
            print(f"\n  [ENCODE] Done in {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"\n  [ENCODE] FAILED: {e}")
            print("  Continuing with remaining steps...")

    if "splice" in to_run:
        t0 = time.time()
        try:
            from .build_splice_db import build_splice_db
            build_splice_db()
            print(f"\n  [Splice DB] Done in {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"\n  [Splice DB] FAILED: {e}")
            print("  Continuing with remaining steps...")

    if "lfc_labels" in to_run:
        t0 = time.time()
        try:
            from .build_lfc_labels import build_lfc_labels
            build_lfc_labels()
            print(f"\n  [LFC Labels] Done in {time.time()-t0:.0f}s")
        except Exception as e:
            import traceback
            print(f"\n  [LFC Labels] FAILED: {e}")
            traceback.print_exc()

    if "features" in to_run:
        t0 = time.time()
        try:
            from .build_features import build_features
            build_features()
            print(f"\n  [Feature Engineering] Done in {time.time()-t0:.0f}s")
        except Exception as e:
            import traceback
            print(f"\n  [Feature Engineering] FAILED: {e}")
            traceback.print_exc()

    if "train" in to_run:
        t0 = time.time()
        try:
            from .train_omics_model import train_omics_model
            train_omics_model()
            print(f"\n  [Model Training] Done in {time.time()-t0:.0f}s")
        except Exception as e:
            import traceback
            print(f"\n  [Model Training] FAILED: {e}")
            traceback.print_exc()

    if "phase4" in to_run:
        t0 = time.time()
        try:
            from .score_phase4 import run_phase4
            run_phase4()
            print(f"\n  [Phase 4] Done in {time.time()-t0:.0f}s")
        except Exception as e:
            import traceback
            print(f"\n  [Phase 4] FAILED: {e}")
            traceback.print_exc()

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Pipeline finished in {elapsed/60:.1f} min")
    print_status()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OmicsCRISPR Phase 1 -Multi-Omics Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m omics_pipeline.run_pipeline              # full pipeline
  python -m omics_pipeline.run_pipeline --steps depmap encode
  python -m omics_pipeline.run_pipeline --status     # check download status
        """,
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=["depmap", "encode", "splice", "lfc_labels", "features", "train", "phase4"],
        help="Run only specified steps (default: all)",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print download status and exit",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.status:
        print_status()
        sys.exit(0)
    run_pipeline(steps=args.steps)
