"""
Build a genome-wide splice site database from GENCODE v44 annotation.

Downloads gencode.v44.annotation.gtf.gz (~1.5 GB) from EBI FTP,
then extracts all splice donor (5') and acceptor (3') positions from
annotated exon boundaries across all protein-coding and lncRNA transcripts.

Outputs written to data/omics/splice/:
  splice_sites_grch38.bed  — BED-format: chr  pos  pos+1  strand  type  gene_name
  splice_sites_grch38.pkl  — Python dict for fast coordinate lookup:
                             {(chr, strand): sorted list of int positions}

The pickle is used by the feature engineering module to query:
    "what is the distance from guide cut site to nearest splice site?"

Run from backend/:
    python -m omics_pipeline.build_splice_db

Note: GTF download is ~1.5 GB. Parsing takes ~5 min on a laptop.
Already-downloaded GTF and already-built DB are skipped automatically.
"""
import bisect
import csv
import gzip
import pickle
import time
import urllib.request
from collections import defaultdict

from .config import (
    GENCODE_GTF_GZ,
    GENCODE_GTF_URL,
    GENCODE_VERSION,
    SPLICE_BED_OUT,
    SPLICE_PKL_OUT,
    SPLICE_DIR,
)

CHUNK = 4 << 20  # 4 MB streaming chunk

# GTF feature types that define exon boundaries we want
KEEP_FEATURES = {"exon"}

# Gene biotypes whose splice sites are clinically relevant
KEEP_BIOTYPES = {
    "protein_coding",
    "lncRNA",
    "processed_transcript",
    "retained_intron",
    "nonsense_mediated_decay",
}


# ── Download ──────────────────────────────────────────────────────────────────

def _download_gtf() -> None:
    """Stream-download GENCODE GTF if not already present."""
    if GENCODE_GTF_GZ.exists():
        size_mb = GENCODE_GTF_GZ.stat().st_size >> 20
        print(f"  [skip] GTF already downloaded ({size_mb} MB) -> {GENCODE_GTF_GZ.name}")
        return

    print(f"  Downloading GENCODE v{GENCODE_VERSION} GTF (~1.5 GB)...")
    print(f"  URL: {GENCODE_GTF_URL}")
    req = urllib.request.Request(
        GENCODE_GTF_URL,
        headers={"User-Agent": "OmicsCRISPR/1.0"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            with open(GENCODE_GTF_GZ, "wb") as f:
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        elapsed = time.time() - t0
                        speed = (downloaded >> 20) / max(elapsed, 1)
                        print(
                            f"\r    {pct:5.1f}%  {downloaded >> 20}/{total >> 20} MB"
                            f"  {speed:.1f} MB/s",
                            end="", flush=True,
                        )
            print(f"\r    Done. {downloaded >> 20} MB in {time.time()-t0:.0f}s.          ")
    except Exception as e:
        if GENCODE_GTF_GZ.exists():
            GENCODE_GTF_GZ.unlink()
        raise RuntimeError(f"GTF download failed: {e}") from e


# ── Parse GTF -> splice sites ──────────────────────────────────────────────────

def _parse_attr(attr_str: str) -> dict[str, str]:
    """Parse GTF attribute string into dict. Fast path — no regex."""
    attrs: dict[str, str] = {}
    for field in attr_str.split(";"):
        field = field.strip()
        if not field:
            continue
        space = field.find(" ")
        if space < 0:
            continue
        key = field[:space]
        val = field[space + 1:].strip().strip('"')
        attrs[key] = val
    return attrs


def _extract_splice_sites() -> tuple[list[dict], dict]:
    """
    Parse GTF and extract splice donor/acceptor positions.

    For each exon in a multi-exon transcript:
      - Donor   (5' splice site) = last position of exon (just before intron start)
      - Acceptor (3' splice site) = first position of next exon (just after intron end)

    We work at the transcript level: group exons per transcript, sort by position,
    then derive splice boundaries from adjacent pairs.

    Returns:
        bed_rows — list of dicts for BED output
        lookup   — {(chr, strand): sorted list of int positions}
    """
    print("  Parsing GTF for splice site positions...")
    print("  (grouping exons by transcript — may take 3-5 minutes)")

    # {transcript_id: {"chr": str, "strand": str, "gene": str,
    #                   "biotype": str, "exons": [(start_1based, end_1based)]}}
    transcripts: dict[str, dict] = {}
    n_lines = 0

    with gzip.open(GENCODE_GTF_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            n_lines += 1
            if n_lines % 500_000 == 0:
                print(f"\r    {n_lines:,} GTF lines read...", end="", flush=True)

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            feature = parts[2]
            if feature not in KEEP_FEATURES:
                continue

            chrom  = parts[0]
            start  = int(parts[3])  # 1-based inclusive
            end    = int(parts[4])  # 1-based inclusive
            strand = parts[6]

            attrs = _parse_attr(parts[8])
            biotype = attrs.get("gene_type", attrs.get("gene_biotype", ""))
            if biotype and biotype not in KEEP_BIOTYPES:
                continue

            tx_id   = attrs.get("transcript_id", "")
            gene_nm = attrs.get("gene_name", attrs.get("gene_id", ""))

            if not tx_id:
                continue

            if tx_id not in transcripts:
                transcripts[tx_id] = {
                    "chr":    chrom,
                    "strand": strand,
                    "gene":   gene_nm,
                    "exons":  [],
                }
            transcripts[tx_id]["exons"].append((start, end))

    print(f"\r    Done. {n_lines:,} lines read, {len(transcripts):,} transcripts found.")

    # ── Derive splice sites from adjacent exon pairs ──
    print("  Deriving splice sites from exon adjacencies...")
    bed_rows: list[dict] = []
    lookup: dict[tuple, list] = defaultdict(list)

    seen: set[tuple] = set()  # (chr, pos, strand) deduplication

    for tx_data in transcripts.values():
        exons = sorted(tx_data["exons"])  # sort by start
        if len(exons) < 2:
            continue  # single-exon transcript — no splice sites

        chrom  = tx_data["chr"]
        strand = tx_data["strand"]
        gene   = tx_data["gene"]

        for i in range(len(exons) - 1):
            ex_curr = exons[i]
            ex_next = exons[i + 1]

            # Donor: end of current exon (last exonic base)
            donor_pos = ex_curr[1]
            # Acceptor: start of next exon (first exonic base)
            acc_pos   = ex_next[0]

            for pos, site_type in ((donor_pos, "donor"), (acc_pos, "acceptor")):
                key = (chrom, pos, strand)
                if key in seen:
                    continue
                seen.add(key)

                bed_rows.append({
                    "chr":    chrom,
                    "start":  pos - 1,    # convert to 0-based for BED
                    "end":    pos,
                    "strand": strand,
                    "type":   site_type,
                    "gene":   gene,
                })
                lookup[(chrom, strand)].append(pos)

    # Sort position lists for bisect binary search
    for key in lookup:
        lookup[key].sort()

    print(f"  Extracted {len(bed_rows):,} unique splice sites "
          f"({sum(1 for r in bed_rows if r['type']=='donor'):,} donors, "
          f"{sum(1 for r in bed_rows if r['type']=='acceptor'):,} acceptors)")

    return bed_rows, dict(lookup)


# ── Write outputs ─────────────────────────────────────────────────────────────

def _write_bed(bed_rows: list[dict]) -> None:
    print(f"  Writing BED file -> {SPLICE_BED_OUT.name}")
    bed_rows_sorted = sorted(bed_rows, key=lambda r: (r["chr"], r["start"]))
    with open(SPLICE_BED_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chr", "start", "end", "strand", "type", "gene"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(bed_rows_sorted)
    print(f"  Done. {len(bed_rows_sorted):,} rows.")


def _write_pkl(lookup: dict) -> None:
    print(f"  Writing pickle index -> {SPLICE_PKL_OUT.name}")
    with open(SPLICE_PKL_OUT, "wb") as f:
        pickle.dump(lookup, f, protocol=4)
    size_mb = SPLICE_PKL_OUT.stat().st_size >> 20
    print(f"  Done. {len(lookup):,} (chr, strand) keys, {size_mb} MB.")


# ── Query interface (used by feature engineering) ─────────────────────────────

def load_splice_db() -> dict:
    """Load splice site lookup dict from pickle. Returns empty dict if not built."""
    if not SPLICE_PKL_OUT.exists():
        return {}
    with open(SPLICE_PKL_OUT, "rb") as f:
        return pickle.load(f)


def distance_to_nearest_splice_site(
    lookup: dict,
    chrom: str,
    pos: int,
    strand: str,
) -> int:
    """
    Return distance (bp) from genomic position to nearest splice site.
    Uses binary search on the sorted position list.
    Returns -1 if no splice sites exist for that (chr, strand).
    """
    key = (chrom, strand)
    positions = lookup.get(key)
    if not positions:
        return -1

    idx = bisect.bisect_left(positions, pos)
    candidates = []
    if idx < len(positions):
        candidates.append(abs(positions[idx] - pos))
    if idx > 0:
        candidates.append(abs(positions[idx - 1] - pos))
    return min(candidates) if candidates else -1


# ── Master ────────────────────────────────────────────────────────────────────

def build_splice_db() -> None:
    print(f"\n{'='*60}")
    print(f"  GENCODE v{GENCODE_VERSION} — splice site database")
    print(f"{'='*60}")

    if SPLICE_BED_OUT.exists() and SPLICE_PKL_OUT.exists():
        n_bed = sum(1 for _ in open(SPLICE_BED_OUT, encoding="utf-8")) - 1
        print(f"  [skip] Splice DB already built ({n_bed:,} sites in BED)")
        return

    # Download GTF
    _download_gtf()

    # Parse GTF
    bed_rows, lookup = _extract_splice_sites()

    # Write outputs
    _write_bed(bed_rows)
    _write_pkl(lookup)

    print("\nSplice DB build complete.")
    print(f"Outputs in: {SPLICE_DIR}")

    # Quick sanity check
    print("\nSanity check — distance from chr1:1000000 (+ strand):")
    dist = distance_to_nearest_splice_site(lookup, "chr1", 1_000_000, "+")
    if dist >= 0:
        print(f"  Nearest splice site: {dist:,} bp away")
    else:
        print("  No splice sites found for chr1+ (unexpected — check GTF parse)")


if __name__ == "__main__":
    build_splice_db()
