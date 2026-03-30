"""
Download and merge publicly available gRNA efficiency datasets for training.

Datasets downloaded:
  1. Doench et al. 2016 (Azimuth / Rule Set 2)
     Source: MicrosoftResearch/Azimuth GitHub (MIT License)
     Guides: ~4,400  |  30-mer context → guide = 30mer[4:24]
     Score:  score_drug_gene_rank (gene-rank-normalised efficiency, 0-1)

  2. Doench et al. 2014 (Rule Set 1)
     Source: MicrosoftResearch/Azimuth GitHub (MIT License)
     Guides: ~313   |  20bp spacer only (no flanking context available)
     Score:  Percent Rank (rank-normalised, 0-1)

  3. Kim et al. 2019 (DeepSpCas9, Science Advances eaax9249)
     Source: L-Q-Y/CRISPRtool GitHub (preprocessed CSV)
     Guides: ~12,825  |  30-mer context, indel frequency (%)
     Score:  Indel frequency / 100, clamped to [0, 1]
     Split:  80% → training (Kim2019 source), 20% → data/kim2019_holdout.csv

  4. Moreno-Mateos et al. 2015 (CRISPRscan — zebrafish in-vivo, optional)
     Source: maximilianh/crisporPaper GitHub (Nat Methods 12:982)
     Guides: ~1,020  |  NOT added to training (cross-organism, zebrafish)

Run from backend/ directory:
    python download_datasets.py

Output: data/combined_training_data.csv
  Columns: sequence (20bp), score (0-1), source, thirty_mer (30bp or empty)
Also writes: data/kim2019_holdout.csv (20% held-out Kim 2019 guides)
"""
import sys, re, csv, io, pickle, urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

DOENCH_2016_URL = (
    "https://raw.githubusercontent.com/MicrosoftResearch/Azimuth"
    "/master/azimuth/data/FC_plus_RES_withPredictions.csv"
)
DOENCH_2014_URL = (
    "https://raw.githubusercontent.com/MicrosoftResearch/Azimuth"
    "/master/azimuth/data/V1_suppl_data.txt"
)
# Kim et al. 2019 (DeepSpCas9, Science Advances eaax9249)
KIM_2019_URL = (
    "https://raw.githubusercontent.com/L-Q-Y/CRISPRtool"
    "/main/data/Cas9/Kim2019_train.csv"
)
# 80/20 split seed — ensures reproducible train/holdout split
KIM_2019_HOLDOUT_SEED = 42
KIM_2019_HOLDOUT_FRAC = 0.20

VALID_BASES = set("ACGT")


def _fetch(url: str) -> str:
    print(f"  Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "gRNA-Predictor/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def _fetch_bytes(url: str) -> bytes:
    print(f"  Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "gRNA-Predictor/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _is_valid(seq: str, length: int = 20) -> bool:
    return len(seq) == length and set(seq.upper()) <= VALID_BASES


def _is_valid_30mer(seq: str) -> bool:
    return len(seq) == 30 and set(seq.upper()) <= VALID_BASES


# ─────────────────────────────────────────────────────────────────────────────
# Doench 2016
# ─────────────────────────────────────────────────────────────────────────────

def load_doench2016(raw: str) -> list[dict]:
    """
    Parse FC_plus_RES_withPredictions.csv.
    Saves full 30-mer for flanking-context feature extraction.
    Guide = 30mer[4:24], PAM = 30mer[24:27] must match [ACGT]GG
    Score = score_drug_gene_rank (gene-rank-normalised, 0-1)
    """
    reader  = csv.DictReader(io.StringIO(raw))
    records = []
    seen    = set()
    for row in reader:
        thirty = row.get("30mer", "").strip().upper()
        if len(thirty) < 27:
            continue
        guide = thirty[4:24]
        pam   = thirty[24:27]
        if not re.fullmatch(r"[ACGT]GG", pam):
            continue
        if not (_is_valid(guide) and _is_valid_30mer(thirty)):
            continue
        try:
            score = float(row["score_drug_gene_rank"])
        except (ValueError, KeyError):
            continue
        if not (0.0 <= score <= 1.0):
            continue
        if guide in seen:
            continue
        seen.add(guide)
        records.append({
            "sequence":   guide,
            "score":      score,
            "source":     "Doench2016",
            "thirty_mer": thirty,
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Doench 2014
# ─────────────────────────────────────────────────────────────────────────────

def load_doench2014(raw: str) -> list[dict]:
    """
    Parse V1_suppl_data.txt (tab-separated).
    No 30-mer context available — thirty_mer left empty.
    Score = Percent Rank (rank-normalised, 0-1)
    """
    reader  = csv.DictReader(io.StringIO(raw), delimiter="\t")
    records = []
    seen    = set()
    for row in reader:
        guide = row.get("Spacer Sequence", "").strip().upper()
        if not _is_valid(guide):
            continue
        try:
            score = float(row["Percent Rank"])
        except (ValueError, KeyError):
            continue
        if not (0.0 <= score <= 1.0):
            continue
        if guide in seen:
            continue
        seen.add(guide)
        records.append({
            "sequence":   guide,
            "score":      score,
            "source":     "Doench2014",
            "thirty_mer": "",
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# CRISPRscan / Moreno-Mateos 2015
# ─────────────────────────────────────────────────────────────────────────────

def load_crisprscan(raw: str) -> list[dict]:
    """
    Parse morenoMateos2015.context.tab from CRISPOR benchmarking repo.
    Columns (tab-separated): guide, seq (23-mer = 20bp+NGG), db, pos, modFreq, longSeq
    modFreq is normalised mutagenesis frequency 0-1.
    longSeq is a 100bp context string; we extract 30-mer from it when possible.
    """
    records = []
    seen    = set()
    header  = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        # Parse header to find column indices
        if header is None:
            header = [p.lower() for p in parts]
            continue
        if len(parts) < 5:
            continue
        # Locate columns
        try:
            seq_idx   = header.index("seq") if "seq" in header else 1
            score_idx = header.index("modfreq") if "modfreq" in header else 4
        except ValueError:
            seq_idx, score_idx = 1, 4

        seq23 = parts[seq_idx].strip().upper() if seq_idx < len(parts) else ""
        # seq is 23-mer (20bp guide + NGG PAM); strip PAM
        if len(seq23) == 23 and seq23[20:] in {"AGG", "CGG", "GGG", "TGG"}:
            guide = seq23[:20]
        elif len(seq23) == 20:
            guide = seq23
        else:
            continue
        if not _is_valid(guide):
            continue
        try:
            score = float(parts[score_idx])
        except (ValueError, IndexError):
            continue
        score = score / 100.0 if score > 1.0 else score
        if not (0.0 <= score <= 1.0):
            continue
        if guide in seen:
            continue
        # Extract 30-mer from longSeq context if available
        thirty = ""
        long_idx = header.index("longseq") if "longseq" in header else -1
        if long_idx >= 0 and long_idx < len(parts):
            ctx = parts[long_idx].strip().upper()
            # longSeq is 100bp; find the guide in it and extract 30-mer (4up+20+6down)
            pos = ctx.find(guide)
            if pos >= 4 and pos + 26 <= len(ctx):
                thirty = ctx[pos - 4: pos + 26]
        seen.add(guide)
        records.append({
            "sequence":   guide,
            "score":      score,
            "source":     "CRISPRscan2015",
            "thirty_mer": thirty,
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# DeepHF / Wang et al. 2019
# ─────────────────────────────────────────────────────────────────────────────

def load_deephf(raw_bytes: bytes) -> list[dict]:
    """
    Parse DeepHF wt_seq_data_array.pkl (Wang et al. 2019 Nat Commun).
    The pickle contains a dict with keys 'seq' (list of 30-mer strings) and
    'indel' (list of indel frequencies as percentages 0-100).
    Guide = 30mer[4:24], thirty_mer = full 30-mer.
    Score normalised to 0-1 by dividing by 100.
    """
    try:
        data = pickle.loads(raw_bytes)
    except Exception:
        try:
            import pickle as _pkl
            import io as _io
            data = _pkl.load(_io.BytesIO(raw_bytes), encoding="latin1")
        except Exception:
            return []

    # Determine structure — may be dict or numpy structured array
    seqs, indels = [], []
    if isinstance(data, dict):
        seqs   = data.get("seq", data.get("seqs", []))
        indels = data.get("indel", data.get("indels", data.get("label", [])))
    elif hasattr(data, "dtype") and hasattr(data.dtype, "names"):
        # Structured numpy array
        import numpy as _np
        names = data.dtype.names or []
        seq_col   = next((n for n in names if "seq" in n.lower()), None)
        indel_col = next((n for n in names if "indel" in n.lower() or "label" in n.lower()), None)
        if seq_col and indel_col:
            seqs   = data[seq_col]
            indels = data[indel_col]
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        seqs, indels = data[0], data[1]

    records = []
    seen    = set()
    for seq_raw, indel_val in zip(seqs, indels):
        try:
            seq30 = str(seq_raw).strip().upper()
            score = float(indel_val)
        except (TypeError, ValueError):
            continue
        # Normalise from percent (0-100) to 0-1
        score = score / 100.0 if score > 1.0 else score
        if not (0.0 <= score <= 1.0):
            continue
        # Extract 20bp guide from 30-mer (positions 4:24)
        if len(seq30) >= 24:
            guide = seq30[4:24]
        elif len(seq30) == 20:
            guide = seq30
            seq30 = ""
        else:
            continue
        if not _is_valid(guide):
            continue
        if guide in seen:
            continue
        seen.add(guide)
        thirty = seq30 if len(seq30) == 30 and _is_valid_30mer(seq30) else ""
        records.append({
            "sequence":   guide,
            "score":      score,
            "source":     "DeepHF2019",
            "thirty_mer": thirty,
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Kim et al. 2019 (DeepSpCas9)
# ─────────────────────────────────────────────────────────────────────────────

def load_kim2019(raw: str, holdout_frac: float = KIM_2019_HOLDOUT_FRAC,
                 seed: int = KIM_2019_HOLDOUT_SEED):
    """Parse Kim 2019 (DeepSpCas9) CSV.

    30-mer format: 4bp upstream + 20bp guide + 3bp PAM + 3bp downstream.
    Scores are rank-normalised to within-dataset percentile ranks (0–1) so
    they are on the same ordinal scale as Doench2016 gene-rank-normalised scores.
    Returns (train_records, holdout_records) split 80/20 by default.
    """
    import random as _random
    import numpy as _np
    _random.seed(seed)

    all_recs = []
    raw_scores = []
    seen = set()
    for row in csv.DictReader(io.StringIO(raw)):
        thirty = row.get("Target sequence", "").strip().upper()
        if len(thirty) != 30 or not (set(thirty) <= VALID_BASES):
            continue
        guide = thirty[4:24]
        pam   = thirty[24:27]
        if not re.fullmatch(r"[ACGT]GG", pam):
            continue
        try:
            sc = float(row["Indel frequency"])   # keep raw % for ranking
        except (ValueError, KeyError):
            continue
        if guide in seen:
            continue
        seen.add(guide)
        all_recs.append({"sequence": guide, "score": sc,
                         "source": "Kim2019", "thirty_mer": thirty})
        raw_scores.append(sc)

    # Rank-normalise to percentiles (matches Doench2016 gene-rank scale)
    from scipy.stats import rankdata as _rankdata
    ranks = _rankdata(raw_scores, method="average")
    n     = len(ranks)
    for rec, rank in zip(all_recs, ranks):
        rec["score"] = float(rank / n)

    _random.shuffle(all_recs)
    n_hold = max(1, int(len(all_recs) * holdout_frac))
    return all_recs[n_hold:], all_recs[:n_hold]   # (train, holdout)


# ─────────────────────────────────────────────────────────────────────────────
# Merge and save
# ─────────────────────────────────────────────────────────────────────────────

def download_and_merge(out_path: Path) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    holdout_path = DATA_DIR / "kim2019_holdout.csv"

    print("\n[1/3] Doench 2016 (Azimuth FC_plus_RES)...")
    try:
        records_2016 = load_doench2016(_fetch(DOENCH_2016_URL))
        print(f"  Parsed {len(records_2016)} valid guides (with 30-mer context).")
    except Exception as e:
        print(f"  WARN: {e}"); records_2016 = []

    print("\n[2/3] Doench 2014 (V1 supplementary)...")
    try:
        records_2014 = load_doench2014(_fetch(DOENCH_2014_URL))
        print(f"  Parsed {len(records_2014)} valid guides.")
    except Exception as e:
        print(f"  WARN: {e}"); records_2014 = []

    print("\n[3/3] Kim 2019 (DeepSpCas9, Science Advances)...")
    try:
        records_kim_train, records_kim_hold = load_kim2019(_fetch(KIM_2019_URL))
        print(f"  Parsed {len(records_kim_train)} training + "
              f"{len(records_kim_hold)} holdout guides.")
    except Exception as e:
        print(f"  WARN: {e} (Kim 2019 optional — continuing)")
        records_kim_train, records_kim_hold = [], []

    if not records_2016 and not records_2014:
        print("\nERROR: No data downloaded. Check network connectivity.")
        return 0

    # Merge training records — priority: Doench2016 > Doench2014 > Kim2019
    all_records: dict[str, dict] = {}
    for r in records_kim_train:
        all_records[r["sequence"]] = r
    for r in records_2014:
        all_records[r["sequence"]] = r
    for r in records_2016:
        all_records[r["sequence"]] = r

    merged = list(all_records.values())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sequence", "score", "source", "thirty_mer"]
        )
        writer.writeheader()
        writer.writerows(merged)

    # Save Kim 2019 holdout separately
    if records_kim_hold:
        with open(holdout_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["sequence", "score", "source", "thirty_mer"]
            )
            writer.writeheader()
            writer.writerows(records_kim_hold)
        print(f"  Holdout saved to {holdout_path} ({len(records_kim_hold)} guides)")

    print(f"\nSaved {len(merged)} unique training guides to {out_path}")
    print(f"  Doench 2016: {sum(1 for r in merged if r['source']=='Doench2016')}")
    print(f"  Doench 2014: {sum(1 for r in merged if r['source']=='Doench2014')}")
    print(f"  Kim 2019:    {sum(1 for r in merged if r['source']=='Kim2019')}")
    n_ctx = sum(1 for r in merged if r.get("thirty_mer"))
    print(f"  Guides with 30-mer context: {n_ctx}")
    return len(merged)


if __name__ == "__main__":
    out = DATA_DIR / "combined_training_data.csv"
    n   = download_and_merge(out)
    if n == 0:
        sys.exit(1)
    print("\nDone. Run: python train_model.py")
