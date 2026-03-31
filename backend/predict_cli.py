#!/usr/bin/env python3
"""
gRNA Predictor — command-line interface.

Runs the full XGBoost prediction pipeline from the terminal without
starting the FastAPI server.

Examples
--------
# Basic — top 5 NGG guides:
  python predict_cli.py --sequence ATCGATCG...

# Custom PAM, top 10, CSV output:
  python predict_cli.py -s ATCG... --pam NNGRRT --top-n 10 --output csv

# With proximity ranking toward base-edit site at position 150:
  python predict_cli.py -s ATCG... --target-position 150 --proximity-weight 0.5

# Read sequence from FASTA file:
  python predict_cli.py --fasta gene.fa --pam NGG
"""
import argparse
import math
import sys
import os

# Allow running from both backend/ and repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from app.services.sequence_parser import find_all_grnas
from app.models.ai_models       import predict_efficiency, get_model_info
from app.services.off_target     import specificity_score

_CAS12A_PAMS  = {"TTTV"}
_SIGMA        = 50.0
_MAX_CANDS    = 300


def _cut_site(c: dict, pam: str) -> int:
    pos    = c["position"]
    strand = c["strand"]
    offset = 18 if pam in _CAS12A_PAMS else (17 if strand == "+" else 3)
    return pos + offset + 1


def _proximity(distance: int) -> float:
    return math.exp(-(distance ** 2) / (2.0 * _SIGMA ** 2))


def _read_fasta(path: str) -> str:
    """Read a single-sequence FASTA file and return the sequence string."""
    lines = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            lines.append(line.upper())
    seq = "".join(lines)
    if not seq:
        raise ValueError(f"No sequence found in FASTA file: {path}")
    return seq


def _score_candidates(candidates, seq, pam, target_pos, prox_w):
    scored = predict_efficiency(candidates, full_sequence=seq)
    for c in scored:
        cs = _cut_site(c, pam)
        c["cut_site"] = cs
        spec = specificity_score(c["sequence"])
        c["off_target_score"] = round(spec, 3)
        eff_adj = c["score"] * spec
        if target_pos is not None:
            d = abs(cs - target_pos)
            c["distance_to_target"] = d
            c["combined_score"] = round(
                (1 - prox_w) * eff_adj + prox_w * _proximity(d), 4
            )
        else:
            c["distance_to_target"] = None
            c["combined_score"] = round(eff_adj, 4)
    return scored


def _print_table(ranked, target_pos):
    header = (
        f"{'#':>3} {'Sequence':<22} {'Str':>3} {'Pos':>6} "
        f"{'Effic':>6} {'Spec':>6} {'Score':>7} {'GC%':>5}"
    )
    if target_pos is not None:
        header += f"  {'Dist':>6}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for i, g in enumerate(ranked):
        row = (
            f"{i+1:>3} {g['sequence']:<22} {g['strand']:>3} {g['position']:>6} "
            f"{g['score']:>6.3f} {g['off_target_score']:>6.3f} "
            f"{g['combined_score']:>7.4f} {g['gc_content']*100:>5.1f}"
        )
        if target_pos is not None:
            row += f"  {g.get('distance_to_target', ''):>6}"
        print(row)


def _print_csv(ranked, target_pos):
    cols = [
        "rank", "sequence", "pam_sequence", "strand", "position",
        "efficiency_score", "specificity_score", "combined_score",
        "gc_content_pct", "cut_site",
    ]
    if target_pos is not None:
        cols.append("distance_to_target")
    print(",".join(cols))
    for i, g in enumerate(ranked):
        row = [
            str(i + 1),
            g["sequence"],
            g.get("pam_sequence", ""),
            g["strand"],
            str(g["position"]),
            str(g["score"]),
            str(g["off_target_score"]),
            str(g["combined_score"]),
            str(round(g["gc_content"] * 100, 1)),
            str(g["cut_site"]),
        ]
        if target_pos is not None:
            row.append(str(g.get("distance_to_target", "")))
        print(",".join(row))


def main():
    parser = argparse.ArgumentParser(
        description="gRNA Predictor — CRISPR guide RNA design & XGBoost efficiency scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    seq_group = parser.add_mutually_exclusive_group(required=True)
    seq_group.add_argument("--sequence", "-s",
                           help="Input DNA sequence (5'→3', IUPAC bases, any length)")
    seq_group.add_argument("--fasta", "-f",
                           help="Path to single-sequence FASTA file")

    parser.add_argument("--pam", "-p",
                        default="NGG",
                        choices=["NGG", "NAG", "NNGRRT", "TTTV"],
                        help="PAM type (default: NGG / SpCas9)")
    parser.add_argument("--top-n", "-n",
                        type=int, default=5,
                        help="Number of top guides to return (default: 5)")
    parser.add_argument("--target-position", "-t",
                        type=int, default=None,
                        help="Genomic position of desired edit site (1-indexed). "
                             "Enables proximity-weighted ranking.")
    parser.add_argument("--proximity-weight", "-w",
                        type=float, default=0.4,
                        help="Weight for proximity score [0–1] (default: 0.4). "
                             "Ignored if --target-position is not set.")
    parser.add_argument("--output", "-o",
                        choices=["table", "csv"],
                        default="table",
                        help="Output format (default: table)")

    args = parser.parse_args()

    # --- Load sequence ---
    if args.fasta:
        try:
            seq = _read_fasta(args.fasta)
        except (OSError, ValueError) as e:
            print(f"Error reading FASTA: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        seq = args.sequence.upper().replace(" ", "").replace("\n", "")

    valid_bases = set("ACGTNRYSWKMBDHV")
    bad = set(seq) - valid_bases
    if bad:
        print(f"Warning: non-IUPAC characters found and will be treated as N: {bad}",
              file=sys.stderr)

    # --- Find candidates ---
    candidates = find_all_grnas(seq, pam=args.pam)
    if not candidates:
        print(f"No valid PAM ({args.pam}) sites found in the provided sequence.",
              file=sys.stderr)
        sys.exit(1)

    if len(candidates) > _MAX_CANDS:
        gc_ok = [c for c in candidates if 0.35 <= c["gc_content"] <= 0.75]
        candidates = (gc_ok or candidates)[:_MAX_CANDS]

    # --- Score ---
    scored = _score_candidates(candidates, seq, args.pam,
                               args.target_position, args.proximity_weight)
    ranked = sorted(scored, key=lambda x: x["combined_score"], reverse=True)[:args.top_n]

    # --- Output ---
    if args.output == "table":
        print(f"\n{get_model_info()}\n")
        print(f"Candidates found : {len(scored)}   |   Showing top {len(ranked)}")
        print(f"PAM              : {args.pam}")
        print(f"Sequence length  : {len(seq)} bp")
        if args.target_position is not None:
            print(f"Target position  : {args.target_position} bp  "
                  f"(proximity weight = {args.proximity_weight})")
        print()
        _print_table(ranked, args.target_position)
        print()
        print("Columns: Effic=XGBoost efficiency  Spec=off-target specificity  "
              "Score=combined (eff×spec + proximity)")
    else:
        _print_csv(ranked, args.target_position)


if __name__ == "__main__":
    main()
