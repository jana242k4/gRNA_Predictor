"""
Create a LinkedIn-ready 2-page benchmark PDF for the gRNA Predictor tool.

Usage:
    cd backend && source ../.venv/Scripts/activate
    python create_benchmark_pdf.py
    # → benchmark_results/gRNA_Predictor_Benchmarks.pdf
"""
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).parent / "benchmark_results"
PDF_OUT = OUT_DIR / "gRNA_Predictor_Benchmarks.pdf"

PLT_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "text.color":       "#e6edf3",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "font.family":      "sans-serif",
    "figure.dpi":       150,
}

ACCENT  = "#58a6ff"   # blue
GREEN   = "#3fb950"
ORANGE  = "#d29922"
RED     = "#f85149"
MUTED   = "#8b949e"


def load_png(path):
    """Load a PNG for embedding in matplotlib."""
    img = plt.imread(str(path))
    return img


def page1(pdf):
    """Page 1: Title card + benchmark table + key numbers."""
    fig = plt.figure(figsize=(11, 8.5))
    plt.rcParams.update(PLT_STYLE)
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── Title block ─────────────────────────────────────────────────────
    ax.text(0.5, 0.95, "gRNA Predictor",
            ha="center", va="top", fontsize=34, fontweight="bold",
            color=ACCENT, transform=ax.transAxes)
    ax.text(0.5, 0.885,
            "AI-Powered CRISPR Guide RNA Designer with Gaussian Proximity Ranking",
            ha="center", va="top", fontsize=13, color=MUTED, transform=ax.transAxes)

    # Divider
    ax.axhline(0.86, xmin=0.05, xmax=0.95, color="#30363d", linewidth=1)

    # ── Key metric boxes ────────────────────────────────────────────────
    metrics = [
        ("r = 0.640", "Kim 2019 independent\n(n=1,828 novel guides)", GREEN),
        ("r = 0.537", "Doench held-out\n(20%, n=938)", ACCENT),
        ("450-dim", "Feature vector\n(XGBoost, 500 trees)", ORANGE),
        ("27 / 27", "Unit tests\npassing", GREEN),
    ]
    for j, (val, label, color) in enumerate(metrics):
        x = 0.1 + j * 0.22
        box = FancyBboxPatch((x - 0.08, 0.70), 0.17, 0.12,
                             boxstyle="round,pad=0.01",
                             facecolor="#161b22", edgecolor=color,
                             linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x + 0.005, 0.775, val, ha="center", va="center",
                fontsize=16, fontweight="bold", color=color,
                transform=ax.transAxes)
        ax.text(x + 0.005, 0.718, label, ha="center", va="center",
                fontsize=7.5, color=MUTED, transform=ax.transAxes)

    # ── Benchmark table ─────────────────────────────────────────────────
    ax.text(0.06, 0.67, "Independent Validation Results", fontsize=12,
            fontweight="bold", color="#e6edf3", transform=ax.transAxes)

    headers = ["Dataset", "n", "Spearman r", "Notes"]
    rows = [
        ["Kim 2019 novel-only ★", "1,828", "0.640", "0% Doench overlap — primary benchmark"],
        ["Doench 2016 held-out", "938", "0.537", "Our 20% held-out (honest)"],
        ["Azimuth (same split)", "938", "0.654", "Asymmetric: Azimuth trained on 100% Doench"],
        ["Chari 2015 (293T)", "10", "0.770", "CI ≈ ±0.74 — treat as supplementary"],
        ["Chari 2015 (K562)", "10", "0.758", "CI ≈ ±0.74"],
        ["Xu 2015 (human)", "35", "0.424", ""],
        ["CRISPRscan (zebrafish)", "1,020", "0.081", "Expected low — trained on human"],
    ]
    col_x  = [0.06, 0.38, 0.49, 0.58]
    row_h  = 0.048
    header_y = 0.630
    col_widths = [0.31, 0.10, 0.08, 0.41]

    # Header row
    header_box = FancyBboxPatch((0.05, header_y - 0.005), 0.90, row_h + 0.005,
                                boxstyle="round,pad=0.002",
                                facecolor="#21262d", edgecolor="#30363d",
                                linewidth=0.8, transform=ax.transAxes)
    ax.add_patch(header_box)
    for col, h in zip(col_x, headers):
        ax.text(col + 0.005, header_y + row_h * 0.4, h, fontsize=9,
                fontweight="bold", color=ACCENT, transform=ax.transAxes)

    for r_idx, row in enumerate(rows):
        y = header_y - (r_idx + 1) * row_h
        bg_color = "#0d1117" if r_idx % 2 == 0 else "#161b22"
        row_bg = FancyBboxPatch((0.05, y - 0.003), 0.90, row_h,
                                boxstyle="square,pad=0",
                                facecolor=bg_color, edgecolor="none",
                                transform=ax.transAxes)
        ax.add_patch(row_bg)
        for col, cell in zip(col_x, row):
            # Highlight the primary benchmark row
            txt_color = GREEN if "★" in row[0] and col == col_x[0] else "#e6edf3"
            if col == col_x[2]:  # r value
                val = float(cell) if cell else 0
                txt_color = GREEN if val >= 0.60 else (ACCENT if val >= 0.50 else MUTED)
            ax.text(col + 0.005, y + row_h * 0.3, cell,
                    fontsize=8, color=txt_color, transform=ax.transAxes)

    # ── Novelty blurb ────────────────────────────────────────────────────
    y_blurb = header_y - (len(rows) + 1) * row_h - 0.01
    ax.axhline(y_blurb + 0.035, xmin=0.05, xmax=0.95,
               color="#30363d", linewidth=0.8)
    ax.text(0.06, y_blurb + 0.015,
            "Novel contribution: Gaussian proximity-weighted ranking\n"
            "combined_score = (1−w) × efficiency + w × exp(−d²/2σ²)   "
            "where σ = 50 bp, w ∈ [0,1] (user-tunable).\n"
            "No existing tool (Azimuth, CRISPRscan, CRISPOR) automates this efficiency-proximity tradeoff.",
            fontsize=8.5, color=MUTED, transform=ax.transAxes,
            linespacing=1.6)

    # Footer
    ax.text(0.5, 0.02, "github.com/YOUR_USERNAME/gRNA_Predictor  ·  In-browser ONNX demo available",
            ha="center", fontsize=8, color="#484f58", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def page2(pdf):
    """Page 2: 3 publication figures side by side."""
    plt.rcParams.update(PLT_STYLE)

    fig1_path = OUT_DIR / "fig1_performance_scatter.png"
    fig4_path = OUT_DIR / "fig4_ablation_bar.png"
    fig5_path = OUT_DIR / "fig5_benchmark_heatmap.png"

    fig = plt.figure(figsize=(17, 6.5))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.08,
                           left=0.01, right=0.99, top=0.88, bottom=0.04)

    titles = [
        "Fig 1 — Predicted vs Experimental Efficiency",
        "Fig 4 — Feature Ablation (SHAP)",
        "Fig 5 — Benchmark Comparison Heatmap",
    ]
    paths = [fig1_path, fig4_path, fig5_path]

    for i, (path, title) in enumerate(zip(paths, titles)):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("#0d1117")
        if path.exists():
            img = load_png(path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"(Figure not found:\n{path.name})",
                    ha="center", va="center", color=RED,
                    transform=ax.transAxes, fontsize=9)
        ax.axis("off")
        ax.set_title(title, fontsize=9, color=MUTED, pad=4)

    fig.suptitle(
        "gRNA Predictor — Publication Figures",
        fontsize=13, fontweight="bold", color=ACCENT, y=0.97
    )

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLT_STYLE)

    with PdfPages(PDF_OUT) as pdf:
        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "gRNA Predictor — Benchmark Results"
        d["Author"]  = "gRNA Predictor"
        d["Subject"] = "CRISPR guide RNA efficiency prediction — XGBoost 450-dim"
        d["Keywords"]= "CRISPR gRNA Cas9 XGBoost ONNX benchmarks Doench Kim2019"

        page1(pdf)
        page2(pdf)

    size_kb = PDF_OUT.stat().st_size / 1024
    print(f"Saved: {PDF_OUT}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
