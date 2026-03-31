"""
gRNA Predictor — Biology & Code Documentation PDF
===================================================
Generates a comprehensive reference PDF covering:
  - CRISPR biology and PAM recognition
  - Guide RNA design rules
  - Feature engineering (all 450 dimensions)
  - Thermodynamics (SantaLucia 1998)
  - Off-target specificity heuristic
  - ML model and training data
  - Multi-objective scoring formula
  - Benchmark results
  - How to use the tool

Run:
    cd backend
    source ../.venv/Scripts/activate
    python create_documentation_pdf.py

Output:
    backend/documentation/gRNA_Predictor_Biology_Guide.pdf
"""

from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT_DIR = Path(__file__).parent / "documentation"
OUT_DIR.mkdir(exist_ok=True)
PDF_PATH = OUT_DIR / "gRNA_Predictor_Biology_Guide.pdf"

# ── Colour palette ──────────────────────────────────────────────────────────
TEAL   = "#00838f"
DARK   = "#0e1117"
LIGHT  = "#e8f4f8"
GREY   = "#546e7a"
ORANGE = "#ef6c00"
GREEN  = "#2e7d32"
RED    = "#b71c1c"
WHITE  = "white"

def new_page(fig_size=(11, 8.5)):
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.axis('off')
    return fig, ax

def header(ax, title, subtitle="", y=0.97):
    ax.text(0.5, y, title, transform=ax.transAxes,
            fontsize=22, fontweight='bold', color=TEAL,
            ha='center', va='top', fontfamily='monospace')
    if subtitle:
        ax.text(0.5, y - 0.055, subtitle, transform=ax.transAxes,
                fontsize=11, color=GREY, ha='center', va='top')

def section_box(ax, x, y, w, h, title, lines, title_color=TEAL, font_size=9.2):
    """Draw a rounded-corner box with a bold title and bulleted lines."""
    rect = mpatches.FancyBboxPatch((x, y - h), w, h,
        boxstyle="round,pad=0.01", linewidth=1.2,
        edgecolor=TEAL, facecolor="#12252a", transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x + 0.012, y - 0.016, title, transform=ax.transAxes,
            fontsize=10.5, fontweight='bold', color=title_color, va='top')
    for i, line in enumerate(lines):
        ax.text(x + 0.018, y - 0.040 - i * 0.018, f"• {line}",
                transform=ax.transAxes, fontsize=font_size,
                color=WHITE, va='top', wrap=True)

def divider(ax, y, color=TEAL):
    ax.plot([0.03, 0.97], [y, y], color=color, linewidth=0.6,
            transform=ax.transAxes, clip_on=False)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Title & Tool Overview
# ────────────────────────────────────────────────────────────────────────────
def page_title(pdf):
    fig, ax = new_page()

    ax.text(0.5, 0.90, "gRNA Predictor", transform=ax.transAxes,
            fontsize=36, fontweight='bold', color=TEAL,
            ha='center', va='top', fontfamily='monospace')
    ax.text(0.5, 0.82, "AI-powered CRISPR Guide RNA Design",
            transform=ax.transAxes, fontsize=16, color=LIGHT,
            ha='center', va='top')
    ax.text(0.5, 0.76, "Biology & Implementation Reference",
            transform=ax.transAxes, fontsize=13, color=GREY,
            ha='center', va='top')

    divider(ax, 0.72)

    # Quick stats row
    stats = [
        ("450", "feature dimensions"),
        ("4,692", "training guides"),
        ("r = 0.640", "Kim2019 independent\nvalidation"),
        ("r = 0.537", "held-out Doench\nbenchmark"),
        ("4 PAMs", "NGG / NAG /\nNNGRRT / TTTV"),
    ]
    for i, (val, lbl) in enumerate(stats):
        cx = 0.10 + i * 0.18
        ax.text(cx, 0.66, val, transform=ax.transAxes,
                fontsize=17, fontweight='bold', color=TEAL, ha='center')
        ax.text(cx, 0.60, lbl, transform=ax.transAxes,
                fontsize=8, color=GREY, ha='center')

    divider(ax, 0.57)

    overview = [
        ("What it does",
         "Predicts CRISPR guide RNA editing efficiency for any DNA target sequence.\n"
         "Combines ML efficiency scoring, off-target specificity estimation, and optional\n"
         "proximity-to-target ranking into a single combined score."),
        ("Architecture",
         "Python/FastAPI backend (local) OR pure-JS in-browser inference (GitHub Pages).\n"
         "Frontend: React 18 + Material UI. ML: XGBoost trained on Doench 2016 + 2014.\n"
         "Browser: 450-dim features + pure-JS XGBoost tree traversal (~3 ms, no WASM)."),
        ("Supported nucleases",
         "SpCas9 (NGG PAM) — most common, human cell optimised.\n"
         "SpCas9-NAG variant — reduced efficiency, alternative PAM sites.\n"
         "SaCas9 (NNGRRT PAM) — smaller nuclease, useful for AAV delivery.\n"
         "Cas12a/Cpf1 (TTTV 5′-PAM) — staggered cuts, T-rich PAM preference."),
    ]
    y0 = 0.54
    for title, body in overview:
        ax.text(0.05, y0, title, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=TEAL, va='top')
        for j, line in enumerate(body.split('\n')):
            ax.text(0.07, y0 - 0.028 - j * 0.022, line,
                    transform=ax.transAxes, fontsize=9, color=WHITE, va='top')
        y0 -= 0.12

    ax.text(0.5, 0.04, "github.com/jana242k4/gRNA_Predictor  |  jana242k4.github.io/gRNA_Predictor",
            transform=ax.transAxes, fontsize=8, color=GREY, ha='center')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 2 — CRISPR Biology
# ────────────────────────────────────────────────────────────────────────────
def page_crispr_biology(pdf):
    fig, ax = new_page()
    header(ax, "CRISPR-Cas9 Biology", "How the system works at the molecular level")
    divider(ax, 0.90)

    # Schematic: DNA strand
    ax.text(0.5, 0.85, "Genomic DNA target layout (SpCas9, NGG PAM)",
            transform=ax.transAxes, fontsize=10, color=GREY, ha='center')

    # Draw DNA schematic
    schema_y = 0.76
    elements = [
        (0.04, 0.10, GREY,   "5′ flanking\n(upstream context)"),
        (0.15, 0.16, "#1565c0","Upstream\n4 bp"),
        (0.32, 0.28, TEAL,   "20 bp Guide RNA\n(protospacer)"),
        (0.61, 0.07, ORANGE, "PAM\nNGG"),
        (0.69, 0.10, GREY,   "3′ flanking\n(downstream context)"),
    ]
    for x, w, color, label in elements:
        rect = mpatches.FancyBboxPatch(
            (x, schema_y - 0.025), w, 0.05,
            boxstyle="round,pad=0.005", facecolor=color, edgecolor=WHITE,
            linewidth=0.8, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w/2, schema_y - 0.06, label,
                transform=ax.transAxes, fontsize=7.5, color=WHITE,
                ha='center', va='top')

    # Cut site arrow
    ax.annotate('', xy=(0.58, schema_y + 0.04), xytext=(0.58, schema_y + 0.08),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
    ax.text(0.58, schema_y + 0.10, "Cut\n(pos 17)", transform=ax.transAxes,
            fontsize=7.5, color=RED, ha='center', va='bottom')

    divider(ax, 0.62)

    boxes = [
        (0.03, 0.60, 0.44, 0.23, "PAM Recognition — Why It Matters", [
            "PAM = Protospacer Adjacent Motif (3′ of target for Cas9)",
            "Cas9 scans DNA by binding & checking for PAM sequence",
            "Only unwinds DNA and checks guide match if PAM is found",
            "NGG occurs ~every 8 bp, giving ~1 guide per ~8 bp on each strand",
            "NAG is tolerated with ~5× lower efficiency than NGG",
            "SaCas9 NNGRRT = N-N-G-[AG]-[AG]-T (6 bp, more specific)",
            "Cas12a TTTV: PAM is 5′ of guide, T-rich, RNase activity too",
        ]),
        (0.52, 0.60, 0.44, 0.23, "The Seed Region — Most Critical 12 bp", [
            "Seed = PAM-proximal 12 bp of the 20 bp guide (positions 9–20)",
            "Mismatches here are NOT tolerated — aborts cleavage",
            "Mismatches in PAM-distal region (pos 1–8) are partially tolerated",
            "High AT content in seed → more off-target risk (weaker binding)",
            "Optimal seed GC: 40–60% for balance of efficiency & specificity",
            "This is why the model has BOTH full-guide and seed-only GC features",
            "Seed region Tm is computed separately as a dedicated feature",
        ]),
        (0.03, 0.33, 0.44, 0.23, "The 30-mer Context Window", [
            "Not just the 20 bp guide — flanking sequence matters too",
            "4 bp upstream (−4 to −1) influence unwinding efficiency",
            "6 bp downstream (+1 to +6 after PAM) affect R-loop stability",
            "Full window: [4 up][20 guide][6 down] = 30 bp total",
            "Used for: upstream one-hot, downstream one-hot, full-window Tm",
            "Microhomology score uses cut-site flanks from the 30-mer",
            "Doench 2016 dataset provides 30-mer context for each guide",
        ]),
        (0.52, 0.33, 0.44, 0.23, "Cut Site Location", [
            "SpCas9 cuts between positions 17 and 18 of the guide (3 bp upstream of PAM)",
            "Cas12a cuts ~18 bp into the guide, creating a staggered end",
            "Cut site = guide_position + 17 + 1 (1-indexed)",
            "Important for HDR (homology-directed repair) template design",
            "Proximity ranking uses Euclidean distance |cut_site − target_pos|",
            "Gaussian weight: exp(−d² / 2σ²), σ = 50 bp — guides within 50 bp preferred",
            "Closer guides get higher combined score when target position is given",
        ]),
    ]
    for args in boxes:
        section_box(ax, *args)

    ax.text(0.5, 0.03, "Key reference: Doench et al. (2016) Nat Biotechnol 34:184–191",
            transform=ax.transAxes, fontsize=8, color=GREY, ha='center')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Guide RNA Design Rules
# ────────────────────────────────────────────────────────────────────────────
def page_design_rules(pdf):
    fig, ax = new_page()
    header(ax, "Guide RNA Design Rules", "Empirical rules captured in the 450-dim feature vector")
    divider(ax, 0.90)

    rules = [
        ("GC Content 40–60%",
         GREEN, 0.03, 0.86, 0.44, 0.13, [
            "GC content drives hybridisation stability of guide:target duplex",
            "< 25% GC: weak binding, poor editing",
            "> 75% GC: secondary structure, off-target risk",
            "Optimal 40–60% balances stability and specificity",
            "Feature indices: [80] full guide GC,  [98] seed GC",
         ]),
        ("Avoid Poly-T Tracts",
         RED, 0.52, 0.86, 0.44, 0.13, [
            "Four or more consecutive T's = TTTT terminates RNA Pol III",
            "U6 promoter (used to express gRNA) is an RNA Pol III promoter",
            "Poly-T in the guide causes premature transcription termination",
            "Feature index: [99] poly-T flag (1.0 if TTTT is present)",
            "Guides with TTTT are expected to have near-zero activity",
         ]),
        ("Seed Region AT Content",
         ORANGE, 0.03, 0.69, 0.44, 0.13, [
            "Seed = PAM-proximal 12 bp (positions 9–20 of the 20 bp guide)",
            "High AT in seed reduces specificity — more off-target editing",
            "Each A or T in seed contributes a specificity penalty of 0.28/12",
            "Off-target score = 1 − seedPenalty − gcPenalty − ... (6 components)",
            "Feature: seed GC [98], also captured in positional one-hot [0:80]",
         ]),
        ("GC Clamp at 3′ End",
         TEAL, 0.52, 0.69, 0.44, 0.13, [
            "Last 4 bp of guide (positions 17–20, PAM-proximal) should be GC-rich",
            "GC at 3′ end strengthens critical seed-region binding to target",
            "> 2 consecutive 3′-end G/C residues add extra off-target risk (GC run penalty)",
            "Optimal: 1–2 G/C in last 4 bp",
            "Feature index: [444] GC clamp (fraction of last 4 bp that are G/C)",
         ]),
        ("RNA Secondary Structure",
         "#7b1fa2", 0.03, 0.52, 0.44, 0.13, [
            "Guide RNA can fold back on itself if self-complementary",
            "A hairpin in the guide reduces accessibility for Cas9 loading",
            "Hairpin proxy: longest palindromic stem ≥ 4 bp in the 20 bp guide",
            "Score = min(1.0, maxStemLength / 10.0) — longer stem = worse",
            "Feature index: [445] RNA hairpin proxy",
         ]),
        ("Microhomology at Cut Site",
         "#37474f", 0.52, 0.52, 0.44, 0.13, [
            "Microhomology = short identical sequences flanking the cut site",
            "Promotes NHEJ (error-prone repair) through microhomology-mediated end joining",
            "Used to predict repair outcome: frameshift vs. in-frame indel",
            "Computed from 30-mer: left 6 bp vs right 6 bp around cut position 21",
            "Feature index: [446] microhomology score (0–1, longer match = higher)",
         ]),
    ]
    for title, color, x, y, w, h, lines in rules:
        section_box(ax, x, y, w, h, title, lines, title_color=color)

    divider(ax, 0.37)

    # Quick reference table
    ax.text(0.5, 0.34, "Quick Reference — Feature ↔ Biological Rule",
            transform=ax.transAxes, fontsize=11, color=TEAL, fontweight='bold', ha='center')

    table_data = [
        ["Feature", "Index", "Optimal Value", "Why It Matters"],
        ["GC content", "[80]", "0.40 – 0.60", "Duplex stability"],
        ["Seed GC", "[98]", "0.40 – 0.60", "Specificity"],
        ["Poly-T flag", "[99]", "0.0 (no TTTT)", "Transcription termination"],
        ["GC clamp", "[444]", "0.25 – 0.50", "3′ binding strength"],
        ["Hairpin proxy", "[445]", "< 0.3", "Guide accessibility"],
        ["Microhomology", "[446]", "any", "Repair outcome prediction"],
        ["Full-guide Tm", "[81]", "0.4 – 0.7 (norm)", "Thermodynamic stability"],
        ["Seed Tm", "[448]", "0.4 – 0.7 (norm)", "Critical region stability"],
    ]
    col_widths = [0.28, 0.10, 0.22, 0.28]
    col_x = [0.04, 0.32, 0.42, 0.65]
    for row_i, row in enumerate(table_data):
        y_row = 0.30 - row_i * 0.028
        bg = TEAL if row_i == 0 else ("#1a3040" if row_i % 2 == 0 else "#12252a")
        rect = mpatches.FancyBboxPatch((0.03, y_row - 0.014), 0.94, 0.026,
            boxstyle="square,pad=0", facecolor=bg, transform=ax.transAxes)
        ax.add_patch(rect)
        for ci, (cell, cx) in enumerate(zip(row, col_x)):
            ax.text(cx + 0.005, y_row - 0.001, cell, transform=ax.transAxes,
                    fontsize=8.5 if row_i > 0 else 9,
                    color=WHITE if row_i == 0 else LIGHT,
                    fontweight='bold' if row_i == 0 else 'normal', va='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Feature Engineering Overview (450 dims)
# ────────────────────────────────────────────────────────────────────────────
def page_features(pdf):
    fig, ax = new_page()
    header(ax, "Feature Engineering — 450 Dimensions",
           "Every feature in the XGBoost model, with biological rationale")
    divider(ax, 0.90)

    features = [
        ("[0:80]",  80,  TEAL,   "Positional One-Hot (20 pos × 4 bases)",
         "Encodes identity of each base at each position. Captures position-specific\n"
         "preferences (e.g., Cas9 favors G at position 20, disfavors T at position 1)."),
        ("[80]",     1,  GREEN,  "GC Content",
         "Fraction of G+C in the 20 bp guide. Single most important bulk property.\n"
         "Drives hybridisation stability."),
        ("[81]",     1,  GREEN,  "Normalised Melting Temperature (SantaLucia 1998)",
         "Full nearest-neighbor Tm of the guide, normalised to [0,1] over [40°C, 80°C].\n"
         "Captures thermodynamic stability more accurately than GC% alone."),
        ("[82:98]", 16,  ORANGE, "Dinucleotide Frequencies (16 pairs AA…TT)",
         "Counts of each dinucleotide (AA, AC, …, TT) normalised by n−1.\n"
         "Encodes short-range sequence composition; some dinucs correlate with efficiency."),
        ("[98]",     1,  ORANGE, "Seed Region GC (last 12 bp)",
         "GC fraction of the PAM-proximal 12 bp seed. Seed binding is all-or-nothing\n"
         "for Cas9; seed GC is a stronger efficiency predictor than overall GC."),
        ("[99]",     1,  RED,    "Poly-T Flag",
         "1.0 if the guide contains TTTT, else 0.0.\n"
         "Binary signal for RNA Pol III termination risk."),
        ("[100:404]",304,"#7b1fa2","Position-Specific Dinucleotide One-Hot (19 × 16)",
         "For each of the 19 consecutive dinucleotides in the guide, a 16-dim one-hot\n"
         "vector. 304 features total. Captures context-dependent dinucleotide effects\n"
         "at specific positions — the richest feature block in the model."),
        ("[404:420]",16, "#1565c0","Upstream Context One-Hot (4 bp × 4 = 16)",
         "One-hot encoding of the 4 bp immediately upstream of the guide (from 30-mer).\n"
         "Upstream context affects R-loop initiation and PAM scanning efficiency."),
        ("[420:444]",24, "#1565c0","Downstream Context One-Hot (6 bp × 4 = 24)",
         "One-hot encoding of the 6 bp immediately downstream of the PAM (from 30-mer).\n"
         "Downstream context affects post-cleavage stability and repair outcomes."),
        ("[444]",    1,  TEAL,   "GC Clamp (last 4 bp)",
         "Fraction of G+C in the final 4 bp of the guide (positions 17–20).\n"
         "Strong predictor: 3′ GC content anchors the seed region to target DNA."),
        ("[445]",    1,  "#ef9a00","RNA Hairpin Proxy",
         "Length of the longest palindromic stem (≥4 bp) in the guide, scaled to [0,1].\n"
         "Hairpin = guide self-folds → lower Cas9 loading → lower efficiency."),
        ("[446]",    1,  "#ef9a00","Microhomology Score (from 30-mer)",
         "Longest identical sequence flanking the cut site (up to 6 bp each side),\n"
         "from the 30-mer context. Predicts NHEJ repair pathway preference."),
        ("[447]",    1,  "#00796b","Tm PAM-Distal Segment (guide[0:8])",
         "Normalised Tm of the first 8 bp (positions 1–8, PAM-distal).\n"
         "Captures stability of the mismatch-tolerant distal region separately."),
        ("[448]",    1,  "#00796b","Tm Seed Segment (guide[12:20])",
         "Normalised Tm of the last 8 bp = seed region (positions 13–20).\n"
         "Critical: seed stability is the main on-target efficiency driver."),
        ("[449]",    1,  "#00796b","Tm Full 30-mer Context",
         "Normalised Tm of the entire 30-mer window (4 up + 20 guide + 6 down).\n"
         "Captures the stability of the full target:guide duplex in context."),
    ]

    total = sum(f[1] for f in features)
    assert total == 450, f"Feature count mismatch: {total}"

    y_cur = 0.87
    for idx, n, color, name, desc in features:
        # bar proportional to feature count
        bar_w = 0.06 * (n / 304)  # normalise to widest block
        bar_w = max(0.005, bar_w)
        rect = mpatches.FancyBboxPatch((0.03, y_cur - 0.012), bar_w, 0.016,
            boxstyle="round,pad=0.002", facecolor=color, alpha=0.85,
            transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.03 + bar_w + 0.008, y_cur - 0.003,
                f"{idx}  ({n} dim)  {name}",
                transform=ax.transAxes, fontsize=8.8, color=WHITE,
                fontweight='bold', va='center')
        for di, dline in enumerate(desc.split('\n')):
            ax.text(0.10, y_cur - 0.022 - di * 0.015, dline,
                    transform=ax.transAxes, fontsize=7.5, color=GREY, va='top')
        y_cur -= 0.055 if '\n' in desc else 0.048

    # Total bar
    ax.text(0.5, 0.015, f"Total: {total} features  |  Vector shape per guide: Float32[450]",
            transform=ax.transAxes, fontsize=9, color=TEAL, ha='center', fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Thermodynamics & SantaLucia 1998
# ────────────────────────────────────────────────────────────────────────────
def page_thermodynamics(pdf):
    fig, ax = new_page()
    header(ax, "Thermodynamics — SantaLucia 1998",
           "Nearest-neighbor model used for Tm calculation (features [81], [447], [448], [449])")
    divider(ax, 0.90)

    # Formula
    ax.text(0.5, 0.86,
            r"$T_m = \dfrac{\Delta H}{\Delta S + R \cdot \ln(C_T/4)} - 273.15°C$",
            transform=ax.transAxes, fontsize=15, color=TEAL, ha='center',
            usetex=False)
    ax.text(0.5, 0.80,
            "  ΔH = sum of nearest-neighbor enthalpies (kcal/mol)   "
            "  ΔS = sum of nearest-neighbor entropies (cal/mol·K)   "
            "  R = 1.987 cal/mol·K   C_T = 250 nM",
            transform=ax.transAxes, fontsize=8, color=GREY, ha='center')

    divider(ax, 0.76)

    # NN table
    ax.text(0.5, 0.73, "SantaLucia 1998 — 16 Nearest-Neighbor Parameters",
            transform=ax.transAxes, fontsize=11, color=WHITE, ha='center', fontweight='bold')

    NN = {
        'AA': (-7.9, -22.2), 'AT': (-7.2, -20.4), 'TA': (-7.2, -21.3), 'CA': (-8.5, -22.7),
        'GT': (-8.4, -22.4), 'CT': (-7.8, -21.0), 'GA': (-8.2, -22.2), 'CG': (-10.6, -27.2),
        'GC': (-9.8, -24.4), 'GG': (-8.0, -19.9), 'AC': (-7.8, -21.0), 'TC': (-7.9, -22.2),
        'AG': (-8.2, -22.2), 'TG': (-8.5, -22.7), 'TT': (-7.9, -22.2), 'CC': (-8.0, -19.9),
    }
    cols = 4
    dinucs = list(NN.keys())
    dh_vals = [NN[d][0] for d in dinucs]
    dh_min, dh_max = min(dh_vals), max(dh_vals)

    for i, (di, (dH, dS)) in enumerate(NN.items()):
        row, col = divmod(i, cols)
        x = 0.04 + col * 0.235
        y = 0.69 - row * 0.065
        # colour by ΔH (more negative = more stable = deeper teal)
        frac = (dH - dh_min) / (dh_max - dh_min + 1e-9)
        cell_color = (0.04 + 0.08*frac, 0.36 + 0.24*frac, 0.40 + 0.20*frac)
        rect = mpatches.FancyBboxPatch((x, y - 0.055), 0.21, 0.058,
            boxstyle="round,pad=0.005", facecolor=cell_color,
            edgecolor=TEAL, linewidth=0.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.015, y - 0.014, di, transform=ax.transAxes,
                fontsize=12, color=WHITE, fontweight='bold', va='top')
        ax.text(x + 0.065, y - 0.015, f"ΔH = {dH:.1f} kcal/mol",
                transform=ax.transAxes, fontsize=7.5, color=LIGHT, va='top')
        ax.text(x + 0.065, y - 0.030, f"ΔS = {dS:.1f} cal/mol·K",
                transform=ax.transAxes, fontsize=7.5, color=LIGHT, va='top')

    divider(ax, 0.13)
    notes = [
        "Terminal corrections: A/T terminal pair → ΔH += 2.3 kcal/mol, ΔS += 4.1 cal/mol·K",
        "G/C terminal pair → ΔH += 0.1 kcal/mol, ΔS −= 2.8 cal/mol·K",
        "Tm normalisation: Tm_norm = clip((Tm − 40) / (80 − 40), 0, 1)",
        "Three Tm features computed: PAM-distal guide[0:8], seed guide[12:20], full 30-mer",
        "CG dinucleotide has highest stability (ΔH = −10.6) — most negative = strongest bond",
    ]
    for i, note in enumerate(notes):
        ax.text(0.04, 0.10 - i * 0.019, f"• {note}",
                transform=ax.transAxes, fontsize=8, color=GREY, va='top')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 6 — Off-Target Specificity Heuristic
# ────────────────────────────────────────────────────────────────────────────
def page_offtarget(pdf):
    fig, ax = new_page()
    header(ax, "Off-Target Specificity Heuristic",
           "6-component penalty model: specificity_score ∈ [0, 1]")
    divider(ax, 0.90)

    ax.text(0.5, 0.86,
            "specificity = max(0, min(1,  1 − seedPen − gcPen − gcRunPen − hpPen − hairpinPen − gqPen))",
            transform=ax.transAxes, fontsize=10, color=TEAL, ha='center', fontfamily='monospace')

    components = [
        ("1. Seed AT Content Penalty", ORANGE, 0.03, 0.80, 0.43, 0.16, [
            "seed = last 12 bp of guide (PAM-proximal, positions 9–20)",
            "seedAT = count(A or T in seed) / 12",
            "seedPen = seedAT × 0.28",
            "Rationale: high AT in seed → weaker binding → more off-target tolerance",
            "Range: 0.0 (all GC seed) to 0.28 (all AT seed)",
            "Biggest single penalty component",
        ]),
        ("2. Extreme GC Penalty", TEAL, 0.52, 0.80, 0.43, 0.16, [
            "gc = count(G or C in 20 bp guide) / 20",
            "gcPen = 0.20 if gc < 0.25 or gc > 0.75",
            "gcPen = 0.10 if gc < 0.35 or gc > 0.65",
            "gcPen = 0.05 if gc < 0.40 or gc > 0.60",
            "Rationale: extreme GC → aberrant folding or non-specific binding",
            "Ideal window: 40–60% GC → no penalty",
        ]),
        ("3. PAM-Proximal GC Run Penalty", "#1565c0", 0.03, 0.60, 0.43, 0.14, [
            "gcRun = consecutive G/C count scanning 3′ end backwards",
            "gcRunPen = min(0.20, max(0.0, (gcRun − 2) × 0.08))",
            "Penalty starts when ≥ 3 consecutive G/C at 3′ end",
            "Rationale: long 3′ GC run → non-specific DNA binding",
        ]),
        ("4. Homopolymer Penalty", "#37474f", 0.52, 0.60, 0.43, 0.14, [
            "For each base B ∈ {A, C, G, T}: if BBBB in guide → hpPen += 0.08",
            "Maximum: 0.32 (all four homopolymers present — unlikely)",
            "AAAA, CCCC, TTTT, GGGG each add 0.08 penalty",
            "Rationale: repetitive runs → secondary structure or off-target matches",
        ]),
        ("5. RNA Hairpin Penalty", "#7b1fa2", 0.03, 0.42, 0.43, 0.12, [
            "Scans guide for palindromic stems ≥ 4 bp with ≥ 4 bp loop",
            "hairpinPen = 0.12 if any hairpin found, else 0.0",
            "Binary: presence/absence of a 4-bp stem",
            "Rationale: hairpin → guide partially folds → less Cas9 loading",
        ]),
        ("6. G-Quadruplex Penalty", "#2e7d32", 0.52, 0.42, 0.43, 0.12, [
            "gqPen = 0.08 if seed region contains 'GGG', else 0.0",
            "G-tracts form G-quadruplex secondary structures in RNA",
            "Reduces guide availability for target hybridisation",
            "Rationale: GGG in seed is the highest-risk motif for G4 formation",
        ]),
    ]
    for title, color, x, y, w, h, lines in components:
        section_box(ax, x, y, w, h, title, lines, title_color=color)

    divider(ax, 0.27)
    ax.text(0.5, 0.24, "How Specificity Score Is Used In The Combined Score",
            transform=ax.transAxes, fontsize=11, color=WHITE, ha='center', fontweight='bold')
    ax.text(0.5, 0.20,
            "eff_adj  =  efficiency_score  ×  specificity_score",
            transform=ax.transAxes, fontsize=12, color=TEAL, ha='center', fontfamily='monospace')
    ax.text(0.5, 0.15,
            "A guide scoring 0.85 efficiency but 0.60 specificity → eff_adj = 0.51",
            transform=ax.transAxes, fontsize=9, color=GREY, ha='center')
    ax.text(0.5, 0.11,
            "This multiplicative coupling means a highly off-target guide is never ranked first,",
            transform=ax.transAxes, fontsize=9, color=GREY, ha='center')
    ax.text(0.5, 0.07,
            "even if the XGBoost model predicts high editing efficiency.",
            transform=ax.transAxes, fontsize=9, color=GREY, ha='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 7 — ML Model & Training Data
# ────────────────────────────────────────────────────────────────────────────
def page_ml_model(pdf):
    fig, ax = new_page()
    header(ax, "Machine Learning Model", "XGBoost trained on real experimental CRISPR screen data")
    divider(ax, 0.90)

    # Model architecture boxes
    section_box(ax, 0.03, 0.87, 0.44, 0.18, "XGBoost Regressor — Hyperparameters", [
        "n_estimators = 500  (500 decision trees in ensemble)",
        "learning_rate = 0.03  (conservative step size, reduces overfitting)",
        "max_depth = 5  (maximum tree depth per estimator)",
        "subsample = 0.8  (80% of training rows sampled per tree)",
        "colsample_bytree = 0.8  (80% of features sampled per tree)",
        "objective = 'reg:squarederror'  (MSE loss)",
        "Input shape: Float32[N, 450]  →  Output: Float32[N] (efficiency score)",
    ])

    section_box(ax, 0.52, 0.87, 0.44, 0.18, "Why XGBoost?", [
        "Handles tabular features well — no normalisation needed",
        "Captures non-linear interactions between features",
        "Faster to train than deep learning on 4,692-sample dataset",
        "Regularisation (subsample + colsample) prevents overfitting",
        "SHAP values give interpretable feature importance",
        "Azimuth (the field standard) also uses a tree-based ensemble",
        "Exported to JSON tree format for pure-JS in-browser inference",
    ])

    section_box(ax, 0.03, 0.64, 0.44, 0.18, "Training Data — Doench 2016 (primary)", [
        "Source: Doench et al. (2016) Nat Biotechnol 34:184",
        "Genome-scale pooled CRISPR screen in human cells",
        "Provides: 20 bp guide, 30-mer context, depletion log-fold-change",
        "After filtering for valid 30-mers: n = 4,379 guides",
        "Normalised to [0, 1] efficiency score",
        "Gold-standard dataset; Azimuth was also trained on this data",
        "Split: 80% train, 20% hold-out (fixed random seed 42)",
    ])

    section_box(ax, 0.52, 0.64, 0.44, 0.18, "Training Data — Doench 2014 (supplementary)", [
        "Source: Doench et al. (2014) Nat Biotechnol 32:1262",
        "Earlier Cas9 screen, shorter guides, limited context",
        "313 guides added to training set (no 30-mer → context = zeros)",
        "Combined training set: 4,379 + 313 = 4,692 guides",
        "Both sources normalised to same [0, 1] scale",
        "Kim 2019 (11,808 guides) in the CSV but NOT used for training:",
        "  → score scale mismatch; reserved as independent validation",
    ])

    section_box(ax, 0.03, 0.41, 0.44, 0.16, "Independent Validation Datasets", [
        "Kim 2019 (novel-only, n=1,828): r = 0.640  ← key benchmark",
        "0% overlap with Doench training data — true out-of-distribution test",
        "Chari 2015 (293T, K562, SKNAS, U2OS): r = 0.746–0.806",
        "Xu 2015 (human cells): r = 0.424 (n=35, small sample)",
        "CRISPRscan zebrafish: r = 0.081 (expected — model trained on human)",
        "Precision@k (Kim2019): P@1=0.70, P@3=0.65, P@5=0.63, P@10=0.61",
    ])

    section_box(ax, 0.52, 0.41, 0.44, 0.16, "Comparison to Azimuth (Doench 2016 baseline)", [
        "Azimuth was trained on ALL Doench 2016 data (no hold-out)",
        "Our model: r=0.537 on 20% held-out Doench (fair comparison)",
        "Azimuth: r=0.654 on same held-out set (trained on full dataset)",
        "Full Doench set (biased, ours seen it): r=0.805 vs Azimuth r=0.699",
        "Kim2019 independent: r=0.640 ours vs ~0.59 Azimuth (est.)",
        "Our advantage: 30-mer context + proximity ranking + specificity",
    ])

    divider(ax, 0.22)
    ax.text(0.5, 0.19, "Training Pipeline",
            transform=ax.transAxes, fontsize=11, color=TEAL, fontweight='bold', ha='center')
    pipeline = [
        "1. Download Doench 2016 & 2014 CSVs (download_datasets.py)",
        "2. Merge → combined_training_data.csv (11,991 rows total, but only 4,692 Doench used)",
        "3. Extract 450-dim features per guide (feature_engineering.py)",
        "4. Stratified 80/20 train/test split → train XGBoost → save xgb_model.pkl",
        "5. Export PKL → flat JSON trees (export_js_model.py) → xgb_trees.json (294 KB)",
        "6. Deploy xgb_trees.json to frontend/public/ → traversed in pure JS at runtime",
    ]
    for i, step in enumerate(pipeline):
        ax.text(0.05, 0.16 - i * 0.022, step,
                transform=ax.transAxes, fontsize=8.5, color=WHITE, va='top')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 8 — Multi-Objective Scoring Formula
# ────────────────────────────────────────────────────────────────────────────
def page_scoring(pdf):
    fig, ax = new_page()
    header(ax, "Multi-Objective Scoring Formula",
           "How efficiency, specificity, and proximity are combined")
    divider(ax, 0.90)

    # Formula breakdown
    formulas = [
        ("Step 1 — Raw Efficiency Score", TEAL,
         "score = XGBoost(features450)   ∈ [0.0, 1.0]",
         "XGBoost predicts the probability of efficient editing from 450 features.\n"
         "Higher = guide expected to cut more efficiently in human cells.\n"
         "Capped to [0,1] via min/max clipping."),
        ("Step 2 — Specificity Score", ORANGE,
         "spec = 1 − seedPen − gcPen − gcRunPen − hpPen − hairpinPen − gqPen   ∈ [0.0, 1.0]",
         "Heuristic 6-component penalty (see Page 6).\n"
         "1.0 = maximally specific, 0.0 = high off-target risk."),
        ("Step 3 — Adjusted Efficiency", GREEN,
         "eff_adj = score × spec   ∈ [0.0, 1.0]",
         "Multiplicative coupling: a guide must be BOTH efficient AND specific.\n"
         "Score 0.85 × spec 0.6 → eff_adj 0.51 (penalised for off-target risk)."),
        ("Step 4a — Combined Score (no target position)", GREY,
         "combined = eff_adj",
         "When no target position is given, combined_score = eff_adj.\n"
         "Guides ranked purely by efficiency × specificity."),
        ("Step 4b — Combined Score (with target position)", "#7b1fa2",
         "combined = (1−w) × eff_adj  +  w × exp(−d² / 2σ²)   σ = 50 bp",
         "When a target position is given, proximity to that position is also rewarded.\n"
         "d = |cut_site − target_position|  (base pair distance)\n"
         "σ = 50 bp Gaussian width — guides within ~50 bp get strongest proximity boost.\n"
         "w = proximity_weight (default 0.4); user-adjustable 0.0 to 1.0."),
    ]

    y0 = 0.87
    for title, color, formula, explanation in formulas:
        ax.text(0.04, y0, title, transform=ax.transAxes,
                fontsize=10, color=color, fontweight='bold', va='top')
        ax.text(0.06, y0 - 0.028,
                formula, transform=ax.transAxes,
                fontsize=9.5, color=WHITE, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a1520',
                          edgecolor=color, linewidth=0.8))
        for j, line in enumerate(explanation.split('\n')):
            ax.text(0.06, y0 - 0.065 - j * 0.020, line,
                    transform=ax.transAxes, fontsize=8.5, color=GREY, va='top')
        y0 -= 0.14 if '\n' in explanation else 0.10

    divider(ax, 0.23)

    # Worked example
    ax.text(0.5, 0.20, "Worked Example", transform=ax.transAxes,
            fontsize=11, color=WHITE, ha='center', fontweight='bold')
    example = [
        ("XGBoost score",       "0.820"),
        ("Specificity score",   "0.730"),
        ("eff_adj",             "0.820 × 0.730 = 0.599"),
        ("Target pos = 500,  cut_site = 483,  d = 17 bp", ""),
        ("Proximity",           "exp(−17² / (2×50²)) = exp(−0.0578) = 0.944"),
        ("Combined (w=0.4)",    "(1−0.4)×0.599 + 0.4×0.944 = 0.359 + 0.378 = 0.737"),
    ]
    col_x = [0.05, 0.70]
    for i, (label, val) in enumerate(example):
        y_r = 0.16 - i * 0.025
        ax.text(col_x[0], y_r, label, transform=ax.transAxes,
                fontsize=8.5, color=LIGHT, va='top')
        if val:
            ax.text(col_x[1], y_r, val, transform=ax.transAxes,
                    fontsize=8.5, color=TEAL, va='top', fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 9 — Benchmark Results
# ────────────────────────────────────────────────────────────────────────────
def page_benchmarks(pdf):
    fig, ax = new_page()
    header(ax, "Benchmark Results",
           "Performance on held-out and independent datasets")
    divider(ax, 0.90)

    # Pearson r bar chart
    datasets = [
        ("Kim 2019 novel\n(n=1,828, 0% Doench)", 0.640, TEAL, "★ Key independent benchmark"),
        ("Doench 2016 held-out\n(n=938, 20% split)", 0.537, "#1565c0", "Fair: held-out only"),
        ("Chari 2015 U2OS\n(n=10)", 0.806, GREEN, "Small n"),
        ("Chari 2015 SKNAS\n(n=10)", 0.746, GREEN, "Small n"),
        ("Chari 2015 K562\n(n=10)", 0.758, GREEN, "Small n"),
        ("Chari 2015 293T\n(n=10)", 0.770, GREEN, "Small n"),
        ("Xu 2015 human\n(n=35)", 0.424, ORANGE, "Small n, different assay"),
        ("CRISPRscan zebrafish\n(n=1,020)", 0.081, GREY, "Expected: trained on human"),
    ]
    azimuth = {
        "Doench 2016 held-out\n(n=938, 20% split)": 0.654,
    }

    bar_x0 = 0.28
    bar_max_w = 0.62
    r_scale = bar_max_w  # r=1.0 → full width
    for i, (name, r, color, note) in enumerate(datasets):
        y = 0.86 - i * 0.085
        ax.text(0.25, y - 0.005, name, transform=ax.transAxes,
                fontsize=8, color=WHITE, ha='right', va='top')
        # ours bar
        bw = r * r_scale
        rect = mpatches.FancyBboxPatch(
            (bar_x0, y - 0.030), bw, 0.028,
            boxstyle="round,pad=0.003", facecolor=color, alpha=0.85,
            transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(bar_x0 + bw + 0.005, y - 0.014, f"r={r:.3f}",
                transform=ax.transAxes, fontsize=8.5, color=WHITE, va='center',
                fontweight='bold')
        ax.text(bar_x0 + bw + 0.065, y - 0.014, note,
                transform=ax.transAxes, fontsize=7.5, color=GREY, va='center')
        # Azimuth comparison
        if name in azimuth:
            az_r = azimuth[name]
            az_w = az_r * r_scale
            az_rect = mpatches.FancyBboxPatch(
                (bar_x0, y - 0.054), az_w, 0.018,
                boxstyle="round,pad=0.002", facecolor=RED, alpha=0.6,
                transform=ax.transAxes)
            ax.add_patch(az_rect)
            ax.text(bar_x0 + az_w + 0.005, y - 0.044, f"Azimuth r={az_r:.3f}",
                    transform=ax.transAxes, fontsize=7.5, color=RED, va='center')

    # Legend
    our_patch = mpatches.Patch(color=TEAL, label='gRNA Predictor (ours)')
    az_patch  = mpatches.Patch(color=RED,  alpha=0.6, label='Azimuth (trained on ALL Doench)')
    ax.legend(handles=[our_patch, az_patch], loc='lower right',
              fontsize=8, facecolor='#0a1520', edgecolor=TEAL,
              labelcolor=WHITE, bbox_to_anchor=(0.98, 0.02))

    divider(ax, 0.28)

    # Precision@k table
    ax.text(0.5, 0.25, "Precision@k — Kim 2019 (threshold = 80th percentile efficiency)",
            transform=ax.transAxes, fontsize=10, color=WHITE, ha='center', fontweight='bold')
    ax.text(0.5, 0.21,
            "P@k = fraction of top-k predicted guides whose true efficiency exceeds the 80th percentile",
            transform=ax.transAxes, fontsize=8, color=GREY, ha='center')

    pk_headers = ["@k=1", "@k=3", "@k=5", "@k=10"]
    pk_vals    = ["0.70", "0.65", "0.63", "0.61"]
    for i, (h, v) in enumerate(zip(pk_headers, pk_vals)):
        cx = 0.25 + i * 0.13
        rect = mpatches.FancyBboxPatch((cx - 0.055, 0.135), 0.11, 0.048,
            boxstyle="round,pad=0.005", facecolor="#12252a",
            edgecolor=TEAL, linewidth=0.8, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(cx, 0.165, h, transform=ax.transAxes,
                fontsize=9, color=GREY, ha='center', va='center')
        ax.text(cx, 0.148, v, transform=ax.transAxes,
                fontsize=14, color=TEAL, ha='center', va='center', fontweight='bold')

    ax.text(0.5, 0.08,
            "Kim 2019 URL: https://raw.githubusercontent.com/L-Q-Y/CRISPRtool/main/data/Cas9/Kim2019_train.csv",
            transform=ax.transAxes, fontsize=7.5, color=GREY, ha='center')
    ax.text(0.5, 0.04,
            "Run python compare_azimuth.py and python independent_validation.py to reproduce these numbers",
            transform=ax.transAxes, fontsize=8, color=TEAL, ha='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 10 — How To Use
# ────────────────────────────────────────────────────────────────────────────
def page_how_to_use(pdf):
    fig, ax = new_page()
    header(ax, "How To Use gRNA Predictor",
           "Step-by-step guide for designing CRISPR experiments")
    divider(ax, 0.90)

    steps = [
        ("Step 1 — Paste Your DNA Sequence", TEAL,
         "Paste a raw DNA sequence (A, C, G, T only — spaces are ignored). The tool\n"
         "scans both strands automatically. Minimum ~40 bp; practical range: 100 bp–10 kbp.\n"
         "If you have a gene of interest, paste the full exon or coding region."),
        ("Step 2 — Choose PAM Type", "#1565c0",
         "Select the nuclease PAM:\n"
         "  NGG  → SpCas9 (default, most common, human cells)\n"
         "  NAG  → SpCas9 alternative PAM (lower efficiency, more sites)\n"
         "  NNGRRT → SaCas9 (smaller nuclease, for AAV delivery vectors)\n"
         "  TTTV → Cas12a/Cpf1 (T-rich PAM, staggered DSB, 5′-PAM)"),
        ("Step 3 — Optionally Set Target Position", ORANGE,
         "If you want to edit at a specific location (e.g., introduce a mutation at codon 248),\n"
         "enter the 1-based position in the sequence. The tool will add a Gaussian proximity\n"
         "bonus for guides whose cut site is near your target.\n"
         "Default proximity weight = 0.4 (40% proximity, 60% efficiency×specificity)."),
        ("Step 4 — Click PREDICT GRNAS", GREEN,
         "Results appear in a table showing the top 5 guides ranked by combined_score.\n"
         "For each guide you see: sequence, PAM, position, strand, efficiency score,\n"
         "off-target specificity, cut site, distance to target, combined score, and GC%."),
        ("Step 5 — Interpret the Results Table", "#7b1fa2",
         "combined_score: primary ranking metric (higher = better, max 1.0)\n"
         "score: raw XGBoost efficiency prediction\n"
         "off_target_score: specificity heuristic (1.0 = most specific)\n"
         "cut_site: 1-based position where Cas9 cuts (choose for HDR templates)\n"
         "distance_to_target: base pairs from cut site to your target position\n"
         "GC%: guide GC content (aim for 40–60%)"),
    ]

    y0 = 0.87
    for title, color, body in steps:
        rect = mpatches.FancyBboxPatch(
            (0.03, y0 - 0.11), 0.93, 0.11,
            boxstyle="round,pad=0.01", facecolor="#0d1e28",
            edgecolor=color, linewidth=1.0, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.05, y0 - 0.016, title,
                transform=ax.transAxes, fontsize=10, color=color,
                fontweight='bold', va='top')
        for j, line in enumerate(body.split('\n')):
            ax.text(0.06, y0 - 0.042 - j * 0.020, line,
                    transform=ax.transAxes, fontsize=8.5, color=WHITE, va='top')
        y0 -= 0.135

    divider(ax, 0.22)

    ax.text(0.5, 0.19, "Local Development Setup", transform=ax.transAxes,
            fontsize=11, color=WHITE, ha='center', fontweight='bold')
    cmds = [
        "# Backend (Python 3.13 + FastAPI)",
        "cd backend && source ../.venv/Scripts/activate",
        "uvicorn app.main:app --reload",
        "",
        "# Frontend (React 18 + Vite, separate terminal)",
        "cd frontend && npm run dev",
        "",
        "# Retrain model (after downloading fresh data)",
        "python download_datasets.py && python train_model.py",
    ]
    ax.text(0.05, 0.17, '\n'.join(cmds), transform=ax.transAxes,
            fontsize=8, color=LIGHT, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0a1520',
                      edgecolor=TEAL, linewidth=0.8))

    ax.text(0.5, 0.02,
            "Live demo: jana242k4.github.io/gRNA_Predictor  |  Source: github.com/jana242k4/gRNA_Predictor",
            transform=ax.transAxes, fontsize=8, color=GREY, ha='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Generating PDF -> {PDF_PATH}")
    with PdfPages(PDF_PATH) as pdf:
        page_title(pdf)
        print("  [1/10] Title & Overview")
        page_crispr_biology(pdf)
        print("  [2/10] CRISPR Biology")
        page_design_rules(pdf)
        print("  [3/10] Design Rules")
        page_features(pdf)
        print("  [4/10] Feature Engineering")
        page_thermodynamics(pdf)
        print("  [5/10] Thermodynamics")
        page_offtarget(pdf)
        print("  [6/10] Off-Target Heuristic")
        page_ml_model(pdf)
        print("  [7/10] ML Model & Training")
        page_scoring(pdf)
        print("  [8/10] Scoring Formula")
        page_benchmarks(pdf)
        print("  [9/10] Benchmark Results")
        page_how_to_use(pdf)
        print("  [10/10] How To Use")

    print(f"\nDone!  {PDF_PATH}")
    print(f"Size:  {PDF_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
