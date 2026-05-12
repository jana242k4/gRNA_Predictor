# gRNA Predictor

[![Deploy to GitHub Pages](https://github.com/jana242k4/gRNA_Predictor/actions/workflows/deploy.yml/badge.svg)](https://github.com/jana242k4/gRNA_Predictor/actions/workflows/deploy.yml)
[![Backend Tests](https://github.com/jana242k4/gRNA_Predictor/actions/workflows/test.yml/badge.svg)](https://github.com/jana242k4/gRNA_Predictor/actions/workflows/test.yml)

**Live demo → [jana242k4.github.io/gRNA_Predictor](https://jana242k4.github.io/gRNA_Predictor)**

ML-assisted CRISPR guide RNA (sgRNA) design and ranking tool with XGBoost-based efficiency prediction, **CFD off-target specificity scoring** (Doench 2016 Table S19), **cell-type-aware OmicsCRISPR scoring**, and **Gaussian proximity-weighted ranking** — backed by a FastAPI server (Hugging Face Spaces) for full ML inference, with a pure-JavaScript fallback for offline use.

---

## Current Limitations

- Off-target scoring uses the CFD mismatch matrix (sequence-intrinsic) and does not perform genome-wide alignment
- Efficiency predictions trained on human cell lines (Doench 2016 + 2014 + Kim 2019); lower accuracy expected in non-human organisms
- OmicsCRISPR cell-type suitability panel only applies to guides in the DepMap Avana sgRNA library
- Tool is intended for research and educational use, not clinical or experimental decision-making

---

## Features

- **XGBoost ML scoring** — 452-dimensional feature model trained on 11,991 experimental guides (Doench 2016 + Doench 2014 + Kim 2019, per-source z-score normalised)
- **FastAPI backend (Hugging Face Spaces)** — primary inference path; full 452-dim XGBoost with 30-mer context features
- **Pure-JS offline fallback** — model exported as 572 KB JSON tree structure; traversed in-browser if backend is unreachable (no WASM, no installation)
- **CFD off-target scoring** — Cutting Frequency Determination matrix from Doench 2016 Table S19 (20 positions × 16 mismatch types); product-of-weights formulation; seed-region mismatches (positions 1–12 from PAM) weighted 2×
- **OmicsCRISPR panel** — three-branch PyTorch model (CNN + feature MLP + omics MLP) scoring cell-type suitability across K562, CD4/CD8 T cells, NK cells, B cells; splice-site risk annotation; Integrated Gradients feature attribution
- **Multi-objective ranking** — `combined_score = (1−w) × (efficiency × specificity) + w × proximity`; off-target risk penalises efficiency multiplicatively
- **Novel proximity ranking** — guides re-ranked by Gaussian decay from a user-specified genomic target; adjustable weight *w* (default 0.4)
- **Multi-PAM support** — SpCas9 (NGG/NAG), SaCas9 (NNGRRT), Cas12a (TTTV)
- **Both strands** — detects guides on forward and reverse complement
- **Benchmark comparison panel** — in-app table comparing this tool against Azimuth, CRISPOR, and CRISPRscan on independent datasets
- **Security-hardened API** — slowapi rate limiting, CORS restriction, input validation (rejects non-ACGT, RNA, short sequences), HTTP security headers

---

## Benchmarks

| Dataset | n | Spearman r | Notes |
|---------|---|-----------|-------|
| **Kim 2019 novel-only** | **1,828** | **0.757** | Primary independent benchmark; 0% Doench overlap |
| Doench 2016 held-out (20%) | 938 | 0.708 | Our held-out; Azimuth sees 100% Doench |
| Azimuth (same held-out) | 938 | 0.654 | Azimuth trained on ALL Doench — asymmetric comparison |
| Chari 2015 (293T avg.) | 10 | 0.794 | CI ≈ ±0.74 — treat as supplementary |
| Xu 2015 | 35 | 0.424 | |
| CRISPRscan zebrafish | 1,020 | 0.081 | Expected low — model trained on human |

> **Kim 2019** (DeepSpCas9, n=12,825) is the honest independent benchmark because neither this model nor Azimuth was trained on it.

### Feature ablation (true retrain, Doench held-out, baseline r=0.708)

Each row retrains the full model from scratch with that feature group removed.

| Feature removed | Δr |
|----------------|-----|
| Dinucleotide freq. | −0.050 |
| Segmented Tm (3 windows) | −0.049 |
| GC clamp (3' end) | −0.038 |
| Positional dinucs | −0.038 |
| Positional one-hot | −0.038 |

---

## Novelty

**Multi-objective scoring** integrates three components no existing tool combines automatically:

```
eff_adj        = ML_efficiency × CFD_specificity_score
combined_score = (1 − w) × eff_adj + w × exp(−d² / 2σ²)
```

where *d* = distance from guide cut site to desired edit position, σ = 50 bp (Paquet 2016; Richardson 2016), and *w* is user-tunable (default 0.4).

| Tool | Efficiency | CFD Specificity | Proximity ranking | Cell-type scoring |
|------|-----------|----------------|------------------|------------------|
| **This tool** | XGBoost 452-dim | Doench 2016 CFD | Gaussian decay | OmicsCRISPR (5 types) |
| Azimuth | Linear regression | None | None | None |
| CRISPOR | Doench 2016 | CRISPOR off-target list | None | None |
| CRISPRscan | Linear model | None | Visual only | None |

---

## Quick Start

### Option A — Live demo (no installation)

Open **[jana242k4.github.io/gRNA_Predictor](https://jana242k4.github.io/gRNA_Predictor)**.

Predictions run via the Hugging Face Spaces FastAPI backend. If the Space is paused (after extended inactivity), it resumes in ~30 seconds — the spinner will show. If the backend is unreachable, the tool falls back to in-browser pure-JS XGBoost automatically.

### Option B — Full local stack (backend + frontend)

```bash
# Clone
git clone https://github.com/jana242k4/gRNA_Predictor.git
cd gRNA_Predictor

# Backend (Python 3.13 + FastAPI)
# Activate venv (.venv/, NOT venv/)
source .venv/Scripts/activate          # Windows Git Bash / WSL
# source .venv/bin/activate            # macOS / Linux

pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload
# → http://localhost:8000/docs

# Frontend (React 18 + Vite, separate terminal)
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

---

## Model Details

```
Architecture:   XGBoost Regressor
Trees:          500   Learning rate: 0.03   Max depth: 5
Features:       452-dimensional vector
Training data:  11,991 guides — Doench 2016 + Doench 2014 + Kim 2019
                (per-source z-score normalised before merging)
Held-out:       20% gene-stratified split, random_state=42
Browser model:  572 KB JSON tree dump (500 trees)
```

### Feature vector (452 dims)

| Dimensions | Feature | Reference |
|-----------|---------|-----------|
| 0–79 | Positional one-hot (20 pos × 4 bases) | Doench 2016 |
| 80 | GC content | Doench 2014 |
| 81 | Nearest-neighbour Tm (full guide, SantaLucia 1998) | SantaLucia 1998 |
| 82–97 | Dinucleotide frequencies (16 pairs) | Doench 2016 |
| 98 | Seed region GC (last 12 bp) | Hsu 2013 |
| 99 | Poly-T flag (TTTT present) | Brummelkamp 2002 |
| 100–403 | Position-specific dinucleotide one-hot (19 × 16) | Doench 2016 / Azimuth |
| 404–419 | Upstream 4-bp context one-hot | Doench 2016 |
| 420–443 | Downstream 6-bp context one-hot | Kim 2019 |
| 444 | GC clamp (last 4 bp) | Doench 2016 |
| 445 | Tm asymmetry — abs(Tm[0:10] − Tm[10:20]) / 20 | This work |
| 446 | Microhomology score at cut site | Bae 2014 |
| 447 | Tm PAM-distal 8 bp (positions 1–8) | SantaLucia 1998 |
| 448 | Seed ΔG at 37°C — last 8 bp duplex (SantaLucia 1998 NN) | This work |
| 449 | Tm full 30-mer context | SantaLucia 1998 |
| 450 | PAM-proximal 10 bp GC (positions 11–20) | This work |
| 451 | PAM-distal 10 bp GC (positions 1–10) | This work |

### Off-target scoring (CFD)

Off-target specificity uses the **Cutting Frequency Determination (CFD)** matrix from Doench 2016 (Supplementary Table S19). For each position, the score is the product of per-position mismatch weights. A perfect 20-nt match scores 1.0; each mismatch reduces the score. Seed-region mismatches (positions 1–12 from PAM) apply a 2× penalty reflecting known Cas9 sensitivity. PAM variants (non-NGG) are penalised by the published PAM score.

---

## Project Structure

```
gRNA_Predictor/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── endpoints.py            # POST /api/v1/predict, GET /api/v1/benchmark
│   │   │   └── omics_endpoints.py      # /api/v1/omics/* (predict, explain, gene)
│   │   ├── models/
│   │   │   ├── rf_model.pkl            # Trained XGBoost model (452-dim)
│   │   │   └── schemas.py              # Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── feature_engineering.py  # 452-dim feature extraction
│   │   │   ├── sequence_parser.py      # PAM detection (NGG/NAG/NNGRRT/TTTV)
│   │   │   ├── off_target.py           # CFD off-target scoring (Doench 2016)
│   │   │   └── scorer.py               # ML scoring + combined ranking
│   │   └── utils/biology_utils.py      # Tm, ΔG, thermodynamic utilities
│   ├── omics_pipeline/
│   │   └── omics_inference.py          # OmicsCRISPR three-branch PyTorch model
│   ├── data/
│   │   ├── combined_training_data.csv  # 11,991 guides (Doench 2016/2014 + Kim 2019)
│   │   └── kim2019_holdout.csv         # 2,565 Kim 2019 independent guides
│   ├── benchmark_results/              # Figures + metrics
│   ├── train_model.py                  # Retrain (z-score normalised, --cv flag)
│   ├── export_js_model.py              # PKL → xgb_trees.json
│   ├── compare_azimuth.py              # Head-to-head vs Azimuth + permutation/Wilcoxon tests
│   ├── independent_validation.py       # Cross-dataset validation with bootstrap CIs
│   ├── shap_analysis.py                # SHAP importance + true retrain ablation
│   ├── predict_cli.py                  # CLI interface (--sequence / --fasta / --pam / --top-n)
│   ├── hf_Dockerfile                   # Hugging Face Spaces Docker image
│   └── hf_README.md                    # HF Space metadata (sdk: docker, app_port: 8000)
├── frontend/
│   ├── public/
│   │   └── xgb_trees.json              # 572 KB model (pure JS offline inference)
│   └── src/
│       ├── utils/
│       │   ├── featureEngineering.js   # 452-dim JS port (mirrors backend)
│       │   ├── sequenceParser.js       # PAM detection JS port
│       │   ├── xgbPredictor.js         # Pure-JS XGBoost tree traversal (offline)
│       │   └── onnxPredictor.js        # Offline orchestrator
│       ├── components/
│       │   ├── OmicsPanel.jsx          # Cell-type suitability + Integrated Gradients
│       │   └── BenchmarkPanel.jsx      # In-app comparison vs Azimuth/CRISPOR/CRISPRscan
│       └── services/api.js             # Axios client → HF Spaces backend; JS fallback on error
└── .github/workflows/
    ├── deploy.yml                      # GitHub Pages auto-deploy
    └── test.yml                        # Backend pytest CI
```

---

## Citations

- Doench JG et al. (2016) *Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9.* Nat Biotechnol **34**:184–191
- Doench JG et al. (2014) *Rational design of highly active sgRNAs for CRISPR-Cas9–mediated gene inactivation.* Nat Biotechnol **32**:1262–1267
- Kim HK et al. (2019) *Deep learning improves prediction of CRISPR-Cpf1 guide RNA activity.* Nat Biotechnol **37**:238–246
- Hsu PD et al. (2013) *DNA targeting specificity of RNA-guided Cas9 nucleases.* Nat Biotechnol **31**:827–832
- Bae S et al. (2014) *Microhomology-based choice of Cas9 nuclease target sites.* Nat Methods **11**:705–706
- Paquet D et al. (2016) *Efficient introduction of specific homozygous and heterozygous mutations using CRISPR/Cas9.* Nature **533**:125–129
- Richardson CD et al. (2016) *Enhancing homology-directed genome editing by catalytically active and inactive CRISPR-Cas9.* Nat Biotechnol **34**:339–344
- SantaLucia J (1998) *A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics.* PNAS **95**:1460–1465

---

## Reproducing Benchmarks

```bash
cd backend && source ../.venv/Scripts/activate

python -m pytest tests/ -v                  # 113/113 unit tests
python train_model.py --cv                  # 5-fold CV → benchmark_results/cv_results.json
python compare_azimuth.py                   # vs Azimuth + permutation/Wilcoxon tests
python independent_validation.py            # Kim 2019 (bootstrap CI), Chari 2015, Xu 2015
python shap_analysis.py                     # SHAP + true retrain ablation → true_ablation.txt
python publication_figures.py               # Regenerate all figures
```

## Re-exporting the browser model

```bash
cd backend && source ../.venv/Scripts/activate
python export_js_model.py
# → frontend/public/xgb_trees.json (572 KB, 500 trees)
```
