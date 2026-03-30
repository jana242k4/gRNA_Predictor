# gRNA Predictor

[![Deploy to GitHub Pages](https://github.com/jana242k4/gRNA_Predictor/actions/workflows/deploy.yml/badge.svg)](https://github.com/jana242k4/gRNA_Predictor/actions/workflows/deploy.yml)
[![Backend Tests](https://github.com/jana242k4/gRNA_Predictor/actions/workflows/test.yml/badge.svg)](https://github.com/jana242k4/gRNA_Predictor/actions/workflows/test.yml)

**Live demo → [jana242k4.github.io/gRNA_Predictor](https://jana242k4.github.io/gRNA_Predictor)**

AI-powered CRISPR guide RNA (sgRNA) designer with XGBoost-based efficiency prediction and **Gaussian proximity-weighted ranking** — runs entirely in the browser via ONNX, no server required.

---

## Features

- **XGBoost ML scoring** — 450-dimensional feature model trained on 4,692 experimental guides (Doench 2016 + 2014)
- **Novel proximity ranking** — guides re-ranked by Gaussian decay from a user-specified genomic target; optimises the efficiency–proximity tradeoff via adjustable weight *w*
- **Multi-PAM support** — SpCas9 (NGG/NAG), SaCas9 (NNGRRT), Cas12a (TTTV)
- **Both strands** — detects guides on forward and reverse complement
- **Runs offline** — in-browser ONNX inference when no local backend is running

---

## Benchmarks

| Dataset | n | Spearman r | Notes |
|---------|---|-----------|-------|
| **Kim 2019 novel-only** | **1,828** | **0.640** | Primary independent benchmark; 0% Doench overlap |
| Doench 2016 held-out (20%) | 938 | 0.537 | Our held-out; Azimuth sees 100% Doench |
| Azimuth (same held-out) | 938 | 0.654 | Azimuth trained on ALL Doench — asymmetric comparison |
| Chari 2015 (293T) | 10 | 0.770 | CI ≈ ±0.74 — treat as supplementary |
| Xu 2015 | 35 | 0.424 | |
| CRISPRscan zebrafish | 1,020 | 0.081 | Expected low — model trained on human |

> **Kim 2019** (DeepSpCas9, n=12,825) is the honest independent benchmark because neither this model nor Azimuth was trained on it.

### Feature ablation (Doench held-out, baseline r=0.537)

| Feature removed | Δr |
|----------------|-----|
| Dinucleotide freq. | −0.050 |
| Segmented Tm (3 windows) | −0.049 |
| GC clamp (3' end) | −0.038 |
| Positional dinucs | −0.038 |
| Positional one-hot | −0.038 |

---

## Novelty

**Gaussian proximity-weighted guide ranking** is the key novel contribution. No existing tool (Azimuth, CRISPRscan, CRISPOR, CHOPCHOP) automates this step:

```
combined_score = (1 − w) × efficiency + w × exp(−d² / 2σ²)
```

where *d* = distance from guide cut site to desired edit position, σ = 50 bp (Paquet 2016; Richardson 2016), and *w* is user-tunable. CRISPOR shows distances visually; this tool **quantifies and optimises the tradeoff automatically**.

---

## Quick Start

### Option A — Live demo (no installation)
Open the [GitHub Pages URL](https://jana242k4.github.io/gRNA_Predictor). The model runs in-browser via WebAssembly ONNX.

### Option B — Full local stack (backend + frontend)

```bash
# Clone
git clone https://github.com/jana242k4/gRNA_Predictor.git
cd gRNA_Predictor

# Backend
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload
# → http://localhost:8000/docs

# Frontend (new terminal)
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

---

## Model Details

```
Architecture:   XGBoost Regressor
Trees:          500   Learning rate: 0.03   Max depth: 5
Features:       450-dimensional vector
Training data:  4,692 guides — Doench 2016 (4,379) + Doench 2014 (313)
Held-out:       2,565 Kim 2019 guides (independent validation set)
```

### Feature vector (450 dims)

| Dimensions | Feature | Reference |
|-----------|---------|-----------|
| 0–79 | Positional one-hot (20 pos × 4 bases) | Doench 2016 |
| 80 | GC content | Doench 2014 |
| 81 | Nearest-neighbour Tm (full guide) | SantaLucia 1998 |
| 82–97 | Dinucleotide frequencies | Doench 2016 |
| 98 | Seed region GC (last 12 bp) | Hsu 2013 |
| 99 | Poly-T flag | Brummelkamp 2002 |
| 100–403 | Position-specific dinucleotide one-hot | Doench 2016 / Azimuth |
| 404–419 | Upstream 4-bp context | Doench 2016 |
| 420–443 | Downstream 6-bp context | Kim 2019 |
| 444 | GC clamp (last 4 bp) | Doench 2016 |
| 445 | RNA hairpin proxy | Zuker 2003 |
| 446 | Microhomology at cut site | Bae 2014 |
| 447–449 | Segmented Tm (PAM-distal / seed / 30-mer) | Doench 2016 ext. |

---

## Project Structure

```
gRNA_Predictor/
├── backend/
│   ├── app/
│   │   ├── api/endpoints.py         # POST /api/v1/predict
│   │   ├── models/
│   │   │   ├── xgb_model.pkl        # Trained XGBoost model
│   │   │   └── schemas.py           # Pydantic request/response
│   │   ├── services/
│   │   │   ├── feature_engineering.py  # 450-dim features
│   │   │   ├── sequence_parser.py      # PAM detection
│   │   │   └── scorer.py               # Heuristic fallback
│   │   └── utils/biology_utils.py   # Tm, GC, hairpin
│   ├── data/
│   │   ├── combined_training_data.csv  # 11,991 guides
│   │   └── kim2019_holdout.csv         # 2,565 independent guides
│   ├── benchmark_results/           # Figures + metrics
│   ├── train_model.py               # Retrain from scratch
│   ├── compare_azimuth.py           # Head-to-head vs Azimuth
│   ├── independent_validation.py    # Cross-dataset validation
│   ├── shap_analysis.py             # Feature importance
│   └── export_onnx.py               # PKL → ONNX for browser
├── frontend/
│   ├── public/xgb_model.onnx        # Model for in-browser inference
│   └── src/
│       ├── utils/
│       │   ├── featureEngineering.js  # 450-dim JS port
│       │   ├── sequenceParser.js      # PAM detection JS port
│       │   └── onnxPredictor.js       # ONNX Runtime Web wrapper
│       └── services/api.js            # API with ONNX fallback
└── .github/workflows/
    ├── deploy.yml                   # GitHub Pages auto-deploy
    └── test.yml                     # Backend pytest CI
```

---

## Citations

- Doench JG et al. (2016) *Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9.* Nat Biotechnol **34**:184–191
- Doench JG et al. (2014) *Rational design of highly active sgRNAs for CRISPR-Cas9–mediated gene inactivation.* Nat Biotechnol **32**:1262–1267
- Kim HK et al. (2019) *Deep learning improves prediction of CRISPR-Cpf1 guide RNA activity.* Nat Biotechnol **37**:238–246
- Paquet D et al. (2016) *Efficient introduction of specific homozygous and heterozygous mutations using CRISPR/Cas9.* Nature **533**:125–129
- Richardson CD et al. (2016) *Enhancing homology-directed genome editing by catalytically active and inactive CRISPR-Cas9.* Nat Biotechnol **34**:339–344
- SantaLucia J (1998) *A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics.* PNAS **95**:1460–1465

---

## Reproducing Benchmarks

```bash
cd backend && source ../.venv/Scripts/activate
python -m pytest tests/ -v                  # 27/27 unit tests
python compare_azimuth.py                   # vs Azimuth benchmark
python independent_validation.py            # Kim 2019, Chari 2015, Xu 2015
python shap_analysis.py                     # SHAP feature importance
python publication_figures.py               # Regenerate all figures
python create_benchmark_pdf.py              # LinkedIn-ready PDF
```

## Re-exporting the ONNX model

```bash
cd backend && source ../.venv/Scripts/activate
python export_onnx.py
# → frontend/public/xgb_model.onnx (580 KB, verified 0.000000 max diff)
```
