# CLAUDE.md — gRNA Predictor / OmicsCRISPR

This file governs how Claude Code behaves in this repository. All rules are mandatory with no exceptions.

## Project Identity

- **Backend**: Python 3.13 + FastAPI + Uvicorn, port 8000
- **Frontend**: React 18 + Vite + Tailwind CSS v4, port 5173
- **ML**: XGBoost (450-dim features) + PyTorch three-branch OmicsCRISPR model
- **Venv**: `.venv/` — activate with `source .venv/Scripts/activate` (NOT `venv/`)
- **GitHub Pages**: `https://jana242k4.github.io/gRNA_Predictor/`
- **Backend (Render)**: `https://grna-predictor-api.onrender.com`

## Git Rules

- **NEVER include `Co-Authored-By: Claude` in any commit message.** No exceptions.
- Do not amend published commits. Create new commits only.
- Do not force-push unless the user explicitly requests it.

## Code Style

- No MUI (`@mui/material`) imports — all UI uses Tailwind CSS.
- Feature dimensions: **450 exactly**. Any change requires updating BOTH `backend/app/services/feature_engineering.py` AND `frontend/src/utils/featureEngineering.js` in the same commit.
- `TRAIN_SOURCES = {"Doench2016", "Doench2014"}` — every script that touches training data must preserve this constant (currently in `train_model.py`, `compare_azimuth.py`, `shap_analysis.py`, `publication_figures.py`).
- All benchmark splits: gene-stratified 80/20, `random_state=42`.
- Write no comments unless the WHY is non-obvious. Never write multi-line docstring blocks.

---

## Security Rules (from Security-First Vibe Coding)

### 1. Secrets & Environment Variables
- All API keys, tokens, database URLs must live in `.env` files only.
- `.env` must be in `.gitignore`. Never commit secrets.
- Frontend: only `VITE_`-prefixed variables in client code — and only for **non-secret** values (e.g. public API base URL).
- Backend secrets accessed via `os.environ` / Pydantic `Settings` only, never hardcoded.
- Maintain `.env.example` with required variable names and empty values.

### 2. Rate Limiting
- Every public-facing endpoint must have rate limiting via `slowapi`.
- Default limits:
  - `/api/v1/predict`: 30 req/min per IP
  - `/api/v1/omics/predict`: 20 req/min per IP
  - `/api/v1/omics/explain`: 10 req/min per IP (heavy compute)
  - `/api/v1/omics/gene/*`: 30 req/min per IP
  - Health check `/health`: unlimited
- Return `429 Too Many Requests` with `Retry-After` header. Never silently swallow.

### 3. Input Validation & Sanitisation
- Validate ALL inputs server-side with Pydantic. Client-side validation is UX only, never security.
- Validate: data type, length limits, allowed characters, required fields.
- Reject and return `400 Bad Request` for invalid input; log the attempt.
- Never interpolate user input into raw queries.
- For sequence inputs: reject non-ACGT characters, detect RNA (U present), enforce minimum length.

### 4. CORS
- Do NOT use wildcard `*` in production CORS.
- Whitelist only: `http://localhost:5173`, `http://localhost:3000`, `https://jana242k4.github.io`
- Restrict allowed HTTP methods to only what each endpoint needs.

### 5. HTTP Security Headers
Set these headers on every response:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'`
- Remove `X-Powered-By` / `Server` headers that leak framework info.

### 6. Error Handling & Logging
- Never return stack traces, raw error messages, or internal file paths to clients in production.
- Return generic messages to users. Log full detail server-side.
- Use `4xx` for client errors, `5xx` for server errors. Never use 500 for validation failures.

### 7. XSS Prevention (Frontend)
- Do NOT use `dangerouslySetInnerHTML` without DOMPurify sanitisation.
- Never use `eval()` or `new Function()` with dynamic content.
- No inline `<script>` tags with dynamic user content.

### 8. Dependency Security
- Run `pip-audit` and `npm audit` before every release.
- Fix high/critical vulnerabilities before shipping.
- Pin dependency versions in `requirements.txt` and `package-lock.json`.

---

## Deployment Checklist

Before every deploy, verify:
- [ ] `.env` is not committed to git
- [ ] All secrets set in Render / GitHub environment variable config
- [ ] Debug mode and development logging are OFF in production
- [ ] HTTPS enforced (no plain HTTP)
- [ ] Rate limiting active on all public endpoints
- [ ] CORS restricted to known origins
- [ ] `npm audit` / `pip-audit` clean (no high/critical)

---

## Testing

- Run tests with: `cd backend && source ../.venv/Scripts/activate && python -m pytest tests/ -v`
- All 108+ tests must pass before committing.
- Do not mock the XGBoost model in integration tests — load the real pkl.
- Benchmark numbers to maintain (do not regress):
  - Kim2019 novel-only Pearson r ≥ 0.640
  - Doench held-out (20%) Pearson r ≥ 0.530

---

## Key File Map

| File | Purpose |
|------|---------|
| `backend/app/main.py` | FastAPI app, CORS, rate limiter, lifespan |
| `backend/app/api/endpoints.py` | POST /api/v1/predict |
| `backend/app/api/omics_endpoints.py` | /omics/* endpoints |
| `backend/app/services/feature_engineering.py` | 450-dim feature extraction |
| `backend/app/services/off_target.py` | Specificity scoring |
| `backend/app/services/sequence_parser.py` | PAM detection, both strands |
| `backend/app/services/scorer.py` | ML scoring, combined ranking |
| `backend/app/models/ai_models.py` | XGBoost model loader |
| `backend/train_model.py` | Model training (Doench2016+2014) |
| `backend/omics_pipeline/omics_inference.py` | OmicsCRISPR three-branch model |
| `frontend/src/services/api.js` | Axios client + offline fallback |
| `frontend/src/utils/featureEngineering.js` | 450-dim JS mirror of backend |
| `frontend/src/utils/onnxPredictor.js` | Offline XGBoost inference |
