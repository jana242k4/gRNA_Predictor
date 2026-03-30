# AI-Powered gRNA Designer - Project Roadmap & Structure

## 1. Project Architecture Overview

The application follows a standard modern web architecture separating the frontend client and the AI-powered backend API.

- **Frontend (Client)**: Built with React + Vite + Material UI. Connects to the backend via REST API.
- **Backend (API)**: Built with Python (FastAPI). Hosts the prediction logic, ML model inference, and sequence parsing.
- **ML Models**: Scikit-Learn Random Forest regressor trained on empirical CRISPR datasets (e.g., Doench 2016 / Azimuth).

---

## 2. Updated Implementation Pipeline

### Phase 1: Core Parsing & Pipeline Setup
- [x] **Backend Skeleton**: Setup FastAPI base, basic health endpoints.
- [x] **Frontend Skeleton**: Setup React/Vite layout with input forms.
- [x] **Sequence Parsing**: Extract valid `SpCas9` (NGG) PAM sites.
- [x] **Mock Scoring**: Establish API data models and basic rule heuristics.

### Phase 2: ML Model Integration & Scoring (Priority A)
1. **Dataset Preparation**: Acquire publicly available dataset (e.g., Azimuth ~15k guides with cleavage efficiencies).
2. **Feature Engineering**: Write Python functions to extract features from a 20bp guide:
   - Positional nucleotide one-hot encoding.
   - GC Content percentage.
   - Melting temperature estimates.
   - Dinucleotide counts (e.g., 'GG', 'AA').
3. **Model Training (Scikit-Learn)**: Train a `RandomForestRegressor` and save the serialized weights (e.g., `model.pkl`).
4. **API Integration**: Load the `.pkl` file at FastAPI startup. In the `/predict` route, transform the input guides into feature vectors and pass them to the ML model for true predictions.

### Phase 3: Comprehensive Testing (Priority B)
1. **Unit Testing Backend (`pytest`)**:
   - Test PAM extraction (`NGG` at correct positions).
   - Test edge cases (short sequences, non-DNA characters, no PAMs found).
   - Test scoring boundaries (ensure scores act rationally).
2. **API Endpoint Testing**:
   - Use `FastAPI.TestClient` to test the `/predict` route.
   - Verify HTTP 400 errors for bad inputs.
3. **Frontend Integration Testing**:
   - Ensure the React UI gracefully handles loading states and errors (e.g., backend is down).

### Phase 4: Feature Additions & UX Polish (Priority D)
1. **Cas12a Support**: Update the parser to support Cas12a 'TTTV' PAMs on the 5' end. Update UI to allow users to select "SpCas9" or "Cas12a".
2. **Off-target Risk Warning**: Implement a basic warning flag if the 'seed region' (10-12bp near the PAM) has risky motifs (like poly-T tracts).
3. **UI Upgrades**:
   - **CSV/TSV Export**: Add a button on the React `ResultsTable` to download the predictions as a file.
   - **Copy to Clipboard**: Add a small clipboard icon next to each sequence in the table.

### Phase 5: Production Deployment (Priority C)
1. **Dockerize Backend**: Add the `RandomForest` weights, `requirements.txt`, and source code to a `Dockerfile` optimized for Render Web Services.
2. **Build Frontend**: Run `npm run build` with the environment API URL pointed to Render.
3. **Deploy via Action/CLI**: Host the frontend statically on GitHub Pages.

---

## 3. Current Focus
We are currently focusing on **Phase 2 (ML Scoring)**, **Phase 3 (Testing)**, and **Phase 4 (Features)**. Once these are completely implemented and verified, we will move to **Phase 5 (Deployment)**.
