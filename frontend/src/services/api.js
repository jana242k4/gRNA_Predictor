import axios from 'axios'
import { predictOffline } from '../utils/onnxPredictor.js'

// In production (GitHub Pages) this is the Render API URL (set in .env.production).
// In local dev it falls back to localhost:8000.
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'

const apiClient = axios.create({
  baseURL: BASE_URL,
  // 60 s — Render free tier has a ~30 s cold-start after 15 min idle.
  // Users see the spinner, then results appear once the backend wakes up.
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

/**
 * Predict top gRNAs for a given DNA sequence.
 *
 * Tries the configured backend (Render in prod, localhost in dev).
 * Falls back to in-browser XGBoost JS inference only if the backend
 * is completely unreachable (network error, not a slow cold start).
 */
export async function predictGRNAs(
  sequence,
  pam = 'NGG',
  topN = 5,
  targetPosition = null,
  proximityWeight = 0.4
) {
  const body = {
    sequence: sequence.toUpperCase().replace(/\s+/g, ''),
    pam,
    top_n: topN,
    proximity_weight: proximityWeight,
  }
  if (targetPosition !== null && targetPosition !== '' && !isNaN(parseInt(targetPosition, 10))) {
    body.target_position = parseInt(targetPosition, 10)
  }

  try {
    const response = await apiClient.post('/predict', body)
    return response.data
  } catch (err) {
    // Only fall back to in-browser inference on network errors (backend not reachable).
    // Timeouts and HTTP errors are surfaced directly so the user knows what happened.
    if (err.code === 'ERR_NETWORK' || err.code === 'ECONNREFUSED') {
      console.info('[gRNA Predictor] Backend unreachable — using in-browser XGBoost JS inference')
      const tgt = body.target_position ?? null
      return predictOffline(
        body.sequence, pam, topN, tgt, proximityWeight,
        import.meta.env.BASE_URL || '/'
      )
    }
    throw err
  }
}

export async function checkHealth() {
  const response = await axios.get(
    (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace('/api/v1', '') + '/health'
  )
  return response.data
}
