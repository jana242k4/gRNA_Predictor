import axios from 'axios'
import { predictOffline } from '../utils/onnxPredictor.js'

// In production (GitHub Pages) this is the Render API URL (set in .env.production).
// In local dev it falls back to localhost:8000.
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'

const apiClient = axios.create({
  baseURL: BASE_URL,
  // 90 s — Render free tier can take up to ~50 s to cold-start after 15 min idle.
  timeout: 90000,
  headers: { 'Content-Type': 'application/json' },
})

// Error codes that mean the backend is simply not reachable — fall back to offline.
const OFFLINE_CODES = new Set(['ERR_NETWORK', 'ECONNREFUSED', 'ECONNABORTED', 'ERR_BAD_RESPONSE'])

/**
 * Predict top gRNAs for a given DNA sequence.
 *
 * Tries the configured backend (Render in prod, localhost in dev).
 * Falls back to in-browser XGBoost JS inference when the backend is
 * unreachable or times out (Render free-tier cold start).
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
    // Fall back to in-browser inference when the backend is unreachable or timed out.
    // HTTP 4xx/5xx errors (bad request, server error) are re-thrown so the user sees them.
    if (OFFLINE_CODES.has(err.code) || err.code?.startsWith('ERR_NETWORK')) {
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
