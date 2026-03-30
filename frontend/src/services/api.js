import axios from 'axios'
import { predictOffline } from '../utils/onnxPredictor.js'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

/**
 * Predict top gRNAs for a given DNA sequence.
 * Tries the local FastAPI backend first; falls back to in-browser ONNX
 * inference when the backend is unreachable (e.g. GitHub Pages hosting).
 *
 * @param {string} sequence           - Raw DNA sequence (ACGTN)
 * @param {string} pam                - PAM sequence (default: NGG)
 * @param {number} topN               - Number of top results to return
 * @param {number|null} targetPosition - Optional 1-indexed target position for proximity ranking
 * @param {number} proximityWeight    - Weight for proximity vs efficiency (0.0-1.0, default 0.4)
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
  } catch (_err) {
    // Backend unreachable — use in-browser ONNX (GitHub Pages / offline mode)
    console.info('[gRNA Predictor] Backend unavailable — using in-browser ONNX inference')
    const tgt = body.target_position ?? null
    return predictOffline(
      body.sequence, pam, topN, tgt, proximityWeight,
      import.meta.env.BASE_URL || '/'
    )
  }
}

export async function checkHealth() {
  const response = await axios.get(
    (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace('/api/v1', '') + '/health'
  )
  return response.data
}
