import axios from 'axios'
import { predictOffline } from '../utils/onnxPredictor.js'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 180000,   // 3 min — large sequences (>3000 bp) need time on HF Spaces
  headers: { 'Content-Type': 'application/json' },
})

// Only true network-down errors fall back to JS inference.
// Timeouts (ECONNABORTED) and bad responses surface as errors — they mean
// the backend is reachable but slow/overloaded; running in-browser XGBoost
// on a large sequence causes an OOM crash.
const OFFLINE_CODES = new Set(['ERR_NETWORK', 'ECONNREFUSED'])

// Sequences longer than this are refused for offline inference — the in-browser
// XGBoost model (572 KB JSON) uses too much heap for large candidate sets.
const OFFLINE_MAX_BP = 3000

export async function predictGRNAs(sequence, pam = 'NGG', topN = 5, targetPosition = null, proximityWeight = 0.4) {
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
    const isNetworkDown = OFFLINE_CODES.has(err.code) || err.code?.startsWith('ERR_NETWORK')
    if (isNetworkDown) {
      if (body.sequence.length > OFFLINE_MAX_BP) {
        throw new Error(
          `Sequence is ${body.sequence.length} bp — offline inference is limited to ${OFFLINE_MAX_BP} bp ` +
          `to avoid browser memory crashes. The prediction server (Hugging Face Spaces) may be starting ` +
          `up; please wait ~30 seconds and try again.`
        )
      }
      console.info('[OmicsCRISPR] Backend unreachable — using in-browser XGBoost JS inference')
      const tgt = body.target_position ?? null
      return predictOffline(body.sequence, pam, topN, tgt, proximityWeight, import.meta.env.BASE_URL || '/')
    }
    // Timeout: backend is alive but slow (large sequence, cold start).
    if (err.code === 'ECONNABORTED') {
      throw new Error(
        'The prediction server is taking longer than expected. ' +
        'This can happen with long sequences or after a cold start. ' +
        'Please try again in a few seconds.'
      )
    }
    throw err
  }
}

export async function omicsPredict(sequence, cellTypes = null) {
  const body = { sequence: sequence.toUpperCase().slice(0, 20) }
  if (cellTypes?.length) body.cell_types = cellTypes
  try {
    const res = await apiClient.post('/omics/predict', body)
    return res.data
  } catch {
    return null
  }
}

export async function omicsExplain(sequence, cellType = 'K562') {
  try {
    const res = await apiClient.post('/omics/explain', { sequence: sequence.toUpperCase().slice(0, 20), cell_type: cellType })
    return res.data
  } catch {
    return null
  }
}

export async function omicsGene(gene, cellType = 'K562', topN = 10) {
  try {
    const res = await apiClient.get(`/omics/gene/${encodeURIComponent(gene)}`, { params: { cell_type: cellType, top_n: topN } })
    return res.data
  } catch (err) {
    if (err.response?.status === 404) return { not_found: true }
    return null
  }
}

export async function checkHealth() {
  const res = await axios.get((import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace('/api/v1', '') + '/health')
  return res.data
}

let _benchmarkCache = null
export async function fetchBenchmark() {
  if (_benchmarkCache) return _benchmarkCache
  try {
    const res = await apiClient.get('/benchmark')
    _benchmarkCache = res.data
    return res.data
  } catch {
    return null
  }
}
