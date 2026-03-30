/**
 * In-browser XGBoost inference via ONNX Runtime Web.
 * Used as a fallback when the local FastAPI backend is unreachable
 * (e.g. when running on GitHub Pages).
 *
 * ONNX inference runs inside a Web Worker to keep the main thread responsive.
 * If the worker fails (OOM, WASM unsupported), falls back to a GC heuristic.
 */
import { findAllGRNAs } from './sequenceParser.js'

// Vite's ?worker suffix compiles the file as a module worker
import OnnxWorker from '../workers/onnxWorker.js?worker'

const SIGMA           = 50.0
const CAS12A_PAMS     = new Set(['TTTV'])
const MAX_CANDIDATES  = 200
const MODEL_INFO_TEXT = 'XGBoost 450-dim (in-browser ONNX) — Doench 2016 + 2014, n=4,692'

let _worker      = null
let _reqId       = 0
let _pending     = new Map()   // reqId → { resolve, reject }
let _onnxFailed  = false

function getWorker() {
  if (_worker) return _worker
  _worker = new OnnxWorker()
  _worker.onmessage = ({ data }) => {
    const handler = _pending.get(data.requestId)
    _pending.delete(data.requestId)
    if (!handler) return
    if (data.error) handler.reject(new Error(data.error))
    else            handler.resolve(data.scores)
  }
  _worker.onerror = (err) => {
    // Terminate worker on unrecoverable error; flag for heuristic fallback
    console.warn('[gRNA Predictor] Worker error:', err.message)
    _onnxFailed = true
    for (const { reject } of _pending.values()) reject(new Error(err.message))
    _pending.clear()
    _worker.terminate()
    _worker = null
  }
  return _worker
}

function runInWorker(sequences, thirtyMers, modelUrl) {
  return new Promise((resolve, reject) => {
    const id = _reqId++
    _pending.set(id, { resolve, reject })
    getWorker().postMessage({ requestId: id, sequences, thirtyMers, modelUrl })
  })
}

// ---------------------------------------------------------------------------
// Off-target specificity heuristic — JS port of backend/app/services/off_target.py
// Returns float in [0.0, 1.0]: 1.0 = highly specific, 0.0 = high off-target risk
// ---------------------------------------------------------------------------
function _reverseComplement(seq) {
  const comp = { A: 'T', T: 'A', C: 'G', G: 'C' }
  return seq.split('').reverse().map(b => comp[b] || b).join('')
}

function _hasHairpin(seq, minStem = 4) {
  for (let i = 0; i <= seq.length - minStem; i++) {
    const stem = seq.slice(i, i + minStem)
    const rc   = _reverseComplement(stem)
    const j    = seq.indexOf(rc)
    if (j !== -1 && j !== i) return true
  }
  return false
}

function specificityScore(seq20) {
  const seq  = seq20.toUpperCase().slice(0, 20)
  const seed = seq.slice(-12)

  const seedAt  = seed.split('').filter(b => b === 'A' || b === 'T').length / 12.0
  const seedPen = seedAt * 0.28

  const gc = seq.split('').filter(b => b === 'G' || b === 'C').length / 20.0
  let gcPen = 0.0
  if (gc < 0.25 || gc > 0.75)       gcPen = 0.20
  else if (gc < 0.35 || gc > 0.65)  gcPen = 0.10
  else if (gc < 0.40 || gc > 0.60)  gcPen = 0.05

  let gcRun = 0
  for (const b of seq.slice(-6).split('').reverse()) {
    if (b === 'G' || b === 'C') gcRun++
    else break
  }
  const gcRunPen = Math.min(0.20, Math.max(0.0, gcRun - 2) * 0.08)

  let hpPen = 0.0
  for (const b of ['A', 'C', 'G', 'T']) {
    if (seq.includes(b.repeat(4))) hpPen += 0.08
  }

  const hairpinPen = _hasHairpin(seq, 4) ? 0.12 : 0.0
  const gqPen      = seed.includes('GGG') ? 0.08 : 0.0

  return Math.min(1.0, Math.max(0.0,
    1.0 - seedPen - gcPen - gcRunPen - hpPen - hairpinPen - gqPen
  ))
}

function cutSite(position, strand, pam) {
  const isCas12a = CAS12A_PAMS.has(pam.toUpperCase())
  const offset   = isCas12a ? 18 : (strand === '+' ? 17 : 3)
  return position + offset + 1
}

function proximityScore(distance) {
  return Math.exp(-(distance ** 2) / (2.0 * SIGMA ** 2))
}

function heuristicScores(candidates) {
  return candidates.map(c => {
    const seq      = c.sequence
    const gc       = seq.split('').filter(b => b === 'G' || b === 'C').length / 20.0
    const gcScore  = Math.max(0.1, 1.0 - Math.abs(gc - 0.55) * 2.0)
    const seedGc   = seq.slice(-12).split('').filter(b => b === 'G' || b === 'C').length / 12.0
    const seedScore = Math.max(0.1, 1.0 - Math.abs(seedGc - 0.50) * 2.0)
    return 0.6 * gcScore + 0.4 * seedScore
  })
}

/**
 * Run in-browser prediction — mirrors the FastAPI /predict response shape.
 *
 * @param {string}      sequence        - DNA sequence
 * @param {string}      pam             - PAM (NGG / NAG / NNGRRT / TTTV)
 * @param {number}      topN            - Number of top results to return
 * @param {number|null} targetPosition  - 1-indexed target for proximity ranking
 * @param {number}      proximityWeight - w ∈ [0,1]
 * @param {string}      modelBase       - Vite BASE_URL (for model file path)
 * @returns {object} PredictResponse-shaped object
 */
export async function predictOffline(
  sequence,
  pam = 'NGG',
  topN = 5,
  targetPosition = null,
  proximityWeight = 0.4,
  modelBase = '/'
) {
  const seq = sequence.toUpperCase().replace(/\s+/g, '')

  // 1. Find candidate guides
  let candidates = findAllGRNAs(seq, pam)
  if (!candidates.length) {
    throw new Error(`No valid PAM (${pam}) sites found in the provided sequence.`)
  }

  // 2. Pre-filter by GC
  if (candidates.length > MAX_CANDIDATES) {
    const gcFiltered = candidates.filter(c => c.gc_content >= 0.35 && c.gc_content <= 0.75)
    candidates = (gcFiltered.length ? gcFiltered : candidates).slice(0, MAX_CANDIDATES)
  }

  // 3. Score via worker (ONNX) or heuristic fallback
  let rawScores
  let modelLabel = 'XGBoost-ONNX'

  if (!_onnxFailed) {
    try {
      const modelUrl  = `${modelBase}xgb_model.onnx`
      const sequences = candidates.map(c => c.sequence)
      // Extract 30-mer context: 4 bp upstream + 20 bp guide + 6 bp downstream
      const thirtyMers = candidates.map(c => {
        const start = c.position - 4
        const end   = c.position + 20 + 6
        if (start < 0 || end > seq.length) return ''
        return seq.slice(start, end)
      })
      rawScores = await runInWorker(sequences, thirtyMers, modelUrl)
    } catch (err) {
      console.warn('[gRNA Predictor] ONNX worker failed, using heuristic:', err.message)
      _onnxFailed = true
    }
  }

  if (_onnxFailed || !rawScores) {
    rawScores  = heuristicScores(candidates)
    modelLabel = 'Heuristic'
  }

  // 4. Compute cut sites, specificity, combined scores
  for (let i = 0; i < candidates.length; i++) {
    const c     = candidates[i]
    const score = Math.min(1.0, Math.max(0.0, rawScores[i]))
    c.score           = Math.round(score * 10000) / 10000
    c.model_used      = modelLabel
    c.cut_site        = cutSite(c.position, c.strand, pam)
    const spec        = specificityScore(c.sequence)
    c.off_target_score = Math.round(spec * 1000) / 1000
    const effAdj      = c.score * spec

    if (targetPosition !== null) {
      const dist           = Math.abs(c.cut_site - targetPosition)
      c.distance_to_target = dist
      c.combined_score     = Math.round(
        ((1.0 - proximityWeight) * effAdj + proximityWeight * proximityScore(dist))
        * 10000) / 10000
    } else {
      c.distance_to_target = null
      c.combined_score     = Math.round(effAdj * 10000) / 10000
    }
  }

  // 5. Rank and return top N
  const ranked = [...candidates]
    .sort((a, b) => b.combined_score - a.combined_score)
    .slice(0, topN)

  return {
    total_candidates: candidates.length,
    sequence_length:  seq.length,
    pam_used:         pam,
    target_position:  targetPosition,
    proximity_weight: targetPosition !== null ? proximityWeight : null,
    model_info:       _onnxFailed
      ? 'Heuristic scorer (ONNX unavailable in this browser)'
      : MODEL_INFO_TEXT,
    top_grnas: ranked.map((g, i) => ({
      rank:               i + 1,
      sequence:           g.sequence,
      pam_sequence:       g.pam_sequence,
      position:           g.position,
      strand:             g.strand,
      score:              g.score,
      gc_content:         Math.round(g.gc_content * 1000) / 1000,
      model_used:         g.model_used,
      cut_site:           g.cut_site,
      distance_to_target: g.distance_to_target,
      combined_score:     g.combined_score,
      off_target_score:   g.off_target_score,
    })),
  }
}
