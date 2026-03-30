/**
 * In-browser XGBoost inference via ONNX Runtime Web.
 * Used as a fallback when the local FastAPI backend is unreachable
 * (e.g. when running on GitHub Pages).
 *
 * The model is loaded from /xgb_model.onnx (served as a static asset).
 * WASM runtime files are loaded from the jsDelivr CDN.
 */
import * as ort from 'onnxruntime-web'
import { findAllGRNAs } from './sequenceParser.js'
import { extractFeaturesBatch } from './featureEngineering.js'

// Force single-threaded WASM to avoid loading the 25 MB JSEP variant.
// numThreads=1 → ORT loads ort-wasm-simd.wasm (~5 MB) instead of
// ort-wasm-simd-threaded.jsep.wasm (~25 MB), preventing tab OOM.
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths  = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/'

const SIGMA           = 50.0
const CAS12A_PAMS     = new Set(['TTTV'])
const MAX_CANDIDATES  = 200   // reduced from 300 to keep memory usage low
const MODEL_INFO_TEXT = 'XGBoost 450-dim (in-browser ONNX) — Doench 2016 + 2014, n=4,692'

let _session     = null   // cached ONNX session
let _onnxFailed  = false  // if ONNX OOMs, fall back to heuristic for rest of session

async function getSession(base) {
  if (_session) return _session
  const url = `${base}xgb_model.onnx`
  _session = await ort.InferenceSession.create(url, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'basic',   // reduces memory during model loading
  })
  return _session
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
  const seq = seq20.toUpperCase().slice(0, 20)
  const seed = seq.slice(-12)

  // 1. Seed region AT content
  const seedAt   = seed.split('').filter(b => b === 'A' || b === 'T').length / 12.0
  const seedPen  = seedAt * 0.28

  // 2. Global GC content
  const gc = seq.split('').filter(b => b === 'G' || b === 'C').length / 20.0
  let gcPen = 0.0
  if (gc < 0.25 || gc > 0.75)       gcPen = 0.20
  else if (gc < 0.35 || gc > 0.65)  gcPen = 0.10
  else if (gc < 0.40 || gc > 0.60)  gcPen = 0.05

  // 3. PAM-proximal GC run (last 6 bp)
  let gcRun = 0
  for (const b of seq.slice(-6).split('').reverse()) {
    if (b === 'G' || b === 'C') gcRun++
    else break
  }
  const gcRunPen = Math.min(0.20, Math.max(0.0, gcRun - 2) * 0.08)

  // 4. Homopolymer runs
  let hpPen = 0.0
  for (const b of ['A', 'C', 'G', 'T']) {
    if (seq.includes(b.repeat(4))) hpPen += 0.08
  }

  // 5. Hairpin self-complementarity
  const hairpinPen = _hasHairpin(seq, 4) ? 0.12 : 0.0

  // 6. G-quadruplex risk (GGG in seed)
  const gqPen = seed.includes('GGG') ? 0.08 : 0.0

  return Math.min(1.0, Math.max(0.0,
    1.0 - seedPen - gcPen - gcRunPen - hpPen - hairpinPen - gqPen
  ))
}

function cutSite(position, strand, pam) {
  const isCas12a = CAS12A_PAMS.has(pam.toUpperCase())
  let offset
  if (isCas12a) {
    offset = 18
  } else {
    offset = strand === '+' ? 17 : 3
  }
  return position + offset + 1  // 1-indexed
}

function proximityScore(distance) {
  return Math.exp(-(distance ** 2) / (2.0 * SIGMA ** 2))
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

  // 2. Pre-filter by GC (mirrors backend pre-filter)
  if (candidates.length > MAX_CANDIDATES) {
    const gcFiltered = candidates.filter(c => c.gc_content >= 0.35 && c.gc_content <= 0.75)
    candidates = (gcFiltered.length ? gcFiltered : candidates).slice(0, MAX_CANDIDATES)
  }

  // 3. Score each candidate (ONNX → heuristic fallback)
  const N = candidates.length
  let rawScores
  let modelLabel = 'XGBoost-ONNX'

  if (!_onnxFailed) {
    try {
      const sequences  = candidates.map(c => c.sequence)
      const thirtyMers = candidates.map(() => '')
      const features   = extractFeaturesBatch(sequences, thirtyMers)
      const session    = await getSession(modelBase)
      const tensor     = new ort.Tensor('float32', features, [N, 450])
      const feeds      = { [session.inputNames[0]]: tensor }
      const results    = await session.run(feeds)
      rawScores        = results[session.outputNames[0]].data
    } catch (err) {
      console.warn('[gRNA Predictor] ONNX inference failed, using heuristic scorer:', err.message)
      _onnxFailed = true
      _session    = null  // release cached session
    }
  }

  if (_onnxFailed || !rawScores) {
    // Heuristic fallback: GC-optimality + seed GC score
    rawScores  = new Float32Array(N)
    modelLabel = 'Heuristic'
    for (let i = 0; i < N; i++) {
      const seq = candidates[i].sequence
      const gc  = seq.split('').filter(b => b === 'G' || b === 'C').length / 20.0
      // GC score: optimal 0.40–0.70, peak at 0.55
      const gcScore  = Math.max(0.1, 1.0 - Math.abs(gc - 0.55) * 2.0)
      // Seed GC (last 12 bp)
      const seedGc   = seq.slice(-12).split('').filter(b => b === 'G' || b === 'C').length / 12.0
      const seedScore = Math.max(0.1, 1.0 - Math.abs(seedGc - 0.50) * 2.0)
      rawScores[i] = 0.6 * gcScore + 0.4 * seedScore
    }
  }

  // 4. Compute cut sites, proximity, combined scores
  for (let i = 0; i < N; i++) {
    const c = candidates[i]
    const score = Math.min(1.0, Math.max(0.0, rawScores[i]))
    c.score      = Math.round(score * 10000) / 10000
    c.model_used = modelLabel
    c.cut_site   = cutSite(c.position, c.strand, pam)
    const spec   = specificityScore(c.sequence)
    c.off_target_score = Math.round(spec * 1000) / 1000

    // Efficiency adjusted by off-target specificity (multiplicative penalty)
    const effAdj = c.score * spec

    if (targetPosition !== null) {
      const dist = Math.abs(c.cut_site - targetPosition)
      c.distance_to_target = dist
      c.combined_score = Math.round(
        ((1.0 - proximityWeight) * effAdj + proximityWeight * proximityScore(dist))
        * 10000) / 10000
    } else {
      c.distance_to_target = null
      // No target: rank by specificity-adjusted efficiency
      c.combined_score = Math.round(effAdj * 10000) / 10000
    }
  }

  // 6. Rank and return top N — always by combined_score (incorporates off-target)
  const ranked = [...candidates].sort((a, b) => b.combined_score - a.combined_score).slice(0, topN)

  return {
    total_candidates:  candidates.length,
    sequence_length:   seq.length,
    pam_used:          pam,
    target_position:   targetPosition,
    proximity_weight:  targetPosition !== null ? proximityWeight : null,
    model_info:        _onnxFailed
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
