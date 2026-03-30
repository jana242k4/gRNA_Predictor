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

// Point WASM files to CDN so they are not bundled into the JS chunk
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/'

const SIGMA           = 50.0
const CAS12A_PAMS     = new Set(['TTTV'])
const MAX_CANDIDATES  = 300
const MODEL_INFO_TEXT = 'XGBoost 450-dim (in-browser ONNX) — Doench 2016 + 2014, n=4,692'

let _session = null  // cached ONNX session

async function getSession(base) {
  if (_session) return _session
  const url = `${base}xgb_model.onnx`
  _session = await ort.InferenceSession.create(url, {
    executionProviders: ['wasm'],
  })
  return _session
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

  // 3. Extract 450-dim features
  const sequences  = candidates.map(c => c.sequence)
  const thirtyMers = candidates.map(() => '')  // no 30-mer in user input
  const features   = extractFeaturesBatch(sequences, thirtyMers)  // Float32Array N×450

  // 4. Run ONNX inference
  const session = await getSession(modelBase)
  const N = candidates.length
  const tensor = new ort.Tensor('float32', features, [N, 450])

  // The input name from skl2onnx export is 'float_input'
  const inputName = session.inputNames[0]
  const feeds = { [inputName]: tensor }
  const results = await session.run(feeds)

  // Output name is typically 'variable' (skl2onnx regression)
  const outputName = session.outputNames[0]
  const rawScores = results[outputName].data  // Float32Array length N

  // 5. Compute cut sites, proximity, combined scores
  for (let i = 0; i < N; i++) {
    const c = candidates[i]
    const score = Math.min(1.0, Math.max(0.0, rawScores[i]))
    c.score      = Math.round(score * 10000) / 10000
    c.model_used = 'XGBoost-ONNX'
    c.cut_site   = cutSite(c.position, c.strand, pam)
    c.off_target_score = Math.round(
      Math.max(0, 0.8 - (c.sequence.slice(-12).split('').filter(b => b==='A'||b==='T').length) * 0.04)
      * 1000) / 1000

    if (targetPosition !== null) {
      const dist = Math.abs(c.cut_site - targetPosition)
      c.distance_to_target = dist
      c.combined_score = Math.round(
        ((1.0 - proximityWeight) * c.score + proximityWeight * proximityScore(dist))
        * 10000) / 10000
    } else {
      c.distance_to_target = null
      c.combined_score = null
    }
  }

  // 6. Rank and return top N
  const sortKey = targetPosition !== null ? 'combined_score' : 'score'
  const ranked = [...candidates].sort((a, b) => b[sortKey] - a[sortKey]).slice(0, topN)

  return {
    total_candidates:  candidates.length,
    sequence_length:   seq.length,
    pam_used:          pam,
    target_position:   targetPosition,
    proximity_weight:  targetPosition !== null ? proximityWeight : null,
    model_info:        MODEL_INFO_TEXT,
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
