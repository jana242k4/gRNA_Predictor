/**
 * In-browser gRNA prediction (fallback when backend is unreachable).
 *
 * Uses pure-JavaScript XGBoost tree traversal — no WebAssembly, no Workers.
 * Fetches xgb_trees.json (~296 KB) once and caches it.  Inference is
 * time-sliced (25 candidates per chunk, yields every ~3 ms) so the main
 * thread stays responsive.
 *
 * This path is only reached if the FastAPI backend (Render or localhost)
 * returns a network error.  Timeouts are surfaced directly so the user
 * knows to wait for the Render cold-start.
 */
import { findAllGRNAs }         from './sequenceParser.js'
import { extractFeaturesBatch }  from './featureEngineering.js'
import { xgbPredict }            from './xgbPredictor.js'

const SIGMA          = 50.0
const CAS12A_PAMS    = new Set(['TTTV'])
const MAX_CANDIDATES = 200
const MODEL_INFO     = 'XGBoost 452-dim (in-browser JS) — Doench 2016 + 2014 + Kim2019, n=11,991'

// ---------------------------------------------------------------------------
// Off-target specificity heuristic (used in all modes)
// ---------------------------------------------------------------------------
function _rc(seq) {
  const comp = { A: 'T', T: 'A', C: 'G', G: 'C' }
  return seq.split('').reverse().map(b => comp[b] || b).join('')
}

function specificityScore(seq20) {
  const seq  = seq20.toUpperCase().slice(0, 20)
  const seed = seq.slice(-12)

  const seedPen = (seed.split('').filter(b => b === 'A' || b === 'T').length / 12.0) * 0.28

  const gc = seq.split('').filter(b => b === 'G' || b === 'C').length / 20.0
  let gcPen = 0.0
  if      (gc < 0.25 || gc > 0.75) gcPen = 0.20
  else if (gc < 0.35 || gc > 0.65) gcPen = 0.10
  else if (gc < 0.40 || gc > 0.60) gcPen = 0.05

  let gcRun = 0
  for (const b of seq.slice(-6).split('').reverse()) {
    if (b === 'G' || b === 'C') gcRun++; else break
  }
  const gcRunPen = Math.min(0.20, Math.max(0.0, gcRun - 2) * 0.08)

  let hpPen = 0.0
  for (const b of ['A', 'C', 'G', 'T']) if (seq.includes(b.repeat(4))) hpPen += 0.08

  const hairpinPen = (() => {
    for (let i = 0; i <= seq.length - 4; i++) {
      const rc = _rc(seq.slice(i, i + 4))
      const j  = seq.indexOf(rc)
      if (j !== -1 && j !== i) return 0.12
    }
    return 0.0
  })()
  const gqPen = seed.includes('GGG') ? 0.08 : 0.0

  // Sequence complexity: low-entropy / repetitive guides hit more loci
  // (mirrors backend off_target.py _complexity_penalty)
  const _kmerEntropy = (s, k) => {
    const kmers = {}
    for (let i = 0; i <= s.length - k; i++) {
      const km = s.slice(i, i + k)
      kmers[km] = (kmers[km] || 0) + 1
    }
    const total = s.length - k + 1
    const H = -Object.values(kmers).reduce((acc, c) => acc + (c/total) * Math.log2(c/total), 0)
    return total > 1 ? H / Math.log2(total) : 1.0
  }
  let dimerPen = 0.0
  for (let i = 0; i < seq.length - 1; i++) {
    const d = seq.slice(i, i + 2)
    if (seq.includes(d + d + d)) { dimerPen = 0.10; break }
  }
  const entropy     = _kmerEntropy(seq, 3)
  const entropyPen  = entropy < 0.70 ? Math.max(0.0, (0.70 - entropy) * 0.20) : 0.0
  const complexityPen = dimerPen + entropyPen

  return Math.min(1.0, Math.max(0.0, 1.0 - seedPen - gcPen - gcRunPen - hpPen - hairpinPen - gqPen - complexityPen))
}

function heuristicScore(seq) {
  const gc       = seq.split('').filter(b => b === 'G' || b === 'C').length / 20.0
  const seedGc   = seq.slice(-12).split('').filter(b => b === 'G' || b === 'C').length / 12.0
  const polyT    = seq.includes('TTTT') ? 0.3 : 0.0
  return Math.max(0.05,
    0.6 * Math.max(0.1, 1.0 - Math.abs(gc - 0.55) * 2.0) +
    0.4 * Math.max(0.1, 1.0 - Math.abs(seedGc - 0.50) * 2.0) - polyT
  )
}

function cutSite(position, strand, pam) {
  const isCas12a = CAS12A_PAMS.has(pam.toUpperCase())
  return position + (isCas12a ? 18 : strand === '+' ? 17 : 3) + 1
}

function proximityScore(d) {
  return Math.exp(-(d ** 2) / (2.0 * SIGMA ** 2))
}

// ---------------------------------------------------------------------------
// Main export — mirrors FastAPI /predict response shape
// ---------------------------------------------------------------------------
export async function predictOffline(
  sequence, pam = 'NGG', topN = 5,
  targetPosition = null, proximityWeight = 0.4, modelBase = '/'
) {
  const seq = sequence.toUpperCase().replace(/\s+/g, '')

  let candidates = findAllGRNAs(seq, pam)
  if (!candidates.length)
    throw new Error(`No valid PAM (${pam}) sites found in the provided sequence.`)

  if (candidates.length > MAX_CANDIDATES) {
    const gc = candidates.filter(c => c.gc_content >= 0.35 && c.gc_content <= 0.75)
    candidates = (gc.length ? gc : candidates).slice(0, MAX_CANDIDATES)
  }

  // XGBoost inference (time-sliced, non-blocking)
  let rawScores, modelLabel, modelInfo
  try {
    const seqs    = candidates.map(c => c.sequence)
    const tm30s   = candidates.map(c => {
      const s = c.position - 4, e = c.position + 26
      return (s >= 0 && e <= seq.length) ? seq.slice(s, e) : ''
    })
    const feats   = extractFeaturesBatch(seqs, tm30s)
    const raw     = await xgbPredict(feats, modelBase)
    rawScores  = Array.from(raw)
    modelLabel = 'XGBoost-JS'
    modelInfo  = MODEL_INFO
  } catch (err) {
    console.warn('[gRNA Predictor] XGBoost fallback failed, using heuristic:', err.message)
    rawScores  = candidates.map(c => heuristicScore(c.sequence))
    modelLabel = 'Heuristic'
    modelInfo  = 'Heuristic scorer (XGBoost model unavailable)'
  }

  for (let i = 0; i < candidates.length; i++) {
    const c    = candidates[i]
    const sc   = Math.min(1.0, Math.max(0.0, rawScores[i]))
    c.score            = Math.round(sc * 10000) / 10000
    c.model_used       = modelLabel
    c.cut_site         = cutSite(c.position, c.strand, pam)
    const spec         = specificityScore(c.sequence)
    c.off_target_score = Math.round(spec * 1000) / 1000
    const effAdj       = sc * spec

    if (targetPosition !== null) {
      const d            = Math.abs(c.cut_site - targetPosition)
      c.distance_to_target = d
      c.combined_score   = Math.round(
        ((1 - proximityWeight) * effAdj + proximityWeight * proximityScore(d)) * 10000) / 10000
    } else {
      c.distance_to_target = null
      c.combined_score     = Math.round(effAdj * 10000) / 10000
    }
  }

  const ranked = [...candidates].sort((a, b) => b.combined_score - a.combined_score).slice(0, topN)

  return {
    total_candidates: candidates.length,
    sequence_length:  seq.length,
    pam_used:         pam,
    target_position:  targetPosition,
    proximity_weight: targetPosition !== null ? proximityWeight : null,
    model_info:       modelInfo,
    top_grnas: ranked.map((g, i) => ({
      rank: i + 1, sequence: g.sequence, pam_sequence: g.pam_sequence,
      position: g.position, strand: g.strand, score: g.score,
      gc_content: Math.round(g.gc_content * 1000) / 1000,
      model_used: g.model_used, cut_site: g.cut_site,
      distance_to_target: g.distance_to_target,
      combined_score: g.combined_score, off_target_score: g.off_target_score,
    })),
  }
}
