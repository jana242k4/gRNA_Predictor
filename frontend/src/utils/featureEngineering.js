/**
 * 450-dim feature engineering for gRNA efficiency prediction.
 * JavaScript port of backend/app/services/feature_engineering.py
 *
 * Feature vector layout:
 *   [0:80]    Positional one-hot (20 positions × 4 bases)
 *   [80]      GC content
 *   [81]      Normalised Tm (SantaLucia 1998, full guide)
 *   [82:98]   Dinucleotide frequencies (16 pairs)
 *   [98]      Seed region GC (last 12 bp)
 *   [99]      Poly-T flag (TTTT present)
 *   [100:404] Position-specific dinucleotide one-hot (19 × 16)
 *   [404:420] Upstream context one-hot (4 bp)
 *   [420:444] Downstream context one-hot (6 bp)
 *   [444]     GC clamp (last 4 bp)
 *   [445]     RNA hairpin proxy
 *   [446]     Microhomology (from 30-mer)
 *   [447]     Tm PAM-distal (guide[0:8])
 *   [448]     Tm PAM-proximal 8 bp (guide[12:20])
 *   [449]     Tm full 30-mer context
 */

// SantaLucia 1998 DNA-DNA nearest-neighbor parameters
// [ΔH kcal/mol, ΔS cal/mol·K]
const NN = {
  AA:[-7.9,-22.2], AT:[-7.2,-20.4], TA:[-7.2,-21.3], CA:[-8.5,-22.7],
  GT:[-8.4,-22.4], CT:[-7.8,-21.0], GA:[-8.2,-22.2], CG:[-10.6,-27.2],
  GC:[-9.8,-24.4], GG:[-8.0,-19.9], AC:[-7.8,-21.0], TC:[-7.9,-22.2],
  AG:[-8.2,-22.2], TG:[-8.5,-22.7], TT:[-7.9,-22.2], CC:[-8.0,-19.9],
}
const R_GAS    = 1.987   // cal/mol·K
const CT_CONC  = 250e-9  // 250 nM strand concentration
const TM_MIN   = 40.0
const TM_MAX   = 80.0
const GUIDE_LEN = 20

const BASES    = ['A','C','G','T']
const DINUCS   = BASES.flatMap(a => BASES.map(b => a+b))  // 16 pairs
const BASE_IDX = Object.fromEntries(BASES.map((b,i) => [b,i]))
const DINUC_IDX = Object.fromEntries(DINUCS.map((d,i) => [d,i]))

const COMPLEMENT = {A:'T', T:'A', G:'C', C:'G', N:'N'}


// ── Tm calculation (SantaLucia 1998) ─────────────────────────────────────────

export function nearestNeighborTm(seq, oligoConcNM = 250.0) {
  const s = seq.toUpperCase()
  const n = s.length
  if (n < 2) return 0.0

  let dH = 0.0, dS = 0.0
  for (let i = 0; i < n - 1; i++) {
    const di = s[i] + s[i+1]
    if (di in NN) { dH += NN[di][0]; dS += NN[di][1] }
  }
  // Terminal corrections
  for (const end of [s[0], s[n-1]]) {
    if (end === 'A' || end === 'T') { dH += 2.3; dS += 4.1 }
    else if (end === 'G' || end === 'C') { dH += 0.1; dS -= 2.8 }
  }
  const CT = oligoConcNM * 1e-9
  const dHcal = dH * 1000.0
  const denom = dS + R_GAS * Math.log(CT / 4.0)
  if (Math.abs(denom) < 1e-10) return 0.0
  return dHcal / denom - 273.15
}

function tmNorm(seq) {
  if (!seq || seq.length < 4) return 0.0
  const tm = nearestNeighborTm(seq)
  return Math.min(1.0, Math.max(0.0, (tm - TM_MIN) / (TM_MAX - TM_MIN)))
}


// ── Individual feature functions ──────────────────────────────────────────────

function positionalOnehot(seq) {
  const vec = new Float32Array(GUIDE_LEN * 4)
  for (let i = 0; i < seq.length && i < GUIDE_LEN; i++) {
    const idx = BASE_IDX[seq[i]]
    if (idx !== undefined) vec[i * 4 + idx] = 1.0
  }
  return vec
}

function gcContent(seq) {
  if (!seq.length) return 0.0
  let gc = 0
  for (const b of seq) if (b === 'G' || b === 'C') gc++
  return gc / seq.length
}

function dinucFreq(seq) {
  const vec = new Float32Array(16)
  const n = seq.length - 1
  if (n <= 0) return vec
  for (let i = 0; i < n; i++) {
    const idx = DINUC_IDX[seq[i] + seq[i+1]]
    if (idx !== undefined) vec[idx]++
  }
  for (let i = 0; i < 16; i++) vec[i] /= n
  return vec
}

function positionalDinucOnehot(seq) {
  const vec = new Float32Array(19 * 16)
  for (let i = 0; i < Math.min(19, seq.length - 1); i++) {
    const idx = DINUC_IDX[seq[i] + seq[i+1]]
    if (idx !== undefined) vec[i * 16 + idx] = 1.0
  }
  return vec
}

function contextOnehot(ctx, length) {
  const vec = new Float32Array(length * 4)
  const s = (ctx || '').toUpperCase()
  for (let i = 0; i < Math.min(length, s.length); i++) {
    const idx = BASE_IDX[s[i]]
    if (idx !== undefined) vec[i * 4 + idx] = 1.0
  }
  return vec
}

function gcClamp(seq) {
  const tail = seq.slice(-4).toUpperCase()
  let gc = 0
  for (const b of tail) if (b === 'G' || b === 'C') gc++
  return gc / 4.0
}

function hairpinProxy(seq) {
  const s = seq.toUpperCase()
  const n = s.length
  let maxStem = 0
  for (let i = 0; i < n - 8; i++) {
    for (let stemLen = 2; i + stemLen * 2 + 4 <= n; stemLen++) {
      const arm5 = s.slice(i, i + stemLen)
      const arm3Start = i + stemLen + 4
      const arm3 = s.slice(arm3Start, arm3Start + stemLen)
      // Check if arm3 is reverse complement of arm5
      let rc = ''
      for (let k = arm3.length - 1; k >= 0; k--) rc += COMPLEMENT[arm3[k]] || 'N'
      if (arm5 === rc) maxStem = Math.max(maxStem, stemLen)
    }
  }
  return Math.min(1.0, maxStem / 10.0)
}

function microhomology(thirtyMer) {
  if (!thirtyMer || thirtyMer.length < 27) return 0.0
  const tm = thirtyMer.toUpperCase()
  const cut = 21  // SpCas9 cut at guide pos 17 in the 30-mer (4+17=21)
  const left  = tm.slice(Math.max(0, cut - 6), cut)
  const right = tm.slice(cut, Math.min(tm.length, cut + 6))
  let maxMH = 0
  for (let length = 1; length <= Math.min(left.length, right.length); length++) {
    if (left.slice(-length) === right.slice(0, length)) maxMH = length
  }
  return Math.min(1.0, maxMH / 6.0)
}


// ── Main feature extractor ────────────────────────────────────────────────────

/**
 * Extract 450-dim feature vector for a single gRNA.
 * @param {string} sequence   - 20-bp guide sequence
 * @param {string} thirtyMer  - Optional 30-mer context (4bp up + 20 guide + 6bp down)
 * @returns {Float32Array} length 450
 */
export function extractFeatures(sequence, thirtyMer = '') {
  const seq = sequence.toUpperCase().slice(0, GUIDE_LEN).padEnd(GUIDE_LEN, 'N')

  const onehot   = positionalOnehot(seq)                          // 80
  const gc       = new Float32Array([gcContent(seq)])             // 1
  const tm       = new Float32Array([tmNorm(seq)])                // 1
  const dinuc    = dinucFreq(seq)                                 // 16
  const seedGC   = new Float32Array([gcContent(seq.slice(-12))])  // 1
  const polyT    = new Float32Array([seq.includes('TTTT') ? 1.0 : 0.0]) // 1
  const posDinuc = positionalDinucOnehot(seq)                     // 304

  let upstream, downstream
  if (thirtyMer && thirtyMer.length >= 30) {
    const tm30 = thirtyMer.toUpperCase()
    upstream   = contextOnehot(tm30.slice(0, 4), 4)    // 16
    downstream = contextOnehot(tm30.slice(24), 6)      // 24
  } else {
    upstream   = new Float32Array(16)
    downstream = new Float32Array(24)
  }

  const gcClampV = new Float32Array([gcClamp(seq)])       // 1
  const hairpin  = new Float32Array([hairpinProxy(seq)])  // 1
  const mhScore  = new Float32Array([microhomology(thirtyMer || '')]) // 1

  const tmDistal    = new Float32Array([tmNorm(seq.slice(0, 8))])   // 1
  const tmProximal  = new Float32Array([tmNorm(seq.slice(12))])     // 1
  const tmCtx       = thirtyMer && thirtyMer.length >= 30
    ? new Float32Array([tmNorm(thirtyMer.toUpperCase())])
    : new Float32Array(1)                                            // 1

  // Concatenate all into a single Float32Array of length 450
  const parts = [onehot, gc, tm, dinuc, seedGC, polyT, posDinuc,
                 upstream, downstream, gcClampV, hairpin, mhScore,
                 tmDistal, tmProximal, tmCtx]
  const total = parts.reduce((s, p) => s + p.length, 0)
  const result = new Float32Array(total)
  let offset = 0
  for (const p of parts) { result.set(p, offset); offset += p.length }
  return result  // length 450
}

/**
 * Extract features for a batch of guides.
 * @param {string[]} sequences   - Array of 20-bp guide sequences
 * @param {string[]} thirtyMers  - Parallel array of 30-mer context strings
 * @returns {Float32Array} shape (N × 450), row-major
 */
export function extractFeaturesBatch(sequences, thirtyMers = null) {
  const tms = thirtyMers || sequences.map(() => '')
  const N   = sequences.length
  const result = new Float32Array(N * 450)
  for (let i = 0; i < N; i++) {
    result.set(extractFeatures(sequences[i], tms[i]), i * 450)
  }
  return result
}
