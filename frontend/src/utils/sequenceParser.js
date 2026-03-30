/**
 * PAM site detection and guide RNA extraction.
 * JavaScript port of backend/app/services/sequence_parser.py
 *
 * Supports:
 *   NGG    — SpCas9,  PAM 3' of guide
 *   NAG    — SpCas9 alt (reduced efficiency), PAM 3'
 *   NNGRRT — SaCas9, PAM 3' of guide
 *   TTTV   — Cas12a, PAM 5' of guide
 */

const GUIDE_LENGTH = 20
const MAX_N_BASES  = 2   // guides with more than 2 N bases are skipped

const COMPLEMENT = {A:'T', T:'A', G:'C', C:'G', N:'N'}

// IUPAC expansion (used to build RegExp patterns)
const IUPAC = {
  N:'[ACGT]', R:'[AG]', Y:'[CT]', S:'[GC]', W:'[AT]',
  K:'[GT]',   M:'[AC]', B:'[CGT]',D:'[AGT]',H:'[ACT]',V:'[ACG]',
  A:'A', C:'C', G:'G', T:'T',
}

function pamToRegex(pam) {
  return pam.toUpperCase().split('').map(c => IUPAC[c] || c).join('')
}

export function reverseComplement(seq) {
  return seq.toUpperCase().split('').reverse().map(b => COMPLEMENT[b] || 'N').join('')
}

export function gcContent(seq) {
  if (!seq.length) return 0.0
  let gc = 0
  for (const b of seq.toUpperCase()) if (b === 'G' || b === 'C') gc++
  return gc / seq.length
}


// ── Site finders ──────────────────────────────────────────────────────────────

/**
 * Find Cas9 sites: [20bp guide][PAM]
 * Returns array of {guide, pamSeq, pos, strand}
 */
function findCas9Sites(seq, pam, strand) {
  const s = seq.toUpperCase()
  const pamRe = pamToRegex(pam)
  // Lookahead so overlapping matches are found
  const fullRe = new RegExp(`(?=([ACGTN]{${GUIDE_LENGTH}}(${pamRe})))`, 'g')
  const results = []
  let m
  while ((m = fullRe.exec(s)) !== null) {
    const guide    = m[1].slice(0, GUIDE_LENGTH)
    const pamFound = m[2]
    const pos      = m.index
    const nCount   = (guide.match(/N/g) || []).length
    if (nCount > MAX_N_BASES) continue
    results.push({ guide, pamSeq: pamFound, pos, strand })
  }
  return results
}

/**
 * Find Cas12a sites: [PAM][20bp guide]
 * Returns array of {guide, pamSeq, pos, strand}
 * pos = start of GUIDE in the sequence (after PAM)
 */
function findCas12aSites(seq, pam, strand) {
  const s = seq.toUpperCase()
  const pamRe = pamToRegex(pam)
  const fullRe = new RegExp(`(?=((${pamRe})([ACGTN]{${GUIDE_LENGTH}})))`, 'g')
  const results = []
  let m
  while ((m = fullRe.exec(s)) !== null) {
    const pamFound = m[2]
    const guide    = m[3]
    const pos      = m.index + pamFound.length  // guide start
    const nCount   = (guide.match(/N/g) || []).length
    if (nCount > MAX_N_BASES) continue
    results.push({ guide, pamSeq: pamFound, pos, strand })
  }
  return results
}


// ── Public API ────────────────────────────────────────────────────────────────

const CAS12A_PAMS = new Set(['TTTV'])

/**
 * Find all gRNAs in a DNA sequence for the given PAM (both strands).
 * @param {string} sequence - Input DNA sequence
 * @param {string} pam      - PAM string (NGG/NAG/NNGRRT/TTTV)
 * @returns {Array} Array of candidate objects with sequence, pamSequence, position, strand, gcContent
 */
export function findAllGRNAs(sequence, pam = 'NGG') {
  const pamUpper  = pam.toUpperCase()
  const isCas12a  = CAS12A_PAMS.has(pamUpper)
  const finder    = isCas12a ? findCas12aSites : findCas9Sites
  const seqUpper  = sequence.toUpperCase()
  const seqLen    = seqUpper.length
  const candidates = []

  // Forward strand
  for (const site of finder(seqUpper, pamUpper, '+')) {
    candidates.push({
      sequence:    site.guide,
      pam_sequence: site.pamSeq,
      position:    site.pos,
      strand:      '+',
      gc_content:  gcContent(site.guide),
    })
  }

  // Reverse strand
  const rcSeq = reverseComplement(seqUpper)
  for (const site of finder(rcSeq, pamUpper, '-')) {
    // pos is guide start in RC space; both Cas9 and Cas12a use same formula
    // (for Cas12a, pos already skips the PAM via findCas12aSites)
    const fwdPos = seqLen - site.pos - GUIDE_LENGTH
    candidates.push({
      sequence:    site.guide,
      pam_sequence: site.pamSeq,
      position:    Math.max(0, fwdPos),
      strand:      '-',
      gc_content:  gcContent(site.guide),
    })
  }

  return candidates
}
