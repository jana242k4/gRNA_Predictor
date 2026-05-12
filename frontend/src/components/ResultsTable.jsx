import React, { useState } from 'react'
import OmicsPanel from './OmicsPanel'

function exportCSV(guides, filename = 'omicscrispr_results.csv') {
  const headers = ['Rank','Sequence','PAM','Efficiency%','GC%','Specificity%','Cut site','Distance (bp)','Score','Strand','Frameshift est.']
  const rows = guides.map((g, i) => [
    i + 1,
    g.sequence,
    g.pam_sequence ?? g.pam ?? '',
    ((g.efficiency_score ?? g.score ?? 0) * 100).toFixed(1),
    (g.gc_content ?? 50).toFixed(1),
    ((g.specificity_score ?? 1) * 100).toFixed(1),
    g.cut_site ?? '',
    g.distance_to_target ?? '',
    (g.combined_score ?? g.efficiency_score ?? 0).toFixed(4),
    g.strand ?? '+',
    '~67%',
  ])
  const csv = [headers, ...rows].map(r => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

function scoreColor(score) {
  if (score >= 0.65) return 'badge-high'
  if (score >= 0.40) return 'badge-medium'
  return 'badge-low'
}

function gcColor(gc) {
  if (gc >= 40 && gc <= 70) return 'text-on-surface'
  return 'text-error'
}

function ExpandButton({ open, onClick }) {
  return (
    <button
      onClick={onClick}
      className="w-7 h-7 flex items-center justify-center rounded-full hover:bg-surface-container-high transition-colors"
      title={open ? 'Collapse' : 'Explain score + omics data'}
    >
      <span className="material-symbols-outlined text-sm text-on-surface-variant">
        {open ? 'expand_less' : 'expand_more'}
      </span>
    </button>
  )
}

function SeqCell({ sequence, pam }) {
  return (
    <span className="seq-mono text-xs text-on-surface">
      {sequence}
      <span className="pam-highlight ml-0.5">{pam}</span>
    </span>
  )
}

// ── Per-guide score breakdown ──────────────────────────────────────────────────

function _gc(seq) {
  if (!seq || !seq.length) return 0
  return [...seq].filter(b => b === 'G' || b === 'C').length / seq.length
}

function Factor({ label, value, ok, warn, note }) {
  const valColor = warn ? 'text-error' : ok ? 'text-primary' : 'text-on-surface'
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-on-surface-variant">{label}</span>
      <span className={`text-xs font-mono font-semibold ${valColor}`}>{value}</span>
      <span className="text-xs text-on-surface-variant/60">{note}</span>
    </div>
  )
}

function ScoreBreakdown({ guide }) {
  const seq  = (guide.sequence || '').toUpperCase()
  const gc   = guide.gc_content ?? _gc(seq) * 100
  const seedGC  = _gc(seq.slice(-12)) * 100
  const gcClamp = _gc(seq.slice(-4)) * 100
  const polyT   = seq.includes('TTTT')
  const proxGC  = _gc(seq.slice(10, 20)) * 100
  const spec    = (guide.specificity_score ?? 1) * 100

  const factors = [
    {
      label: 'GC content',
      value: `${gc.toFixed(0)}%`,
      ok:   gc >= 40 && gc <= 70,
      warn: gc < 25 || gc > 85,
      note: '40–70% optimal (Doench 2016)',
    },
    {
      label: "Seed GC (3′ 12 bp)",
      value: `${seedGC.toFixed(0)}%`,
      ok:   seedGC >= 35 && seedGC <= 75,
      warn: seedGC < 20 || seedGC > 90,
      note: 'Cas9 binding seed region',
    },
    {
      label: "GC clamp (3′ 4 bp)",
      value: `${gcClamp.toFixed(0)}%`,
      ok:   gcClamp >= 50,
      warn: gcClamp === 0,
      note: "G/C at 3′ end stabilises R-loop",
    },
    {
      label: 'Poly-T stretch',
      value: polyT ? 'Detected ⚠' : 'None',
      ok:   !polyT,
      warn: polyT,
      note: 'TTTT → Pol-III termination',
    },
    {
      label: 'PAM-proximal GC',
      value: `${proxGC.toFixed(0)}%`,
      ok:   proxGC >= 40,
      warn: proxGC < 20,
      note: 'Positions 11–20 (seed window)',
    },
    {
      label: 'CFD specificity',
      value: `${spec.toFixed(0)}%`,
      ok:   spec >= 80,
      warn: spec < 50,
      note: 'Doench 2016 mismatch matrix',
    },
  ]

  const specLabel = spec >= 80 ? 'Low off-target risk'
                  : spec >= 50 ? 'Moderate — verify key sites'
                  :              'High — experimental validation required'
  const specColor = spec >= 80 ? 'text-primary' : spec >= 50 ? 'text-tertiary' : 'text-error'

  return (
    <div className="flex flex-col gap-4">
      {/* Score factors grid */}
      <div>
        <p className="text-xs font-medium text-on-surface-variant uppercase tracking-wide mb-2">
          Score factors
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-3">
          {factors.map(f => <Factor key={f.label} {...f} />)}
        </div>
      </div>

      {/* Off-target + repair outcome */}
      <div className="flex flex-wrap gap-6 pt-2 border-t border-outline-variant/30">
        <div className="flex flex-col gap-0.5">
          <span className="text-xs text-on-surface-variant uppercase tracking-wide font-medium">Off-target</span>
          <span className={`text-xs font-semibold ${specColor}`}>{specLabel}</span>
          <span className="text-xs text-on-surface-variant/60">Sequence-intrinsic only — no genome-wide search</span>
        </div>

        <div className="flex flex-col gap-0.5">
          <span className="text-xs text-on-surface-variant uppercase tracking-wide font-medium">Frameshift probability</span>
          <span className="text-xs font-semibold text-on-surface">~67% (typical SpCas9 NHEJ)</span>
          <span className="text-xs text-on-surface-variant/60">Higher microhomology → more in-frame deletions</span>
        </div>

        <div className="flex flex-col gap-0.5">
          <span className="text-xs text-on-surface-variant uppercase tracking-wide font-medium">Prediction uncertainty</span>
          <span className="text-xs font-semibold text-on-surface">Scores ≤ 0.05 apart are not meaningfully different</span>
          <span className="text-xs text-on-surface-variant/60">Model Pearson r = 0.708 on independent test set</span>
        </div>
      </div>

      {/* Organism scope notice */}
      <div className="flex items-start gap-2 bg-surface-container px-3 py-2 rounded-xl text-xs text-on-surface-variant">
        <span className="material-symbols-outlined text-sm mt-0.5 flex-shrink-0">info</span>
        <span>
          Efficiency model trained on <strong className="text-on-surface">human cell lines</strong> (HEK293, K562) using Doench 2016 + 2014 + Kim 2019 data.
          Predictions for non-human organisms (plants, zebrafish, bacteria) are less reliable.
          Wet-lab validation is always required before experimental use.
        </span>
      </div>
    </div>
  )
}


// ── Main table ─────────────────────────────────────────────────────────────────

export default function ResultsTable({ data, inputSeq }) {
  const [expanded, setExpanded] = useState({})

  const guides    = data?.top_grnas ?? []
  const offline   = data?.offline === true
  const modelInfo = data?.model_info ?? {}

  const toggle = (idx) => setExpanded(prev => ({ ...prev, [idx]: !prev[idx] }))

  if (guides.length === 0) {
    return (
      <div className="bg-surface-container-low rounded-3xl p-6 text-center text-on-surface-variant text-sm">
        No guide RNAs found. Try a longer sequence or different PAM.
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Summary bar */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-lg">check_circle</span>
          <span className="text-sm text-on-surface font-medium">{guides.length} guides found</span>
        </div>
        {offline && (
          <span className="px-2 py-0.5 rounded text-xs bg-tertiary-container/30 text-tertiary">
            Offline mode
          </span>
        )}
        {modelInfo.model && (
          <span className="text-xs text-on-surface-variant">{modelInfo.model}</span>
        )}
        {modelInfo.pearson_r && (
          <span className="text-xs text-on-surface-variant">r={modelInfo.pearson_r}</span>
        )}
        <span className="text-xs text-on-surface-variant/60 hidden sm:inline">
          Human cell lines · Expand any row to explain score
        </span>
        <button
          onClick={() => exportCSV(guides)}
          className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium
                     border border-outline-variant text-on-surface-variant
                     hover:bg-surface-container-high transition-colors"
        >
          <span className="material-symbols-outlined text-sm">download</span>
          Export CSV
        </button>
      </div>

      {/* Table */}
      <div className="bg-surface-container-low rounded-3xl overflow-hidden">
        {/* Header */}
        <div className="hidden sm:grid sm:grid-cols-[2rem_1fr_5rem_3.5rem_3.5rem_4rem_5rem_5rem_2.5rem_2rem]
                        gap-x-3 px-4 py-2.5 border-b border-outline-variant
                        text-xs font-medium text-on-surface-variant uppercase tracking-wide">
          <span>#</span>
          <span>Sequence + PAM</span>
          <span>Effic.</span>
          <span>GC%</span>
          <span>Spec.</span>
          <span>Cut</span>
          <span>Dist.</span>
          <span>Score</span>
          <span>Str.</span>
          <span></span>
        </div>

        {guides.map((g, i) => (
          <React.Fragment key={i}>
            <div
              className={`grid grid-cols-[2rem_1fr_5rem_3.5rem_3.5rem_4rem_5rem_5rem_2.5rem_2rem]
                          gap-x-3 px-4 py-3 items-center border-b border-outline-variant/40
                          hover:bg-surface-container-high/40 transition-colors
                          ${expanded[i] ? 'bg-surface-container-high/20' : ''}`}
            >
              <span className="text-xs text-on-surface-variant tabular-nums">{i + 1}</span>

              <SeqCell sequence={g.sequence} pam={g.pam_sequence ?? g.pam ?? ''} />

              {/* Efficiency mini-bar */}
              <div className="flex items-center gap-1.5">
                <div className="flex-1 h-1 bg-surface-container-highest rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-primary-container"
                    style={{ width: `${Math.round((g.efficiency_score ?? g.score ?? 0) * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-on-surface tabular-nums w-7 text-right">
                  {((g.efficiency_score ?? g.score ?? 0) * 100).toFixed(0)}
                </span>
              </div>

              <span className={`text-xs font-mono tabular-nums ${gcColor(g.gc_content ?? 50)}`}>
                {(g.gc_content ?? 50).toFixed(0)}%
              </span>

              <span className="text-xs font-mono tabular-nums text-on-surface">
                {((g.specificity_score ?? 1) * 100).toFixed(0)}%
              </span>

              <span className="text-xs font-mono tabular-nums text-on-surface-variant">
                {g.cut_site ?? '—'}
              </span>

              <span className="text-xs font-mono tabular-nums text-on-surface-variant">
                {g.distance_to_target != null ? g.distance_to_target : '—'}
              </span>

              <span className={`text-xs font-mono tabular-nums px-2 py-0.5 rounded ${scoreColor(g.combined_score ?? g.efficiency_score ?? 0)}`}>
                {(g.combined_score ?? g.efficiency_score ?? 0).toFixed(3)}
              </span>

              <span className="text-xs text-on-surface-variant">{g.strand ?? '+'}</span>

              <ExpandButton open={!!expanded[i]} onClick={() => toggle(i)} />
            </div>

            {expanded[i] && (
              <div className="px-6 py-4 bg-surface-container/50 border-b border-outline-variant/40 animate-fade-up flex flex-col gap-5">
                <ScoreBreakdown guide={g} />
                <div className="border-t border-outline-variant/30 pt-4">
                  <p className="text-xs font-medium text-on-surface-variant uppercase tracking-wide mb-3">
                    Cell-type suitability (OmicsCRISPR)
                  </p>
                  <OmicsPanel sequence={g.sequence} />
                </div>
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {modelInfo.citation && (
        <p className="text-xs text-on-surface-variant px-1">{modelInfo.citation}</p>
      )}
    </div>
  )
}
