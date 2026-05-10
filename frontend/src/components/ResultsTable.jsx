import React, { useState } from 'react'
import OmicsPanel from './OmicsPanel'

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
      title={open ? 'Collapse' : 'Show omics data'}
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
          <span className="text-xs text-on-surface-variant">Pearson r={modelInfo.pearson_r}</span>
        )}
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
              <div className="px-6 pb-4 bg-surface-container/50 border-b border-outline-variant/40 animate-fade-up">
                <OmicsPanel sequence={g.sequence} />
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
