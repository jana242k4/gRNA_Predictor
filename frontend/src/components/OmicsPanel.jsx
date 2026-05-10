import React, { useEffect, useState } from 'react'
import { omicsPredict, omicsExplain } from '../services/api'

const CELL_COLORS = {
  K562:       '#cdbdff',
  T_cell_CD4: '#ffb688',
  T_cell_CD8: '#f4a0c8',
  NK_cell:    '#80cbc4',
  B_cell:     '#90caf9',
}

function SuitBar({ value, color }) {
  const pct = Math.round((value ?? 0.5) * 100)
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="text-xs font-mono text-on-surface-variant w-8 text-right">{pct}%</span>
    </div>
  )
}

function SpliceRiskBadge({ risk }) {
  if (risk == null) return null
  const level = risk >= 0.6 ? 'high' : risk >= 0.3 ? 'med' : 'low'
  const styles = {
    high: 'bg-error-container text-on-error-container',
    med:  'bg-tertiary-container/30 text-tertiary',
    low:  'bg-surface-container text-on-surface-variant',
  }
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${styles[level]}`}>
      Splice risk {Math.round(risk * 100)}%
    </span>
  )
}

export default function OmicsPanel({ sequence }) {
  const [data, setData]       = useState(null)
  const [explain, setExplain] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showIG, setShowIG]   = useState(false)

  useEffect(() => {
    if (!sequence) return
    setData(null)
    setExplain(null)
    setLoading(true)
    omicsPredict(sequence).then(d => {
      setData(d)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [sequence])

  const loadExplain = async () => {
    if (explain || !sequence) return
    const d = await omicsExplain(sequence, 'K562')
    setExplain(d)
    setShowIG(true)
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-3 text-sm text-on-surface-variant">
        <span className="w-4 h-4 border-2 border-outline border-t-primary rounded-full animate-spin" />
        Loading omics data…
      </div>
    )
  }

  if (!data) return (
    <p className="text-xs text-on-surface-variant py-2">
      Omics data not available for this guide (not in DepMap library).
    </p>
  )

  const spliceRow = data.predictions?.[0]
  const spliceRisk = spliceRow?.splice_risk
  const nearestDist = spliceRow?.nearest_splice_dist_bp
  const nearestType = spliceRow?.nearest_splice_type

  return (
    <div className="flex flex-col gap-4 py-2">
      {/* Meta row */}
      <div className="flex items-center gap-3 flex-wrap">
        {data.gene && (
          <span className="text-xs font-medium text-on-surface bg-surface-container-high px-2 py-0.5 rounded">
            {data.gene}
          </span>
        )}
        {data.chr && (
          <span className="text-xs text-on-surface-variant">{data.chr}{data.strand}</span>
        )}
        {data.efficacy != null && (
          <span className="text-xs text-on-surface-variant">
            Efficacy: <span className="text-on-surface font-mono">{data.efficacy.toFixed(2)}</span>
          </span>
        )}
        <SpliceRiskBadge risk={spliceRisk} />
        {nearestDist != null && (
          <span className="text-xs text-on-surface-variant">
            {nearestDist} bp to {nearestType ?? 'splice site'}
          </span>
        )}
      </div>

      {/* Cell-type suitability */}
      <div>
        <p className="text-xs font-medium text-on-surface-variant uppercase tracking-wide mb-2">
          Cell-type suitability
        </p>
        <div className="flex flex-col gap-2">
          {(data.predictions ?? []).map(pred => (
            <div key={pred.cell_type}>
              <div className="flex justify-between items-center mb-0.5">
                <span className="text-xs text-on-surface" style={{ color: CELL_COLORS[pred.cell_type] }}>
                  {pred.cell_type.replace(/_/g, ' ')}
                </span>
                {pred.omics_score != null && (
                  <span className="text-xs font-mono text-on-surface-variant">
                    model: {pred.omics_score.toFixed(3)}
                  </span>
                )}
              </div>
              <SuitBar value={pred.suitability_score} color={CELL_COLORS[pred.cell_type]} />
            </div>
          ))}
        </div>
      </div>

      {/* Feature attribution */}
      <div>
        <button
          onClick={() => { loadExplain(); setShowIG(v => !v) }}
          className="text-xs text-primary hover:text-on-primary-container transition-colors flex items-center gap-1"
        >
          <span className="material-symbols-outlined text-sm">
            {showIG ? 'expand_less' : 'expand_more'}
          </span>
          Feature attribution (Integrated Gradients, K562)
        </button>
        {showIG && explain && (
          <div className="mt-3 flex flex-col gap-2">
            <div className="flex gap-4 text-xs text-on-surface-variant mb-1">
              <span>CNN: {Math.round((explain.branch_contributions?.sequence_cnn ?? 0) * 100)}%</span>
              <span>Feature MLP: {Math.round((explain.branch_contributions?.feature_mlp ?? 0) * 100)}%</span>
              <span>Omics MLP: {Math.round((explain.branch_contributions?.omics_mlp ?? 0) * 100)}%</span>
            </div>
            {(explain.feature_groups ?? []).slice(0, 8).map(g => {
              const abs = Math.abs(g.attribution)
              const pct = Math.min(abs * 400, 100)
              const positive = g.attribution >= 0
              return (
                <div key={g.group} className="flex items-center gap-2">
                  <span className="text-xs text-on-surface-variant w-44 truncate flex-shrink-0">{g.group}</span>
                  <div className="flex-1 h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${pct}%`,
                        background: positive ? '#cdbdff' : '#ffb4ab',
                      }}
                    />
                  </div>
                  <span className={`text-xs font-mono w-14 text-right ${positive ? 'text-primary' : 'text-error'}`}>
                    {g.attribution > 0 ? '+' : ''}{g.attribution.toFixed(3)}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
