import React, { useMemo } from 'react'

const COLORS = [
  'rgba(124,77,255,0.45)',
  'rgba(255,182,136,0.45)',
  'rgba(105,224,106,0.45)',
  'rgba(255,180,171,0.45)',
  'rgba(144,202,249,0.45)',
]
const CHUNK = 60

export default function GeneMap({ sequence, grnas }) {
  if (!sequence || !grnas || grnas.length === 0) return null

  const highlights = useMemo(() => {
    const map = new Array(sequence.length).fill(null)
    grnas.forEach((g, idx) => {
      const start = g.position
      for (let i = start; i < start + 20 && i < sequence.length; i++) {
        if (map[i] === null) map[i] = idx
      }
    })
    return map
  }, [sequence, grnas])

  const chunks = []
  for (let i = 0; i < sequence.length; i += CHUNK) {
    chunks.push({ start: i, bases: sequence.slice(i, i + CHUNK) })
  }

  return (
    <div className="bg-surface-container-low rounded-3xl p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-primary">map</span>
        <h3 className="text-sm font-semibold text-on-surface">Sequence Map</h3>
      </div>
      <p className="text-xs text-on-surface-variant">
        Highlighted regions show top-ranked guide RNA target sites.
      </p>
      <div className="overflow-x-auto">
        {chunks.map(({ start, bases }) => (
          <div key={start} className="flex items-center mb-1">
            <span className="text-xs font-mono text-on-surface-variant w-12 flex-shrink-0">
              {start + 1}
            </span>
            <span className="font-mono text-xs leading-6">
              {Array.from(bases).map((base, i) => {
                const absIdx = start + i
                const gIdx = highlights[absIdx]
                return (
                  <span
                    key={i}
                    title={gIdx !== null ? `Guide #${gIdx + 1}: ${grnas[gIdx].sequence}` : undefined}
                    style={{ background: gIdx !== null ? COLORS[gIdx % COLORS.length] : undefined }}
                    className={`${gIdx !== null ? 'text-on-surface cursor-pointer' : 'text-on-surface-variant'}`}
                  >
                    {base}
                  </span>
                )
              })}
            </span>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-1">
        {grnas.slice(0, 5).map((g, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded-sm flex-shrink-0"
              style={{ background: COLORS[i % COLORS.length].replace('0.45', '0.8') }}
            />
            <span className="text-xs text-on-surface-variant font-mono">{g.sequence?.slice(0, 8)}…</span>
          </div>
        ))}
      </div>
    </div>
  )
}
