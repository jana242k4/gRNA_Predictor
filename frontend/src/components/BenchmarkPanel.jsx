import React, { useState, useEffect, useCallback } from 'react'
import { fetchBenchmark } from '../services/api'

const fmt = (v, digits = 3) => v != null ? v.toFixed(digits) : '—'

export default function BenchmarkPanel() {
  const [open, setOpen]       = useState(false)
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const load = useCallback(() => {
    setLoading(true)
    setError(null)
    fetchBenchmark()
      .then(d => { setData(d); setLoading(false) })
      .catch(() => { setError('Could not load benchmark data — the server may be starting up.'); setLoading(false) })
  }, [])

  useEffect(() => {
    if (open && !data && !loading && !error) load()
  }, [open, data, loading, error, load])

  return (
    <div className="rounded-2xl border border-white/10 bg-surface overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-6 py-4 text-left hover:bg-white/5 transition-colors"
      >
        <div>
          <span className="font-semibold text-on-surface">Performance Benchmarks</span>
          <span className="ml-3 text-sm text-on-surface-variant">
            How this tool compares to Azimuth, CRISPOR &amp; CRISPRscan
          </span>
        </div>
        <svg
          className={`w-5 h-5 text-on-surface-variant transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="px-6 pb-6">
          {loading && (
            <p className="text-sm text-on-surface-variant py-4">Loading benchmark data…</p>
          )}

          {error && (
            <div className="flex items-center gap-3 py-4">
              <p className="text-sm text-on-surface-variant flex-1">{error}</p>
              <button
                onClick={load}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium
                           border border-outline-variant text-on-surface-variant
                           hover:bg-surface-container-high transition-colors"
              >
                <span className="material-symbols-outlined text-sm">refresh</span>
                Retry
              </button>
            </div>
          )}

          {data && (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-white/10 text-on-surface-variant">
                      <th className="text-left py-2 pr-4 font-medium">Tool</th>
                      <th className="text-right py-2 px-3 font-medium whitespace-nowrap">Pearson r<br/><span className="text-xs font-normal">Doench held-out</span></th>
                      <th className="text-right py-2 px-3 font-medium whitespace-nowrap">Pearson r<br/><span className="text-xs font-normal">Kim2019 novel</span></th>
                      <th className="text-right py-2 px-3 font-medium whitespace-nowrap">Spearman r<br/><span className="text-xs font-normal">All guides</span></th>
                      <th className="text-left py-2 pl-4 font-medium">Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.tools.map((t, i) => (
                      <tr key={i} className={`border-b border-white/5 ${i === 0 ? 'text-primary font-medium' : 'text-on-surface'}`}>
                        <td className="py-2 pr-4 whitespace-nowrap">{t.name}</td>
                        <td className="text-right py-2 px-3 tabular-nums">{fmt(t.pearson_doench_heldout)}</td>
                        <td className="text-right py-2 px-3 tabular-nums">{fmt(t.pearson_kim2019_novel)}</td>
                        <td className="text-right py-2 px-3 tabular-nums">{fmt(t.spearman_all)}</td>
                        <td className="py-2 pl-4 text-xs text-on-surface-variant max-w-xs">{t.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 space-y-1">
                {data.caveats.map((c, i) => (
                  <p key={i} className="text-xs text-on-surface-variant leading-relaxed">
                    {i === 0 ? '⚠ ' : '• '}{c}
                  </p>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
