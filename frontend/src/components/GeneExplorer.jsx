import React, { useState } from 'react'
import { omicsGene } from '../services/api'

const CELL_TYPES = ['K562', 'T_cell_CD4', 'T_cell_CD8', 'NK_cell', 'B_cell']

function exportCSV(rows, gene, cellType) {
  const headers = ['Guide sequence','Efficacy','Suitability%','Splice risk%','Chr','Strand']
  const data = rows.map(r => [
    r.guide_id,
    r.efficacy != null ? r.efficacy.toFixed(3) : '',
    r.suitability_score != null ? (r.suitability_score * 100).toFixed(1) : '',
    r.splice_risk != null ? (r.splice_risk * 100).toFixed(1) : '',
    r.chr ?? '',
    r.strand ?? '',
  ])
  const csv = [headers, ...data].map(r => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url; a.download = `${gene}_${cellType}_guides.csv`; a.click()
  URL.revokeObjectURL(url)
}

function SuitBar({ value }) {
  const pct = Math.round((value ?? 0) * 100)
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
        <div className="h-full rounded-full bg-primary-container" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-on-surface-variant w-8 text-right">{pct}%</span>
    </div>
  )
}

export default function GeneExplorer() {
  const [gene, setGene]         = useState('')
  const [cellType, setCellType] = useState('K562')
  const [topN, setTopN]         = useState(10)
  const [results, setResults]   = useState(null)
  const [loading, setLoading]   = useState(false)
  const [notFound, setNotFound] = useState(false)

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!gene.trim() || loading) return
    setLoading(true)
    setResults(null)
    setNotFound(false)
    const data = await omicsGene(gene.trim().toUpperCase(), cellType, topN)
    setLoading(false)
    if (!data || data.not_found) {
      setNotFound(true)
    } else {
      setResults(data)
    }
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Search form */}
      <form onSubmit={handleSearch} className="bg-surface-container-low rounded-3xl p-6 flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary">travel_explore</span>
          <h2 className="text-base font-semibold text-on-surface">Gene Explorer</h2>
        </div>
        <p className="text-sm text-on-surface-variant">
          Search the DepMap guide library for top-ranked gRNAs targeting a specific gene.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-[1fr_10rem_6rem_auto] gap-3 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-on-surface-variant uppercase tracking-wide">Gene symbol</label>
            <input
              type="text"
              value={gene}
              onChange={e => setGene(e.target.value)}
              placeholder="e.g. BRCA1, TP53, EGFR…"
              className="bg-surface-container rounded-xl px-3 py-2.5 text-sm text-on-surface
                         border border-outline-variant focus:outline-none focus:border-primary transition-colors"
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-on-surface-variant uppercase tracking-wide">Cell type</label>
            <select
              value={cellType}
              onChange={e => setCellType(e.target.value)}
              className="bg-surface-container rounded-xl px-3 py-2.5 text-sm text-on-surface
                         border border-outline-variant focus:outline-none focus:border-primary transition-colors"
            >
              {CELL_TYPES.map(ct => (
                <option key={ct} value={ct}>{ct.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-on-surface-variant uppercase tracking-wide">Top N</label>
            <input
              type="number"
              value={topN}
              onChange={e => setTopN(Math.max(1, Math.min(50, parseInt(e.target.value) || 10)))}
              min={1} max={50}
              className="bg-surface-container rounded-xl px-3 py-2.5 text-sm text-on-surface
                         border border-outline-variant focus:outline-none focus:border-primary transition-colors"
            />
          </div>

          <button
            type="submit"
            disabled={!gene.trim() || loading}
            className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-primary-container
                       text-on-primary-container font-medium text-sm hover:opacity-90
                       active:scale-95 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="w-4 h-4 border-2 border-on-primary-container/30 border-t-on-primary-container rounded-full animate-spin" />
            ) : (
              <span className="material-symbols-outlined text-base">search</span>
            )}
            Search
          </button>
        </div>
      </form>

      {/* Not found */}
      {notFound && (
        <div className="bg-surface-container-low rounded-3xl p-6 text-center">
          <span className="material-symbols-outlined text-3xl text-on-surface-variant">search_off</span>
          <p className="text-sm text-on-surface-variant mt-2">
            Gene <strong className="text-on-surface">{gene}</strong> not found in the DepMap library.
          </p>
          <p className="text-xs text-on-surface-variant mt-1">
            The omics library contains guides from the DepMap CRISPR screen.
          </p>
        </div>
      )}

      {/* Results */}
      {results && Array.isArray(results) && results.length > 0 && (
        <div className="bg-surface-container-low rounded-3xl overflow-hidden">
          <div className="px-5 py-3 border-b border-outline-variant flex items-center gap-3">
            <span className="text-sm font-semibold text-on-surface">{gene}</span>
            <span className="text-xs text-on-surface-variant">
              Top {results.length} guides · {cellType.replace(/_/g, ' ')}
            </span>
            <button
              onClick={() => exportCSV(results, gene, cellType)}
              className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium
                         border border-outline-variant text-on-surface-variant
                         hover:bg-surface-container-high transition-colors"
            >
              <span className="material-symbols-outlined text-sm">download</span>
              Export CSV
            </button>
          </div>

          {/* Table header */}
          <div className="grid grid-cols-[1fr_4.5rem_6rem_4rem_4rem_3rem] gap-x-3
                          px-5 py-2 text-xs font-medium text-on-surface-variant uppercase tracking-wide
                          border-b border-outline-variant">
            <span>Guide sequence</span>
            <span>Efficacy</span>
            <span>Suitability</span>
            <span>Splice risk</span>
            <span>Chr</span>
            <span>Strand</span>
          </div>

          {results.map((r, i) => (
            <div
              key={r.guide_id ?? i}
              className="grid grid-cols-[1fr_4.5rem_6rem_4rem_4rem_3rem] gap-x-3
                         px-5 py-3 items-center border-b border-outline-variant/40
                         hover:bg-surface-container-high/40 transition-colors"
            >
              <span className="seq-mono text-xs text-on-surface">{r.guide_id}</span>

              <span className="text-xs font-mono tabular-nums text-on-surface">
                {r.efficacy != null ? r.efficacy.toFixed(2) : '—'}
              </span>

              <SuitBar value={r.suitability_score} />

              <span className={`text-xs font-mono tabular-nums ${
                (r.splice_risk ?? 0) >= 0.6 ? 'text-error' :
                (r.splice_risk ?? 0) >= 0.3 ? 'text-tertiary' : 'text-on-surface-variant'
              }`}>
                {r.splice_risk != null ? (r.splice_risk * 100).toFixed(0) + '%' : '—'}
              </span>

              <span className="text-xs text-on-surface-variant">{r.chr ?? '—'}</span>
              <span className="text-xs text-on-surface-variant">{r.strand ?? '—'}</span>
            </div>
          ))}
        </div>
      )}

      {/* No results but request succeeded */}
      {results && Array.isArray(results) && results.length === 0 && (
        <div className="bg-surface-container-low rounded-3xl p-6 text-center text-sm text-on-surface-variant">
          No guides found for {gene} in {cellType.replace(/_/g, ' ')}.
        </div>
      )}
    </div>
  )
}
