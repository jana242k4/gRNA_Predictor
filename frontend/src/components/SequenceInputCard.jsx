import React, { useState, useRef } from 'react'

const PAM_OPTIONS = [
  { value: 'NGG',    label: 'NGG — SpCas9' },
  { value: 'NAG',    label: 'NAG — SpCas9 alt' },
  { value: 'NNGRRT', label: 'NNGRRT — SaCas9' },
  { value: 'TTTV',   label: 'TTTV — Cas12a (5\')' },
]

const CELL_TYPES = ['K562', 'T_cell_CD4', 'T_cell_CD8', 'NK_cell', 'B_cell']

const EXAMPLE = 'ATGGAGGAGCCGCAGTCAGATCCTAGCGGTAATCTACTGGGACGGAACAGCTTTGAGGTGCGTGTTTGTGCCTGTCCTGGGAGAGACCGGCGCACAGAGGAAGAGAATCTCCGCAAGAAAGGGGAGCCTCACCACGAGCTGCCCCCAGGGAGCACTAAGCGAGCACTGCCCAACAACACCAGCTCCTCTCCCCAGCCAAAGAAGAAACCACTGGATGGAGAATATTTCACCCTTCAGATCCGTGGGCGTGAGCGCTTCGAGATGTTCCGAGAGCTGAATGAGGCC'

export default function SequenceInputCard({ onPredict, loading, error }) {
  const [sequence, setSequence]           = useState('')
  const [pam, setPam]                     = useState('NGG')
  const [targetPos, setTargetPos]         = useState('')
  const [proximityWeight, setProxWeight]  = useState(0.4)
  const [topN, setTopN]                   = useState(10)
  const [selectedCells, setSelectedCells] = useState([])
  const textareaRef = useRef(null)

  const seqClean = sequence.toUpperCase().replace(/\s+/g, '')
  const seqLen   = seqClean.length
  const valid    = seqLen >= 23

  const toggleCell = (ct) =>
    setSelectedCells(prev => prev.includes(ct) ? prev.filter(c => c !== ct) : [...prev, ct])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!valid || loading) return
    onPredict(seqClean, pam, targetPos || null, proximityWeight, topN)
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-surface-container-low rounded-3xl p-6 flex flex-col gap-5"
    >
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-primary">genetics</span>
        <h2 className="text-base font-semibold text-on-surface">Design gRNAs</h2>
      </div>

      {/* Sequence textarea */}
      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-medium text-on-surface-variant uppercase tracking-wide">
          Target DNA sequence
        </label>
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={sequence}
            onChange={e => setSequence(e.target.value)}
            placeholder="Paste DNA sequence (min 23 bp)…"
            rows={4}
            className="w-full bg-surface-container rounded-2xl px-4 py-3 text-sm font-mono text-on-surface
                       placeholder-outline resize-none border border-outline-variant
                       focus:outline-none focus:border-primary transition-colors"
            spellCheck={false}
          />
          <div className="absolute bottom-3 right-4 text-xs text-on-surface-variant tabular-nums select-none">
            {seqLen} bp
          </div>
        </div>
        <button
          type="button"
          onClick={() => setSequence(EXAMPLE)}
          className="self-start text-xs text-primary hover:text-on-primary-container transition-colors"
        >
          Load example
        </button>
      </div>

      {/* PAM + Top-N row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="flex flex-col gap-1 sm:col-span-2">
          <label className="text-xs font-medium text-on-surface-variant uppercase tracking-wide">PAM</label>
          <select
            value={pam}
            onChange={e => setPam(e.target.value)}
            className="bg-surface-container rounded-xl px-3 py-2.5 text-sm text-on-surface
                       border border-outline-variant focus:outline-none focus:border-primary transition-colors"
          >
            {PAM_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-on-surface-variant uppercase tracking-wide">
            Target bp (opt.)
          </label>
          <input
            type="number"
            value={targetPos}
            onChange={e => setTargetPos(e.target.value)}
            placeholder="e.g. 150"
            min={1}
            className="bg-surface-container rounded-xl px-3 py-2.5 text-sm text-on-surface
                       border border-outline-variant focus:outline-none focus:border-primary transition-colors"
          />
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-on-surface-variant uppercase tracking-wide">
            Top N
          </label>
          <input
            type="number"
            value={topN}
            onChange={e => setTopN(Math.max(1, Math.min(20, parseInt(e.target.value) || 10)))}
            min={1}
            max={20}
            className="bg-surface-container rounded-xl px-3 py-2.5 text-sm text-on-surface
                       border border-outline-variant focus:outline-none focus:border-primary transition-colors"
          />
        </div>
      </div>

      {/* Proximity weight slider */}
      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between items-center">
          <label className="text-xs font-medium text-on-surface-variant uppercase tracking-wide">
            Proximity weight
          </label>
          <span className="text-xs text-primary font-mono">{proximityWeight.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0} max={1} step={0.05}
          value={proximityWeight}
          onChange={e => setProxWeight(parseFloat(e.target.value))}
          className="w-full accent-primary-container"
        />
        <div className="flex justify-between text-xs text-on-surface-variant">
          <span>Score only</span>
          <span>Position only</span>
        </div>
      </div>

      {/* Cell type chips */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-medium text-on-surface-variant uppercase tracking-wide">
          Cell types for omics scoring
        </label>
        <div className="flex flex-wrap gap-2">
          {CELL_TYPES.map(ct => (
            <button
              key={ct}
              type="button"
              onClick={() => toggleCell(ct)}
              className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                selectedCells.includes(ct)
                  ? 'bg-secondary-container text-on-secondary-container border-secondary-container'
                  : 'bg-transparent text-on-surface-variant border-outline-variant hover:border-outline'
              }`}
            >
              {ct.replace(/_/g, ' ')}
            </button>
          ))}
          <span className="text-xs text-on-surface-variant self-center ml-1">
            {selectedCells.length === 0 ? '(all)' : `${selectedCells.length} selected`}
          </span>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-error-container rounded-xl px-4 py-3 flex items-start gap-2">
          <span className="material-symbols-outlined text-on-error-container text-base mt-0.5">error</span>
          <p className="text-sm text-on-error-container">{error}</p>
        </div>
      )}

      {/* Submit */}
      <button
        type="submit"
        disabled={!valid || loading}
        className="self-end flex items-center gap-2 px-6 py-2.5 rounded-full
                   bg-primary-container text-on-primary-container font-medium text-sm
                   hover:opacity-90 active:scale-95 transition-all
                   disabled:opacity-40 disabled:cursor-not-allowed
                   animate-[pulse-glow_2s_ease-in-out_infinite]"
      >
        {loading ? (
          <>
            <span className="w-4 h-4 border-2 border-on-primary-container/30 border-t-on-primary-container rounded-full animate-spin" />
            Predicting…
          </>
        ) : (
          <>
            <span className="material-symbols-outlined text-base">search</span>
            Find gRNAs
          </>
        )}
      </button>
    </form>
  )
}
