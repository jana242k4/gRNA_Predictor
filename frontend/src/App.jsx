import React, { useState, useCallback } from 'react'
import SequenceInputCard from './components/SequenceInputCard'
import ResultsTable from './components/ResultsTable'
import GeneExplorer from './components/GeneExplorer'
import BenchmarkPanel from './components/BenchmarkPanel'
import { predictGRNAs } from './services/api'

const TABS = ['Guide Design', 'Gene Explorer']

export default function App() {
  const [activeTab, setActiveTab] = useState(0)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [results, setResults]     = useState(null)
  const [inputSeq, setInputSeq]   = useState('')

  const handlePredict = useCallback(async (sequence, pam, targetPosition, proximityWeight, topN = 10) => {
    setLoading(true)
    setError(null)
    setResults(null)
    setInputSeq(sequence)
    try {
      const data = await predictGRNAs(sequence, pam, topN, targetPosition, proximityWeight)
      setResults(data)
    } catch (err) {
      let msg = err.response?.data?.detail || err.response?.data?.error || err.message || 'Unexpected error.'
      if (typeof msg !== 'string') msg = JSON.stringify(msg)
      if (err.code === 'ECONNABORTED' || msg.toLowerCase().includes('timeout')) {
        msg = 'The prediction server may be starting up — please try again in a few seconds.'
      } else if (err.code === 'ERR_NETWORK') {
        msg = 'Cannot reach the prediction server. Running in offline mode.'
      }
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div className="min-h-screen bg-background text-on-surface">
      {/* Header */}
      <header className="border-b border-outline-variant bg-surface-container-low sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center gap-4 h-16">
          <div className="flex items-center gap-3">
            <span className="material-symbols-outlined text-primary text-3xl select-none">biotech</span>
            <div>
              <h1 className="text-base font-semibold text-on-surface leading-tight">gRNA Predictor</h1>
              <p className="text-xs text-on-surface-variant leading-tight">OmicsCRISPR — Cell-type-aware guide scoring</p>
            </div>
          </div>

          {/* Tabs */}
          <nav className="flex gap-1 ml-8">
            {TABS.map((tab, i) => (
              <button
                key={tab}
                onClick={() => setActiveTab(i)}
                className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
                  activeTab === i
                    ? 'bg-secondary-container text-on-secondary-container'
                    : 'text-on-surface-variant hover:bg-surface-container-high'
                }`}
              >
                {tab}
              </button>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <a
              href="https://github.com/janathilakarathna/gRNA_Predictor"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-full hover:bg-surface-container-high text-on-surface-variant transition-colors"
              title="GitHub"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.342-3.369-1.342-.454-1.155-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0 1 12 6.836a9.59 9.59 0 0 1 2.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.202 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
              </svg>
            </a>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {activeTab === 0 && (
          <div className="flex flex-col gap-6">
            <SequenceInputCard
              onPredict={handlePredict}
              loading={loading}
              error={error}
            />
            {results && (
              <ResultsTable data={results} inputSeq={inputSeq} />
            )}
            <BenchmarkPanel />
          </div>
        )}
        {activeTab === 1 && <GeneExplorer />}
      </main>

      {/* Footer */}
      <footer className="border-t border-outline-variant mt-12 py-6 text-center text-xs text-on-surface-variant">
        gRNA Predictor · OmicsCRISPR · Trained on Doench 2016 + 2014 · Kim2019 r=0.640
      </footer>
    </div>
  )
}
