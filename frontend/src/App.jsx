import React, { useState } from 'react'
import {
  Box,
  Container,
  Typography,
  Divider,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Alert,
  Collapse,
} from '@mui/material'
import BiotechIcon from '@mui/icons-material/Biotech'
import SequenceInput from './components/SequenceInput'
import ResultsTable from './components/ResultsTable'
import GeneMap from './components/GeneMap'
import CitationCard from './components/CitationCard'
import { predictGRNAs } from './services/api'

export default function App() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)
  const [inputSeq, setInputSeq] = useState('')

  const handlePredict = async (sequence, pam, targetPosition, proximityWeight) => {
    setLoading(true)
    setError(null)
    setResults(null)
    setInputSeq(sequence)
    try {
      const data = await predictGRNAs(sequence, pam, 5, targetPosition, proximityWeight)
      setResults(data)
    } catch (err) {
      let msg =
        err.response?.data?.detail ||
        err.response?.data?.error ||
        err.message ||
        'An unexpected error occurred.'
      if (typeof msg !== 'string') msg = JSON.stringify(msg)
      if (err.code === 'ECONNABORTED' || msg.toLowerCase().includes('timeout')) {
        msg = 'Request timed out. The prediction server may be starting up — please try again in a few seconds.'
      } else if (err.code === 'ERR_NETWORK' || msg.toLowerCase().includes('network')) {
        msg = 'Cannot reach the prediction server. Please check your connection and try again.'
      }
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', py: 4 }}>
      <Container maxWidth="lg">
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 4 }}>
          <BiotechIcon sx={{ fontSize: 48, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" color="primary.main">
              gRNA Predictor
            </Typography>
            <Typography variant="body2" color="text.secondary">
              AI-powered CRISPR guide RNA design & efficiency scoring
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* Input Card */}
          <Paper sx={{ p: 3 }} elevation={2}>
            <SequenceInput
              onPredict={handlePredict}
              loading={loading}
              error={error}
            />
          </Paper>

          {/* Results */}
          <Collapse in={!!results}>
            {results && (
              <>
                <Paper sx={{ p: 3 }} elevation={2}>
                  <ResultsTable data={results} />
                </Paper>

                <Paper sx={{ p: 3 }} elevation={2}>
                  <GeneMap sequence={inputSeq} grnas={results.top_grnas} />
                </Paper>

                <CitationCard modelInfo={results.model_info} />
              </>
            )}
          </Collapse>
        </Box>
      </Container>
    </Box>
  )
}
