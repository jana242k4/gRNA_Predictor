import React, { useState } from 'react'
import {
  Box,
  TextField,
  Button,
  Typography,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Alert,
  Slider,
  Tooltip,
  Divider,
} from '@mui/material'
import SearchIcon from '@mui/icons-material/Search'
import DnaIcon from '@mui/icons-material/Biotech'
import MyLocationIcon from '@mui/icons-material/MyLocation'

const EXAMPLE_SEQ =
  'ATCGATCGATCGATCGATCGGGATCGATCGATCGATCGATCGGGATCGATCGATCGATCGATCGGGATCGATCGATCGATCGATCGGGATCGATCGATCGATCGATCGGG'

export default function SequenceInput({ onPredict, loading, error }) {
  const [sequence, setSequence]               = useState('')
  const [pam, setPam]                         = useState('NGG')
  const [targetPosition, setTargetPosition]   = useState('')
  const [proximityWeight, setProximityWeight] = useState(0.4)

  const handleSubmit = () => {
    if (sequence.trim()) onPredict(sequence.trim(), pam, targetPosition, proximityWeight)
  }

  const handleExample = () => setSequence(EXAMPLE_SEQ)

  const seqLen   = sequence.replace(/\s+/g, '').length
  const gcCount  = (sequence.match(/[GCgc]/g) || []).length
  const gcPct    = seqLen > 0 ? ((gcCount / seqLen) * 100).toFixed(1) : null

  const targetVal    = parseInt(targetPosition, 10)
  const targetValid  = targetPosition === '' || (!isNaN(targetVal) && targetVal >= 1 && targetVal <= seqLen)
  const hasTarget    = targetPosition !== '' && !isNaN(targetVal) && targetVal >= 1

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <DnaIcon sx={{ color: 'primary.main' }} />
        <Typography variant="h6">Input DNA Sequence</Typography>
      </Box>

      <TextField
        multiline
        minRows={4}
        maxRows={10}
        fullWidth
        value={sequence}
        onChange={(e) => setSequence(e.target.value)}
        placeholder="Paste your DNA sequence here (5' to 3', ACGTN only)..."
        variant="outlined"
        InputProps={{
          style: { fontFamily: 'Roboto Mono, monospace', fontSize: '0.85rem', letterSpacing: '0.05em' },
        }}
      />

      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
        {seqLen > 0 && (
          <>
            <Chip label={`${seqLen} bp`} size="small" color="primary" variant="outlined" />
            {gcPct && <Chip label={`GC: ${gcPct}%`} size="small" color="secondary" variant="outlined" />}
          </>
        )}
        <Button size="small" variant="text" onClick={handleExample} sx={{ ml: 'auto' }}>
          Load Example
        </Button>
      </Box>

      {/* PAM + Predict row */}
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>PAM Sequence</InputLabel>
          <Select value={pam} label="PAM Sequence" onChange={(e) => setPam(e.target.value)}>
            <MenuItem value="NGG">NGG (SpCas9)</MenuItem>
            <MenuItem value="NAG">NAG (SpCas9 alt)</MenuItem>
            <MenuItem value="NNGRRT">NNGRRT (SaCas9)</MenuItem>
            <MenuItem value="TTTV">TTTV (Cas12a / Cpf1)</MenuItem>
          </Select>
        </FormControl>

        <Button
          variant="contained"
          size="large"
          startIcon={loading ? <CircularProgress size={18} color="inherit" /> : <SearchIcon />}
          onClick={handleSubmit}
          disabled={loading || seqLen < 23 || !targetValid}
          sx={{ flex: 1, maxWidth: 200 }}
        >
          {loading ? 'Predicting...' : 'Predict gRNAs'}
        </Button>
      </Box>

      <Divider />

      {/* Target Position section */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <MyLocationIcon sx={{ color: hasTarget ? 'warning.main' : 'text.disabled', fontSize: 20 }} />
          <Typography variant="body2" color={hasTarget ? 'text.primary' : 'text.secondary'}>
            Target Position
            <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
              (optional — rank by proximity to a specific genomic coordinate)
            </Typography>
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
          <Tooltip
            title={
              seqLen > 0
                ? `Enter a position between 1 and ${seqLen}`
                : 'Enter a sequence first'
            }
            placement="top"
          >
            <TextField
              size="small"
              label="Target Position (bp)"
              type="number"
              value={targetPosition}
              onChange={(e) => setTargetPosition(e.target.value)}
              disabled={seqLen < 23}
              error={!targetValid}
              helperText={
                !targetValid
                  ? `Must be 1–${seqLen}`
                  : hasTarget
                  ? `Guide cut sites will be ranked by distance to position ${targetVal}`
                  : 'Leave blank to rank by efficiency score only'
              }
              InputProps={{ inputProps: { min: 1, max: seqLen || undefined } }}
              sx={{ width: 200 }}
            />
          </Tooltip>

          {hasTarget && (
            <Box sx={{ flex: 1, maxWidth: 320, pt: 0.5 }}>
              <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                Proximity weight: <strong>{Math.round(proximityWeight * 100)}%</strong>
                {' '}proximity · <strong>{Math.round((1 - proximityWeight) * 100)}%</strong> efficiency
              </Typography>
              <Slider
                value={proximityWeight}
                min={0}
                max={1}
                step={0.05}
                onChange={(_, v) => setProximityWeight(v)}
                marks={[
                  { value: 0,   label: 'Efficiency' },
                  { value: 0.5, label: '50/50' },
                  { value: 1,   label: 'Proximity' },
                ]}
                size="small"
                color="warning"
              />
            </Box>
          )}
        </Box>
      </Box>

      {error && (
        <Alert severity="error" onClose={() => {}}>
          {error}
        </Alert>
      )}
    </Box>
  )
}
