import React, { useState } from 'react'
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Chip,
  LinearProgress,
  Tooltip,
  Paper,
  Button,
  IconButton,
  Snackbar,
  Alert,
} from '@mui/material'
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents'
import DownloadIcon from '@mui/icons-material/Download'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import CheckIcon from '@mui/icons-material/Check'
import MyLocationIcon from '@mui/icons-material/MyLocation'

function ScoreBar({ value, color }) {
  const pct      = Math.round(value * 100)
  const barColor = color || (pct >= 70 ? 'success' : pct >= 45 ? 'warning' : 'error')
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <LinearProgress
        variant="determinate"
        value={pct}
        color={barColor}
        sx={{ flex: 1, height: 8, borderRadius: 4 }}
      />
      <Typography variant="caption" sx={{ minWidth: 36, color: `${barColor}.main` }}>
        {pct}%
      </Typography>
    </Box>
  )
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <Tooltip title={copied ? 'Copied!' : 'Copy sequence'}>
      <IconButton size="small" onClick={handleCopy} sx={{ ml: 0.5 }}>
        {copied
          ? <CheckIcon fontSize="inherit" sx={{ color: 'success.main' }} />
          : <ContentCopyIcon fontSize="inherit" sx={{ color: 'text.disabled' }} />}
      </IconButton>
    </Tooltip>
  )
}

function SpecificityBadge({ value }) {
  if (value === null || value === undefined) return null
  const pct   = Math.round(value * 100)
  const color = pct >= 70 ? 'success' : pct >= 45 ? 'warning' : 'error'
  const label = pct >= 70 ? 'Low risk' : pct >= 45 ? 'Moderate' : 'High risk'
  return (
    <Tooltip title={`Off-target specificity: ${pct}%. Higher = more specific (lower off-target risk).`}>
      <Chip label={`${label} (${pct}%)`} size="small" color={color} variant="outlined" />
    </Tooltip>
  )
}

function DistanceBadge({ distance }) {
  if (distance === null || distance === undefined) return null
  const color = distance <= 25 ? 'success' : distance <= 75 ? 'warning' : 'error'
  return (
    <Tooltip title={`${distance} bp from your target position`}>
      <Chip
        label={`${distance} bp`}
        size="small"
        color={color}
        variant="outlined"
        icon={<MyLocationIcon style={{ fontSize: 12 }} />}
      />
    </Tooltip>
  )
}

function exportCSV(grnas, pamUsed, hasTarget) {
  const baseHeaders = ['Rank', 'Guide Sequence', 'PAM', 'Position', 'Cut Site', 'Strand', 'GC%', 'Efficiency Score', 'Off-Target Specificity', 'Model']
  const headers     = hasTarget ? [...baseHeaders, 'Distance to Target (bp)', 'Combined Score'] : baseHeaders

  const rows = grnas.map((g) => {
    const row = [
      g.rank, g.sequence, g.pam_sequence, g.position, g.cut_site,
      g.strand, (g.gc_content * 100).toFixed(1), g.score.toFixed(4),
      g.off_target_score != null ? g.off_target_score.toFixed(3) : '',
      g.model_used,
    ]
    if (hasTarget) {
      row.push(g.distance_to_target ?? '')
      row.push(g.combined_score != null ? g.combined_score.toFixed(4) : '')
    }
    return row
  })

  const csv  = [headers, ...rows].map((r) => r.map((v) => `"${v}"`).join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url  = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href  = url
  link.setAttribute('download', `grna_predictions_${pamUsed}.csv`)
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

export default function ResultsTable({ data }) {
  const [snackOpen, setSnackOpen] = useState(false)
  if (!data || !data.top_grnas || data.top_grnas.length === 0) return null

  const hasTarget = data.target_position != null

  return (
    <Box>
      {/* Header row */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, flexWrap: 'wrap' }}>
        <EmojiEventsIcon sx={{ color: 'secondary.main' }} />
        <Typography variant="h6">Top Guide RNAs</Typography>
        <Chip label={`${data.top_grnas.length} of ${data.total_candidates} candidates`} size="small" sx={{ ml: 1 }} />
        {hasTarget && (
          <Chip
            icon={<MyLocationIcon />}
            label={`Target pos ${data.target_position} · ${Math.round(data.proximity_weight * 100)}% proximity`}
            size="small"
            color="warning"
            variant="outlined"
          />
        )}
        <Button
          size="small"
          variant="outlined"
          startIcon={<DownloadIcon />}
          onClick={() => exportCSV(data.top_grnas, data.pam_used, hasTarget)}
          sx={{ ml: 'auto' }}
        >
          Export CSV
        </Button>
      </Box>

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>#</TableCell>
              <TableCell>Guide Sequence (5'→3')</TableCell>
              <TableCell>PAM</TableCell>
              <TableCell>
                <Tooltip title="0-indexed start of guide on forward strand"><span>Position</span></Tooltip>
              </TableCell>
              <TableCell>
                <Tooltip title="Predicted genomic cut site (1-indexed). SpCas9 cuts between guide nt 17-18, 3 bp upstream of PAM.">
                  <span>Cut Site</span>
                </Tooltip>
              </TableCell>
              <TableCell>Strand</TableCell>
              <TableCell>GC%</TableCell>
              <TableCell sx={{ minWidth: 140 }}>
                <Tooltip title="ML efficiency score (0-1). Higher is better."><span>Efficiency</span></Tooltip>
              </TableCell>
              <TableCell sx={{ minWidth: 130 }}>
                <Tooltip title="Heuristic off-target specificity (0-1). Green ≥70% = low risk; yellow 45-70% = moderate; red <45% = high risk. Based on seed-region AT content, GC composition, homopolymers, and hairpin structure.">
                  <span>Off-Target</span>
                </Tooltip>
              </TableCell>
              {hasTarget && (
                <>
                  <TableCell>
                    <Tooltip title="Distance from cut site to your target position. Green ≤25 bp, yellow ≤75 bp, red >75 bp.">
                      <span>Distance</span>
                    </Tooltip>
                  </TableCell>
                  <TableCell sx={{ minWidth: 140 }}>
                    <Tooltip title={`Combined score = ${Math.round((1 - data.proximity_weight) * 100)}% efficiency + ${Math.round(data.proximity_weight * 100)}% proximity (Gaussian decay, σ=50 bp)`}>
                      <span>Combined Score</span>
                    </Tooltip>
                  </TableCell>
                </>
              )}
            </TableRow>
          </TableHead>
          <TableBody>
            {data.top_grnas.map((g) => (
              <TableRow key={g.rank} sx={{ '&:hover': { bgcolor: 'action.hover' } }}>
                <TableCell>
                  <Chip
                    label={g.rank}
                    size="small"
                    color={g.rank === 1 ? 'primary' : 'default'}
                    variant={g.rank === 1 ? 'filled' : 'outlined'}
                  />
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Tooltip title={`Model: ${g.model_used}`}>
                      <Typography sx={{ fontFamily: 'Roboto Mono, monospace', fontSize: '0.8rem', letterSpacing: '0.05em', color: 'primary.main' }}>
                        {g.sequence}
                      </Typography>
                    </Tooltip>
                    <CopyButton text={g.sequence} />
                  </Box>
                </TableCell>
                <TableCell>
                  <Typography sx={{ fontFamily: 'Roboto Mono, monospace', fontSize: '0.8rem', color: 'secondary.main' }}>
                    {g.pam_sequence}
                  </Typography>
                </TableCell>
                <TableCell>{g.position}</TableCell>
                <TableCell>
                  <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono, monospace', fontSize: '0.8rem' }}>
                    {g.cut_site}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={g.strand === '+' ? '+ (fwd)' : '- (rev)'}
                    size="small"
                    color={g.strand === '+' ? 'info' : 'warning'}
                    variant="outlined"
                  />
                </TableCell>
                <TableCell>{(g.gc_content * 100).toFixed(1)}%</TableCell>
                <TableCell><ScoreBar value={g.score} /></TableCell>
                <TableCell><SpecificityBadge value={g.off_target_score} /></TableCell>
                {hasTarget && (
                  <>
                    <TableCell><DistanceBadge distance={g.distance_to_target} /></TableCell>
                    <TableCell>
                      {g.combined_score != null && <ScoreBar value={g.combined_score} color="warning" />}
                    </TableCell>
                  </>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Snackbar open={snackOpen} autoHideDuration={2000} onClose={() => setSnackOpen(false)}>
        <Alert severity="success" sx={{ width: '100%' }}>Sequence copied!</Alert>
      </Snackbar>
    </Box>
  )
}
