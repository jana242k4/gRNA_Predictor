import React, { useMemo } from 'react'
import { Box, Typography, Tooltip, Paper } from '@mui/material'

const COLORS = ['#00e5ff', '#69f0ae', '#ffb74d', '#ef5350', '#ce93d8']
const CHUNK = 60 // bases per line

export default function GeneMap({ sequence, grnas }) {
  if (!sequence || !grnas || grnas.length === 0) return null

  const highlights = useMemo(() => {
    const map = new Array(sequence.length).fill(null)
    grnas.forEach((g, idx) => {
      const start = g.position
      const end = start + 20
      for (let i = start; i < end && i < sequence.length; i++) {
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
    <Box>
      <Typography variant="h6" sx={{ mb: 1 }}>
        Gene Map
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
        Colored regions indicate top-ranked guide RNA target sites.
      </Typography>
      <Paper variant="outlined" sx={{ p: 2, overflowX: 'auto' }}>
        {chunks.map(({ start, bases }) => (
          <Box key={start} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            <Typography
              variant="caption"
              sx={{ minWidth: 52, color: 'text.disabled', fontFamily: 'monospace', mr: 1 }}
            >
              {start + 1}
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'nowrap' }}>
              {Array.from(bases).map((base, i) => {
                const absIdx = start + i
                const gIdx = highlights[absIdx]
                const bg = gIdx !== null ? COLORS[gIdx % COLORS.length] : 'transparent'
                return (
                  <Tooltip
                    key={i}
                    title={gIdx !== null ? `Guide #${gIdx + 1}: ${grnas[gIdx].sequence}` : ''}
                    placement="top"
                  >
                    <Typography
                      component="span"
                      sx={{
                        fontFamily: 'Roboto Mono, monospace',
                        fontSize: '0.75rem',
                        color: gIdx !== null ? '#0a0e1a' : 'text.secondary',
                        bgcolor: bg,
                        px: 0,
                        borderRadius: '2px',
                        lineHeight: 1.6,
                        cursor: gIdx !== null ? 'pointer' : 'default',
                      }}
                    >
                      {base}
                    </Typography>
                  </Tooltip>
                )
              })}
            </Box>
          </Box>
        ))}
      </Paper>
    </Box>
  )
}
