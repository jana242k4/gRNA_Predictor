import React from 'react'
import { Box, Typography, Chip, Divider, Link, Paper } from '@mui/material'
import InfoIcon from '@mui/icons-material/Info'
import SchoolIcon from '@mui/icons-material/School'

const CITATIONS = [
  {
    label: 'Doench et al. 2016',
    desc: 'Optimized sgRNA design to maximize activity and minimize off-target effects (Rule Set 2).',
    href: 'https://www.nature.com/articles/nbt.3437',
  },
  {
    label: 'Xu et al. 2015',
    desc: 'Sequence determinants of improved CRISPR sgRNA design.',
    href: 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4380695/',
  },
  {
    label: 'Kim et al. 2019 (DeepCRISPR)',
    desc: 'Deep learning model for predicting sgRNA on-target activity and off-target effects.',
    href: 'https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1459-4',
  },
]

export default function CitationCard({ modelInfo }) {
  return (
    <Paper variant="outlined" sx={{ p: 2.5 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
        <InfoIcon sx={{ color: 'primary.main' }} fontSize="small" />
        <Typography variant="h6" sx={{ fontSize: '1rem' }}>
          Scoring Methodology
        </Typography>
      </Box>

      {modelInfo && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {modelInfo}
        </Typography>
      )}

      <Divider sx={{ mb: 1.5 }} />

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <SchoolIcon fontSize="small" sx={{ color: 'secondary.main' }} />
        <Typography variant="subtitle2">Key References</Typography>
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {CITATIONS.map((c) => (
          <Box key={c.label} sx={{ display: 'flex', gap: 1, alignItems: 'flex-start' }}>
            <Chip
              label={c.label}
              size="small"
              component={Link}
              href={c.href}
              target="_blank"
              rel="noopener"
              clickable
              color="primary"
              variant="outlined"
              sx={{ flexShrink: 0, mt: 0.25 }}
            />
            <Typography variant="caption" color="text.secondary">
              {c.desc}
            </Typography>
          </Box>
        ))}
      </Box>
    </Paper>
  )
}
