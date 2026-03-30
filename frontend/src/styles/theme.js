import { createTheme } from '@mui/material/styles'

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00e5ff',
    },
    secondary: {
      main: '#69f0ae',
    },
    background: {
      default: '#0a0e1a',
      paper: '#111827',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto Mono", monospace',
    h4: { fontWeight: 700 },
    h6: { fontWeight: 600 },
  },
  shape: {
    borderRadius: 12,
  },
})

export default theme
