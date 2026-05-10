/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'background':                '#14121c',
        'surface':                   '#14121c',
        'surface-dim':               '#14121c',
        'surface-container-lowest':  '#0f0d16',
        'surface-container-low':     '#1d1a24',
        'surface-container':         '#211e28',
        'surface-container-high':    '#2b2833',
        'surface-container-highest': '#36333e',
        'surface-variant':           '#36333e',
        'surface-bright':            '#3b3743',
        'primary':                   '#cdbdff',
        'primary-container':         '#7c4dff',
        'on-primary':                '#370096',
        'on-primary-container':      '#fcf6ff',
        'secondary':                 '#d3bcfc',
        'secondary-container':       '#523f76',
        'on-secondary-container':    '#c4aeed',
        'tertiary':                  '#ffb688',
        'tertiary-container':        '#b55800',
        'on-tertiary-container':     '#fff7f4',
        'on-background':             '#e6e0ee',
        'on-surface':                '#e6e0ee',
        'on-surface-variant':        '#cac3d8',
        'outline':                   '#948ea1',
        'outline-variant':           '#494455',
        'error':                     '#ffb4ab',
        'error-container':           '#93000a',
        'on-error-container':        '#ffdad6',
        'inverse-surface':           '#e6e0ee',
        'inverse-primary':           '#6833ea',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      fontSize: {
        'code-sm': ['12px', { lineHeight: '16px', fontWeight: '400' }],
        'code-md': ['14px', { lineHeight: '20px', letterSpacing: '0.05em', fontWeight: '500' }],
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(124,77,255,0.4)' },
          '50%':       { boxShadow: '0 0 20px rgba(124,77,255,0.35)' },
        },
        'fade-up': {
          '0%':   { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'bar-grow': {
          '0%':   { width: '0%' },
          '100%': { width: 'var(--bar-w)' },
        },
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'fade-up':    'fade-up 0.5s cubic-bezier(0.22,1,0.36,1) both',
        'bar-grow':   'bar-grow 1.2s cubic-bezier(0.65,0,0.35,1) forwards',
      },
    },
  },
  plugins: [],
}
