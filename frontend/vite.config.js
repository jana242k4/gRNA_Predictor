import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // Use '/' for local dev, '/gRNA_Predictor/' for GitHub Pages production build
  base: command === 'build' ? '/gRNA_Predictor/' : '/',
  build: {
    chunkSizeWarningLimit: 1000,
  },
}))
