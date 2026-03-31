import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // Use '/' for local dev, '/gRNA_Predictor/' for GitHub Pages production build
  base: command === 'build' ? '/gRNA_Predictor/' : '/',
  build: {
    chunkSizeWarningLimit: 1000,
  },
  // Inject the Render API URL for production builds (GitHub Pages calls Render backend)
  define: command === 'build' ? {
    'import.meta.env.VITE_API_BASE_URL': JSON.stringify(
      'https://grna-predictor-api.onrender.com/api/v1'
    ),
  } : {},
}))
