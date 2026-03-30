import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // Use '/' for local dev, '/gRNA_Predictor/' for GitHub Pages production build
  base: command === 'build' ? '/gRNA_Predictor/' : '/',
  build: {
    // onnxruntime-web copies its WASM backends into the build; suppress the
    // size warning — these files are not loaded at runtime because wasmPaths
    // in onnxPredictor.js points to the jsDelivr CDN instead.
    chunkSizeWarningLimit: 30000,
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
}))
