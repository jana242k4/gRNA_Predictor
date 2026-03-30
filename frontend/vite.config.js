import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // Use '/' for local dev, '/gRNA_Predictor/' for GitHub Pages production build
  base: command === 'build' ? '/gRNA_Predictor/' : '/',
  build: {
    // The 25 MB JSEP WASM asset is copied to dist/ but never loaded at runtime:
    // onnxPredictor.js sets numThreads=1 (→ ORT selects ort-wasm-simd.wasm ~5 MB
    // from CDN) and wasmPaths points to jsDelivr CDN, not the local copy.
    chunkSizeWarningLimit: 30000,
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
}))
