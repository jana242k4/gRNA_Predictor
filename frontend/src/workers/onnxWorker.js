/**
 * Web Worker: ONNX inference + feature extraction off the main thread.
 *
 * Running WASM compilation and ML inference inside a worker prevents the
 * "Page Unresponsive" freeze that occurs when these run on the main thread.
 */
import * as ort from 'onnxruntime-web'
import { extractFeaturesBatch } from '../utils/featureEngineering.js'

// Single-threaded WASM → loads ort-wasm-simd.wasm (~5 MB) from CDN
// instead of ort-wasm-simd-threaded.jsep.wasm (~25 MB)
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths  = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/'

let _session = null

async function getSession(modelUrl) {
  if (_session) return _session
  _session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'basic',
  })
  return _session
}

self.onmessage = async ({ data }) => {
  // Warmup: pre-load the ONNX session before the first real inference request
  if (data.type === 'warmup') {
    try { await getSession(data.modelUrl) } catch {}
    return
  }

  const { requestId, sequences, thirtyMers, modelUrl } = data

  try {
    // Feature extraction (450-dim) and ONNX inference both run here in the worker
    const features = extractFeaturesBatch(sequences, thirtyMers)
    const session  = await getSession(modelUrl)
    const N        = sequences.length
    const tensor   = new ort.Tensor('float32', features, [N, 450])
    const feeds    = { [session.inputNames[0]]: tensor }
    const results  = await session.run(feeds)
    const scores   = Array.from(results[session.outputNames[0]].data)

    self.postMessage({ requestId, scores })
  } catch (err) {
    self.postMessage({ requestId, error: err.message })
  }
}
