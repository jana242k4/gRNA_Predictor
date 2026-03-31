/**
 * Web Worker — runs XGBoost feature extraction + inference off the main thread.
 * Pure JavaScript: no WebAssembly, no native modules, no memory spikes.
 *
 * Protocol (postMessage):
 *   IN  { sequences: string[], thirtyMers: string[], modelBase: string }
 *   OUT { ok: true,  scores: Float32Array }   (buffer is transferred, not copied)
 *   OUT { ok: false, error: string }
 */
import { extractFeaturesBatch } from '../utils/featureEngineering.js'
import { xgbPredict }           from '../utils/xgbPredictor.js'

self.onmessage = async ({ data }) => {
  const { sequences, thirtyMers, modelBase } = data
  try {
    const features = extractFeaturesBatch(sequences, thirtyMers)
    const scores   = await xgbPredict(features, modelBase)
    // Transfer the ArrayBuffer so it isn't copied (zero-copy IPC)
    self.postMessage({ ok: true, scores }, [scores.buffer])
  } catch (err) {
    self.postMessage({ ok: false, error: err.message ?? String(err) })
  }
}
