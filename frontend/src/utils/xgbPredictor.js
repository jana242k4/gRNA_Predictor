/**
 * Pure-JavaScript XGBoost inference — no WebAssembly, no Workers, no memory spikes.
 *
 * Uses cooperative time-slicing: processes CHUNK_SIZE candidates then yields the
 * event loop via setTimeout(0).  Each chunk takes ~3 ms, so Chrome never sees a
 * synchronous block long enough to show "Page Unresponsive" (<5 s threshold).
 * Total time for 200 candidates: ~40 ms spread across ~8 yields.
 *
 * Model format — flat parallel arrays (DFS order per tree):
 *   features[i]   = feature index at node i, or -1 for leaf nodes
 *   thresholds[i] = split threshold (internal) or leaf value (leaf)
 *   left[i]       = child index when feature < threshold
 *   right[i]      = child index when feature >= threshold
 */

const CHUNK_SIZE = 25   // candidates per slice (~3 ms each on slow devices)

let _model = null

async function loadModel(modelBase) {
  if (_model) return _model
  const url  = `${modelBase}xgb_trees.json`
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`Failed to load XGBoost model: ${resp.status}`)
  _model = await resp.json()
  return _model
}

/**
 * Run XGBoost inference on a batch of feature vectors.
 * Yields the event loop every CHUNK_SIZE rows so the UI stays responsive.
 *
 * @param {Float32Array} featuresFlat - Row-major [N × 450]
 * @param {string}       modelBase    - Vite BASE_URL (e.g. '/gRNA_Predictor/')
 * @returns {Promise<Float32Array>}   Raw regression scores, shape [N]
 */
export async function xgbPredict(featuresFlat, modelBase = '/') {
  const model = await loadModel(modelBase)
  const { numTrees, treeOffsets, features, thresholds, left, right } = model

  const DIM = 452
  const N   = Math.round(featuresFlat.length / DIM)
  const out = new Float32Array(N)

  for (let chunkStart = 0; chunkStart < N; chunkStart += CHUNK_SIZE) {
    // Yield to the event loop BEFORE each chunk so the browser can repaint/respond
    await new Promise(resolve => setTimeout(resolve, 0))

    const chunkEnd = Math.min(chunkStart + CHUNK_SIZE, N)

    for (let i = chunkStart; i < chunkEnd; i++) {
      const base = i * DIM
      let score  = 0.0

      for (let t = 0; t < numTrees; t++) {
        const offset = treeOffsets[t]
        let   node   = 0

        while (features[offset + node] !== -1) {
          const feat   = features[offset + node]
          const thresh = thresholds[offset + node]
          node = featuresFlat[base + feat] < thresh
            ? left[offset + node]
            : right[offset + node]
        }
        score += thresholds[offset + node]  // leaf value
      }

      out[i] = score
    }
  }

  return out
}
