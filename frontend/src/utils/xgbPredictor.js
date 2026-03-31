/**
 * Pure-JavaScript XGBoost inference — no WebAssembly, no Workers, no memory spikes.
 *
 * Loads the exported tree structure (xgb_trees.json, ~294 KB) once and caches it.
 * Tree traversal is ~2–5 ms for 200 candidates × 500 trees on modern V8.
 *
 * Format: flat parallel arrays (DFS order per tree):
 *   features[i]   = feature index, or -1 for leaf nodes
 *   thresholds[i] = split threshold, or leaf value when features[i] === -1
 *   left[i]       = index of left  child (feature <  threshold)
 *   right[i]      = index of right child (feature >= threshold)
 */

let _model = null

async function loadModel(modelBase) {
  if (_model) return _model
  const url = `${modelBase}xgb_trees.json`
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`Failed to load XGBoost trees: ${resp.status}`)
  _model = await resp.json()
  return _model
}

/**
 * Run XGBoost inference on a batch of feature vectors.
 *
 * @param {Float32Array} featuresFlat - Row-major matrix, shape [N × 450]
 * @param {string}       modelBase    - Vite BASE_URL (e.g. '/gRNA_Predictor/')
 * @returns {Float32Array} Raw regression scores, shape [N]
 */
export async function xgbPredict(featuresFlat, modelBase = '/') {
  const model = await loadModel(modelBase)
  // Yield the event loop once after the (async) model load so the UI
  // stays responsive before we enter the synchronous traversal loops.
  await new Promise(resolve => setTimeout(resolve, 0))

  const { numTrees, treeOffsets, features, thresholds, left, right } = model
  const DIM = 450
  const N   = Math.round(featuresFlat.length / DIM)
  const out = new Float32Array(N)

  for (let i = 0; i < N; i++) {
    const base = i * DIM
    let score  = 0.0

    for (let t = 0; t < numTrees; t++) {
      const offset = treeOffsets[t]
      let   node   = 0

      // Traverse one tree
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

  return out
}
