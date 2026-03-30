"""
Export the trained XGBoost model to ONNX format for in-browser inference.

Produces: frontend/public/xgb_model.onnx

Usage:
    cd backend
    source ../.venv/Scripts/activate
    python export_onnx.py
"""
import pickle
import numpy as np
from pathlib import Path

import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

PKL_PATH  = Path(__file__).parent / "app/models/xgb_model.pkl"
ONNX_PATH = Path(__file__).parent.parent / "frontend/public/xgb_model.onnx"
N_FEATURES = 450

print(f"Loading {PKL_PATH} ...")
with open(PKL_PATH, "rb") as f:
    model = pickle.load(f)

print("Converting to ONNX ...")
onnx_model = onnxmltools.convert_xgboost(
    model,
    name="gRNA_efficiency_xgboost",
    initial_types=[("float_input", FloatTensorType([None, N_FEATURES]))]
)

ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
onnxmltools.utils.save_model(onnx_model, str(ONNX_PATH))
size_kb = ONNX_PATH.stat().st_size / 1024
print(f"Saved: {ONNX_PATH}  ({size_kb:.0f} KB)")

# ── Verification: compare PKL vs ONNX predictions ────────────────────────────
print("\nVerifying ONNX predictions match pickle ...")
import onnxruntime as ort  # noqa: E402 (installed with onnxmltools)

rng  = np.random.default_rng(42)
X    = rng.random((20, N_FEATURES), dtype=np.float32)

pkl_preds  = model.predict(X)

sess       = ort.InferenceSession(str(ONNX_PATH))
input_name = sess.get_inputs()[0].name
out_name   = sess.get_outputs()[0].name
onnx_preds = sess.run([out_name], {input_name: X})[0].ravel()

max_diff   = float(np.max(np.abs(pkl_preds - onnx_preds)))
print(f"Max absolute difference (pkl vs ONNX): {max_diff:.6f}")
if max_diff < 1e-4:
    print("PASS — predictions match within tolerance.")
else:
    print("WARNING — predictions differ by more than 1e-4; review ONNX conversion.")

print("\nInput names: ", [i.name for i in sess.get_inputs()])
print("Output names:", [o.name for o in sess.get_outputs()])
print("\nDone.")
