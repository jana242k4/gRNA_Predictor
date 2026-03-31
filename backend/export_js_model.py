"""
Export the trained XGBoost model as a compact JSON tree structure
for pure-JavaScript inference in the browser (no WASM required).

Tree format (flat parallel arrays, DFS order per tree):
  features[i]   : feature index (-1 for leaf nodes)
  thresholds[i] : split threshold, or leaf value when features[i] == -1
  left[i]       : index of left  (yes / feature < threshold) child
  right[i]      : index of right (no  / feature >= threshold) child
                  both -1 for leaf nodes

Traversal in JavaScript:
  node = 0
  while features[offset + node] != -1:
      if features_vec[features[offset+node]] < thresholds[offset+node]:
          node = left[offset+node]
      else:
          node = right[offset+node]
  leaf_value = thresholds[offset+node]

Usage:
    cd backend
    source ../.venv/Scripts/activate
    python export_js_model.py

Output: frontend/public/xgb_trees.json  (~600 KB)
"""

import json
import pickle
from pathlib import Path

PKL_PATH  = Path(__file__).parent / "app/models/xgb_model.pkl"
OUT_PATH  = Path(__file__).parent.parent / "frontend/public/xgb_trees.json"


def flatten_tree(root: dict):
    """DFS-flatten a single XGBoost tree JSON into parallel flat arrays."""
    features   = []
    thresholds = []
    left_ch    = []
    right_ch   = []

    def dfs(node):
        idx = len(features)
        if "leaf" in node:
            # Leaf node
            features.append(-1)
            thresholds.append(float(node["leaf"]))
            left_ch.append(-1)
            right_ch.append(-1)
        else:
            # Internal node — extract feature index from "fN" string
            feat = int(node["split"][1:])
            thresh = float(node["split_condition"])
            features.append(feat)
            thresholds.append(thresh)
            left_ch.append(-1)   # filled in after recursive calls
            right_ch.append(-1)
            # children[0] = "yes" child (feature < threshold)
            # children[1] = "no"  child (feature >= threshold)
            li = dfs(node["children"][0])
            ri = dfs(node["children"][1])
            left_ch[idx]  = li
            right_ch[idx] = ri
        return idx

    dfs(root)
    return features, thresholds, left_ch, right_ch


def main():
    print(f"Loading model from {PKL_PATH}")
    with open(PKL_PATH, "rb") as f:
        model = pickle.load(f)

    booster = model.get_booster()
    raw_dumps = booster.get_dump(dump_format="json")
    tree_jsons = [json.loads(t) for t in raw_dumps]
    print(f"  Trees: {len(tree_jsons)}")

    all_features   = []
    all_thresholds = []
    all_lefts      = []
    all_rights     = []
    tree_offsets   = [0]

    for i, tree in enumerate(tree_jsons):
        feat, thresh, lch, rch = flatten_tree(tree)
        all_features.extend(feat)
        all_thresholds.extend(thresh)
        all_lefts.extend(lch)
        all_rights.extend(rch)
        tree_offsets.append(len(all_features))

    total_nodes = len(all_features)
    leaf_count  = sum(1 for f in all_features if f == -1)
    print(f"  Total nodes: {total_nodes}  (leaves: {leaf_count})")

    # Round thresholds to 6 significant figures to reduce JSON size
    all_thresholds = [round(v, 6) for v in all_thresholds]

    model_data = {
        "numTrees":    len(tree_jsons),
        "treeOffsets": tree_offsets,
        "features":    all_features,    # int  list, -1 for leaves
        "thresholds":  all_thresholds,  # float list (split threshold or leaf value)
        "left":        all_lefts,       # int  list, -1 for leaves
        "right":       all_rights,      # int  list, -1 for leaves
    }

    print(f"Writing to {OUT_PATH}")
    with open(OUT_PATH, "w") as f:
        json.dump(model_data, f, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"  File size: {size_kb:.0f} KB")
    print("Done.")


if __name__ == "__main__":
    main()
