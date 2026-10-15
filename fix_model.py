"""Rewrite a .keras config so it loads on older/newer Keras 3 releases.
Usage: python fix_model.py model_dir out.keras
Removes serialized fields newer Keras adds (GlorotUniform input/output_axes,
BatchNormalization renorm*). Weights are untouched."""
import json, sys, zipfile
from pathlib import Path
d, out = Path(sys.argv[1]), sys.argv[2]
cfg = json.loads((d/"config.json").read_text())
DROP = {"GlorotUniform": ("input_axes","output_axes"),
        "BatchNormalization": ("renorm","renorm_clipping","renorm_momentum")}
def strip(o):
    if isinstance(o, dict):
        if isinstance(o.get("config"), dict):
            for k in DROP.get(o.get("class_name"), ()) + ("quantization_config",):
                o["config"].pop(k, None)
        for v in o.values(): strip(v)
    elif isinstance(o, list):
        for v in o: strip(v)
strip(cfg)
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    z.writestr("config.json", json.dumps(cfg))
    z.write(d/"metadata.json", "metadata.json")
    z.write(d/"model.weights.h5", "model.weights.h5")
