"""Hand Gesture Recognition with true live webcam prediction.

The webcam streams individual frames continuously; each frame is predicted
as it arrives. No recording or submit button is required.
"""
import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_gesture_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
IMAGE_SIZE = (224, 224)

DISPLAY_NAMES = {
    "good": "Thumbs Up 👍",
    "no": "Thumbs Down 👎",
    "ok": "Victory Sign ✌️",
    "raise_hand": "Raise Hand ✋",
    "stop": "Stop 🛑",
}


def _compat_model_path():
    """Return a path Keras can load on any Keras 3 release.

    The model was saved with a newer Keras that writes config fields older
    releases reject (GlorotUniform input_axes/output_axes, BatchNormalization
    renorm*, quantization_config). We strip those from config.json; the
    weights are untouched. The patched copy is cached next to the model.
    """
    patched = BASE_DIR / "hand_gesture_model.compat.keras"
    if patched.is_file():
        return patched
    drop = {"GlorotUniform": ("input_axes", "output_axes"),
            "BatchNormalization": ("renorm", "renorm_clipping", "renorm_momentum")}

    def strip(o):
        if isinstance(o, dict):
            if isinstance(o.get("config"), dict):
                for k in drop.get(o.get("class_name"), ()) + ("quantization_config",):
                    o["config"].pop(k, None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for v in o:
                strip(v)

    with zipfile.ZipFile(MODEL_PATH) as src, zipfile.ZipFile(patched, "w", zipfile.ZIP_STORED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "config.json":
                cfg = json.loads(data)
                strip(cfg)
                data = json.dumps(cfg).encode()
            dst.writestr(item.filename, data)
    return patched


def _assemble_model_parts():
    """GitHub's web uploader caps files near 25 MB, so the model may be
    stored as hand_gesture_model.keras.part00, .part01, ... Join them."""
    parts = sorted(BASE_DIR.glob("*hand_gesture_model.keras.part*"))
    if parts and not MODEL_PATH.is_file():
        with MODEL_PATH.open("wb") as out:
            for p in parts:
                out.write(p.read_bytes())


@lru_cache(maxsize=1)
def get_model():
    _assemble_model_parts()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH.name}")
    # compile=False avoids restoring training-only optimizer/loss state.
    try:
        return load_model(MODEL_PATH, compile=False)
    except Exception:
        return load_model(_compat_model_path(), compile=False)


def load_classes():
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    return [mapping[str(i)] for i in range(len(mapping))]


CLASS_NAMES = load_classes()


def classify_frame(frame):
    """Classify one frame received from the live webcam stream."""
    if frame is None:
        return "Show your hand to the camera", {}

    try:
        # Gradio Image streaming supplies a NumPy array in RGB order.
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            return "Waiting for a valid camera frame…", {}
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        from PIL import Image
        image = Image.fromarray(rgb).convert("RGB").resize(IMAGE_SIZE)
    except (TypeError, ValueError, OSError):
        return "Could not read camera frame", {}

    batch = preprocess_input(np.asarray(image, dtype=np.float32))[None, ...]
    scores = np.asarray(get_model().predict(batch, verbose=0))[0]
    if scores.ndim != 1 or len(scores) != len(CLASS_NAMES) or not np.all(np.isfinite(scores)):
        return "Model output error", {}

    best_idx = int(np.argmax(scores))
    label = DISPLAY_NAMES.get(CLASS_NAMES[best_idx], CLASS_NAMES[best_idx])
    confidence = float(scores[best_idx])
    distribution = {
        DISPLAY_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i]): float(scores[i])
        for i in range(len(CLASS_NAMES))
    }
    return f"{label}  ·  {confidence:.1%} confidence", distribution


def classify_still(image):
    if image is None:
        raise gr.Error("Upload or capture an image first.")
    # Reuse the same prediction path as the live webcam.
    rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    label, scores = classify_frame(rgb)
    return label, scores


with gr.Blocks(title="Live Hand Gesture Recognition") as demo:
    gr.Markdown("# ✋ Live Hand Gesture Recognition")
    gr.Markdown(
        "Live mode predicts continuously from incoming webcam frames. "
        "You do **not** need to record a video or press Submit. "
        "Keep your hand visible with reasonable lighting."
    )

    with gr.Tab("🔴 Live prediction"):
        camera = gr.Image(
            sources=["webcam"],
            type="numpy",
            streaming=True,
            label="Live webcam — start the camera stream",
        )
        live_label = gr.Textbox(label="Current prediction", value="Start the camera to begin…")
        live_scores = gr.Label(num_top_classes=5, label="Class probabilities")
        camera.stream(
            fn=classify_frame,
            inputs=camera,
            outputs=[live_label, live_scores],
            stream_every=0.25,
            concurrency_limit=1,
            trigger_mode="always_last",
        )

    with gr.Tab("🖼️ Single image"):
        image_input = gr.Image(
            sources=["upload", "webcam"],
            type="pil",
            label="Upload an image or take a single photo",
        )
        predict_button = gr.Button("Predict image", variant="primary")
        still_label = gr.Textbox(label="Prediction")
        still_scores = gr.Label(num_top_classes=5, label="Class probabilities")
        predict_button.click(
            fn=classify_still,
            inputs=image_input,
            outputs=[still_label, still_scores],
        )

if __name__ == "__main__":
    on_colab = "COLAB_GPU" in os.environ or Path("/content").exists()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=on_colab,  # public temp link only on Colab; hosts give a real URL
        show_error=True,
    )
