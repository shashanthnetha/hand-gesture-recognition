---
title: Live Hand Gesture Recognition
emoji: ✋
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# ✋ Live Hand Gesture Recognition

Gradio app that classifies hand gestures from a live webcam stream (or a single image) using a MobileNetV2 transfer-learning model.

Classes: Thumbs Up, Thumbs Down, Victory Sign, Raise Hand, Stop.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860. `hand_gesture_model.keras` and `class_names.json` must sit next to `app.py`.

## Notes

- `app_original.py` is the original Colab version, kept unchanged.
- `app.py` adds a loader fallback: the model was saved with Keras 3.15, and older Keras rejects some config fields. If loading fails, the app strips those fields (weights untouched) and retries. `fix_model.py` does the same offline.
- On Colab it still creates the temporary public link; on a host it serves on `$PORT` (default 7860).
