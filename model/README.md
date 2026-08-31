# Onion V3 Model Checkpoint

This directory is where the trained Onion V3 checkpoint must live for the
backend to load it.

## Expected layout

```
model/best_model/
├── config.json
├── model.safetensors
├── preprocessor_config.json
└── metadata.json
```

## Important

The actual trained checkpoint files (`config.json`, `model.safetensors`,
`preprocessor_config.json`, `metadata.json`) are **not included in this
handoff package** — they were not provided as input to this packaging
step, and this package does not retrain or fabricate them.

Copy your already-trained V3 checkpoint from:

```
/content/drive/MyDrive/Produce Quality Project/Models/onion_v3/best_model
```

into `model/best_model/` before running the backend. The application
reads this path via the `ONION_MODEL_DIR` environment variable, which
defaults to `model/best_model` (see `app/inference.py`).

**Do not** point `ONION_MODEL_DIR` at `google/vit-base-patch16-224` — that
is the base pretrained architecture the model was fine-tuned from, not
the trained Onion V3 checkpoint. Doing so would silently swap in an
untrained model.

**Do not** modify `model.safetensors` or any other file in this
directory — it is the frozen, already-trained V3 checkpoint.
