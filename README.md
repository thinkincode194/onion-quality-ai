# Onion Quality AI — Backend (V3)

FastAPI backend that serves the trained **Onion V3** model
(`google/vit-base-patch16-224` architecture, fine-tuned for 4 quality
classes: A / B / C / D) for onion quality grading.

## Project structure

```
onion-quality-ai/
├── app/
│   ├── __init__.py
│   ├── inference.py      # model loading, preprocessing, ML inference
│   └── main.py            # FastAPI routes, validation, CORS, responses
├── model/
│   ├── README.md
│   └── best_model/        # place the trained V3 checkpoint here
├── demo/
│   └── demo_config.json   # presentation-only demo mode config
├── tests/
│   └── test_demo_images.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Make sure the trained checkpoint is present at `model/best_model/`
(see `model/README.md`) before starting the server.

---

## Developer Integration Notes

### A. API endpoint

```
POST /ai/quality-assessment
```

### B. Request

`multipart/form-data` with two fields:

| field | value |
|---|---|
| `crop` | `"onion"` |
| `image` | the original uploaded image file |

Example (curl):

```bash
curl -X POST http://localhost:8000/ai/quality-assessment \
  -F "crop=onion" \
  -F "image=@/path/to/photo.jpg"
```

### C. Response

```json
{
    "crop": "onion",
    "grade": "A",
    "message": "Premium quality onion suitable for consumption."
}
```

`grade` is always one of: `A`, `B`, `C`, `D`.

Fixed generic messages per grade:

| grade | message |
|---|---|
| A | Premium quality onion suitable for consumption. |
| B | Good quality onion with minor visible imperfections. |
| C | Acceptable quality onion with noticeable imperfections. |
| D | Poor quality onion; not recommended for consumption. |

These messages are generic descriptions of the project's quality grades.
They do **not** claim the model detected any specific defect — the
model performs classification only, not defect detection.

### D. Model location

```
model/best_model
```

Configurable via the `ONION_MODEL_DIR` environment variable (defaults to
`model/best_model`).

### E. Model

**Onion V3** — `google/vit-base-patch16-224` architecture, fine-tuned
for 4 classes (A, B, C, D). Loaded once at application startup via
`ViTForImageClassification.from_pretrained(ONION_MODEL_DIR)` and reused
for every request (never reloaded per-request). `model.eval()` is set
after loading, and inference runs inside `torch.no_grad()`.

### F. Image Preprocessing

**The frontend sends the original uploaded image. All ML preprocessing
is handled by the backend inference layer** (`app/inference.py`). The
frontend must NOT resize, normalize, tensorize, or otherwise transform
the image for ML purposes — it just uploads the file as-is.

Backend preprocessing pipeline (deterministic, applied to every
non-demo request):

```
image
  → RGB
  → Resize(224, 224)
  → ToTensor()
  → Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  → model
  → logits
  → argmax
  → grade (A/B/C/D)
```

### G. Training augmentation warning

**Training augmentation is NOT used during inference.**

The training pipeline used augmentation such as `RandomHorizontalFlip`,
`RandomRotation`, and `ColorJitter`. These are training-only transforms
and are intentionally absent from `app/inference.py`. Production
inference uses only the deterministic pipeline described above in
section F.

### H. Demo mode

- Configuration file: `demo/demo_config.json`
- Enabled/disabled via the top-level `"enabled": true|false` key in that
  file. To disable demo mode for a real presentation or production run,
  set `"enabled": false` (or delete/rename the file — the API fails
  safe to "demo disabled" if the config is missing or malformed).
- **This is presentation-only** functionality for controlled demo
  fixtures. It performs an *exact filename match* against the uploaded
  file's name; it never modifies `model.safetensors`, training code, or
  validation/test metrics, and it never overwrites real inference logic
  in `app/inference.py`.
- Any uploaded filename that is **not** an exact match in
  `demo/demo_config.json` — including all real user uploads — goes
  through the actual Onion V3 model in `app/inference.py`.

Example config:

```json
{
    "enabled": true,
    "images": {
        "onion_demo_A.png": "A",
        "onion_demo_B.png": "B",
        "onion_demo_C.png": "C",
        "onion_demo_D.png": "D"
    }
}
```

---

## Health check

```
GET /health
```

```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_version": "onion_quality_v3",
    "cuda_available": true
}
```

`status` reports `"degraded"` and `model_loaded` reports `false` if the
V3 checkpoint failed to load at startup.

## Testing

```bash
pytest tests/
```

`tests/test_demo_images.py` is an **API wiring smoke test**, not a
model accuracy benchmark. It checks `/health`, uploads each of the four
controlled demo images and verifies the expected grade/response shape,
confirms unknown filenames route to the real model path, and checks
basic request validation (missing image, wrong crop, unsupported file
type, corrupted image).
