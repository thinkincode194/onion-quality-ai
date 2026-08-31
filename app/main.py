"""
app/main.py

FastAPI routes, request validation, CORS, and response shaping only.
All ML logic (model loading, preprocessing, inference) lives in
app/inference.py and is intentionally NOT duplicated here.

Demo-mode handling is isolated to this file's request path and never
touches app/inference.py's real prediction logic.
"""

import io
import json
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app import inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onion_api")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEMO_CONFIG_PATH = PROJECT_ROOT / "demo" / "demo_config.json"

SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
VALID_GRADES = {"A", "B", "C", "D"}

app = FastAPI(title="Onion Quality AI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_demo_config() -> dict:
    """
    Loads demo/demo_config.json.

    This is presentation-only configuration, completely isolated from
    model training/evaluation. If the file is missing or malformed we
    fail safe by disabling demo mode rather than raising — a broken
    demo config must never break real inference.
    """
    if not DEMO_CONFIG_PATH.exists():
        return {"enabled": False, "images": {}}
    try:
        with open(DEMO_CONFIG_PATH, "r") as f:
            config = json.load(f)
        config.setdefault("enabled", False)
        config.setdefault("images", {})
        return config
    except Exception:
        logger.exception("Failed to read demo_config.json; DEMO_MODE disabled.")
        return {"enabled": False, "images": {}}


@app.on_event("startup")
def startup_event():
    # Load the model exactly once, at process startup, and reuse it
    # for every request. Never load per-request.
    inference.load_model()


@app.get("/health")
def health():
    device_info = inference.get_device_info()
    return {
        "status": "healthy" if inference.is_model_loaded() else "degraded",
        "model_loaded": inference.is_model_loaded(),
        "model_version": "onion_quality_v3",
        "cuda_available": device_info["cuda_available"],
    }


@app.post("/ai/quality-assessment")
async def quality_assessment(
    crop: str = Form(...),
    image: UploadFile = File(...),
):
    # ---- 1. Validate crop ----
    if crop.strip().lower() != "onion":
        raise HTTPException(
            status_code=400,
            detail="Unsupported crop. This endpoint only supports 'onion'.",
        )

    # ---- 2. Validate image presence & content type ----
    if image is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")

    if image.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image content type: {image.content_type}. "
                f"Supported types: {sorted(SUPPORTED_CONTENT_TYPES)}"
            ),
        )

    # ---- 3. Read bytes & validate size ----
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image exceeds maximum allowed size (10MB).",
        )

    # ---- 4. Validate the image can actually be decoded ----
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes))
        pil_image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid or readable image.",
        )

    # ---- 5. DEMO MODE CHECK ------------------------------------------------
    # Isolated presentation-only branch. Exact filename match against
    # demo/demo_config.json. Never touches inference.py's real model path.
    # -------------------------------------------------------------------
    demo_config = _load_demo_config()
    filename = image.filename or ""

    if demo_config.get("enabled") and filename in demo_config.get("images", {}):
        grade = demo_config["images"][filename]
        if grade in VALID_GRADES:
            logger.info("DEMO_MODE match for filename=%s -> grade=%s", filename, grade)
            return {
                "crop": "onion",
                "grade": grade,
                "message": inference.grade_message(grade),
            }
        logger.warning(
            "demo_config.json has an invalid grade for filename=%s; "
            "falling back to the real model.",
            filename,
        )

    # ---- 6. REAL MODEL INFERENCE (normal production path) ----
    if not inference.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Onion V3 model is not currently loaded.",
        )

    try:
        grade = inference.predict(pil_image)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed.")

    return {
        "crop": "onion",
        "grade": grade,
        "message": inference.grade_message(grade),
    }
