"""
app/inference.py

Owns everything ML-related for Onion V3:
- loading the local trained checkpoint (once, at startup)
- deterministic evaluation/inference preprocessing
- running the model and mapping logits -> grade

This module intentionally contains NO FastAPI code, NO request handling,
and NO demo-mode logic. Those live in app/main.py, so the ML layer stays
isolated and easy to reason about / test independently.
"""

import logging
import os

import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image

logger = logging.getLogger("onion_inference")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Local trained Onion V3 checkpoint. Must NOT default to the base
# "google/vit-base-patch16-224" checkpoint — that is the pretrained
# starting point, not our fine-tuned model.
ONION_MODEL_DIR = os.environ.get("ONION_MODEL_DIR", "model/best_model")

LABELS = {0: "A", 1: "B", 2: "C", 3: "D"}

GRADE_MESSAGES = {
    "A": "Premium quality onion suitable for consumption.",
    "B": "Good quality onion with minor visible imperfections.",
    "C": "Acceptable quality onion with noticeable imperfections.",
    "D": "Poor quality onion; not recommended for consumption.",
}

# ----------------------------------------------------------------------
# Deterministic V3 evaluation/inference preprocessing.
#
# IMPORTANT: This is the ONLY preprocessing applied during production
# inference. Training-time augmentation (RandomHorizontalFlip,
# RandomRotation, ColorJitter, etc.) is NOT part of this pipeline and
# must never be added here — it would make predictions non-deterministic
# and inconsistent with how the model was evaluated.
# ----------------------------------------------------------------------
_INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Module-level singletons populated by load_model() at app startup.
_model = None
_device = None
_model_load_error = None


def load_model() -> None:
    """
    Loads the Onion V3 checkpoint exactly once. Intended to be called
    from the FastAPI startup event, not per-request.
    """
    global _model, _device, _model_load_error

    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = ViTForImageClassification.from_pretrained(ONION_MODEL_DIR)
        _model.to(_device)
        _model.eval()  # inference mode — disables dropout etc.
        _model_load_error = None
        logger.info("Onion V3 model loaded from '%s' on device=%s", ONION_MODEL_DIR, _device)
    except Exception as exc:  # noqa: BLE001 - we want to capture and surface any load failure
        _model = None
        _model_load_error = str(exc)
        logger.exception("Failed to load Onion V3 model from '%s'", ONION_MODEL_DIR)


def is_model_loaded() -> bool:
    return _model is not None


def get_model_load_error():
    return _model_load_error


def get_device_info() -> dict:
    return {
        "cuda_available": torch.cuda.is_available(),
        "device": str(_device) if _device is not None else None,
    }


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    RGB -> Resize(224,224) -> ToTensor -> Normalize(mean=0.5, std=0.5)

    This must exactly match the V3 evaluation pipeline used during
    training/validation. Do not add any random/augmentation transforms.
    """
    rgb_image = image.convert("RGB")
    tensor = _INFERENCE_TRANSFORM(rgb_image)
    return tensor.unsqueeze(0)  # add batch dimension


def predict(image: Image.Image) -> str:
    """
    Runs the REAL Onion V3 model on a single image and returns the
    predicted grade ("A" | "B" | "C" | "D").

    Raises RuntimeError if the model is not loaded or produces an
    unexpected output — callers (app/main.py) are responsible for
    translating that into an HTTP error rather than fabricating a grade.
    """
    if _model is None:
        raise RuntimeError("Onion V3 model is not loaded")

    input_tensor = preprocess_image(image).to(_device)

    with torch.no_grad():
        outputs = _model(pixel_values=input_tensor)
        logits = outputs.logits
        predicted_idx = int(torch.argmax(logits, dim=-1).item())

    grade = LABELS.get(predicted_idx)
    if grade is None:
        raise RuntimeError(f"Unexpected class index from model: {predicted_idx}")
    return grade


def grade_message(grade: str) -> str:
    """Generic, fixed description of a quality grade (not model output)."""
    return GRADE_MESSAGES.get(grade, "")
