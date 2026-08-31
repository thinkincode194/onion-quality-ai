"""
tests/test_demo_images.py

Integration smoke tests for the Onion Quality AI API.

These verify API *wiring* — health reporting, request validation, the
demo-mode path, and the response contract — using the four controlled
demo fixtures plus a couple of edge cases.

This is explicitly NOT a model accuracy benchmark. It does not measure
real-world classification accuracy of the Onion V3 model; it only
confirms the API behaves correctly end-to-end.
"""

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_CONFIG_PATH = PROJECT_ROOT / "demo" / "demo_config.json"
VALID_GRADES = {"A", "B", "C", "D"}


def _make_dummy_image_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(128, 64, 32)).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------
# 1 & 2. Health check + model loaded confirmation
# ---------------------------------------------------------------------
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "model_version" in body
    assert body["model_version"] == "onion_quality_v3"


def test_health_reports_model_loaded_flag_type():
    response = client.get("/health")
    body = response.json()
    assert isinstance(body["model_loaded"], bool)


# ---------------------------------------------------------------------
# 3-7. Upload each controlled demo image, call the endpoint, verify
#      status/grade/response shape individually.
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename",
    [
        "onion_demo_A.png",
        "onion_demo_B.png",
        "onion_demo_C.png",
        "onion_demo_D.png",
    ],
)
def test_demo_images_return_expected_grade(filename):
    if not DEMO_CONFIG_PATH.exists():
        pytest.skip("demo_config.json not present")

    demo_config = json.loads(DEMO_CONFIG_PATH.read_text())
    if not demo_config.get("enabled"):
        pytest.skip("DEMO_MODE disabled")

    expected_grade = demo_config["images"].get(filename)
    if expected_grade is None:
        pytest.skip(f"{filename} not configured in demo_config.json")

    files = {"image": (filename, _make_dummy_image_bytes(), "image/png")}
    data = {"crop": "onion"}
    response = client.post("/ai/quality-assessment", data=data, files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["crop"] == "onion"
    assert body["grade"] == expected_grade
    assert body["grade"] in VALID_GRADES
    assert "message" in body and body["message"]


def test_unknown_filename_goes_to_real_model_path():
    """
    A filename not present in demo_config.json must use the real V3
    model. If the model isn't loaded in this test environment, the API
    should fail loudly (503) rather than fabricate a grade.
    """
    files = {"image": ("user_uploaded_onion.jpg", _make_dummy_image_bytes(), "image/jpeg")}
    data = {"crop": "onion"}
    response = client.post("/ai/quality-assessment", data=data, files=files)

    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert body["crop"] == "onion"
        assert body["grade"] in VALID_GRADES
        assert "message" in body and body["message"]


# ---------------------------------------------------------------------
# Basic request validation
# ---------------------------------------------------------------------
def test_missing_image_is_rejected():
    response = client.post("/ai/quality-assessment", data={"crop": "onion"})
    assert response.status_code in (400, 422)


def test_invalid_crop_is_rejected():
    files = {"image": ("test.png", _make_dummy_image_bytes(), "image/png")}
    data = {"crop": "tomato"}
    response = client.post("/ai/quality-assessment", data=data, files=files)
    assert response.status_code == 400


def test_unsupported_content_type_is_rejected():
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    data = {"crop": "onion"}
    response = client.post("/ai/quality-assessment", data=data, files=files)
    assert response.status_code == 400


def test_corrupted_image_is_rejected():
    files = {"image": ("broken.png", b"\x89PNG\r\n\x1a\nnot-really-a-png", "image/png")}
    data = {"crop": "onion"}
    response = client.post("/ai/quality-assessment", data=data, files=files)
    assert response.status_code == 400
