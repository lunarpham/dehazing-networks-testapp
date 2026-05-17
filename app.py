"""
Dehazing Model Testing Web Application
Flask backend for model upload, selection, and inference.
"""

import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory

# ── Configuration ────────────────────────────────────────────────────────────

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.toml"

with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

# Resolve DehazeNet project path and add to sys.path for imports
DEHAZENET_PATH = Path(CONFIG["project"]["dehazenet_path"])
if not DEHAZENET_PATH.is_absolute():
    DEHAZENET_PATH = (APP_DIR / DEHAZENET_PATH).resolve()

if str(DEHAZENET_PATH) not in sys.path:
    sys.path.insert(0, str(DEHAZENET_PATH))

# Now import DehazeNet modules
import torch
import numpy as np
from src.models import build_model, DIRECT_MODELS
from src.core import get_dark_channel, estimate_atmospheric_light, recover_image
from src.utils import load_image, to_tensor, save_image

# ── Paths ────────────────────────────────────────────────────────────────────

UPLOAD_MODELS_DIR = APP_DIR / "uploads" / "models"
UPLOAD_IMAGES_DIR = APP_DIR / "uploads" / "images"
RESULTS_DIR = APP_DIR / "results"
MODELS_META_FILE = UPLOAD_MODELS_DIR / "models.json"

for d in [UPLOAD_MODELS_DIR, UPLOAD_IMAGES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model Metadata Store ────────────────────────────────────────────────────

def _load_models_meta() -> dict:
    """Load model metadata from JSON file."""
    if MODELS_META_FILE.exists():
        with open(MODELS_META_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_models_meta(meta: dict):
    """Save model metadata to JSON file."""
    with open(MODELS_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


# ── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB max upload

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DehazeTest] Using device: {DEVICE}")

# Cache loaded models to avoid reloading on every inference
_model_cache = {}


def _get_model(model_id: str):
    """Load and cache a model by its ID."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    meta = _load_models_meta()
    if model_id not in meta:
        raise ValueError(f"Model not found: {model_id}")

    info = meta[model_id]
    arch_type = info["arch_type"]
    pth_path = UPLOAD_MODELS_DIR / info["filename"]

    # Build model architecture
    config = {"network": {"type": arch_type}}
    model = build_model(config).to(DEVICE)

    # Load weights
    state_dict = torch.load(str(pth_path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    is_direct = arch_type in DIRECT_MODELS

    _model_cache[model_id] = (model, is_direct)
    return model, is_direct


def _run_inference(model, is_direct: bool, img_np: np.ndarray) -> np.ndarray:
    """Run inference on a single image."""
    physics_cfg = CONFIG.get("physics", {})

    img_tensor = to_tensor(img_np).to(DEVICE)

    with torch.no_grad():
        if is_direct:
            dehazed = model(img_tensor)
        else:
            # Transmission-based: predict t(x), then physics inversion
            t_pred = model(img_tensor)

            dark_channel = get_dark_channel(
                img_tensor,
                window_size=physics_cfg.get("dark_channel_window", 15)
            )
            atm_light = estimate_atmospheric_light(
                img_tensor, dark_channel,
                top_percent=physics_cfg.get("atm_light_top_percent", 0.001)
            )
            dehazed = recover_image(
                img_tensor, t_pred, atm_light,
                t0=physics_cfg.get("t_min", 0.1)
            )

    # Convert back to numpy HWC
    result = dehazed.squeeze(0).cpu().numpy()
    result = np.transpose(result, (1, 2, 0))  # CHW -> HWC
    result = np.clip(result, 0.0, 1.0)
    return result


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main SPA page."""
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    """List all uploaded models."""
    meta = _load_models_meta()
    models = []
    for model_id, info in meta.items():
        models.append({
            "id": model_id,
            "name": info["name"],
            "arch_type": info["arch_type"],
            "filename": info["filename"],
            "uploaded_at": info.get("uploaded_at", ""),
        })
    return jsonify({"models": models})


@app.route("/api/models/upload", methods=["POST"])
def upload_model():
    """Upload a .pth model file with architecture type."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    arch_type = request.form.get("arch_type", "dehazenet_plus")

    if not file.filename.endswith(".pth"):
        return jsonify({"error": "Only .pth files are accepted"}), 400

    valid_types = ["dehazenet", "dehazenet_plus", "dehazenet_direct", "dehazenet_hybrid", "aodnet", "aodnet_enhanced"]
    if arch_type not in valid_types:
        return jsonify({"error": f"Invalid architecture type. Must be one of: {valid_types}"}), 400

    # Generate unique filename to avoid collisions
    model_id = str(uuid.uuid4())[:8]
    original_name = Path(file.filename).stem
    safe_filename = f"{original_name}_{model_id}.pth"

    save_path = UPLOAD_MODELS_DIR / safe_filename
    file.save(str(save_path))

    # Validate that the weights can actually be loaded into the architecture
    try:
        config = {"network": {"type": arch_type}}
        test_model = build_model(config)
        state_dict = torch.load(str(save_path), map_location="cpu", weights_only=True)
        test_model.load_state_dict(state_dict)
        del test_model
    except Exception as e:
        # Clean up the invalid file
        save_path.unlink(missing_ok=True)
        return jsonify({
            "error": f"Weight file is incompatible with '{arch_type}' architecture: {str(e)}"
        }), 400

    # Save metadata
    meta = _load_models_meta()
    meta[model_id] = {
        "name": original_name,
        "arch_type": arch_type,
        "filename": safe_filename,
        "uploaded_at": datetime.now().isoformat(),
    }
    _save_models_meta(meta)

    return jsonify({
        "id": model_id,
        "name": original_name,
        "arch_type": arch_type,
        "message": "Model uploaded successfully",
    })


@app.route("/api/models/<model_id>", methods=["DELETE"])
def delete_model(model_id):
    """Delete an uploaded model."""
    meta = _load_models_meta()

    if model_id not in meta:
        return jsonify({"error": "Model not found"}), 404

    # Remove the file
    info = meta[model_id]
    pth_path = UPLOAD_MODELS_DIR / info["filename"]
    if pth_path.exists():
        pth_path.unlink()

    # Remove from cache
    _model_cache.pop(model_id, None)

    # Remove from metadata
    del meta[model_id]
    _save_models_meta(meta)

    return jsonify({"message": "Model deleted"})


@app.route("/api/infer", methods=["POST"])
def infer():
    """Run inference on an uploaded image with the selected model."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    model_id = request.form.get("model_id")
    if not model_id:
        return jsonify({"error": "No model selected"}), 400

    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save the uploaded image
    img_id = str(uuid.uuid4())[:8]
    ext = Path(image_file.filename).suffix or ".png"
    input_filename = f"{img_id}_input{ext}"
    input_path = UPLOAD_IMAGES_DIR / input_filename
    image_file.save(str(input_path))

    try:
        # Load model
        model, is_direct = _get_model(model_id)

        # Load image and run inference
        img_np = load_image(str(input_path))
        result_np = _run_inference(model, is_direct, img_np)

        # Save result
        output_filename = f"{img_id}_output.png"
        output_path = RESULTS_DIR / output_filename

        # Convert to uint8 and save via PIL
        from PIL import Image
        result_uint8 = (result_np * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(result_uint8).save(str(output_path))

        return jsonify({
            "input_url": f"/api/images/uploads/{input_filename}",
            "output_url": f"/api/images/results/{output_filename}",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/images/uploads/<filename>")
def serve_upload(filename):
    """Serve uploaded images."""
    return send_from_directory(str(UPLOAD_IMAGES_DIR), filename)


@app.route("/api/images/results/<filename>")
def serve_result(filename):
    """Serve result images."""
    return send_from_directory(str(RESULTS_DIR), filename)


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server_cfg = CONFIG.get("server", {})
    app.run(
        host=server_cfg.get("host", "127.0.0.1"),
        port=server_cfg.get("port", 5000),
        debug=server_cfg.get("debug", True),
    )
