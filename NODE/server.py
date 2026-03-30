from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
CHECKPOINT_PATH = REPO_ROOT / "W.pt"
SCALE = 2


class PredictRequest(BaseModel):
    imageDataUrl: str
    filename: str | None = None


class SRCNN(nn.Module):
    def __init__(self, num_channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

        for layer in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)


def load_model() -> SRCNN:
    model = SRCNN(num_channels=4)
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise RuntimeError(f"Checkpoint is missing model_state_dict: {CHECKPOINT_PATH}")
        model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL = load_model()

app = FastAPI(title="SR ROI Lab", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _decode_data_url(data_url: str) -> Image.Image:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    _, encoded = data_url.split(",", 1)
    try:
        raw = base64.b64decode(encoded)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Unable to decode image payload") from exc

    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Unable to read image") from exc


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise HTTPException(status_code=400, detail="Expected an RGB image")

    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    gray = rgb_tensor.mean(dim=1, keepdim=True)
    four_channel = torch.cat([rgb_tensor, gray], dim=1)
    return four_channel


def _tensor_to_data_url(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().clamp(0.0, 1.0).squeeze(0)
    if tensor.shape[0] not in {3, 4}:
        raise HTTPException(status_code=500, detail="Unexpected tensor shape")
    rgb = tensor[:3].permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _bicubic_upscale(tensor: torch.Tensor, scale_factor: int = SCALE) -> torch.Tensor:
    return F.interpolate(tensor, scale_factor=scale_factor, mode="bicubic", align_corners=False)


def _forward_tiled(model: nn.Module, tensor: torch.Tensor, tile_size: int = 192, overlap: int = 24) -> torch.Tensor:
    batch, channels, height, width = tensor.shape
    tile_size = min(tile_size, height, width)
    overlap = min(overlap, tile_size // 2)
    stride = max(1, tile_size - overlap)

    output = torch.zeros((batch, channels, height, width), dtype=torch.float32)
    weight = torch.zeros((batch, channels, height, width), dtype=torch.float32)

    y_positions = list(range(0, max(1, height - tile_size + 1), stride))
    x_positions = list(range(0, max(1, width - tile_size + 1), stride))

    if not y_positions or y_positions[-1] != max(0, height - tile_size):
        y_positions.append(max(0, height - tile_size))
    if not x_positions or x_positions[-1] != max(0, width - tile_size):
        x_positions.append(max(0, width - tile_size))

    for top in sorted(set(y_positions)):
        for left in sorted(set(x_positions)):
            patch = tensor[:, :, top : top + tile_size, left : left + tile_size]
            patch_output = model(patch).detach().to(torch.float32)
            patch_height = patch_output.shape[-2]
            patch_width = patch_output.shape[-1]
            output[:, :, top : top + patch_height, left : left + patch_width] += patch_output
            weight[:, :, top : top + patch_height, left : left + patch_width] += 1.0

    return output / torch.clamp(weight, min=1.0)


@app.post("/api/predict")
def predict(payload: PredictRequest) -> JSONResponse:
    image = _decode_data_url(payload.imageDataUrl)
    lr_tensor = _image_to_tensor(image)
    bicubic = _bicubic_upscale(lr_tensor, scale_factor=SCALE)

    start = time.perf_counter()
    with torch.no_grad():
        sr = _forward_tiled(MODEL, bicubic)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    result = {
        "filename": payload.filename or "uploaded-image",
        "inputWidth": int(image.width),
        "inputHeight": int(image.height),
        "scale": SCALE,
        "modelName": "SRCNN x2",
        "checkpoint": CHECKPOINT_PATH.name if CHECKPOINT_PATH.exists() else "random-init",
        "processingMs": round(elapsed_ms, 2),
        "bicubicDataUrl": _tensor_to_data_url(bicubic),
        "srDataUrl": _tensor_to_data_url(sr),
        "notice": "This workspace contains SRCNN checkpoints, so the site uses the repo's local 4-channel SRCNN model for prediction.",
    }
    return JSONResponse(result)