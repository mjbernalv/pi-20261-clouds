from __future__ import annotations
import torch.nn.functional as F
from pathlib import Path
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import time
import os
import io


CLASSES = ["Fish", "Flower", "Gravel", "Sugar"]
DEFAULT_THRESHOLD = 0.5

_model = None
_config: dict = {}
_threshold: float = DEFAULT_THRESHOLD


# ── Model architecture (must match training) ───────────────────────────────────

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.relu(self.gate_proj(gate) + self.skip_proj(skip)))
        return skip * attention


class AttentionResidualUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 4,
                 features: list | None = None, dropout: float = 0.10) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        current = in_channels
        for f in features:
            self.downs.append(ResidualConvBlock(current, f))
            current = f

        self.bottleneck = ResidualConvBlock(features[-1], features[-1] * 2, dropout=dropout)

        self.up_transpose = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        decoder = features[-1] * 2
        for f in reversed(features):
            self.up_transpose.append(nn.ConvTranspose2d(decoder, f, 2, 2))
            self.attentions.append(AttentionGate(f, f, max(f // 2, 16)))
            self.up_blocks.append(ResidualConvBlock(f * 2, f, dropout=dropout / 2))
            decoder = f

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(len(self.up_transpose)):
            x = self.up_transpose[i](x)
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            skip = self.attentions[i](x, skip)
            x = torch.cat([skip, x], dim=1)
            x = self.up_blocks[i](x)

        return self.final_conv(x)


# ── Model loading ──────────────────────────────────────────────────────────────

def get_model() -> AttentionResidualUNet:
    global _model, _config, _threshold
    if _model is not None:
        return _model

    path = os.environ.get(
        "MODEL_PATH",
        str(Path(__file__).parent.parent / "model" / "model.pt"),
    )
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Place your .pt file there or set the MODEL_PATH env var."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    _config = checkpoint.get("config", {})
    _threshold = float(_config.get("threshold", DEFAULT_THRESHOLD))

    _model = AttentionResidualUNet(in_channels=3, out_channels=len(CLASSES))
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    _model.to(device)
    return _model


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resize_mask(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    img = img.resize((target_w, target_h), Image.NEAREST)
    return (np.asarray(img) > 0).astype(np.uint8)


def _mask_to_rle(mask: np.ndarray) -> str:
    pixels = mask.T.flatten().astype(np.uint8)
    padded = np.concatenate([[0], pixels, [0]])
    edges = np.where(padded[1:] != padded[:-1])[0] + 1
    edges[1::2] -= edges[::2]
    return " ".join(str(int(x)) for x in edges)


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(image_bytes: bytes) -> dict:
    started = time.perf_counter()
    model = get_model()
    device = next(model.parameters()).device

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_w, original_h = image.size

    img_size = _config.get("img_size", 384)
    arr = np.asarray(image.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze(0).cpu().numpy()  # (4, H, W)

    predictions = {}
    for idx, cls in enumerate(CLASSES):
        binary = (probs[idx] > _threshold).astype(np.uint8)
        full = _resize_mask(binary, original_w, original_h)
        predictions[cls] = {
            "rle": _mask_to_rle(full) if full.any() else "",
            "pixel_count": int(full.sum()),
        }

    return {
        "width": original_w,
        "height": original_h,
        "predictions": predictions,
        "inference_ms": int((time.perf_counter() - started) * 1000),
    }
