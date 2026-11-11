"""Standalone inference for E2 N7K11 binary-pressure/load models."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


N7_TOPOLOGIES = {
    "N7-A": ((1, 3), (1, 9), (3, 5), (3, 9), (3, 11),
             (5, 7), (5, 11), (5, 13), (7, 13), (13, 11), (11, 9)),
    "N7-B": ((2, 4), (2, 10), (2, 8), (4, 6), (4, 12),
             (4, 10), (6, 14), (6, 12), (14, 12), (12, 10), (10, 8)),
}


def unordered_pair_indices(original_pairs: np.ndarray,
                           selected_pairs: Iterable[tuple[int, int]]) -> np.ndarray:
    """Map physical pairs to canonical columns without assuming orientation/order."""
    original = np.asarray(original_pairs, dtype=int)
    if original.shape != (91, 2):
        raise ValueError(f"original_pairs must have shape (91,2), got {original.shape}")
    lookup = {}
    for index, pair in enumerate(original):
        key = frozenset(map(int, pair))
        if len(key) != 2 or key in lookup:
            raise ValueError("Canonical pairs contain an invalid or duplicate unordered pair")
        lookup[key] = index
    indices = []
    for pair in selected_pairs:
        key = frozenset(map(int, pair))
        if key not in lookup:
            raise KeyError(f"Selected physical pair is absent from canonical pairs: {pair}")
        indices.append(lookup[key])
    if len(set(indices)) != len(indices):
        raise ValueError("Selected topology contains duplicate measurements")
    return np.asarray(indices, dtype=np.int64)


def raw_features(z91: np.ndarray, z_all_0: np.ndarray,
                 pair_indices: np.ndarray) -> np.ndarray:
    """Return exactly [magnitude ratio, sin(delta phase), cos(delta phase)]."""
    z = np.asarray(z91, dtype=np.complex128)
    reference = np.asarray(z_all_0, dtype=np.complex128).reshape(-1)
    indices = np.asarray(pair_indices, dtype=int).reshape(-1)
    if z.ndim == 1:
        z = z[None, :]
    if z.ndim != 2 or z.shape[1] != 91 or reference.shape != (91,):
        raise ValueError("Expected z91 shape (N,91) and z_all_0 shape (91,)")
    if indices.shape != (11,) or np.any(indices < 0) or np.any(indices >= 91):
        raise ValueError("Expected 11 valid canonical pair indices")
    if not np.isfinite(z).all() or not np.isfinite(reference).all() or np.any(reference == 0):
        raise ValueError("Nonfinite or zero complex impedance")
    selected = z[:, indices]
    ref = reference[indices]
    phase_delta = np.angle(selected / ref[None, :])
    features = np.stack((np.abs(selected) / np.abs(ref)[None, :],
                         np.sin(phase_delta), np.cos(phase_delta)), axis=-1)
    if features.shape != (len(z), 11, 3) or not np.isfinite(features).all():
        raise ValueError("Invalid N7K11 features")
    return features


def normalize_features(features: np.ndarray, mean: np.ndarray,
                       std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float64).reshape(1, 1, 3)
    if np.any(std <= 0) or not np.isfinite([mean, std]).all():
        raise ValueError("Invalid checkpoint normalization statistics")
    return ((np.asarray(features) - mean) / std).astype(np.float32)


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout),
        )
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class N7K11DualHeadCNN(nn.Module):
    """Established N7K11 residual backbone with binary and load heads."""
    def __init__(self, max_load_kg: float = 1.0):
        super().__init__()
        self.max_load_kg = float(max_load_kg)
        self.backbone = nn.Sequential(
            ResidualBlock(3, 64), ResidualBlock(64, 128), nn.MaxPool1d(2),
            ResidualBlock(128, 256), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 19))
        self.regressor = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 19))

    def forward(self, x):
        h = self.backbone(x.transpose(1, 2))
        logits = self.classifier(h)
        load_kg = self.max_load_kg * torch.sigmoid(self.regressor(h))
        return logits, load_kg


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model = N7K11DualHeadCNN(checkpoint["model_config"]["max_load_kg"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict_from_checkpoint(checkpoint_path: str | Path, z91: np.ndarray,
                            device: str = "cpu", batch_size: int = 1024) -> dict:
    """Predict probabilities, binary states, and continuous loads from (N,91) Z."""
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    features = raw_features(z91, checkpoint["Z_all_0"], checkpoint["pair_indices"])
    x = normalize_features(features, checkpoint["feature_mean"], checkpoint["feature_std"])
    probabilities, loads = [], []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start:start + batch_size]).to(device)
        logits, load = model(xb)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        loads.append(load.cpu().numpy())
    probability = np.concatenate(probabilities).astype(np.float32)
    load_kg = np.concatenate(loads).astype(np.float32)
    threshold = float(checkpoint["classification_threshold"])
    binary = (probability >= threshold).astype(np.uint8)
    # The continuous head remains available independently, but inference output
    # enforces the requested physical zero for predicted-unpressed sensors.
    gated_load_kg = load_kg * binary
    return {"pressed_probability": probability, "pressed_binary": binary,
            "load_kg": gated_load_kg, "raw_load_kg": load_kg}

