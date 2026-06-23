"""Glove feature extraction: ESP32 JSON frame -> normalized 11D vector."""

from __future__ import annotations

import numpy as np

from config import (
    GLOVE_FEAT_DIM,
    GLOVE_FLEX_DIM,
    GLOVE_FLEX_REF_V,
    GLOVE_FLEX_HALF_RANGE_V,
    GLOVE_ACCEL_HALF_RANGE,
    GLOVE_GYRO_HALF_RANGE,
)


def extract_glove_vector(frame: dict | None) -> np.ndarray | None:
    """Build the normalized 11D feature vector from a glove_frame.

    Expected frame: {"flex": [5], "accel": [3], "gyro": [3]}.
    Returns float32 array (11,) or None if the frame is incomplete.
    Each channel is scaled to roughly [-1, 1] so flex, accel and gyro
    contribute comparably to the cosine distance.
    """
    if frame is None:
        return None

    flex = frame.get("flex")
    accel = frame.get("accel")
    gyro = frame.get("gyro")

    if (not isinstance(flex, list) or len(flex) < GLOVE_FLEX_DIM
            or not isinstance(accel, list) or len(accel) < 3
            or not isinstance(gyro, list) or len(gyro) < 3):
        return None

    out = np.empty(GLOVE_FEAT_DIM, dtype=np.float32)
    for i in range(GLOVE_FLEX_DIM):
        out[i] = (float(flex[i]) - GLOVE_FLEX_REF_V) / GLOVE_FLEX_HALF_RANGE_V
    for i in range(3):
        out[GLOVE_FLEX_DIM + i] = float(accel[i]) / GLOVE_ACCEL_HALF_RANGE
    for i in range(3):
        out[GLOVE_FLEX_DIM + 3 + i] = float(gyro[i]) / GLOVE_GYRO_HALF_RANGE

    np.clip(out, -3.0, 3.0, out=out)   # clamp mechanical-shock outliers
    return out


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return 1.0 - float(np.dot(a, b)) / (na * nb)
