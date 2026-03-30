# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Literal

import numpy as np
import cv2  # type: ignore


def _to_gray_f32(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray / 255.0


def _sobel_mag(gray_f32: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray_f32, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return mag.astype(np.float32)


def _rank_census_lite(gray_f32: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray_f32, (3, 3), 0)
    pad = cv2.copyMakeBorder(g, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    center = pad[1:-1, 1:-1]
    h, w = center.shape[:2]
    count = np.zeros((h, w), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neigh = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            count += (neigh < center).astype(np.float32)
    rank = count / 8.0
    return _sobel_mag(rank)


def build_structure(
    bgr: np.ndarray,
    mode: Literal["sobel_mag", "rank_census_lite"] = "sobel_mag",
) -> np.ndarray:
    m = str(mode).strip().lower()
    gray = _to_gray_f32(bgr)
    if m == "sobel_mag":
        return _sobel_mag(gray)
    if m == "rank_census_lite":
        return _rank_census_lite(gray)
    raise ValueError(f"Unknown structure mode: {mode}")
