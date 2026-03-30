# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2  # type: ignore


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    af -= float(af.mean())
    bf -= float(bf.mean())
    denom = (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-6)
    return float((af * bf).sum() / denom)


def grad_ncc(a: np.ndarray, b: np.ndarray) -> float:
    return _ncc(a, b)


def edge_f1(
    a: np.ndarray,
    b: np.ndarray,
    thresholds: Tuple[int, int] = (50, 150),
) -> float:
    t1, t2 = thresholds
    a_u8 = np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8)
    b_u8 = np.clip(b * 255.0, 0.0, 255.0).astype(np.uint8)
    ea = cv2.Canny(a_u8, t1, t2)
    eb = cv2.Canny(b_u8, t1, t2)
    ea_bin = ea > 0
    eb_bin = eb > 0
    tp = int(np.logical_and(ea_bin, eb_bin).sum())
    fp = int(np.logical_and(ea_bin, ~eb_bin).sum())
    fn = int(np.logical_and(~ea_bin, eb_bin).sum())
    denom = (2 * tp + fp + fn)
    if denom <= 0:
        return 0.0
    return float(2.0 * tp / denom)


def combine_quality(
    ecc_score: Optional[float],
    grad_ncc_score: Optional[float],
    edge_f1_score: Optional[float],
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> float:
    w_ecc, w_ncc, w_edge = weights
    vals = []
    ws = []
    if ecc_score is not None:
        vals.append(float(ecc_score))
        ws.append(w_ecc)
    if grad_ncc_score is not None:
        vals.append(float(grad_ncc_score))
        ws.append(w_ncc)
    if edge_f1_score is not None:
        vals.append(float(edge_f1_score))
        ws.append(w_edge)
    if not vals or sum(ws) <= 0:
        return 0.0
    s = 0.0
    wsum = 0.0
    for v, w in zip(vals, ws):
        s += v * w
        wsum += w
    return float(s / wsum)
