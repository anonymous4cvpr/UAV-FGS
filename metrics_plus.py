#
# Extra evaluation metrics for cleanliness / texture clarity / alignment robustness.
# Does NOT modify or depend on metrics.py behavior.
#
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

METRIC_DIRECTION: Dict[str, str] = {
    # higher is better
    "EdgePSNR": "high",
    "GradientCorr": "high",
    "EdgeF1": "high",
    "EdgeF1_best": "high",
    "EdgeF1_mean": "high",
    "EdgePR_AUC": "high",
    "AlignedPSNR": "high",
    "AlignedEdgePSNR": "high",
    "AlignedGradientCorr": "high",
    "AlignedEdgeF1": "high",
    "AlignedEdgeF1_best": "high",
    "AlignedEdgeF1_mean": "high",
    "AlignedEdgePR_AUC": "high",
    "SGF_StructureScore": "high",
    "SGF_CleanScore": "high",
    "SGF_FidelityScore": "high",
    "SGF_MetricsPlusScore": "high",
    # lower is better
    "EdgeL1": "low",
    "HFAbsMeanDiff": "low",
    "BgLeakRatio": "low",
    "BgLeakRatio_band": "low",
    "EdgeHaloScore": "low",
    "AirArtifactEdgeExcess": "low",
    "AirArtifactHFMean": "low",
    "AirArtifactBrightExcess": "low",
}


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _conv3x3(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = img.shape
    pad = np.pad(img, ((1, 1), (1, 1)), mode="edge")
    out = (
        kernel[0, 0] * pad[0:h, 0:w] +
        kernel[0, 1] * pad[0:h, 1:w + 1] +
        kernel[0, 2] * pad[0:h, 2:w + 2] +
        kernel[1, 0] * pad[1:h + 1, 0:w] +
        kernel[1, 1] * pad[1:h + 1, 1:w + 1] +
        kernel[1, 2] * pad[1:h + 1, 2:w + 2] +
        kernel[2, 0] * pad[2:h + 2, 0:w] +
        kernel[2, 1] * pad[2:h + 2, 1:w + 1] +
        kernel[2, 2] * pad[2:h + 2, 2:w + 2]
    )
    return out


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    gx = _conv3x3(gray, kx)
    gy = _conv3x3(gray, ky)
    mag = np.hypot(gx, gy)
    mag /= (4.0 * math.sqrt(2.0))
    return np.clip(mag, 0.0, 1.0)


def _sobel_xy(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    gx = _conv3x3(gray, kx)
    gy = _conv3x3(gray, ky)
    return gx, gy


def _smooth_gray(gray: np.ndarray, iters: int = 2) -> np.ndarray:
    k = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=np.float32) / 16.0
    out = gray.astype(np.float32)
    for _ in range(max(1, int(iters))):
        out = _conv3x3(out, k)
    return out


def _local_contrast_norm(gray: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    mu = _smooth_gray(gray, iters=2)
    dev = np.abs(gray - mu)
    sigma = _smooth_gray(dev, iters=2)
    norm = (gray - mu) / (sigma + float(eps))
    return np.clip(norm, -4.0, 4.0).astype(np.float32)


def _texture_metrics_lcn(gray: np.ndarray) -> Tuple[float, float]:
    g = _local_contrast_norm(gray)
    gx, gy = _sobel_xy(g)
    gm = np.hypot(gx, gy)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    thr = float(np.mean(gm) + 0.75 * np.std(gm))
    thr = max(thr, 1e-4)
    edge_density = float(np.mean(gm > thr))
    return tenengrad, edge_density


def _edge_anisotropy(gray: np.ndarray) -> float:
    # Directionality score in [0,1]: higher means clearer oriented structures.
    g = _local_contrast_norm(gray)
    gx, gy = _sobel_xy(g)
    mag = np.hypot(gx, gy)
    if mag.size <= 0:
        return 0.0
    thr = float(np.percentile(mag, 75.0))
    mask = mag > max(thr, 1e-6)
    if int(np.sum(mask)) < 16:
        return 0.0
    gxx = gx[mask] * gx[mask]
    gyy = gy[mask] * gy[mask]
    gxy = gx[mask] * gy[mask]
    jxx = float(np.mean(gxx))
    jyy = float(np.mean(gyy))
    jxy = float(np.mean(gxy))
    tr = max(jxx + jyy, 1e-12)
    det = (jxx * jyy) - (jxy * jxy)
    disc = max(0.0, 0.25 * tr * tr - det)
    root = math.sqrt(disc)
    l1 = 0.5 * tr + root
    l2 = max(0.0, 0.5 * tr - root)
    return float(np.clip((l1 - l2) / max(l1 + l2, 1e-12), 0.0, 1.0))


def _laplacian(gray: np.ndarray) -> np.ndarray:
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    return _conv3x3(gray, k)


def _psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _aligned_psnr_rgb(render: np.ndarray, gt: np.ndarray, k: int) -> float:
    h, w, _ = render.shape
    best = -float("inf")

    for dy in range(-k, k + 1):
        if dy >= 0:
            ry0, gy0, hh = dy, 0, h - dy
        else:
            ry0, gy0, hh = 0, -dy, h + dy
        if hh <= 0:
            continue
        for dx in range(-k, k + 1):
            if dx >= 0:
                rx0, gx0, ww = dx, 0, w - dx
            else:
                rx0, gx0, ww = 0, -dx, w + dx
            if ww <= 0:
                continue
            r = render[ry0:ry0 + hh, rx0:rx0 + ww, :]
            g = gt[gy0:gy0 + hh, gx0:gx0 + ww, :]
            mse = float(np.mean((r - g) ** 2))
            psnr = _psnr_from_mse(mse, 1.0)
            if psnr > best:
                best = psnr

    return best


def _aligned_psnr_gray(render: np.ndarray, gt: np.ndarray, k: int) -> float:
    h, w = render.shape
    best = -float("inf")

    for dy in range(-k, k + 1):
        if dy >= 0:
            ry0, gy0, hh = dy, 0, h - dy
        else:
            ry0, gy0, hh = 0, -dy, h + dy
        if hh <= 0:
            continue
        for dx in range(-k, k + 1):
            if dx >= 0:
                rx0, gx0, ww = dx, 0, w - dx
            else:
                rx0, gx0, ww = 0, -dx, w + dx
            if ww <= 0:
                continue
            r = render[ry0:ry0 + hh, rx0:rx0 + ww]
            g = gt[gy0:gy0 + hh, gx0:gx0 + ww]
            mse = float(np.mean((r - g) ** 2))
            psnr = _psnr_from_mse(mse, 1.0)
            if psnr > best:
                best = psnr

    return best


def _crop_by_shift_2d(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = a.shape
    if dy >= 0:
        ay0, by0, hh = dy, 0, h - dy
    else:
        ay0, by0, hh = 0, -dy, h + dy
    if dx >= 0:
        ax0, bx0, ww = dx, 0, w - dx
    else:
        ax0, bx0, ww = 0, -dx, w + dx
    if hh <= 0 or ww <= 0:
        return a[:0, :0], b[:0, :0]
    return a[ay0:ay0 + hh, ax0:ax0 + ww], b[by0:by0 + hh, bx0:bx0 + ww]


def _best_shift_gray_by_psnr(render_gray: np.ndarray, gt_gray: np.ndarray, k: int) -> Tuple[int, int]:
    best = -float("inf")
    best_dx = 0
    best_dy = 0
    h, w = render_gray.shape
    for dy in range(-k, k + 1):
        if dy >= 0:
            ry0, gy0, hh = dy, 0, h - dy
        else:
            ry0, gy0, hh = 0, -dy, h + dy
        if hh <= 0:
            continue
        for dx in range(-k, k + 1):
            if dx >= 0:
                rx0, gx0, ww = dx, 0, w - dx
            else:
                rx0, gx0, ww = 0, -dx, w + dx
            if ww <= 0:
                continue
            r = render_gray[ry0:ry0 + hh, rx0:rx0 + ww]
            g = gt_gray[gy0:gy0 + hh, gx0:gx0 + ww]
            mse = float(np.mean((r - g) ** 2))
            psnr = _psnr_from_mse(mse, 1.0)
            if psnr > best:
                best = psnr
                best_dx = int(dx)
                best_dy = int(dy)
    return best_dx, best_dy


def _bg_leak_ratio(render: np.ndarray, bg: int, thr: float = 3.0 / 255.0) -> float:
    if bg == 0:
        mask = np.max(render, axis=2) <= thr
    else:
        mask = np.min(render, axis=2) >= (1.0 - thr)
    return float(np.mean(mask))


def _bg_leak_mask(render: np.ndarray, bg: int, thr: float = 3.0 / 255.0) -> np.ndarray:
    if bg == 0:
        return np.max(render, axis=2) <= thr
    return np.min(render, axis=2) >= (1.0 - thr)


def _largest_top_connected(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    best_area = 0
    best_coords = []
    for x0 in range(w):
        if not mask[0, x0] or visited[0, x0]:
            continue
        stack = [(0, x0)]
        visited[0, x0] = True
        coords = []
        area = 0
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            area += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if area > best_area:
            best_area = area
            best_coords = coords
    out = np.zeros((h, w), dtype=bool)
    for y, x in best_coords:
        out[y, x] = True
    return out


def _air_mask_from_gt(gray_g: np.ndarray, edge_g: np.ndarray) -> np.ndarray:
    # Robust air proxy:
    # 1) constrain to upper region
    # 2) low-edge candidates
    # 3) keep top-border connected component
    h, w = gray_g.shape
    top_h = max(1, int(round(0.65 * h)))
    top = np.zeros((h, w), dtype=bool)
    top[:top_h, :] = True
    edge_top = edge_g[top]
    if edge_top.size <= 0:
        return top
    q_edge = float(np.percentile(edge_top, 60.0))
    cand = np.logical_and(top, edge_g <= q_edge)
    mask = _largest_top_connected(cand)
    if float(np.mean(mask)) < 0.03:
        q_edge2 = float(np.percentile(edge_top, 75.0))
        cand2 = np.logical_and(top, edge_g <= q_edge2)
        mask = _largest_top_connected(cand2)
    if float(np.mean(mask)) < 0.01:
        mask = np.logical_and(top, edge_g <= float(np.percentile(edge_top, 85.0)))
    return mask


def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    if arr.shape != mask.shape:
        return 0.0
    cnt = int(mask.sum())
    if cnt <= 0:
        return 0.0
    return float(arr[mask].mean())


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _edge_f1(edge_r: np.ndarray, edge_g: np.ndarray, thr: float) -> float:
    pred = edge_r >= thr
    gt = edge_g >= thr
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    denom_p = tp + fp
    denom_r = tp + fn
    if denom_p == 0 or denom_r == 0:
        return 0.0
    precision = tp / denom_p
    recall = tp / denom_r
    denom_f1 = precision + recall
    if denom_f1 <= 0:
        return 0.0
    return 2.0 * precision * recall / denom_f1


def _edge_f1_best(edge_r: np.ndarray, edge_g: np.ndarray, fallback_thr: float) -> float:
    vals = np.concatenate([edge_r.reshape(-1), edge_g.reshape(-1)], axis=0)
    if vals.size == 0:
        return 0.0
    q = np.linspace(0.65, 0.98, 12)
    thrs = np.quantile(vals, q)
    thrs = np.unique(np.clip(thrs, 1e-4, 1.0))
    if thrs.size == 0:
        thrs = np.array([max(1e-4, float(fallback_thr))], dtype=np.float32)
    best = 0.0
    for t in thrs:
        f1 = _edge_f1(edge_r, edge_g, float(t))
        if f1 > best:
            best = f1
    return float(best)


def _edge_threshold_grid(edge_r: np.ndarray, edge_g: np.ndarray, fallback_thr: float) -> np.ndarray:
    vals = np.concatenate([edge_r.reshape(-1), edge_g.reshape(-1)], axis=0)
    if vals.size == 0:
        return np.array([max(1e-4, float(fallback_thr))], dtype=np.float32)
    q = np.linspace(0.60, 0.99, 20)
    thrs = np.quantile(vals, q)
    thrs = np.unique(np.clip(thrs, 1e-4, 1.0))
    if thrs.size == 0:
        thrs = np.array([max(1e-4, float(fallback_thr))], dtype=np.float32)
    return thrs.astype(np.float32)


def _edge_f1_mean(edge_r: np.ndarray, edge_g: np.ndarray, fallback_thr: float) -> float:
    thrs = _edge_threshold_grid(edge_r, edge_g, fallback_thr)
    vals = [_edge_f1(edge_r, edge_g, float(t)) for t in thrs]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _edge_gt_binary(edge_g: np.ndarray) -> np.ndarray:
    if edge_g.size <= 0:
        return np.zeros_like(edge_g, dtype=bool)
    thr = float(np.percentile(edge_g, 80.0))
    thr = max(thr, 1e-4)
    return edge_g >= thr


def _edge_pr_auc(edge_r: np.ndarray, edge_g: np.ndarray, fallback_thr: float) -> float:
    gt = _edge_gt_binary(edge_g)
    gt_pos = int(gt.sum())
    if gt_pos <= 0:
        return 0.0
    thrs = _edge_threshold_grid(edge_r, edge_g, fallback_thr)
    prs: List[float] = []
    rcs: List[float] = []
    for t in thrs:
        pred = edge_r >= float(t)
        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        denom_p = tp + fp
        denom_r = tp + fn
        precision = (tp / denom_p) if denom_p > 0 else 0.0
        recall = (tp / denom_r) if denom_r > 0 else 0.0
        prs.append(float(precision))
        rcs.append(float(recall))
    if not prs:
        return 0.0
    # Monotonic envelope on precision to stabilize AUC.
    pairs = sorted(zip(rcs, prs), key=lambda x: x[0])
    r = np.asarray([p[0] for p in pairs], dtype=np.float32)
    p = np.asarray([p[1] for p in pairs], dtype=np.float32)
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    trap = getattr(np, "trapezoid", np.trapz)
    auc = float(trap(p, r))
    return max(0.0, min(1.0, auc))


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.astype(bool)
    r = max(0, int(radius))
    for _ in range(r):
        pad = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            pad[:-2, :-2] | pad[:-2, 1:-1] | pad[:-2, 2:] |
            pad[1:-1, :-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:] |
            pad[2:, :-2] | pad[2:, 1:-1] | pad[2:, 2:]
        )
    return out


def _boundary_band_mask(edge_g: np.ndarray, radius: int) -> np.ndarray:
    gt = _edge_gt_binary(edge_g)
    if int(gt.sum()) <= 0:
        return np.zeros_like(gt, dtype=bool)
    band = _binary_dilate(gt, max(1, int(radius)))
    return band


def _warn_once(msg: str, warned: set) -> None:
    if msg in warned:
        return
    warned.add(msg)
    print(f"[WARN] {msg}")


def _normalize_extra_iqa_name(name: str) -> str:
    x = str(name).strip().lower().replace("_", "").replace("-", "")
    mapping = {
        "msssim": "msssim",
        "hdrvdp": "hdrvdp3",
    }
    return mapping.get(x, x)


class _ExtraIQAEngine:
    def __init__(self, names: List[str], channel: str, device: str = "cuda"):
        self.names = [_normalize_extra_iqa_name(n) for n in names if str(n).strip()]
        self.channel = "rgb" if str(channel).lower() == "rgb" else "y"
        req = str(device).strip().lower()
        self.device_req = req if req in {"cpu", "cuda", "auto"} else "cuda"
        self.device = "cpu"
        self.warned: set = set()
        self.pyiqa = None
        self.piq = None
        self.flip_api = None
        self.torch = None
        self._pyiqa_metrics: Dict[str, Callable] = {}
        self._piq_metrics: Dict[str, Callable] = {}
        if self.names:
            self._init_backends()

    def _init_backends(self) -> None:
        try:
            import torch  # type: ignore
            self.torch = torch
        except Exception:
            self.torch = None
        cuda_ok = bool(self.torch is not None and self.torch.cuda.is_available())
        if self.device_req == "cpu":
            self.device = "cpu"
        elif self.device_req == "cuda":
            if cuda_ok:
                self.device = "cuda"
            else:
                self.device = "cpu"
                _warn_once("extra_iqa device=cuda but CUDA unavailable, fallback to cpu", self.warned)
        else:
            self.device = "cuda" if cuda_ok else "cpu"

        try:
            import pyiqa  # type: ignore
            self.pyiqa = pyiqa
        except Exception:
            self.pyiqa = None
        try:
            import piq  # type: ignore
            self.piq = piq
        except Exception:
            self.piq = None
        try:
            from flip_evaluator import flip_python_api as flip_api  # type: ignore
            self.flip_api = flip_api
        except Exception:
            self.flip_api = None

    def _to_tensor(self, img: np.ndarray, force_channel: Optional[str] = None):
        import torch
        ch = self.channel if force_channel is None else ("rgb" if str(force_channel).lower() == "rgb" else "y")
        if ch == "rgb":
            arr = np.transpose(img, (2, 0, 1))[None, ...]
        else:
            g = _rgb_to_gray(img)
            arr = g[None, None, ...]
        return torch.from_numpy(arr.astype(np.float32)).to(self.device)

    def _eval_flip_fallback(self, render: np.ndarray, gt: np.ndarray) -> Optional[float]:
        if self.flip_api is None:
            return None
        try:
            _, mean_err, _ = self.flip_api.evaluate(
                gt.astype(np.float32),
                render.astype(np.float32),
                "LDR",
                inputsRGB=True,
                applyMagma=False,
                computeMeanError=True,
            )
            return float(mean_err)
        except Exception as exc:
            _warn_once(f"extra_iqa flip_evaluator failed: {exc}", self.warned)
            return None

    def _create_pyiqa_metric(self, name: str):
        if self.pyiqa is None:
            return None
        if name in self._pyiqa_metrics:
            return self._pyiqa_metrics[name]
        candidates = {
            "flip": ["flip"],
            "dists": ["dists"],
            "fsim": ["fsim"],
            "vif": ["vif", "vifp"],
            "msssim": ["ms_ssim", "ms-ssim"],
            "gmsd": ["gmsd"],
            "haarpsi": ["haarpsi"],
            "niqe": ["niqe"],
            "brisque": ["brisque"],
            "piqe": ["piqe"],
            "hdrvdp3": ["hdrvdp3", "hdrvdp"],
        }.get(name, [name])
        for c in candidates:
            try:
                fn = self.pyiqa.create_metric(c, device=self.device, as_loss=False)
                self._pyiqa_metrics[name] = fn
                return fn
            except Exception:
                continue
        return None

    def _create_piq_metric(self, name: str):
        if self.piq is None:
            return None
        if name in self._piq_metrics:
            return self._piq_metrics[name]
        fn = None
        try:
            if name == "msssim":
                fn = lambda x, y: self.piq.multi_scale_ssim(x, y, data_range=1.0)
            elif name == "vif":
                fn = lambda x, y: self.piq.vif_p(x, y, data_range=1.0)
            elif name == "gmsd":
                fn = lambda x, y: self.piq.gmsd(x, y, data_range=1.0)
            elif name == "fsim":
                fn = lambda x, y: self.piq.fsim(x, y, data_range=1.0)
            elif name == "haarpsi":
                fn = lambda x, y: self.piq.haarpsi(x, y, data_range=1.0)
        except Exception:
            fn = None
        if fn is not None:
            self._piq_metrics[name] = fn
        return fn

    def evaluate_pair(self, render: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        if not self.names:
            return {}
        out: Dict[str, float] = {}
        try:
            x = self._to_tensor(render)
            y = self._to_tensor(gt)
        except Exception as exc:
            _warn_once(f"extra_iqa tensor conversion failed: {exc}", self.warned)
            for n in self.names:
                out[n] = float("nan")
            return out

        fr_names = {"flip", "dists", "fsim", "vif", "msssim", "gmsd", "haarpsi", "hdrvdp3"}
        nr_names = {"niqe", "brisque", "piqe"}
        for n in self.names:
            score = float("nan")
            done = False

            fn_py = self._create_pyiqa_metric(n)
            if fn_py is not None:
                try:
                    val = fn_py(x, y) if n in fr_names else fn_py(x)
                    score = float(val.detach().cpu().reshape(-1)[0].item())
                    done = True
                except Exception as exc:
                    _warn_once(f"extra_iqa pyiqa '{n}' failed: {exc}", self.warned)

            if (not done) and (n in fr_names):
                fn_piq = self._create_piq_metric(n)
                if fn_piq is not None:
                    try:
                        x_use, y_use = x, y
                        # PIQ FSIM expects RGB input; fallback from Y to RGB for robustness.
                        if n == "fsim" and self.channel != "rgb":
                            x_use = self._to_tensor(render, force_channel="rgb")
                            y_use = self._to_tensor(gt, force_channel="rgb")
                        val = fn_piq(x_use, y_use)
                        score = float(val.detach().cpu().reshape(-1)[0].item())
                        done = True
                    except Exception as exc:
                        _warn_once(f"extra_iqa piq '{n}' failed: {exc}", self.warned)

            if (not done) and n == "flip":
                flip_score = self._eval_flip_fallback(render, gt)
                if flip_score is not None:
                    score = flip_score
                    done = True

            if not done:
                _warn_once(f"extra_iqa '{n}' unavailable; writing NaN", self.warned)
            out[n] = score
        return out

def _format_float(v: float) -> str:
    if math.isfinite(v):
        return f"{v:.6f}"
    return str(v)


def _ratio_closeness(r: float) -> float:
    if (not math.isfinite(r)) or r <= 0.0:
        return 0.0
    # 1.0 is best, decays symmetrically for over/under-shoot.
    return float(math.exp(-abs(math.log(r))))


def _score_high_unit(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _score_high_pos(x: float, scale: float) -> float:
    if (not math.isfinite(x)) or x <= 0.0:
        return 0.0
    s = max(1e-6, float(scale))
    return float(1.0 - math.exp(-x / s))


def _score_low_pos(x: float, scale: float) -> float:
    if not math.isfinite(x):
        return 0.0
    s = max(1e-6, float(scale))
    return float(math.exp(-max(0.0, x) / s))


def _iter_pairs(renders_dir: Path, gt_dir: Path) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
    names = sorted([f for f in os.listdir(renders_dir) if (renders_dir / f).is_file()])
    for name in names:
        gt_path = gt_dir / name
        if not gt_path.exists():
            continue
        yield name, _load_rgb(renders_dir / name), _load_rgb(gt_path)


def evaluate(
    model_paths: List[str],
    k: int,
    bg: int,
    edge_thr: float,
    save_json: bool,
    edge_band_radius: int = 5,
    extra_iqa: str = "",
    extra_iqa_space: str = "y",
    extra_iqa_device: str = "cuda",
) -> None:
    full_dict = {}

    for scene_dir in model_paths:
        print("")
        print("Scene:", scene_dir)
        test_dir = Path(scene_dir) / "test"
        if not test_dir.exists():
            print("  [WARN] test/ not found, skip.")
            continue

        full_dict[scene_dir] = {}

        for method in sorted(os.listdir(test_dir)):
            method_dir = test_dir / method
            renders_dir = method_dir / "renders"
            gt_dir = method_dir / "gt"
            if not renders_dir.exists() or not gt_dir.exists():
                continue

            edge_psnrs: List[float] = []
            edge_l1s: List[float] = []
            grad_corrs: List[float] = []
            edge_f1s: List[float] = []
            edge_f1_bests: List[float] = []
            lap_var_r: List[float] = []
            lap_var_g: List[float] = []
            hf_r: List[float] = []
            hf_g: List[float] = []
            aligned_psnrs: List[float] = []
            aligned_edge_psnrs: List[float] = []
            aligned_grad_corrs: List[float] = []
            aligned_edge_f1s: List[float] = []
            aligned_edge_f1_bests: List[float] = []
            edge_f1_means: List[float] = []
            edge_pr_aucs: List[float] = []
            aligned_edge_f1_means: List[float] = []
            aligned_edge_pr_aucs: List[float] = []
            bg_leaks: List[float] = []
            bg_leaks_band: List[float] = []
            edge_halo_scores: List[float] = []
            tex_lcn_ten_r: List[float] = []
            tex_lcn_ten_g: List[float] = []
            tex_lcn_edge_r: List[float] = []
            tex_lcn_edge_g: List[float] = []
            edge_aniso_r: List[float] = []
            edge_aniso_g: List[float] = []
            air_mask_ratios: List[float] = []
            air_edge_excess: List[float] = []
            air_hf_mean: List[float] = []
            air_bright_excess: List[float] = []
            iqa_names = [t.strip() for t in str(extra_iqa).split(",") if t.strip()]
            iqa_engine = _ExtraIQAEngine(iqa_names, channel=extra_iqa_space, device=extra_iqa_device)
            iqa_acc: Dict[str, List[float]] = {n: [] for n in iqa_engine.names}

            for _, render, gt in _iter_pairs(renders_dir, gt_dir):
                if render.shape != gt.shape:
                    raise ValueError(f"Render/GT size mismatch: {render.shape} vs {gt.shape}")
                gray_r = _rgb_to_gray(render)
                gray_g = _rgb_to_gray(gt)

                edge_r = _sobel_mag(gray_r)
                edge_g = _sobel_mag(gray_g)
                mse_edge = float(np.mean((edge_r - edge_g) ** 2))
                edge_psnrs.append(_psnr_from_mse(mse_edge, 1.0))
                edge_l1s.append(float(np.mean(np.abs(edge_r - edge_g))))
                grad_corrs.append(_corrcoef_safe(edge_r, edge_g))
                edge_f1s.append(_edge_f1(edge_r, edge_g, edge_thr))
                edge_f1_bests.append(_edge_f1_best(edge_r, edge_g, edge_thr))
                edge_f1_means.append(_edge_f1_mean(edge_r, edge_g, edge_thr))
                edge_pr_aucs.append(_edge_pr_auc(edge_r, edge_g, edge_thr))

                lap_r = _laplacian(gray_r)
                lap_g = _laplacian(gray_g)
                lap_var_r.append(float(np.var(lap_r)))
                lap_var_g.append(float(np.var(lap_g)))
                hf_r.append(float(np.mean(np.abs(lap_r))))
                hf_g.append(float(np.mean(np.abs(lap_g))))

                aligned_psnrs.append(_aligned_psnr_rgb(render, gt, k))
                aligned_edge_psnrs.append(_aligned_psnr_gray(edge_r, edge_g, k))
                best_dx, best_dy = _best_shift_gray_by_psnr(gray_r, gray_g, k)
                edge_r_al, edge_g_al = _crop_by_shift_2d(edge_r, edge_g, best_dx, best_dy)
                aligned_grad_corrs.append(_corrcoef_safe(edge_r_al, edge_g_al))
                aligned_edge_f1s.append(_edge_f1(edge_r_al, edge_g_al, edge_thr))
                aligned_edge_f1_bests.append(_edge_f1_best(edge_r_al, edge_g_al, edge_thr))
                aligned_edge_f1_means.append(_edge_f1_mean(edge_r_al, edge_g_al, edge_thr))
                aligned_edge_pr_aucs.append(_edge_pr_auc(edge_r_al, edge_g_al, edge_thr))
                bg_leaks.append(_bg_leak_ratio(render, bg))

                # Boundary-band artifact metrics: more sensitive to halo/bleeding than full-image averages.
                band = _boundary_band_mask(edge_g, max(1, int(edge_band_radius)))
                leak_mask = _bg_leak_mask(render, bg).astype(np.float32)
                leak_r_al, _ = _crop_by_shift_2d(leak_mask, leak_mask, best_dx, best_dy)
                _, band_g_al = _crop_by_shift_2d(
                    np.zeros_like(band, dtype=np.float32), band.astype(np.float32), best_dx, best_dy
                )
                band_bool = band_g_al > 0.5
                bg_leaks_band.append(_masked_mean((leak_r_al > 0.5).astype(np.float32), band_bool))
                edge_halo_scores.append(_masked_mean(np.maximum(edge_r_al - edge_g_al, 0.0), band_bool))

                # Brightness-insensitive texture clarity.
                ten_r, ed_r = _texture_metrics_lcn(gray_r)
                ten_g, ed_g = _texture_metrics_lcn(gray_g)
                tex_lcn_ten_r.append(ten_r)
                tex_lcn_ten_g.append(ten_g)
                tex_lcn_edge_r.append(ed_r)
                tex_lcn_edge_g.append(ed_g)
                edge_aniso_r.append(_edge_anisotropy(gray_r))
                edge_aniso_g.append(_edge_anisotropy(gray_g))

                # Air cleanliness: artifact/fog proxies measured on GT-derived empty-air mask.
                air_mask = _air_mask_from_gt(gray_g, edge_g)
                air_mask_ratios.append(float(np.mean(air_mask)))
                air_edge_excess.append(_masked_mean(np.maximum(edge_r - edge_g, 0.0), air_mask))
                air_hf_mean.append(_masked_mean(np.abs(lap_r), air_mask))
                air_bright_excess.append(_masked_mean(np.maximum(gray_r - gray_g, 0.0), air_mask))

                if iqa_engine.names:
                    iqa_vals = iqa_engine.evaluate_pair(render, gt)
                    for n in iqa_engine.names:
                        v = float(iqa_vals.get(n, float("nan")))
                        iqa_acc.setdefault(n, []).append(v)

            if not edge_psnrs:
                continue

            mean_edge_psnr = float(np.mean(edge_psnrs))
            mean_edge_l1 = float(np.mean(edge_l1s))
            mean_grad_corr = float(np.mean(grad_corrs))
            mean_edge_f1 = float(np.mean(edge_f1s))
            mean_edge_f1_best = float(np.mean(edge_f1_bests))
            mean_edge_f1_mean = float(np.mean(edge_f1_means))
            mean_edge_pr_auc = float(np.mean(edge_pr_aucs))
            mean_lap_r = float(np.mean(lap_var_r))
            mean_lap_g = float(np.mean(lap_var_g))
            lap_ratio = mean_lap_r / mean_lap_g if mean_lap_g > 0.0 else float("inf")
            mean_hf_r = float(np.mean(hf_r))
            mean_hf_g = float(np.mean(hf_g))
            mean_hf_diff = float(abs(mean_hf_r - mean_hf_g))
            mean_aligned_psnr = float(np.mean(aligned_psnrs))
            mean_aligned_edge_psnr = float(np.mean(aligned_edge_psnrs))
            mean_aligned_grad_corr = float(np.mean(aligned_grad_corrs))
            mean_aligned_edge_f1 = float(np.mean(aligned_edge_f1s))
            mean_aligned_edge_f1_best = float(np.mean(aligned_edge_f1_bests))
            mean_aligned_edge_f1_mean = float(np.mean(aligned_edge_f1_means))
            mean_aligned_edge_pr_auc = float(np.mean(aligned_edge_pr_aucs))
            mean_bg_leak = float(np.mean(bg_leaks))
            mean_bg_leak_band = float(np.mean(bg_leaks_band))
            mean_edge_halo = float(np.mean(edge_halo_scores))
            mean_tex_ten_r = float(np.mean(tex_lcn_ten_r))
            mean_tex_ten_g = float(np.mean(tex_lcn_ten_g))
            tex_ten_ratio = mean_tex_ten_r / mean_tex_ten_g if mean_tex_ten_g > 0.0 else float("inf")
            mean_tex_ed_r = float(np.mean(tex_lcn_edge_r))
            mean_tex_ed_g = float(np.mean(tex_lcn_edge_g))
            tex_ed_ratio = mean_tex_ed_r / mean_tex_ed_g if mean_tex_ed_g > 0.0 else float("inf")
            mean_aniso_r = float(np.mean(edge_aniso_r))
            mean_aniso_g = float(np.mean(edge_aniso_g))
            aniso_ratio = mean_aniso_r / mean_aniso_g if mean_aniso_g > 0.0 else float("inf")
            mean_air_mask_ratio = float(np.mean(air_mask_ratios))
            mean_air_edge_excess = float(np.mean(air_edge_excess))
            mean_air_hf = float(np.mean(air_hf_mean))
            mean_air_bright_excess = float(np.mean(air_bright_excess))
            iqa_means: Dict[str, float] = {}
            for n, vals in iqa_acc.items():
                if not vals:
                    iqa_means[n] = float("nan")
                    continue
                arr = np.asarray(vals, dtype=np.float32)
                valid = arr[np.isfinite(arr)]
                iqa_means[n] = float(np.mean(valid)) if valid.size > 0 else float("nan")

            # SGF-oriented derived scores (higher is better):
            s_grad = _score_high_unit(0.5 * (mean_aligned_grad_corr + 1.0))
            s_edge_f1 = _score_high_unit(mean_aligned_edge_f1)
            s_edge_f1_best = _score_high_unit(mean_edge_f1_best)
            s_tex_ratio = _ratio_closeness(tex_ten_ratio)
            s_tex_ed_ratio = _ratio_closeness(tex_ed_ratio)
            s_aniso_ratio = _ratio_closeness(aniso_ratio)
            sgf_structure_score = float(
                np.clip(
                    0.24 * s_grad +
                    0.20 * s_edge_f1 +
                    0.10 * s_edge_f1_best +
                    0.24 * s_tex_ratio +
                    0.12 * s_tex_ed_ratio +
                    0.10 * s_aniso_ratio,
                    0.0, 1.0
                )
            )

            c_air_edge = _score_low_pos(mean_air_edge_excess, 0.010)
            c_air_hf = _score_low_pos(mean_air_hf, 0.020)
            c_air_bright = _score_low_pos(mean_air_bright_excess, 0.020)
            c_bg = _score_low_pos(mean_bg_leak, 0.005)
            c_hf_diff = _score_low_pos(mean_hf_diff, 0.030)
            c_lap_ratio = _ratio_closeness(lap_ratio)
            sgf_clean_score = float(
                np.clip(
                    0.22 * c_air_edge +
                    0.22 * c_air_hf +
                    0.20 * c_air_bright +
                    0.16 * c_bg +
                    0.10 * c_hf_diff +
                    0.10 * c_lap_ratio,
                    0.0, 1.0
                )
            )

            f_edge_psnr = _score_high_pos(mean_edge_psnr, 25.0)
            f_aligned_edge = _score_high_pos(mean_aligned_edge_psnr, 25.0)
            f_aligned_rgb = _score_high_pos(mean_aligned_psnr, 25.0)
            sgf_fidelity_score = float(
                np.clip(
                    0.40 * f_edge_psnr +
                    0.35 * f_aligned_edge +
                    0.25 * f_aligned_rgb,
                    0.0, 1.0
                )
            )

            sgf_metrics_plus_score = float(
                np.clip(
                    0.50 * sgf_structure_score +
                    0.35 * sgf_clean_score +
                    0.15 * sgf_fidelity_score,
                    0.0, 1.0
                )
            )

            print("Method:", method)
            print(f"  EdgePSNR(mean): {_format_float(mean_edge_psnr)}")
            print(f"  EdgeL1(mean): {_format_float(mean_edge_l1)}")
            print(f"  GradientCorr(mean): {_format_float(mean_grad_corr)}")
            print(f"  EdgeF1@{edge_thr:.3f}(mean): {_format_float(mean_edge_f1)}")
            print(f"  EdgeF1_best(mean): {_format_float(mean_edge_f1_best)}")
            print(f"  EdgeF1_mean(mean): {_format_float(mean_edge_f1_mean)}")
            print(f"  EdgePR_AUC(mean): {_format_float(mean_edge_pr_auc)}")
            print(f"  LapVar(render)(mean): {_format_float(mean_lap_r)}")
            print(f"  LapVar(gt)(mean): {_format_float(mean_lap_g)}")
            print(f"  LapVarRatio(mean): {_format_float(lap_ratio)}")
            print(f"  HFAbsMean(render)(mean): {_format_float(mean_hf_r)}")
            print(f"  HFAbsMean(gt)(mean): {_format_float(mean_hf_g)}")
            print(f"  HFAbsMeanDiff(mean): {_format_float(mean_hf_diff)}")
            print(f"  AlignedPSNR@{k}(mean): {_format_float(mean_aligned_psnr)}")
            print(f"  AlignedEdgePSNR@{k}(mean): {_format_float(mean_aligned_edge_psnr)}")
            print(f"  AlignedGradientCorr@{k}(mean): {_format_float(mean_aligned_grad_corr)}")
            print(f"  AlignedEdgeF1@{k}(mean): {_format_float(mean_aligned_edge_f1)}")
            print(f"  AlignedEdgeF1_best@{k}(mean): {_format_float(mean_aligned_edge_f1_best)}")
            print(f"  AlignedEdgeF1_mean@{k}(mean): {_format_float(mean_aligned_edge_f1_mean)}")
            print(f"  AlignedEdgePR_AUC@{k}(mean): {_format_float(mean_aligned_edge_pr_auc)}")
            print(f"  BgLeakRatio(mean): {_format_float(mean_bg_leak)}")
            print(f"  BgLeakRatio_band(mean): {_format_float(mean_bg_leak_band)}")
            print(f"  EdgeHaloScore(mean): {_format_float(mean_edge_halo)}")
            print(f"  TextureLCN_Tenengrad(render)(mean): {_format_float(mean_tex_ten_r)}")
            print(f"  TextureLCN_Tenengrad(gt)(mean): {_format_float(mean_tex_ten_g)}")
            print(f"  TextureLCN_TenengradRatio(mean): {_format_float(tex_ten_ratio)}")
            print(f"  TextureLCN_EdgeDensity(render)(mean): {_format_float(mean_tex_ed_r)}")
            print(f"  TextureLCN_EdgeDensity(gt)(mean): {_format_float(mean_tex_ed_g)}")
            print(f"  TextureLCN_EdgeDensityRatio(mean): {_format_float(tex_ed_ratio)}")
            print(f"  EdgeAnisotropy(render)(mean): {_format_float(mean_aniso_r)}")
            print(f"  EdgeAnisotropy(gt)(mean): {_format_float(mean_aniso_g)}")
            print(f"  EdgeAnisotropyRatio(mean): {_format_float(aniso_ratio)}")
            print(f"  AirArtifactEdgeExcess(mean): {_format_float(mean_air_edge_excess)}")
            print(f"  AirArtifactHFMean(mean): {_format_float(mean_air_hf)}")
            print(f"  AirArtifactBrightExcess(mean): {_format_float(mean_air_bright_excess)}")
            print(f"  AirMaskRatio(mean): {_format_float(mean_air_mask_ratio)}")
            print(f"  SGF_StructureScore(mean): {_format_float(sgf_structure_score)}")
            print(f"  SGF_CleanScore(mean): {_format_float(sgf_clean_score)}")
            print(f"  SGF_FidelityScore(mean): {_format_float(sgf_fidelity_score)}")
            print(f"  SGF_MetricsPlusScore(mean): {_format_float(sgf_metrics_plus_score)}")
            if iqa_means:
                for n in sorted(iqa_means.keys()):
                    print(f"  IQA_{n}(mean): {_format_float(iqa_means[n])}")

            full_dict[scene_dir][method] = {
                "EdgePSNR": mean_edge_psnr,
                "EdgeL1": mean_edge_l1,
                "GradientCorr": mean_grad_corr,
                "EdgeF1": mean_edge_f1,
                "EdgeF1_best": mean_edge_f1_best,
                "EdgeF1_mean": mean_edge_f1_mean,
                "EdgePR_AUC": mean_edge_pr_auc,
                "LapVar_render": mean_lap_r,
                "LapVar_gt": mean_lap_g,
                "LapVarRatio": lap_ratio,
                "HFAbsMean_render": mean_hf_r,
                "HFAbsMean_gt": mean_hf_g,
                "HFAbsMeanDiff": mean_hf_diff,
                "AlignedPSNR": mean_aligned_psnr,
                "AlignedEdgePSNR": mean_aligned_edge_psnr,
                "AlignedGradientCorr": mean_aligned_grad_corr,
                "AlignedEdgeF1": mean_aligned_edge_f1,
                "AlignedEdgeF1_best": mean_aligned_edge_f1_best,
                "AlignedEdgeF1_mean": mean_aligned_edge_f1_mean,
                "AlignedEdgePR_AUC": mean_aligned_edge_pr_auc,
                "BgLeakRatio": mean_bg_leak,
                "BgLeakRatio_band": mean_bg_leak_band,
                "EdgeHaloScore": mean_edge_halo,
                "TextureLCN_Tenengrad_render": mean_tex_ten_r,
                "TextureLCN_Tenengrad_gt": mean_tex_ten_g,
                "TextureLCN_TenengradRatio": tex_ten_ratio,
                "TextureLCN_EdgeDensity_render": mean_tex_ed_r,
                "TextureLCN_EdgeDensity_gt": mean_tex_ed_g,
                "TextureLCN_EdgeDensityRatio": tex_ed_ratio,
                "EdgeAnisotropy_render": mean_aniso_r,
                "EdgeAnisotropy_gt": mean_aniso_g,
                "EdgeAnisotropyRatio": aniso_ratio,
                "AirArtifactEdgeExcess": mean_air_edge_excess,
                "AirArtifactHFMean": mean_air_hf,
                "AirArtifactBrightExcess": mean_air_bright_excess,
                "AirMaskRatio": mean_air_mask_ratio,
                "SGF_StructureScore": sgf_structure_score,
                "SGF_CleanScore": sgf_clean_score,
                "SGF_FidelityScore": sgf_fidelity_score,
                "SGF_MetricsPlusScore": sgf_metrics_plus_score,
            }
            for n, v in iqa_means.items():
                full_dict[scene_dir][method][f"IQA_{n}"] = float(v)

        if save_json:
            out_path = Path(scene_dir) / "results_plus.json"
            out_path.write_text(json.dumps(full_dict[scene_dir], indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended metrics: edges, sharpness, alignment, background leak")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    parser.add_argument("--K", type=int, default=8, help="Search window for AlignedPSNR (default: 8)")
    parser.add_argument("--bg", type=int, default=0, choices=[0, 1], help="Background color: 0=black, 1=white")
    parser.add_argument("--edge_thr", type=float, default=0.1, help="Sobel edge threshold for EdgeF1 (default: 0.1)")
    parser.add_argument("--edge_band_radius", type=int, default=5, help="Boundary band radius in px for halo/bg-leak metrics (default: 5)")
    parser.add_argument("--extra_iqa", type=str, default="flip,dists,fsim,vif,ms-ssim,gmsd,haarpsi,niqe,brisque,piqe,hdrvdp3",
                        help="Optional IQA set, comma-separated: flip,dists,fsim,vif,ms-ssim,gmsd,haarpsi,niqe,brisque,piqe,hdrvdp3")
    parser.add_argument("--extra_iqa_space", type=str, default="y", choices=["y", "rgb"],
                        help="IQA input space (default: y)")
    parser.add_argument("--extra_iqa_device", type=str, default="cuda", choices=["cpu", "cuda", "auto"],
                        help="IQA backend device (default: cuda)")
    parser.add_argument("--save_json", action="store_true", default=False, help="Write results_plus.json (default: off)")
    args = parser.parse_args()
    evaluate(
        args.model_paths, args.K, args.bg, args.edge_thr, args.save_json,
        edge_band_radius=args.edge_band_radius,
        extra_iqa=args.extra_iqa,
        extra_iqa_space=args.extra_iqa_space,
        extra_iqa_device=args.extra_iqa_device,
    )


if __name__ == "__main__":
    main()
