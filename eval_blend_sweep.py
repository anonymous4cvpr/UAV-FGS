#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_blend_sweep.py

Evaluate a sweep of blended 3DGS models laid out as:
  sweep_root/
    <strategy_name>/
      <alpha>/            (a 3DGS model dir; may or may not have test/ours_*/renders yet)
        point_cloud/...
        cfg_args

It produces TWO evaluation sets:
  1) render_ref: compare each blend render against RGB-render (structure) and T-render (thermal scalar)
  2) gt_ref    : compare each blend render against RGB-GT (and optionally T-GT if provided)

It can also auto-render missing renders by calling render.py.
"""

from __future__ import annotations

import os
import re
import ast
import sys
import math
import json
import shlex
import time
import shutil
import argparse
import subprocess
import warnings

# Silence torchvision deprecation warnings triggered inside lpipsPyTorch/torchvision
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*weights.*positional parameter.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Arguments other than a weight enum.*deprecated.*",
)

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- optional deps ---
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception as e:
    raise RuntimeError("PIL is required (pip install pillow).") from e

try:
    from skimage.metrics import structural_similarity as sk_ssim  # type: ignore
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr  # type: ignore
except Exception as e:
    raise RuntimeError("scikit-image is required (pip install scikit-image).") from e

try:
    from scipy.stats import spearmanr  # type: ignore
except Exception:
    spearmanr = None

# LPIPS: use lpipsPyTorch (your env has it)
LPIPS_OK = False
_lpips_fn = None
try:
    import torch  # type: ignore
    from lpipsPyTorch import lpips as lpips_fn  # type: ignore

    def _lpips_init() -> Any:
        # lpipsPyTorch expects torch tensors in [0,1] or [-1,1] depending on fn; we use [-1,1]
        # Using alexnet by default
        return "alex"

    _lpips_fn = lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False
    _lpips_fn = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


# -----------------------------
# Helpers: images
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    out = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    out.sort(key=lambda p: p.name)
    return out


def read_image_f32(path: Path) -> np.ndarray:
    # returns float32 RGB in [0,1]
    if cv2 is not None:
        im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read image: {path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return (im.astype(np.float32) / 255.0)
    # PIL fallback
    im = Image.open(path).convert("RGB")
    return (np.asarray(im).astype(np.float32) / 255.0)


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    # img: H,W,3 in [0,1]
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def edge_map_y(img_y: np.ndarray) -> np.ndarray:
    # simple Sobel
    if cv2 is not None:
        gx = cv2.Sobel(img_y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_y, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        return mag
    # numpy sobel
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    from scipy.signal import convolve2d  # type: ignore
    gx = convolve2d(img_y, kx, mode="same", boundary="symm")
    gy = convolve2d(img_y, ky, mode="same", boundary="symm")
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float((a @ b) / denom)


# -----------------------------
# Information / fusion metrics (source-referenced)
# -----------------------------
def _normalize01(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = a.astype(np.float32)
    lo = float(np.percentile(a, 1))
    hi = float(np.percentile(a, 99))
    if hi - lo < eps:
        lo = float(a.min())
        hi = float(a.max())
    if hi - lo < eps:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Mutual information between two scalar images (float)."""
    a = _normalize01(a).reshape(-1)
    b = _normalize01(b).reshape(-1)
    h2, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
    pxy = h2 / (np.sum(h2) + 1e-12)
    px = np.sum(pxy, axis=1, keepdims=True)  # (bins,1)
    py = np.sum(pxy, axis=0, keepdims=True)  # (1,bins)
    denom = px @ py  # outer product (bins,bins)
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log((pxy[nz] / (denom[nz] + 1e-12)) + 1e-12))
    return float(mi)


def normalized_mi(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Normalized mutual information: (H(a)+H(b))/H(a,b)."""
    a = _normalize01(a).reshape(-1)
    b = _normalize01(b).reshape(-1)
    h2, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
    pxy = h2 / (np.sum(h2) + 1e-12)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    def H(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-12)))

    ha = H(px)
    hb = H(py)
    hab = H(pxy.reshape(-1))
    if hab < 1e-12:
        return 0.0
    return float((ha + hb) / hab)


def spatial_frequency(img: np.ndarray) -> float:
    """Spatial Frequency (SF) for a scalar image."""
    img = img.astype(np.float32)
    rf = img[1:, :] - img[:-1, :]
    cf = img[:, 1:] - img[:, :-1]
    RF = float(np.sqrt(np.mean(rf * rf) + 1e-12))
    CF = float(np.sqrt(np.mean(cf * cf) + 1e-12))
    return float(np.sqrt(RF * RF + CF * CF))


def _vifp_ref_dist(ref: np.ndarray, dist: np.ndarray) -> float:
    """
    VIFp (pixel-domain) implementation for scalar images in [0,1].
    Based on common public-domain implementations of Sheikh & Bovik VIF.
    """
    # Use scipy.ndimage for Gaussian filtering
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
    except Exception:
        return float("nan")

    ref = _normalize01(ref).astype(np.float64)
    dist = _normalize01(dist).astype(np.float64)

    sigma_nsq = 2.0
    eps = 1e-10
    num = 0.0
    den = 0.0

    for scale in range(1, 5):
        # filter size / sigma per scale (typical)
        N = 2 ** (5 - scale) + 1
        sd = N / 5.0

        if scale > 1:
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = gaussian_filter(ref, sd)
        mu2 = gaussian_filter(dist, sd)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g = np.maximum(g, 0.0)
        sv_sq = np.maximum(sv_sq, eps)

        # where sigma1_sq is too small, set g=0
        mask = sigma1_sq < eps
        g[mask] = 0.0
        sv_sq[mask] = sigma2_sq[mask]

        # where sigma2_sq is too small, set sv_sq=eps
        mask2 = sigma2_sq < eps
        sv_sq[mask2] = eps

        num += np.sum(np.log10(1.0 + (g * g) * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq))

    if den < eps:
        return 0.0
    return float(num / den)


def _grad_mag_ori(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = a.astype(np.float32)
    if cv2 is not None:
        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    else:
        # simple finite difference
        gx = np.zeros_like(a)
        gy = np.zeros_like(a)
        gx[:, 1:-1] = 0.5 * (a[:, 2:] - a[:, :-2])
        gy[1:-1, :] = 0.5 * (a[2:, :] - a[:-2, :])
    mag = np.sqrt(gx * gx + gy * gy) + 1e-12
    ori = np.arctan2(gy, gx)
    return mag, ori


def qabf_metric(src_a: np.ndarray, src_b: np.ndarray, fused: np.ndarray) -> float:
    """
    QABF edge-based fusion metric (Xydeas & Petrovic style).
    Works on scalar images in [0,1].
    """
    A = _normalize01(src_a)
    B = _normalize01(src_b)
    F = _normalize01(fused)

    GA, OA = _grad_mag_ori(A)
    GB, OB = _grad_mag_ori(B)
    GF, OF = _grad_mag_ori(F)

    # Gradient similarity
    Tg = 0.85  # small constant to stabilize; typical implementations use values ~0.85
    QgA = (2 * GA * GF + Tg) / (GA * GA + GF * GF + Tg)
    QgB = (2 * GB * GF + Tg) / (GB * GB + GF * GF + Tg)

    # Orientation similarity: normalized to [0,1]
    dOA = np.abs(OA - OF)
    dOB = np.abs(OB - OF)
    dOA = np.minimum(dOA, np.pi - dOA)
    dOB = np.minimum(dOB, np.pi - dOB)
    QoA = 1.0 - (dOA / (np.pi / 2.0))
    QoB = 1.0 - (dOB / (np.pi / 2.0))
    QoA = np.clip(QoA, 0.0, 1.0)
    QoB = np.clip(QoB, 0.0, 1.0)

    QA = QgA * QoA
    QB = QgB * QoB

    # Weights based on source saliency (gradient magnitudes)
    WA = GA
    WB = GB
    denom = np.sum(WA + WB) + 1e-12
    q = float(np.sum(QA * WA + QB * WB) / denom)
    return q


# -----------------------------
# Thermal scalar from pseudo-color images
# -----------------------------
def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    # img in [0,1], returns hsv in [0,1]
    if cv2 is not None:
        hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = hsv[..., 0] / 179.0
        hsv[..., 1] = hsv[..., 1] / 255.0
        hsv[..., 2] = hsv[..., 2] / 255.0
        return hsv
    # PIL / numpy implementation
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn + 1e-12

    h = np.zeros_like(mx)
    mask = (mx == r)
    h[mask] = ((g - b)[mask] / diff[mask]) % 6.0
    mask = (mx == g)
    h[mask] = ((b - r)[mask] / diff[mask]) + 2.0
    mask = (mx == b)
    h[mask] = ((r - g)[mask] / diff[mask]) + 4.0
    h = (h / 6.0) % 1.0

    s = diff / (mx + 1e-12)
    v = mx
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def thermal_scalar(img_rgb: np.ndarray, mode: str = "hue_y") -> np.ndarray:
    """
    Convert pseudo-colored thermal render to a single scalar map S.
    - hue: hue channel (0..1)
    - sat: saturation
    - val: value
    - hue_y: hue weighted by luminance (more robust when saturation is small)
    """
    mode = mode.lower()
    hsv = rgb_to_hsv(img_rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    y = rgb_to_y(img_rgb)
    if mode == "hue":
        return h
    if mode == "sat":
        return s
    if mode == "val":
        return v
    if mode == "hue_y":
        return h * (0.3 + 0.7 * y)
    raise ValueError(f"Unknown thermal_scalar mode: {mode}")


def align_scalar(pred: np.ndarray, ref: np.ndarray, mode: str = "linear") -> np.ndarray:
    """
    Align scalar map pred to ref (global). Useful when a method changes contrast.
    - none: no align
    - linear: pred' = a*pred + b by least squares
    - rank: match mean/std after monotonic rank mapping (approx)
    """
    mode = mode.lower()
    if mode == "none":
        return pred
    p = pred.reshape(-1).astype(np.float64)
    r = ref.reshape(-1).astype(np.float64)
    if mode == "linear":
        A = np.stack([p, np.ones_like(p)], axis=1)
        x, *_ = np.linalg.lstsq(A, r, rcond=None)
        a, b = float(x[0]), float(x[1])
        return (a * pred + b).astype(np.float32)
    if mode == "rank":
        # map to z-score with ref mean/std
        pm, ps = float(p.mean()), float(p.std() + 1e-12)
        rm, rs = float(r.mean()), float(r.std() + 1e-12)
        z = (pred - pm) / ps
        return (z * rs + rm).astype(np.float32)
    raise ValueError(f"Unknown thermal_align mode: {mode}")


# -----------------------------
# Metrics
# -----------------------------
def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: H,W float32 [0,1]
    return float(sk_ssim(a, b, data_range=1.0))


def psnr_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return float(sk_psnr(a, b, data_range=1.0))


def psnr_gray(a: np.ndarray, b: np.ndarray) -> float:
    return float(sk_psnr(a, b, data_range=1.0))


def lpips_rgb(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not LPIPS_OK or _lpips_fn is None:
        return None
    # a,b: H,W,3 in [0,1] -> torch [-1,1]
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
    with torch.no_grad():
        v = _lpips_fn(ta, tb, net_type=_lpips_init())
    return float(v.item())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if spearmanr is None:
        # fallback: pearson on ranks
        ra = a.argsort().argsort().astype(np.float64)
        rb = b.argsort().argsort().astype(np.float64)
        return corrcoef(ra, rb)
    return float(spearmanr(a, b).correlation)


# -----------------------------
# Paths / discovery
# -----------------------------
@dataclass
class MethodEntry:
    strategy: str
    alpha: float
    label: str
    model_dir: Path
    renders_dir: Optional[Path] = None
    gt_dir: Optional[Path] = None
    used_iter: Optional[int] = None


def parse_alpha_dirname(name: str) -> Optional[float]:
    try:
        # allow "0", "0.1", "1"
        return float(name)
    except Exception:
        return None


def find_best_ours_dir(model_dir: Path, force_iter: Optional[int] = None) -> Optional[Path]:
    test = model_dir / "test"
    if not test.exists():
        return None
    ours = []
    for d in test.iterdir():
        if d.is_dir() and d.name.startswith("ours_"):
            suf = d.name.split("_", 1)[-1]
            if suf.isdigit():
                it = int(suf)
                if force_iter is not None and it != force_iter:
                    continue
                ours.append((it, d))
    if not ours:
        return None
    ours.sort(key=lambda x: x[0])
    return ours[-1][1]


def resolve_renders_and_gt(path: Path, force_iter: Optional[int] = None) -> Tuple[Path, Path, Optional[int]]:
    """
    Accept either:
      - a model dir containing test/ours_*/renders, gt
      - a direct renders dir (…/renders) whose sibling gt exists
      - a direct gt dir (…/gt) whose sibling renders exists
    """
    if path.name.lower() == "renders":
        renders = path
        gt = path.parent / "gt"
        it = None
        m = re.search(r"ours_(\d+)", str(path.parent))
        if m:
            it = int(m.group(1))
        return renders, gt, it
    if path.name.lower() == "gt":
        gt = path
        renders = path.parent / "renders"
        it = None
        m = re.search(r"ours_(\d+)", str(path.parent))
        if m:
            it = int(m.group(1))
        return renders, gt, it

    # treat as model dir
    ours_dir = find_best_ours_dir(path, force_iter=force_iter)
    if ours_dir is None:
        raise FileNotFoundError(f"No test/ours_* under model dir: {path}")
    renders = ours_dir / "renders"
    gt = ours_dir / "gt"
    it = None
    m = re.search(r"ours_(\d+)", ours_dir.name)
    if m:
        it = int(m.group(1))
    return renders, gt, it


def parse_cfg_args(model_dir: Path) -> Dict[str, Any]:
    cfg = model_dir / "cfg_args"
    if not cfg.exists():
        return {}
    text = cfg.read_text(encoding="utf-8", errors="ignore").strip()
    m = re.search(r"Namespace\((.*)\)\s*$", text)
    if not m:
        return {}
    body = m.group(1)
    # split by commas not inside quotes/brackets
    parts = []
    buf = []
    depth = 0
    in_str = False
    quote = ""
    esc = False
    for ch in body:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if ch == "\\":
            buf.append(ch)
            esc = True
            continue
        if in_str:
            buf.append(ch)
            if ch == quote:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            quote = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
            buf.append(ch)
            continue
        if ch in ")]}":
            depth -= 1
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    out: Dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v.strip("'\"")
    return out


def infer_pointcloud_iters(model_dir: Path) -> List[int]:
    pc = model_dir / "point_cloud"
    if not pc.exists():
        return []
    iters = []
    for d in pc.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            suf = d.name.split("_", 1)[-1]
            if suf.isdigit():
                iters.append(int(suf))
    iters.sort()
    return iters


def choose_iteration(model_dir: Path, preferred: int) -> int:
    if (model_dir / "point_cloud" / f"iteration_{preferred}").exists():
        return preferred
    iters = infer_pointcloud_iters(model_dir)
    if iters:
        return iters[-1]
    return preferred


def discover_methods(sweep_root: Path) -> List[MethodEntry]:
    methods: List[MethodEntry] = []
    for strategy_dir in sorted([p for p in sweep_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        strategy = strategy_dir.name
        for alpha_dir in sorted([p for p in strategy_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            a = parse_alpha_dirname(alpha_dir.name)
            if a is None:
                continue
            methods.append(MethodEntry(strategy=strategy, alpha=a, label=f"{strategy}-a{alpha_dir.name}", model_dir=alpha_dir))
    return methods


# -----------------------------
# Rendering (auto)
# -----------------------------
def mklink_junction(dst: Path, src: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], capture_output=True, text=True)
        return p.returncode == 0
    except Exception:
        return False


def ensure_gt_dedup(gt_dir: Path, shared_gt: Path, mode: str) -> None:
    mode = mode.lower()
    if mode == "keep":
        return
    if mode == "delete":
        if gt_dir.exists():
            shutil.rmtree(gt_dir, ignore_errors=True)
        return
    if mode == "link":
        if gt_dir.exists() and gt_dir.is_dir():
            shutil.rmtree(gt_dir, ignore_errors=True)
        ok = mklink_junction(gt_dir, shared_gt)
        if not ok:
            # fallback: do nothing (renders will still exist)
            return
    else:
        raise ValueError(f"Unknown gt_mode: {mode}")


def run_render(
    render_py: Path,
    model_dir: Path,
    source_path: Path,
    images_subdir: str,
    resolution: int,
    iteration: int,
    python_exe: str,
    extra: str = "",
    verbose: bool = True,
) -> int:
    cmd = [
        python_exe,
        str(render_py),
        "-m", str(model_dir),
        "-s", str(source_path),
        "-i", images_subdir,
        "-r", str(int(resolution)),
        "--iteration", str(int(iteration)),
    ]
    if extra:
        cmd += shlex.split(extra)
    if verbose:
        print("[AUTO_RENDER]", " ".join(cmd))
    p = subprocess.run(cmd)
    return int(p.returncode)


def auto_render_methods(
    methods: List[MethodEntry],
    render_py: Path,
    source_path: Path,
    images_subdir: str,
    resolution: int,
    preferred_iter: int,
    python_exe: str,
    gt_mode: str,
    shared_gt: Optional[Path],
    extra: str = "",
    verbose: bool = True,
) -> None:
    missing = []
    for m in methods:
        # decide expected render dir after rendering
        it = choose_iteration(m.model_dir, preferred_iter)
        rdir = m.model_dir / "test" / f"ours_{it}" / "renders"
        if not rdir.exists() or not any(rdir.glob("*")):
            missing.append(m)

    if not missing:
        return

    bar = tqdm(missing, desc="Auto-render", unit="model")
    for m in bar:
        it = choose_iteration(m.model_dir, preferred_iter)
        m.used_iter = it

        if shared_gt is not None and gt_mode.lower() == "link":
            ensure_gt_dedup(m.model_dir / "test" / f"ours_{it}" / "gt", shared_gt, mode="link")

        bar.set_postfix_str(f"{m.strategy} a={m.alpha:g} it={it}")
        ret = run_render(
            render_py=render_py,
            model_dir=m.model_dir,
            source_path=source_path,
            images_subdir=images_subdir,
            resolution=resolution,
            iteration=it,
            python_exe=python_exe,
            extra=extra,
            verbose=verbose,
        )
        if ret != 0:
            print(f"[WARN] render.py failed ({ret}) for: {m.model_dir}")
            continue

        rdir = m.model_dir / "test" / f"ours_{it}" / "renders"
        gdir = m.model_dir / "test" / f"ours_{it}" / "gt"
        m.renders_dir = rdir if rdir.exists() else None
        m.gt_dir = gdir if gdir.exists() else None

        if shared_gt is not None and gt_mode.lower() in ("delete", "link"):
            ensure_gt_dedup(gdir, shared_gt, mode=gt_mode)


# -----------------------------
# Evaluation
# -----------------------------
def pair_by_name(a_dir: Path, b_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Pair images in two folders.

    Priority:
      1) If filenames overlap, pair by exact filename intersection (stable, safest).
      2) Otherwise fallback to index pairing by sorted filename order.

    Returns list of (a_path, b_path).
    """
    a = list_images(a_dir)
    b = list_images(b_dir)
    if not a or not b:
        return []
    a_map = {p.name: p for p in a}
    b_map = {p.name: p for p in b}
    names = sorted(set(a_map.keys()) & set(b_map.keys()))
    if names:
        return [(a_map[n], b_map[n]) for n in names]
    n = min(len(a), len(b))
    return list(zip(a[:n], b[:n]))


def pair_triple_by_name(m_dir: Path, rgb_dir: Path, t_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Pair images across 3 folders: method vs RGB-ref vs T-ref.

    Priority:
      1) intersection by exact filename across all 3
      2) fallback by index (sorted order) across all 3
    """
    m = list_images(m_dir)
    r = list_images(rgb_dir)
    t = list_images(t_dir)
    if not m or not r or not t:
        return []
    mm = {p.name: p for p in m}
    rr = {p.name: p for p in r}
    tt = {p.name: p for p in t}
    names = sorted(set(mm.keys()) & set(rr.keys()) & set(tt.keys()))
    if names:
        return [(mm[n], rr[n], tt[n]) for n in names]
    n = min(len(m), len(r), len(t))
    return list(zip(m[:n], r[:n], t[:n]))


def common_names(dirs: List[Path]) -> List[str]:
    """Return sorted intersection of filenames across dirs (files with image extensions)."""
    if not dirs:
        return []
    sets = []
    for d in dirs:
        imgs = list_images(d)
        if not imgs:
            return []
        sets.append({p.name for p in imgs})
    inter = set.intersection(*sets)
    return sorted(inter)


def eval_ref_bundle(
    method_renders: Path,
    rgb_renders: Path,
    t_renders: Path,
    thermal_mode: str,
    thermal_align: str,
    sample_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Teacher/source-referenced evaluation bundle.

    We evaluate the fused render against:
      - RGB teacher renders (RGB-ref): standard NVS metrics + structure (EdgeCorr on Y)
      - Thermal teacher renders (T-ref): monotonic-invariant consistency on a scalar map S
      - Fusion metrics (no fused GT): classic IR-VIS fusion metrics using scalar sources:
           A = RGB luminance (Y),
           B = Thermal scalar (normalized),
           F = fused luminance (Y)

    Pairing is done on the *triple filename intersection* (method, rgb_ref, t_ref),
    to ensure all metrics are computed on the same set of frames.

    If sample_names filters everything out, we fall back to all triples to avoid NaNs,
    which otherwise make some strategies "disappear" in plots.
    """
    triples = pair_triple_by_name(method_renders, rgb_renders, t_renders)
    if not triples:
        return {}

    if sample_names is not None:
        sset = set(sample_names)
        triples_f = [(m, r, t) for (m, r, t) in triples if m.name in sset]
        if triples_f:
            triples = triples_f  # only apply if non-empty

    # RGB-ref metrics
    psnr_list: List[float] = []
    ssim_list: List[float] = []
    lpips_list: List[float] = []
    psnr_y_list: List[float] = []
    ssim_y_list: List[float] = []
    edgecorr_y_list: List[float] = []

    # Thermal-ref metrics on scalar S
    spearman_s_list: List[float] = []
    ssim_s_list: List[float] = []
    mi_s_list: List[float] = []
    nmi_s_list: List[float] = []

    # Fusion metrics (classic IR-VIS fusion metrics on scalar representations)
    mi_f_total: List[float] = []
    mi_f_rgb: List[float] = []
    mi_f_t: List[float] = []
    vif_total: List[float] = []
    vif_rgb: List[float] = []
    vif_t: List[float] = []
    qabf_list: List[float] = []
    sf_list: List[float] = []
    ssim_f_total: List[float] = []
    ssim_f_rgb: List[float] = []
    ssim_f_t: List[float] = []

    for (m_img_p, rgb_p, t_p) in triples:
        m = read_image_f32(m_img_p)
        rgb = read_image_f32(rgb_p)
        t = read_image_f32(t_p)

        # ---------- RGB teacher metrics (image-level) ----------
        psnr_list.append(psnr_rgb(m, rgb))
        ssim_list.append(float(sk_ssim(m, rgb, channel_axis=2, data_range=1.0)))

        v = lpips_rgb(m, rgb)
        if v is not None:
            lpips_list.append(v)

        my = rgb_to_y(m)
        rgby = rgb_to_y(rgb)
        psnr_y_list.append(psnr_gray(my, rgby))
        ssim_y_list.append(ssim_gray(my, rgby))
        edgecorr_y_list.append(corrcoef(edge_map_y(my), edge_map_y(rgby)))

        # ---------- Thermal teacher metrics (scalar map S) ----------
        ms = thermal_scalar(m, thermal_mode)
        ts = thermal_scalar(t, thermal_mode)
        ms_aligned = align_scalar(ms, ts, mode=thermal_align)

        spearman_s_list.append(spearman(ms_aligned, ts))
        ssim_s_list.append(ssim_gray(_normalize01(ms_aligned), _normalize01(ts)))
        mi_s_list.append(mutual_information(ms_aligned, ts))
        nmi_s_list.append(normalized_mi(ms_aligned, ts))

        # ---------- Fusion metrics (no fused GT) ----------
        # Sources: A=rgby (visible structure), B=thermal scalar (normalized),
        # Fused:   F=my (fused luminance)
        B = _normalize01(ts)
        F = _normalize01(my)
        A = _normalize01(rgby)

        miA = mutual_information(F, A)
        miB = mutual_information(F, B)
        mi_f_rgb.append(miA)
        mi_f_t.append(miB)
        mi_f_total.append(miA + miB)

        # VIF (pixel)
        vA = _vifp_ref_dist(A, F)
        vB = _vifp_ref_dist(B, F)
        vif_rgb.append(vA)
        vif_t.append(vB)
        vif_total.append(vA + vB)

        # QABF edge metric
        qabf_list.append(qabf_metric(A, B, F))

        # SF on fused
        sf_list.append(spatial_frequency(F))

        # SSIM source-referenced
        sA = ssim_gray(F, A)
        sB = ssim_gray(F, B)
        ssim_f_rgb.append(sA)
        ssim_f_t.append(sB)
        ssim_f_total.append(0.5 * (sA + sB))

    out: Dict[str, float] = {
        # RGB-ref
        "rgb_ref_PSNR": float(np.mean(psnr_list)),
        "rgb_ref_SSIM": float(np.mean(ssim_list)),
        "rgb_ref_PSNR_Y": float(np.mean(psnr_y_list)),
        "rgb_ref_SSIM_Y": float(np.mean(ssim_y_list)),
        "rgb_ref_EdgeCorr_Y": float(np.mean(edgecorr_y_list)),
        # T-ref
        "t_ref_Spearman_S": float(np.mean(spearman_s_list)),
        "t_ref_SSIM_S": float(np.mean(ssim_s_list)),
        "t_ref_MI_S": float(np.mean(mi_s_list)),
        "t_ref_NMI_S": float(np.mean(nmi_s_list)),
        # Fusion metrics
        "fusion_MI_total": float(np.mean(mi_f_total)),
        "fusion_MI_rgb": float(np.mean(mi_f_rgb)),
        "fusion_MI_t": float(np.mean(mi_f_t)),
        "fusion_VIF_total": float(np.mean(vif_total)),
        "fusion_VIF_rgb": float(np.mean(vif_rgb)),
        "fusion_VIF_t": float(np.mean(vif_t)),
        "fusion_QABF": float(np.mean(qabf_list)),
        "fusion_SF": float(np.mean(sf_list)),
        "fusion_SSIM_total": float(np.mean(ssim_f_total)),
        "fusion_SSIM_rgb": float(np.mean(ssim_f_rgb)),
        "fusion_SSIM_t": float(np.mean(ssim_f_t)),
    }
    if lpips_list:
        out["rgb_ref_LPIPS"] = float(np.mean(lpips_list))
    return out


def eval_vs_gt(
    method_renders: Path,
    gt_dir: Path,
    sample_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compare method renders against GT images (RGB GT).

    If sample_names filters everything out, fall back to all pairs to avoid NaNs.
    """
    pairs = pair_by_name(method_renders, gt_dir)
    if not pairs:
        return {}

    if sample_names is not None:
        sset = set(sample_names)
        pairs_f = [(a, b) for (a, b) in pairs if a.name in sset]
        if pairs_f:
            pairs = pairs_f

    psnr: List[float] = []
    ssim: List[float] = []
    lp: List[float] = []
    psnr_y: List[float] = []
    ssim_y: List[float] = []

    for m_p, gt_p in pairs:
        m = read_image_f32(m_p)
        g = read_image_f32(gt_p)
        psnr.append(psnr_rgb(m, g))
        ssim.append(float(sk_ssim(m, g, channel_axis=2, data_range=1.0)))

        y_m = rgb_to_y(m)
        y_g = rgb_to_y(g)
        psnr_y.append(psnr_gray(y_m, y_g))
        ssim_y.append(ssim_gray(y_m, y_g))

        v = lpips_rgb(m, g)
        if v is not None:
            lp.append(v)

    out = {
        "PSNR": float(np.mean(psnr)),
        "SSIM": float(np.mean(ssim)),
        "PSNR_Y": float(np.mean(psnr_y)),
        "SSIM_Y": float(np.mean(ssim_y)),
    }
    if lp:
        out["LPIPS"] = float(np.mean(lp))
    return out

def sample_frame_names(ref_dir: Path, k: int, seed: int) -> List[str]:
    imgs = list_images(ref_dir)
    if not imgs:
        return []
    rng = np.random.RandomState(seed)
    if k <= 0 or k >= len(imgs):
        return [p.name for p in imgs]
    idx = rng.choice(len(imgs), size=k, replace=False)
    idx.sort()
    return [imgs[i].name for i in idx]


# -----------------------------
# Optional model-level metrics merge
# -----------------------------
def _numeric_kv(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = float(v)
            continue
        if isinstance(v, (int, float)):
            fv = float(v)
            if math.isfinite(fv):
                out[k] = fv
    return out


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _extract_results_plus_metrics(path: Path) -> Dict[str, float]:
    """
    Support both:
      - {"ours_XXXX": {...metrics...}}
      - { ...metrics... }.
    """
    obj = _read_json(path)
    if obj is None:
        return {}

    # direct numeric dict
    direct = _numeric_kv(obj)
    if direct:
        return direct

    # nested first numeric dict
    for v in obj.values():
        if isinstance(v, dict):
            nested = _numeric_kv(v)
            if nested:
                return nested
    return {}


def _extract_novel_metrics(model_dir: Path) -> Dict[str, float]:
    candidates = [
        model_dir / "novel_view_metrics.json",
        model_dir / "novel_view_metrics_grid.json",
        model_dir / "novel_views_grid" / "novel_view_metrics_grid.json",
    ]
    for p in candidates:
        obj = _read_json(p)
        if obj is not None:
            return _numeric_kv(obj)
    return {}


def _load_optional_model_metrics(model_dir: Path) -> Dict[str, float]:
    """
    Merge selected metrics from results_plus / novel_view_metrics into one flat dict.
    Prefixed keys avoid collisions with sweep-native metrics.
    """
    out: Dict[str, float] = {}

    mp = _extract_results_plus_metrics(model_dir / "results_plus.json")
    nv = _extract_novel_metrics(model_dir)

    keep_mp = [
        "SGF_MetricsPlusScore",
        "SGF_StructureScore",
        "SGF_CleanScore",
        "SGF_NovelQualityScore",
        "IQA_flip",
        "IQA_fsim",
        "IQA_dists",
        "IQA_hdrvdp3",
        "TextureLCN_TenengradRatio",
        "AlignedEdgeF1_best",
        "AlignedGradientCorr",
        "BgLeakRatio",
        "AirArtifactScore",
    ]
    keep_nv = [
        "SGF_NovelQualityScore_mean",
        "SGF_CleanFarScore_mean",
        "TextureTenengrad_mean",
        "BgSensitivity_mean",
        "SpikeScore_air_mean",
        "TemporalFlicker_local_mean",
        "AirArtifactScore_mean",
    ]

    for k in keep_mp:
        if k in mp:
            out[f"mp_{k}"] = float(mp[k])
    for k in keep_nv:
        # Backward-compatible output key: keep *_mean column names,
        # but accept newer novel_view_metrics keys without the *_mean suffix.
        candidates = [k]
        if k.endswith("_mean"):
            candidates.append(k[:-5])
        for ck in candidates:
            if ck in nv:
                out[f"nv_{k}"] = float(nv[ck])
                break

    return out


# -----------------------------
# Plotting
# -----------------------------
def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def plot_lines(
    rows: List[Dict[str, Any]],
    out_path: Path,
    x_key: str,
    y_key: str,
    group_key: str = "strategy",
    label_key: str = "alpha",
    title: str = "",
) -> None:
    """
    Plot lines grouped by strategy.

    Improvements:
      - Robustly drops NaN/inf points (prevents missing/blank lines).
      - Uses varying markers/linestyles so overlapping curves are still distinguishable
        (helps cases where e.g. all_float overlaps another strategy).
      - If x_key == label_key, skips point text annotation to reduce clutter.
    """
    import matplotlib.pyplot as plt  # type: ignore
    from itertools import cycle

    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get(group_key, "")), []).append(r)

    markers = cycle(["o", "s", "^", "D", "v", "P", "*", "X", "+", "x"])
    linestyles = cycle(["-", "--", "-.", ":"])

    plt.figure(figsize=(9, 6))
    for g, items in groups.items():
        # keep only finite points
        items = [it for it in items if _is_finite_number(it.get(x_key)) and _is_finite_number(it.get(y_key))]
        if not items:
            continue
        items = sorted(items, key=lambda rr: float(rr[x_key]))
        xs = [float(it[x_key]) for it in items]
        ys = [float(it[y_key]) for it in items]

        mk = next(markers)
        ls = next(linestyles)
        plt.plot(xs, ys, marker=mk, linestyle=ls, linewidth=2.0, markersize=5.5, label=g)

        if label_key != x_key:
            for it, x, y in zip(items, xs, ys):
                if _is_finite_number(it.get(label_key)):
                    plt.text(x, y, f'{float(it[label_key]):g}', fontsize=8)

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _load_font(size: int = 16):
    try:
        from PIL import ImageFont  # type: ignore
        # try common fonts on Windows
        for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
            try:
                return ImageFont.truetype(name, size=size)
            except Exception:
                continue
        return ImageFont.load_default()
    except Exception:
        return None




def _open_pil_rgb(path: Path):
    # Keep original resolution; PNG output is lossless.
    return Image.open(path).convert("RGB")


def make_montage(
    out_path: Path,
    methods: List[MethodEntry],
    rgb_renders: Path,
    t_renders: Path,
    rgb_gt: Path,
    t_gt: Path,
    sample_name: str,
) -> None:
    """Create a single (non-paged) montage for one sample view.

    Columns:
      [RGB_GT] [RGB_ref] [alpha... for each strategy] [T_ref] [T_GT]

    Rows:
      One row per strategy.

    Notes:
      - No resizing/downsampling.
      - If some method image is missing, the cell is left blank and marked 'missing'.
    """
    from PIL import ImageDraw  # type: ignore

    strat_map: Dict[str, List[MethodEntry]] = {}
    for m in methods:
        if m.renders_dir is None:
            continue
        strat_map.setdefault(m.strategy, []).append(m)

    if not strat_map:
        return

    # Resolve reference/GT images for this sample
    rgb_ref_path = rgb_renders / sample_name
    t_ref_path = t_renders / sample_name
    rgb_gt_path = rgb_gt / sample_name
    t_gt_path = t_gt / sample_name

    if not (rgb_ref_path.exists() and t_ref_path.exists() and rgb_gt_path.exists() and t_gt_path.exists()):
        # If any side missing, skip silently (caller can choose better samples)
        return

    rgb_ref = _open_pil_rgb(rgb_ref_path)
    t_ref = _open_pil_rgb(t_ref_path)
    rgb_gt_im = _open_pil_rgb(rgb_gt_path)
    t_gt_im = _open_pil_rgb(t_gt_path)

    W, H = rgb_ref.size
    cell_w, cell_h = W, H

    strategies = sorted(strat_map.keys())
    alphas_all = sorted({round(m.alpha, 6) for m in methods})

    pad = 6
    header_h = 40
    left_w = 190
    title_h = 30

    headers = ["RGB_GT", "RGB_ref"] + [f"a={a:g}" for a in alphas_all] + ["T_ref", "T_GT"]
    ncols = len(headers)
    nrows = len(strategies)

    canvas_w = pad + left_w + ncols * (cell_w + pad)
    canvas_h = pad + title_h + header_h + nrows * (cell_h + pad)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font_hdr = _load_font(16)
    font_row = _load_font(14)
    font_title = _load_font(18)
    font_missing = _load_font(14)

    # Title
    title = f"Montage: {sample_name}"
    if font_title is not None:
        draw.text((pad, pad), title, fill=(0, 0, 0), font=font_title)
    else:
        draw.text((pad, pad), title, fill=(0, 0, 0))

    # Column headers
    y_hdr = pad + title_h
    x0 = pad + left_w
    for ci, htxt in enumerate(headers):
        x = x0 + ci * (cell_w + pad)
        tx = x + 6
        ty = y_hdr + 10
        if font_hdr is not None:
            draw.text((tx, ty), htxt, fill=(0, 0, 0), font=font_hdr)
        else:
            draw.text((tx, ty), htxt, fill=(0, 0, 0))

    # Simple cache to avoid reopening images repeatedly
    img_cache: Dict[str, Any] = {}

    def _get(path: Path):
        key = str(path)
        im = img_cache.get(key)
        if im is None:
            im = _open_pil_rgb(path)
            img_cache[key] = im
        return im

    # Paste rows
    for ri, strat in enumerate(strategies):
        y0 = pad + title_h + header_h + ri * (cell_h + pad)

        # row label
        if font_row is not None:
            draw.text((pad + 4, y0 + 4), strat, fill=(0, 0, 0), font=font_row)
        else:
            draw.text((pad + 4, y0 + 4), strat, fill=(0, 0, 0))

        x = x0

        # Left: RGB GT, RGB ref (same for all strategies)
        canvas.paste(rgb_gt_im, (x, y0)); x += cell_w + pad
        canvas.paste(rgb_ref, (x, y0)); x += cell_w + pad

        # Method renders per alpha
        entries = {round(m.alpha, 6): m for m in strat_map[strat]}
        for a in alphas_all:
            m = entries.get(round(a, 6))
            if m is not None and m.renders_dir is not None:
                p = m.renders_dir / sample_name
                if p.exists():
                    canvas.paste(_get(p), (x, y0))
                else:
                    # mark missing
                    if font_missing is not None:
                        draw.text((x + 10, y0 + 10), "missing", fill=(200, 0, 0), font=font_missing)
                    else:
                        draw.text((x + 10, y0 + 10), "missing", fill=(200, 0, 0))
            else:
                if font_missing is not None:
                    draw.text((x + 10, y0 + 10), "n/a", fill=(200, 0, 0), font=font_missing)
                else:
                    draw.text((x + 10, y0 + 10), "n/a", fill=(200, 0, 0))
            x += cell_w + pad

        # Right: T ref, T GT
        canvas.paste(t_ref, (x, y0)); x += cell_w + pad
        canvas.paste(t_gt_im, (x, y0)); x += cell_w + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
def main() -> None:
    ap = argparse.ArgumentParser("Evaluate sweep blends with render-ref + gt-ref metrics.")

    ap.add_argument("--sweep_root", required=True, help="Top folder: sweep_root/strategy/alpha/...")
    ap.add_argument("--rgb_render", required=True, help="RGB model output dir OR explicit renders/gt dir.")
    ap.add_argument("--t_render", required=True, help="T model output dir OR explicit renders/gt dir.")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: sweep_root/eval)")

    # render selection for refs
    ap.add_argument("--rgb_iter", type=int, default=None, help="Force RGB ours_<iter> (optional).")
    ap.add_argument("--t_iter", type=int, default=None, help="Force T ours_<iter> (optional).")

    # auto render
    ap.add_argument("--auto_render", action="store_true", help="Auto run render.py for missing method renders.")
    ap.add_argument("--render_py", default=None, help="Path to render.py (default: next to this script)")
    ap.add_argument("--python", dest="python_exe", default=sys.executable, help="Python executable for render.py")
    ap.add_argument("--render_iter", type=int, default=None, help="Iteration to render (default inferred from T model)")
    ap.add_argument("--render_source", default=None, help="Override -s for render.py (default from RGB cfg_args)")
    ap.add_argument("--render_images", default=None, help="Override -i for render.py (default from RGB cfg_args)")
    ap.add_argument("--render_resolution", type=int, default=None, help="Override -r for render.py (default from RGB cfg_args)")
    ap.add_argument("--render_extra", default="", help="Extra args appended to render.py (string).")
    ap.add_argument("--gt_mode", choices=["keep", "delete", "link"], default="link",
                    help="When auto_render: what to do with per-method gt to save disk. link uses junction to RGB gt.")

    # evaluation
    ap.add_argument("--thermal_scalar", default="hue_y", choices=["hue", "sat", "val", "hue_y"])
    ap.add_argument("--thermal_align", default="linear", choices=["none", "linear", "rank"])
    ap.add_argument("--sample_frames", type=int, default=8, help="Randomly sample K frames for metrics (0=all).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--no_montage", action="store_true", help="Disable montage output.")
    ap.add_argument("--montage_cols", type=int, default=9999, help="(Deprecated) kept for compatibility; montage is not paged anymore.")
    ap.add_argument("--montage_samples", type=int, default=5, help="How many sample views to output as montage images (default: 5).")

    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = Path(args.out_dir) if args.out_dir else (sweep_root / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # resolve references (renders + gt)
    rgb_model_or_dir = Path(args.rgb_render)
    t_model_or_dir = Path(args.t_render)

    rgb_renders, rgb_gt, rgb_it = resolve_renders_and_gt(rgb_model_or_dir, force_iter=args.rgb_iter)
    t_renders, t_gt, t_it = resolve_renders_and_gt(t_model_or_dir, force_iter=args.t_iter)

    print("Resolved refs:")
    print("  RGB renders:", rgb_renders)
    print("  RGB gt     :", rgb_gt)
    print("  T   renders:", t_renders)
    print("  T   gt     :", t_gt)

    # discover methods
    methods = discover_methods(sweep_root)

    # auto render missing
    if args.auto_render:
        render_py = Path(args.render_py) if args.render_py else (Path(__file__).resolve().parent / "render.py")
        if not render_py.exists():
            raise FileNotFoundError(f"render.py not found: {render_py}")

        # infer render params from RGB cfg_args by default
        cfg = parse_cfg_args(rgb_model_or_dir if rgb_model_or_dir.is_dir() else rgb_model_or_dir.parent)
        source_path = Path(args.render_source) if args.render_source else Path(cfg.get("source_path", ""))
        images_subdir = args.render_images if args.render_images else str(cfg.get("images", "images"))
        resolution = int(args.render_resolution if args.render_resolution is not None else int(cfg.get("resolution", 1)))

        if not source_path.exists():
            raise FileNotFoundError(
                f"render_source not found. Pass --render_source explicitly. Got: {source_path}"
            )

        preferred_iter = args.render_iter
        if preferred_iter is None:
            # default: use the iteration of the provided T-model renders (e.g. ours_40000)
            preferred_iter = t_it
        if preferred_iter is None:
            # fallback: inspect T model dir for best ours_*
            try:
                best = find_best_ours_dir(t_model_or_dir, None)
                if best is not None:
                    preferred_iter = int(best.name.split('_')[-1])
            except Exception:
                preferred_iter = None
        if preferred_iter is None:
            preferred_iter = 0

        print(f"[AUTO_RENDER] using source={source_path} images={images_subdir} r={resolution} iter={preferred_iter}")

        # shared gt = RGB gt (avoid per-alpha duplication)
        shared_gt = rgb_gt if rgb_gt.exists() else None

        auto_render_methods(
            methods=methods,
            render_py=render_py,
            source_path=source_path,
            images_subdir=images_subdir,
            resolution=resolution,
            preferred_iter=int(preferred_iter),
            python_exe=args.python_exe,
            gt_mode=args.gt_mode,
            shared_gt=shared_gt,
            extra=args.render_extra,
            verbose=True,
        )

    # after possible rendering, resolve method renders/gt dirs
    for m in methods:
        if m.renders_dir is None:
            # try resolve best ours (might have existed already)
            try:
                r, g, it = resolve_renders_and_gt(m.model_dir, force_iter=args.render_iter)
                m.renders_dir = r if r.exists() else None
                m.gt_dir = g if g.exists() else None
                m.used_iter = it
            except Exception:
                pass

    # filter valid
    methods = [m for m in methods if m.renders_dir is not None and m.renders_dir.exists()]
    if not methods:
        raise RuntimeError(f"No methods with renders found under: {sweep_root}")

    # sample frames:
    # - for GT metrics: sample from RGB GT
    sample_names_gt = sample_frame_names(rgb_gt, args.sample_frames, args.seed)
    if not sample_names_gt:
        sample_names_gt = sample_frame_names(rgb_renders, args.sample_frames, args.seed)

    # - for render-ref metrics & montage: prefer intersection of RGB renders and T renders
    ref_all = common_names([rgb_renders, t_renders])
    if ref_all:
        rng = np.random.RandomState(args.seed)
        if args.sample_frames <= 0 or args.sample_frames >= len(ref_all):
            sample_names_ref = ref_all
        else:
            idx = rng.choice(len(ref_all), size=args.sample_frames, replace=False)
            idx.sort()
            sample_names_ref = [ref_all[i] for i in idx]
    else:
        sample_names_ref = sample_frame_names(rgb_renders, args.sample_frames, args.seed)

    # evaluate

    rows_render_ref = []
    rows_gt_ref = []

    for m in tqdm(methods, desc="Evaluating", unit="method"):
        rr = eval_ref_bundle(
            method_renders=m.renders_dir,  # type: ignore
            rgb_renders=rgb_renders,
            t_renders=t_renders,
            thermal_mode=args.thermal_scalar,
            thermal_align=args.thermal_align,
            sample_names=sample_names_ref,
        )
        gr = eval_vs_gt(
            method_renders=m.renders_dir,  # type: ignore
            gt_dir=rgb_gt,
            sample_names=sample_names_gt,
        )

        rows_render_ref.append({
            "strategy": m.strategy,
            "alpha": m.alpha,
            "label": m.label,
            **{f"{k}_mean": v for k, v in rr.items()},
        })
        rows_gt_ref.append({
            "strategy": m.strategy,
            "alpha": m.alpha,
            "label": m.label,
            **{f"{k}_mean": v for k, v in gr.items()},
        })


    # write csv (pandas optional)
    rr_sorted = sorted(rows_render_ref, key=lambda r: (str(r.get("strategy", "")), float(r.get("alpha", 0.0))))
    gt_sorted = sorted(rows_gt_ref, key=lambda r: (str(r.get("strategy", "")), float(r.get("alpha", 0.0))))

    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        # stable field order: strategy, alpha, label, then the rest
        base = ["strategy", "alpha", "label"]
        extra_keys = sorted({k for row in rows for k in row.keys() if k not in base})
        fields = base + extra_keys
        try:
            import pandas as pd  # type: ignore
            pd.DataFrame(rows)[fields].to_csv(path, index=False)
        except ModuleNotFoundError:
            import csv
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in rows:
                    w.writerow({k: row.get(k, "") for k in fields})

    _write_csv(out_dir / "summary_render_ref.csv", rr_sorted)
    _write_csv(out_dir / "summary_gt_ref.csv", gt_sorted)
    print("[OK] wrote:", out_dir / "summary_render_ref.csv")
    print("[OK] wrote:", out_dir / "summary_gt_ref.csv")

    # merged summary for pipeline compatibility (run_uavfgs_pipeline step14 checks this path)
    rr_map: Dict[Tuple[str, float], Dict[str, Any]] = {}
    gt_map: Dict[Tuple[str, float], Dict[str, Any]] = {}
    model_map: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for r in rr_sorted:
        rr_map[(str(r.get("strategy", "")), float(r.get("alpha", 0.0)))] = r
    for g in gt_sorted:
        gt_map[(str(g.get("strategy", "")), float(g.get("alpha", 0.0)))] = g
    for m in methods:
        key = (str(m.strategy), float(m.alpha))
        model_map[key] = _load_optional_model_metrics(m.model_dir)

    merged_rows: List[Dict[str, Any]] = []
    for m in sorted(methods, key=lambda x: (str(x.strategy), float(x.alpha))):
        key = (str(m.strategy), float(m.alpha))
        row: Dict[str, Any] = {
            "strategy": m.strategy,
            "alpha": m.alpha,
            "label": m.label,
        }
        row.update(rr_map.get(key, {}))
        row.update(gt_map.get(key, {}))
        row.update(model_map.get(key, {}))
        merged_rows.append(row)

    _write_csv(out_dir / "summary.csv", merged_rows)
    print("[OK] wrote:", out_dir / "summary.csv")

    rr_cols = {k for row in rr_sorted for k in row.keys()}
    gt_cols = {k for row in gt_sorted for k in row.keys()}


    # Build a merged table for common Pareto plots: RGB-GT fidelity vs Thermal-teacher consistency
    _tref_map: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for r in rr_sorted:
        try:
            _tref_map[(str(r.get("strategy", "")), float(r.get("alpha", 0.0)))] = r
        except Exception:
            continue

    merged_gt_tref: List[Dict[str, Any]] = []
    for g in gt_sorted:
        key = (str(g.get("strategy", "")), float(g.get("alpha", 0.0)))
        tr = _tref_map.get(key)
        if tr is None:
            continue
        mg = dict(g)
        # bring in key thermal teacher metrics if exist
        for k in ["t_ref_Spearman_S_mean", "t_ref_MI_S_mean", "t_ref_NMI_S_mean", "t_ref_SSIM_S_mean",
                  "fusion_MI_total_mean", "fusion_QABF_mean"]:
            if k in tr:
                mg[k] = tr[k]
        merged_gt_tref.append(mg)



    # Diagnostic: report identical curves (common when two blend methods touch the same parameter set)
    def _report_identical(rows: List[Dict[str, Any]], metric: str) -> None:
        by_strat: Dict[str, List[Tuple[float, float]]] = {}
        for r in rows:
            s = str(r.get("strategy", ""))
            if metric not in r:
                continue
            try:
                a = float(r.get("alpha", 0.0))
                y = float(r.get(metric))
                if not math.isfinite(y):
                    continue
                by_strat.setdefault(s, []).append((a, y))
            except Exception:
                continue
        series: Dict[str, Tuple[List[float], List[float]]] = {}
        for s, pts in by_strat.items():
            pts = sorted(pts, key=lambda x: x[0])
            series[s] = ([p[0] for p in pts], [p[1] for p in pts])

        keys = list(series.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                s1, s2 = keys[i], keys[j]
                x1, y1 = series[s1]
                x2, y2 = series[s2]
                if x1 != x2 or len(y1) != len(y2):
                    continue
                if np.allclose(np.asarray(y1), np.asarray(y2), atol=1e-12, rtol=1e-12):
                    print(f"[INFO] Identical curve for metric '{metric}': {s1} == {s2} (curves overlap)")

    for _m in ["rgb_ref_SSIM_Y_mean", "t_ref_Spearman_S_mean", "fusion_MI_total_mean", "fusion_QABF_mean", "PSNR_Y_mean"]:
        if _m in rr_cols:
            _report_identical(rr_sorted, _m)
        if _m in gt_cols:
            _report_identical(gt_sorted, _m)

    # pareto (teacher/source-ref) : RGB structure vs Thermal consistency
    if "rgb_ref_SSIM_Y_mean" in rr_cols and "t_ref_Spearman_S_mean" in rr_cols:
        plot_lines(
            rows=rr_sorted,
            out_path=out_dir / "pareto_ref_structure_vs_thermal.png",
            x_key="rgb_ref_SSIM_Y_mean",
            y_key="t_ref_Spearman_S_mean",
            title="Teacher-ref Pareto: RGB-structure (SSIM_Y vs RGB-render) vs Thermal (Spearman_S vs T-render)",
        )
        print("[OK] wrote:", out_dir / "pareto_ref_structure_vs_thermal.png")

    # Pareto: RGB-GT fidelity vs Thermal-teacher consistency (useful for choosing alpha)
    if merged_gt_tref and "PSNR_Y_mean" in gt_cols and "t_ref_Spearman_S_mean" in rr_cols:
        plot_lines(
            rows=merged_gt_tref,
            out_path=out_dir / "pareto_gtfidelity_vs_thermal.png",
            x_key="PSNR_Y_mean",
            y_key="t_ref_Spearman_S_mean",
            title="Pareto: RGB-GT fidelity (PSNR_Y) vs Thermal consistency (Spearman_S vs T-render)",
        )
        print("[OK] wrote:", out_dir / "pareto_gtfidelity_vs_thermal.png")


    # REF curves vs alpha (teacher/source referenced)
    ref_metrics = [
        # RGB teacher
        "rgb_ref_PSNR_mean", "rgb_ref_SSIM_mean", "rgb_ref_LPIPS_mean",
        "rgb_ref_PSNR_Y_mean", "rgb_ref_SSIM_Y_mean", "rgb_ref_EdgeCorr_Y_mean",
        # Thermal teacher
        "t_ref_Spearman_S_mean", "t_ref_MI_S_mean", "t_ref_NMI_S_mean", "t_ref_SSIM_S_mean",
        # Fusion metrics (no fused GT)
        "fusion_MI_total_mean", "fusion_VIF_total_mean", "fusion_QABF_mean", "fusion_SF_mean", "fusion_SSIM_total_mean",
    ]
    for metric in ref_metrics:
        if metric in rr_cols:
            plot_lines(
                rows=rr_sorted,
                out_path=out_dir / f"curve_ref_{metric}.png",
                x_key="alpha",
                y_key=metric,
                title=f"Teacher/source-ref metric vs alpha: {metric}",
            )
            print("[OK] wrote:", out_dir / f"curve_ref_{metric}.png")

# GT curves (method vs RGB-GT) vs alpha
    for metric in ["PSNR_mean", "SSIM_mean", "LPIPS_mean", "PSNR_Y_mean", "SSIM_Y_mean"]:
        if metric in gt_cols:
            plot_lines(
                rows=gt_sorted,
                out_path=out_dir / f"curve_{metric}.png",
                x_key="alpha",
                y_key=metric,
                title=f"GT metric vs alpha: {metric}",
            )
            print("[OK] wrote:", out_dir / f"curve_{metric}.png")

    # montage
    if not args.no_montage:
        # choose montage sample names: prefer names present in all four dirs (RGB_ref/T_ref/RGB_GT/T_GT)
        all_names = common_names([rgb_renders, t_renders, rgb_gt, t_gt])
        if not all_names:
            all_names = common_names([rgb_renders, t_renders])
        if not all_names:
            all_names = sample_frame_names(rgb_renders, max(1, int(args.montage_samples)), args.seed + 123)
        else:
            rng = np.random.RandomState(args.seed + 123)
            k = int(args.montage_samples)
            if k <= 0:
                k = 1
            if k >= len(all_names):
                montage_names = all_names
            else:
                idx = rng.choice(len(all_names), size=k, replace=False)
                idx.sort()
                montage_names = [all_names[i] for i in idx]

        if all_names and 'montage_names' not in locals():
            montage_names = all_names[:max(1, int(args.montage_samples))]

        for mi, name in enumerate(montage_names, start=1):
            stem = Path(name).stem
            out_path = out_dir / f"montage_{mi:02d}_{stem}.png"
            make_montage(
                out_path=out_path,
                methods=methods,
                rgb_renders=rgb_renders,
                t_renders=t_renders,
                rgb_gt=rgb_gt,
                t_gt=t_gt,
                sample_name=name,
            )
        print(f"[OK] wrote montage_*.png (count={len(montage_names)})")

    print("Done.")


if __name__ == "__main__":
    main()
