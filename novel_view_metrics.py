#
# Novel-view metrics: render a small novel camera path and compute no-ref cleanliness proxies.
#
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import safe_state


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


def _texture_metrics(gray: np.ndarray) -> Tuple[float, float, float]:
    # Brightness-insensitive texture proxy: evaluate gradients on local-contrast-normalized gray.
    g = _local_contrast_norm(gray)
    gx, gy = _sobel_xy(g)
    gm = np.hypot(gx, gy)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    grad_p90 = float(np.percentile(gm, 90.0))
    thr = float(np.mean(gm) + 0.75 * np.std(gm))
    thr = max(thr, 1e-4)
    edge_density = float(np.mean(gm > thr))
    return tenengrad, grad_p90, edge_density


def _edge_anisotropy(gray: np.ndarray) -> float:
    # Directionality score in [0,1]: higher means clearer oriented structures, lower means isotropic clutter/smear.
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


def _nonair_noise_penalty(gray: np.ndarray, air_mask: np.ndarray) -> float:
    # Optional penalty for isotropic high-frequency clutter in non-air regions.
    if air_mask.shape != gray.shape:
        return 0.0
    non_air = ~air_mask
    if int(np.sum(non_air)) < 64:
        return 0.0
    g = _local_contrast_norm(gray)
    gx, gy = _sobel_xy(g)
    mag = np.hypot(gx, gy)
    if int(np.sum(non_air)) <= 0:
        return 0.0
    thr = float(np.percentile(mag[non_air], 80.0))
    mask = np.logical_and(non_air, mag > max(thr, 1e-6))
    if int(np.sum(mask)) < 32:
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
    anis = float(np.clip((l1 - l2) / max(l1 + l2, 1e-12), 0.0, 1.0))
    iso = 1.0 - anis
    lap = np.abs(_laplacian(gray))
    hf = float(np.mean(lap[mask]))
    return float(iso * hf)


def _laplacian(gray: np.ndarray) -> np.ndarray:
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    return _conv3x3(gray, k)


def _air_mask_from_render(gray: np.ndarray, edge_mag: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    top_h = max(1, int(round(0.65 * h)))
    top = np.zeros((h, w), dtype=bool)
    top[:top_h, :] = True
    edge_thr = float(np.percentile(edge_mag, 55.0))
    air = np.logical_and(top, edge_mag <= edge_thr)
    # Fallback for very close-up scenes: still keep a stable "upper-view" proxy.
    if float(np.mean(air)) < 0.08:
        air = top
    return air


def _air_cleanliness_metrics(gray: np.ndarray, edge_mag: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    air = _air_mask_from_render(gray, edge_mag)
    cnt = int(air.sum())
    if cnt <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    lap = np.abs(_laplacian(gray))
    base = _smooth_gray(gray, iters=3)
    resid = gray - base
    resid_abs = np.abs(resid)

    air_edge = float(np.mean(edge_mag[air]))
    air_hf = float(np.mean(lap[air]))
    air_ratio = float(cnt) / float(gray.size)

    resid_thr = max(float(np.percentile(resid[air], 92.0)), 0.03)
    edge_low = float(np.percentile(edge_mag[air], 65.0))
    blob_mask = np.logical_and(air, np.logical_and(resid >= resid_thr, edge_mag <= edge_low))
    blob_ratio = float(blob_mask.sum()) / float(cnt)

    contrast_low = float(np.percentile(resid_abs[air], 35.0))
    gray_high = float(np.percentile(gray[air], 65.0))
    fog_mask = np.logical_and(air, np.logical_and(resid_abs <= contrast_low, gray >= gray_high))
    fog_ratio = float(fog_mask.sum()) / float(cnt)

    # Lower is cleaner; weighted to emphasize obvious floaters and haze.
    artifact_score = float(air_edge + air_hf + 2.0 * blob_ratio + 1.5 * fog_ratio)
    return air_ratio, air_edge, air_hf, blob_ratio, fog_ratio, artifact_score


def _bg_leak_ratio(render: np.ndarray, bg: int, thr: float = 3.0 / 255.0) -> float:
    if bg == 0:
        mask = np.max(render, axis=2) <= thr
    else:
        mask = np.min(render, axis=2) >= (1.0 - thr)
    return float(np.mean(mask))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def _parse_float_list(raw: str, default_vals: List[float], clip_min: float = None, clip_max: float = None) -> List[float]:
    if raw is None:
        vals = list(default_vals)
    else:
        text = str(raw).strip()
        if not text:
            vals = list(default_vals)
        else:
            vals = []
            for tok in text.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    vals.append(float(tok))
                except ValueError:
                    continue
    out: List[float] = []
    for v in vals:
        x = float(v)
        if clip_min is not None:
            x = max(float(clip_min), x)
        if clip_max is not None:
            x = min(float(clip_max), x)
        out.append(x)
    if not out:
        out = list(default_vals)
    return out


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    t = str(v).strip().lower()
    if t in ("1", "true", "yes", "y", "on"):
        return True
    if t in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def _resolve_torch_device(requested: str) -> torch.device:
    req = str(requested).strip().lower()
    if req not in ("cpu", "cuda"):
        req = "cuda"
    if req == "cuda" and (not torch.cuda.is_available()):
        print("[WARN] novel_view_metrics device=cuda but CUDA unavailable, fallback to cpu")
        req = "cpu"
    return torch.device(req)


def _camera_mats(cam) -> Tuple[np.ndarray, np.ndarray]:
    w2c = cam.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)
    return w2c, c2w


def _camera_center(cam) -> np.ndarray:
    _, c2w = _camera_mats(cam)
    return c2w[:3, 3].copy()


def _project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = _normalize(normal.astype(np.float32))
    out = v.astype(np.float32) - float(np.dot(v, n)) * n
    if float(np.linalg.norm(out)) < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return _normalize(out)


def _aligned_axis_mean(vectors: List[np.ndarray]) -> np.ndarray:
    valid: List[np.ndarray] = []
    for v in vectors:
        vn = _normalize(v.astype(np.float32))
        if float(np.linalg.norm(vn)) >= 1e-8:
            valid.append(vn)
    if not valid:
        return np.zeros((3,), dtype=np.float32)
    ref = valid[0]
    aligned: List[np.ndarray] = []
    for v in valid:
        if float(np.dot(v, ref)) < 0.0:
            v = -v
        aligned.append(v)
    mean = _normalize(np.mean(np.stack(aligned, axis=0), axis=0).astype(np.float32))
    if float(np.linalg.norm(mean)) < 1e-8:
        mean = ref
    return mean


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _build_c2w_lookat(cam_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    z = _normalize(target - cam_pos)
    if float(np.linalg.norm(z)) < 1e-8:
        z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up = _normalize(up_hint)
    if float(np.linalg.norm(up)) < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # Build a right-handed camera basis: x = up x forward, y = forward x x.
    x = np.cross(up, z)
    if float(np.linalg.norm(x)) < 1e-8:
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(alt, z))) > 0.95:
            alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = np.cross(alt, z)
    x = _normalize(x)
    y = _normalize(np.cross(z, x))
    R = np.stack([x, y, z], axis=1)
    R = _orthonormalize(R)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.astype(np.float32)
    c2w[:3, 3] = cam_pos.astype(np.float32)
    return c2w


def _build_c2w_from_forward_right(cam_pos: np.ndarray, forward: np.ndarray, right_hint: np.ndarray) -> np.ndarray:
    z = _normalize(forward.astype(np.float32))
    if float(np.linalg.norm(z)) < 1e-8:
        z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x = right_hint.astype(np.float32) - float(np.dot(right_hint, z)) * z
    if float(np.linalg.norm(x)) < 1e-8:
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(alt, z))) > 0.95:
            alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = alt - float(np.dot(alt, z)) * z
    x = _normalize(x)
    y = _normalize(np.cross(z, x))
    R = np.stack([x, y, z], axis=1)
    R = _orthonormalize(R)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.astype(np.float32)
    c2w[:3, 3] = cam_pos.astype(np.float32)
    return c2w


def _nearest_neighbor_dist(centers: np.ndarray) -> np.ndarray:
    n = centers.shape[0]
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    out = np.full((n,), np.inf, dtype=np.float32)
    chunk = 512
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        a = centers[i0:i1]
        d = np.linalg.norm(a[:, None, :] - centers[None, :, :], axis=2)
        rows = np.arange(i0, i1) - i0
        d[rows, np.arange(i0, i1)] = np.inf
        out[i0:i1] = d.min(axis=1)
    out[~np.isfinite(out)] = 0.0
    return out


def _camera_scene_reference(cams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    centers = np.stack([_camera_center(c) for c in cams], axis=0).astype(np.float32)
    center_med = np.median(centers, axis=0).astype(np.float32)
    centered = centers - center_med[None, :]

    if centered.shape[0] >= 3:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        up = vt[-1].astype(np.float32)
    else:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up = _normalize(up)
    if float(np.linalg.norm(up)) < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Estimate a look-at center by least-squares intersection of camera forward rays.
    fwd = []
    for c in cams:
        _, c2w = _camera_mats(c)
        d = _normalize(c2w[:3, 2].astype(np.float32))
        if float(np.linalg.norm(d)) < 1e-8:
            d = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        fwd.append(d)
    fwd = np.stack(fwd, axis=0)
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3,), dtype=np.float64)
    for p, d in zip(centers.astype(np.float64), fwd.astype(np.float64)):
        m = np.eye(3, dtype=np.float64) - np.outer(d, d)
        A += m
        b += m @ p
    try:
        center = np.linalg.solve(A, b).astype(np.float32)
    except np.linalg.LinAlgError:
        center = center_med.copy()

    # Resolve forward sign so that most rays point toward the estimated center.
    t_med = float(np.median(np.einsum("ij,ij->i", (center[None, :] - centers), fwd)))
    if t_med < 0.0:
        fwd = -fwd

    # Resolve plane-normal sign robustly:
    # 1) cameras should lie mostly on +up side of the scene center;
    # 2) camera forward rays should point mostly toward -up side (down-looking for aerial sets).
    def _up_score(u: np.ndarray) -> Tuple[float, float, float]:
        cam_side = float(np.median(np.dot(centers - center[None, :], u)))
        look_side = float(np.median(np.dot(fwd, u)))
        score = cam_side - look_side
        return score, cam_side, look_side

    s_pos, cam_pos, look_pos = _up_score(up)
    s_neg, cam_neg, look_neg = _up_score(-up)
    if s_neg > s_pos:
        up = -up
        cam_pos, look_pos = cam_neg, look_neg

    # Ambiguous fallback: keep +Z preference to avoid below-ground sampling flips.
    if abs(cam_pos) <= 1e-6 and abs(look_pos) <= 1e-6 and up[2] < 0.0:
        up = -up

    cam_dist = np.linalg.norm(centers - center[None, :], axis=1)
    d50 = float(np.percentile(cam_dist, 50.0)) if cam_dist.size > 0 else 1.0
    d90 = float(np.percentile(cam_dist, 90.0)) if cam_dist.size > 0 else max(1.0, d50)
    d50 = max(0.05, d50)
    d90 = max(d50 + 1e-6, d90)
    return centers, center, up, d50, d90


def _scene_center_radius(scene: Scene, gaussians: GaussianModel) -> Tuple[np.ndarray, float]:
    xyz = gaussians.get_xyz.detach().cpu().numpy() if gaussians.get_xyz.numel() > 0 else np.zeros((0, 3), dtype=np.float32)
    if xyz.shape[0] > 0:
        center = np.mean(xyz, axis=0).astype(np.float32)
        dist = np.linalg.norm(xyz - center[None, :], axis=1)
        # Use a robust percentile to avoid a few outliers dominating far-view distance.
        radius = float(np.percentile(dist, 90.0))
        if not np.isfinite(radius) or radius <= 1e-6:
            radius = max(0.05, float(scene.cameras_extent) * 0.3)
        return center, radius
    cams = scene.getTestCameras() or scene.getTrainCameras()
    if cams:
        ctrs = np.stack([_camera_center(c) for c in cams], axis=0)
        center = np.mean(ctrs, axis=0).astype(np.float32)
        dist = np.linalg.norm(ctrs - center[None, :], axis=1)
        radius = float(np.percentile(dist, 80.0))
        if not np.isfinite(radius) or radius <= 1e-6:
            radius = max(0.05, float(scene.cameras_extent) * 0.3)
        return center, radius
    return np.zeros((3,), dtype=np.float32), max(0.05, float(scene.cameras_extent) * 0.3)


def _connected_components(mask: np.ndarray) -> List[Tuple[int, int, int]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[Tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            minx = maxx = x
            miny = maxy = y
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((area, maxx - minx + 1, maxy - miny + 1))
    return comps


def _spike_score(edge_mag: np.ndarray, thr: float = 0.1, valid_mask: Optional[np.ndarray] = None) -> float:
    mask = edge_mag > thr
    if valid_mask is not None:
        if valid_mask.shape != mask.shape:
            return 0.0
        mask = np.logical_and(mask, valid_mask.astype(bool))
    ds = mask[::4, ::4]
    if valid_mask is not None:
        vm = valid_mask[::4, ::4].astype(bool)
        denom = int(vm.sum())
    else:
        denom = int(ds.size)
    if denom <= 0:
        return 0.0
    comps = _connected_components(ds)
    spike_pixels = 0
    for area, w, h in comps:
        if area < 2:
            continue
        ar = max(w, h) / max(1, min(w, h))
        if ar >= 4.0 and area <= 200:
            spike_pixels += area
    return float(spike_pixels) / float(denom)


def _collect_frame_tags(out_dir: Path, num_frames: int) -> List[Tuple[int, bool]]:
    tags: List[Tuple[int, bool]] = [(0, False) for _ in range(max(0, int(num_frames)))]
    files = sorted(out_dir.glob("*.png"))
    if len(files) < num_frames:
        return tags
    out: List[Tuple[int, bool]] = []
    for i in range(num_frames):
        name = files[i].name
        m = re.search(r"_dst(\d+)", name)
        dist_idx = int(m.group(1)) if m else 0
        is_topdown = "_topdown_" in name
        out.append((dist_idx, is_topdown))
    return out


def _collect_frame_names(out_dir: Path, num_frames: int) -> List[str]:
    names = [f.name for f in sorted(out_dir.glob("*.png"))]
    if len(names) < num_frames:
        return [f"{i:04d}.png" for i in range(max(0, int(num_frames)))]
    return names[:num_frames]


def _grid_local_group_from_name(name: str) -> Optional[Tuple[int, int, int, bool]]:
    # Example: 0000_dir00_pit00_dst00_a123_p30_de12.34.png
    m = re.search(r"_dir(\d+)_pit(\d+)_dst(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), ("_topdown_" in name)


def _local_flicker_stats(grays: List[np.ndarray], frame_names: List[str]) -> Tuple[List[float], float, float]:
    # Local flicker: only compare adjacent distance levels under the same (dir, pit).
    # This avoids penalizing large viewpoint jumps in grid traversal.
    n = min(len(grays), len(frame_names))
    if n <= 1:
        return [], 0.0, 0.0
    groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for i in range(n):
        parsed = _grid_local_group_from_name(frame_names[i])
        if parsed is None:
            continue
        dir_i, pit_i, dst_i, is_topdown = parsed
        if is_topdown:
            continue
        groups.setdefault((dir_i, pit_i), []).append((dst_i, i))

    vals: List[float] = []
    for _k, arr in groups.items():
        if len(arr) < 2:
            continue
        arr = sorted(arr, key=lambda x: x[0])
        for j in range(1, len(arr)):
            i0 = arr[j - 1][1]
            i1 = arr[j][1]
            vals.append(float(np.mean(np.abs(grays[i1] - grays[i0]))))

    if not vals:
        return [], 0.0, 0.0
    return vals, float(np.mean(vals)), float(np.percentile(vals, 90.0))


def _append_mean_p90(results: Dict[str, Any], key: str, values: List[float]) -> None:
    if not values:
        return
    arr = np.asarray(values, dtype=np.float64)
    results[f"{key}_mean"] = float(np.mean(arr))
    results[f"{key}_p90"] = float(np.percentile(arr, 90.0))


def _write_bucket_means(results: Dict[str, Any], key_prefix: str, values: List[float], frame_tags: List[Tuple[int, bool]]) -> None:
    if not values or not frame_tags or len(values) != len(frame_tags):
        return
    bucket_vals: Dict[int, List[float]] = {}
    top_vals: List[float] = []
    for v, (dist_idx, is_topdown) in zip(values, frame_tags):
        if is_topdown:
            top_vals.append(float(v))
            continue
        bucket_vals.setdefault(int(dist_idx), []).append(float(v))
    for dist_idx in sorted(bucket_vals.keys()):
        arr = bucket_vals[dist_idx]
        if not arr:
            continue
        results[f"{key_prefix}_d{dist_idx:02d}_mean"] = float(np.mean(arr))
        results[f"{key_prefix}_d{dist_idx:02d}_count"] = int(len(arr))
    if top_vals:
        results[f"{key_prefix}_topdown_mean"] = float(np.mean(top_vals))
        results[f"{key_prefix}_topdown_count"] = int(len(top_vals))


def _bucket_pairs(results: Dict[str, Any], key_prefix: str) -> List[Tuple[int, float]]:
    pairs: List[Tuple[int, float]] = []
    pat = re.compile(rf"^{re.escape(key_prefix)}_d(\d+)_mean$")
    for k, v in results.items():
        m = pat.match(str(k))
        if not m:
            continue
        try:
            idx = int(m.group(1))
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            pairs.append((idx, fv))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _head_mean(pairs: List[Tuple[int, float]], n: int, fallback: float) -> float:
    if not pairs:
        return float(fallback)
    vals = [v for _, v in pairs[: max(1, int(n))]]
    return float(np.mean(vals)) if vals else float(fallback)


def _tail_mean(pairs: List[Tuple[int, float]], n: int, fallback: float) -> float:
    if not pairs:
        return float(fallback)
    vals = [v for _, v in pairs[-max(1, int(n)) :]]
    return float(np.mean(vals)) if vals else float(fallback)


def _prepare_png_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass


def _find_sibr_viewer_exe(user_path: str = "") -> Optional[Path]:
    if user_path:
        p = Path(user_path).expanduser()
        if p.exists():
            return p
    here = Path(__file__).resolve().parent
    cands = [
        here / "SIBR_viewers" / "install" / "bin" / "SIBR_gaussianViewer_app.exe",
        here / "SIBR_viewers" / "install" / "bin" / "SIBR_gaussianViewer_app",
        Path("SIBR_viewers") / "install" / "bin" / "SIBR_gaussianViewer_app.exe",
        Path("SIBR_viewers") / "install" / "bin" / "SIBR_gaussianViewer_app",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


_SIBR_HELP_CACHE: Dict[str, bool] = {}


def _sibr_supports_gaussian_mode(sibr_exe: Path) -> bool:
    key = str(sibr_exe)
    if key in _SIBR_HELP_CACHE:
        return _SIBR_HELP_CACHE[key]
    try:
        proc = subprocess.run(
            [str(sibr_exe), "--help"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        text = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
        ok = ("gaussian-mode" in text)
        _SIBR_HELP_CACHE[key] = bool(ok)
        return bool(ok)
    except Exception:
        _SIBR_HELP_CACHE[key] = False
        return False


def _camera_to_lookat_line(cam: MiniCam, name: str) -> str:
    w2c = cam.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)
    origin = c2w[:3, 3]
    target = origin + c2w[:3, 2]
    up = c2w[:3, 1]
    fovy_deg = float(cam.FoVy) * 180.0 / math.pi
    # SIBR path render can output blank frames when clip far is too small for large orbit radii.
    znear = min(float(cam.znear), 0.01)
    zfar = max(float(cam.zfar), 10000.0)
    return (
        f"{name} -D origin={origin[0]:.6f},{origin[1]:.6f},{origin[2]:.6f}"
        f" -D target={target[0]:.6f},{target[1]:.6f},{target[2]:.6f}"
        f" -D up={up[0]:.6f},{up[1]:.6f},{up[2]:.6f}"
        f" -D fovy={fovy_deg:.6f} -D clip={znear:.6f},{zfar:.6f}\n"
    )


def _write_lookat_file(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln)


def _lookat_line_from_pose(
    name: str,
    origin: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    fovy_rad: float,
    znear: float,
    zfar: float,
) -> str:
    o = np.asarray(origin, dtype=np.float32).reshape(3)
    t = np.asarray(target, dtype=np.float32).reshape(3)
    u = _normalize(np.asarray(up, dtype=np.float32).reshape(3))
    if float(np.linalg.norm(u)) < 1e-8:
        u = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fovy_deg = float(fovy_rad) * 180.0 / math.pi
    return (
        f"{name} -D origin={o[0]:.6f},{o[1]:.6f},{o[2]:.6f}"
        f" -D target={t[0]:.6f},{t[1]:.6f},{t[2]:.6f}"
        f" -D up={u[0]:.6f},{u[1]:.6f},{u[2]:.6f}"
        f" -D fovy={fovy_deg:.6f} -D clip={float(znear):.6f},{float(zfar):.6f}\n"
    )


def _write_bundle_file(path: Path, cams: List[MiniCam]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(cams) <= 0:
        path.write_text("# Bundle file v0.3\n0 0\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as f:
        f.write("# Bundle file v0.3\n")
        f.write(f"{len(cams)} 0\n")
        for cam in cams:
            h = float(cam.image_height)
            fovy = float(cam.FoVy)
            focal = 0.5 * h / max(math.tan(0.5 * fovy), 1e-8)
            w2c = cam.world_view_transform.transpose(0, 1).detach().cpu().numpy().astype(np.float64)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            f.write(f"{focal:.9f} 0.0 0.0\n")
            f.write(f"{R[0,0]:.9f} {R[0,1]:.9f} {R[0,2]:.9f}\n")
            f.write(f"{R[1,0]:.9f} {R[1,1]:.9f} {R[1,2]:.9f}\n")
            f.write(f"{R[2,0]:.9f} {R[2,1]:.9f} {R[2,2]:.9f}\n")
            f.write(f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f}\n")


def _run_sibr_offline(
    sibr_exe: Path,
    model_path: Path,
    source_path: Path,
    path_file: Path,
    out_dir: Path,
    width: int,
    height: int,
    device_id: int,
    gaussian_mode: str,
    iteration: Optional[int] = None,
) -> Tuple[bool, str, bool, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass
    supports_mode = _sibr_supports_gaussian_mode(sibr_exe)
    mode_attempts: List[bool] = [bool(supports_mode)]
    if supports_mode:
        mode_attempts.append(False)
    attempts: List[Tuple[bool, bool]] = []
    for offscreen in [True, False]:
        for use_mode_flag in mode_attempts:
            attempts.append((offscreen, use_mode_flag))

    last_msg = "unknown"
    for use_offscreen, use_mode_flag in attempts:
        cmd = [
            str(sibr_exe),
            "-m",
            str(model_path),
            "-s",
            str(source_path),
        ]
        if iteration is not None and int(iteration) > 0:
            cmd.extend(["--iteration", str(int(iteration))])
        if use_offscreen:
            cmd.append("--offscreen")
        cmd.extend(
            [
                "--pathFile",
                str(path_file),
                "--outPath",
                str(out_dir),
            ]
        )
        if use_mode_flag:
            cmd.extend(["--gaussian-mode", str(gaussian_mode)])
        cmd.extend(
            [
                "--rendering-size",
                str(int(width)),
                str(int(height)),
                "--force-aspect-ratio",
                "--device",
                str(int(device_id)),
            ]
        )
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        except Exception as ex:
            last_msg = f"spawn_error:{ex}"
            continue
        n_png = len(list(out_dir.glob("*.png")))
        if proc.returncode == 0 and n_png > 0:
            detail = "offscreen" if use_offscreen else "windowed"
            mode_detail = "mode_arg" if use_mode_flag else "mode_default"
            return True, f"ok_{detail}_{mode_detail}", use_mode_flag, use_offscreen
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-6:])
        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-6:])
        tail = stderr_tail if stderr_tail else stdout_tail
        last_msg = f"exit={proc.returncode} png={n_png} offscreen={int(use_offscreen)} mode_arg={int(use_mode_flag)} {tail}".strip()

    return False, last_msg, False, False


def _rename_sibr_frames_to_match(sibr_dir: Path, target_names: List[str]) -> Tuple[int, int]:
    src_files = sorted(sibr_dir.glob("*.png"))
    n_src = len(src_files)
    n_tgt = len(target_names)
    if n_src <= 0 or n_tgt <= 0:
        return n_src, n_tgt
    # If SIBR renders an extra warm-up frame (expected), keep the latest n_tgt frames.
    start = max(0, n_src - n_tgt)
    src_pick = src_files[start:]
    pick_set = {p.name for p in src_pick}
    # Drop unpicked frames (typically warm-up black frame) so folder view stays 1:1 with novel_views_grid.
    for src in src_files:
        if src.name not in pick_set:
            try:
                src.unlink()
            except OSError:
                pass
    n = min(len(src_pick), n_tgt)
    if n <= 0:
        return n_src, n_tgt
    tmp_files: List[Path] = []
    for i in range(n):
        src = src_pick[i]
        tmp = sibr_dir / f"__tmp_sibr_{i:06d}.png"
        try:
            src.rename(tmp)
            tmp_files.append(tmp)
        except OSError:
            tmp_files.append(src)
    for i in range(n):
        dst = sibr_dir / target_names[i]
        try:
            if dst.exists():
                dst.unlink()
            tmp_files[i].rename(dst)
        except OSError:
            pass
    keep_names = set(target_names[:n])
    for leftover in sibr_dir.glob("*.png"):
        if leftover.name not in keep_names:
            try:
                leftover.unlink()
            except OSError:
                pass
    return n_src, n_tgt


def _invert_lookat_forward(lines: List[str]) -> List[str]:
    out: List[str] = []
    pat = re.compile(
        r"^(?P<prefix>.*?-D origin=)(?P<ox>[-+0-9.eE]+),(?P<oy>[-+0-9.eE]+),(?P<oz>[-+0-9.eE]+)"
        r"(?P<m1>.*?-D target=)(?P<tx>[-+0-9.eE]+),(?P<ty>[-+0-9.eE]+),(?P<tz>[-+0-9.eE]+)(?P<suffix>.*)$"
    )
    for ln in lines:
        m = pat.match(ln.rstrip("\n"))
        if not m:
            out.append(ln)
            continue
        ox = float(m.group("ox"))
        oy = float(m.group("oy"))
        oz = float(m.group("oz"))
        tx = float(m.group("tx"))
        ty = float(m.group("ty"))
        tz = float(m.group("tz"))
        ntx = 2.0 * ox - tx
        nty = 2.0 * oy - ty
        ntz = 2.0 * oz - tz
        rebuilt = (
            f"{m.group('prefix')}{ox:.6f},{oy:.6f},{oz:.6f}"
            f"{m.group('m1')}{ntx:.6f},{nty:.6f},{ntz:.6f}{m.group('suffix')}\n"
        )
        out.append(rebuilt)
    return out


def _image_transform(img: np.ndarray, tag: str) -> np.ndarray:
    if tag == "vflip":
        return img[::-1, :, :]
    if tag == "hflip":
        return img[:, ::-1, :]
    if tag == "vhflip":
        return img[::-1, ::-1, :]
    return img


def _score_sibr_alignment(
    ref_dir: Path,
    sibr_dir: Path,
    names: List[str],
    transform_tag: str,
    max_samples: int = 12,
) -> float:
    if not names:
        return float("inf")
    if max_samples <= 0:
        max_samples = 12
    idxs = np.linspace(0, len(names) - 1, num=min(max_samples, len(names)))
    idxs = np.unique(np.round(idxs).astype(np.int32)).tolist()
    errs: List[float] = []
    valid_ratios: List[float] = []
    for i in idxs:
        nm = names[int(i)]
        rp = ref_dir / nm
        sp = sibr_dir / nm
        if (not rp.exists()) or (not sp.exists()):
            continue
        try:
            ra = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
            sa = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0
        except Exception:
            continue
        sa = _image_transform(sa, transform_tag)
        valid_ratios.append(float(np.mean(np.max(sa, axis=2) > 0.03)))
        rg = _rgb_to_gray(ra)
        sg = _rgb_to_gray(sa)
        rg = (rg - float(np.mean(rg))) / (float(np.std(rg)) + 1e-6)
        sg = (sg - float(np.mean(sg))) / (float(np.std(sg)) + 1e-6)
        errs.append(float(np.mean((rg - sg) ** 2)))
    if not errs:
        return float("inf")
    valid_mean = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    valid_frame_ratio = float(np.mean(np.asarray(valid_ratios, dtype=np.float32) > 0.05)) if valid_ratios else 0.0
    # Reject degenerate candidates (mostly black/empty outputs).
    if valid_mean < 0.20 or valid_frame_ratio < 0.70:
        return float("inf")
    # Mild penalty for low visible coverage.
    cov_penalty = 0.25 * max(0.0, 0.20 - valid_mean)
    return float(np.mean(errs) + cov_penalty)


def _apply_sibr_transform_inplace(sibr_dir: Path, transform_tag: str) -> None:
    if transform_tag == "none":
        return
    for fp in sibr_dir.glob("*.png"):
        try:
            with Image.open(fp) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            arr = (_image_transform(arr, transform_tag)).astype(np.uint8)
            Image.fromarray(arr).save(fp)
        except Exception:
            continue


def _qnorm_torch(x: torch.Tensor, qlo: float = 0.02, qhi: float = 0.98, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo = torch.quantile(x, qlo)
    hi = torch.quantile(x, qhi)
    if (not torch.isfinite(lo)) or (not torch.isfinite(hi)):
        return torch.zeros_like(x)
    denom = hi - lo
    if torch.abs(denom) < eps:
        return torch.zeros_like(x)
    return torch.clamp((x - lo) / denom, 0.0, 1.0)


def _build_ellipsoid_proxy_override_color(gaussians: GaussianModel) -> Optional[torch.Tensor]:
    with torch.no_grad():
        dc = gaussians.get_features_dc
        if dc is None or dc.numel() == 0:
            return None
        # Use SH-DC color to keep appearance close to the model's native colors.
        # _features_dc is (N,1,3) in this repo.
        if dc.dim() == 3 and dc.shape[1] == 1:
            dc = dc[:, 0, :]
        dc = dc.float()
        rgb = torch.clamp(dc + 0.5, 0.0, 1.0)
        return rgb


def _render_save_pair(
    cam: MiniCam,
    gaussians: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    out_dir: Path,
    file_name: str,
    frames: List[np.ndarray],
    proxy_enabled: bool,
    proxy_dir: Optional[Path],
    proxy_override_color: Optional[torch.Tensor],
    bg_mode: int,
    bg_sens_means: Optional[List[float]] = None,
    bg_sens_ratios: Optional[List[float]] = None,
    bg_sens_thr: float = 8.0 / 255.0,
    sibr_lookat_lines: Optional[List[str]] = None,
    sibr_cam_name: Optional[str] = None,
) -> None:
    render_out = render(cam, gaussians, pipe, bg_color, use_trained_exp=False)
    img = render_out["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    frames.append(img)
    img_u8 = (img * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(out_dir / file_name)
    if sibr_lookat_lines is not None:
        cam_name = str(sibr_cam_name) if sibr_cam_name else Path(file_name).stem
        sibr_lookat_lines.append(_camera_to_lookat_line(cam, cam_name))

    if bg_sens_means is not None and bg_sens_ratios is not None:
        if int(bg_mode) == 0:
            img_b = img
            bg_white = torch.tensor([1, 1, 1], dtype=torch.float32, device=bg_color.device)
            rw = render(cam, gaussians, pipe, bg_white, use_trained_exp=False)
            img_w = rw["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        else:
            img_w = img
            bg_black = torch.tensor([0, 0, 0], dtype=torch.float32, device=bg_color.device)
            rb = render(cam, gaussians, pipe, bg_black, use_trained_exp=False)
            img_b = rb["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        sens = np.mean(np.abs(img_b - img_w), axis=2)
        bg_sens_means.append(float(np.mean(sens)))
        bg_sens_ratios.append(float(np.mean(sens > float(bg_sens_thr))))

    if proxy_enabled and proxy_dir is not None and proxy_override_color is not None:
        # For ellipsoid diagnostics we force near-opaque alpha so distribution is easier to inspect.
        # Keep it local to proxy pass and restore immediately after render.
        op_backup = None
        aa_backup = None
        try:
            if hasattr(gaussians, "_opacity") and gaussians._opacity is not None:
                op_backup = gaussians._opacity.detach().clone()
                with torch.no_grad():
                    gaussians._opacity.data.fill_(12.0)  # sigmoid(12) ~= 0.999994
            if hasattr(pipe, "antialiasing"):
                aa_backup = bool(pipe.antialiasing)
                pipe.antialiasing = False
            render_proxy = render(
                cam,
                gaussians,
                pipe,
                bg_color,
                scaling_modifier=0.65,
                override_color=proxy_override_color,
                use_trained_exp=False,
            )
        finally:
            if aa_backup is not None:
                pipe.antialiasing = aa_backup
            if op_backup is not None:
                with torch.no_grad():
                    gaussians._opacity.data.copy_(op_backup)
        img_proxy = render_proxy["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        # Harder depth-edge outline for clearer ellipsoid boundaries.
        depth_t = render_proxy.get("depth", None)
        if depth_t is not None:
            try:
                depth = depth_t.detach().float().squeeze().cpu().numpy()
                if depth.ndim == 2 and depth.size > 0:
                    gx, gy = np.gradient(depth)
                    g = np.hypot(gx, gy)
                    p80 = float(np.percentile(g, 80.0))
                    p98 = float(np.percentile(g, 98.0))
                    if p98 > p80:
                        e = np.clip((g - p80) / (p98 - p80), 0.0, 1.0)
                        img_proxy = np.clip(img_proxy * (1.0 - 0.65 * e[..., None]), 0.0, 1.0)
            except Exception:
                pass
        img_proxy_u8 = (img_proxy * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        stem = Path(file_name).stem
        Image.fromarray(img_proxy_u8).save(proxy_dir / f"{stem}_ellip.png")


def _render_mode_orbit(
    scene: Scene,
    gaussians: GaussianModel,
    pipe,
    device: torch.device,
    bg: int,
    n: int,
    out_dir: Path,
    proxy_enabled: bool = False,
    proxy_dir: Optional[Path] = None,
    proxy_override_color: Optional[torch.Tensor] = None,
) -> Tuple[List[np.ndarray], Dict[str, Any], List[str]]:
    cams = scene.getTestCameras() or scene.getTrainCameras()
    if not cams:
        raise RuntimeError("No cameras found to derive a novel path.")
    ref_cam = cams[0]

    w2c = ref_cam.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)
    right = _normalize(c2w[:3, 0])
    up = _normalize(c2w[:3, 1])
    forward = _normalize(c2w[:3, 2])

    center = c2w[:3, 3]
    radius = max(0.05, float(scene.cameras_extent) * 0.3)

    width = int(ref_cam.image_width)
    height = int(ref_cam.image_height)
    fovy = float(ref_cam.FoVy)
    fovx = float(ref_cam.FoVx)
    znear = float(ref_cam.znear)
    zfar = float(ref_cam.zfar)

    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).to(device)
    bg_color = torch.tensor([1, 1, 1] if bg == 1 else [0, 0, 0], dtype=torch.float32, device=device)

    _prepare_png_dir(out_dir)
    if proxy_enabled and proxy_dir is not None:
        _prepare_png_dir(proxy_dir)
    frames: List[np.ndarray] = []
    sibr_lines: List[str] = []
    bg_sens_means: List[float] = []
    bg_sens_ratios: List[float] = []

    for i in range(n):
        theta = 2.0 * math.pi * (i / float(n))
        delta = radius * (math.cos(theta) * right + math.sin(theta) * forward)
        delta += (0.1 * radius * math.sin(2.0 * theta)) * up
        c2w_new = c2w.copy()
        c2w_new[:3, 3] = center + delta
        w2c_new = np.linalg.inv(c2w_new)

        world_view = torch.tensor(w2c_new, dtype=torch.float32, device=device).transpose(0, 1)
        full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full_proj)

        _render_save_pair(
            cam, gaussians, pipe, bg_color, out_dir, f"{i:04d}.png", frames,
            proxy_enabled, proxy_dir, proxy_override_color, int(bg),
            bg_sens_means=bg_sens_means, bg_sens_ratios=bg_sens_ratios,
            sibr_lookat_lines=sibr_lines, sibr_cam_name=f"{i:04d}"
        )

    return frames, {
        "mode": "orbit",
        "num_frames_target": int(n),
        "bg_sensitivity_thr": float(8.0 / 255.0),
        "bg_sensitivity_means": bg_sens_means,
        "bg_sensitivity_ratios": bg_sens_ratios,
    }, sibr_lines


def _render_mode_test_offset(
    scene: Scene,
    gaussians: GaussianModel,
    pipe,
    device: torch.device,
    bg: int,
    n: int,
    out_dir: Path,
    shift_lat: float,
    shift_up: float,
    lookat_blend: float,
    seed: int,
    proxy_enabled: bool = False,
    proxy_dir: Optional[Path] = None,
    proxy_override_color: Optional[torch.Tensor] = None,
) -> Tuple[List[np.ndarray], Dict[str, Any], List[str]]:
    cams = scene.getTestCameras() or scene.getTrainCameras()
    if not cams:
        raise RuntimeError("No cameras found to derive test-offset path.")
    n_total = len(cams)
    n_use = n_total if n <= 0 else min(int(n), n_total)
    if n_use <= 0:
        raise RuntimeError("No cameras selected for test-offset mode.")
    cams = cams[:n_use]

    centers, center_scene, _up_scene, _d50, _d90 = _camera_scene_reference(cams)
    nn_dist = _nearest_neighbor_dist(centers)
    fallback = max(0.05, float(scene.cameras_extent) * 0.05)
    nn_dist = np.maximum(nn_dist, fallback)

    rng = np.random.default_rng(int(seed))
    bg_color = torch.tensor([1, 1, 1] if bg == 1 else [0, 0, 0], dtype=torch.float32, device=device)
    _prepare_png_dir(out_dir)
    if proxy_enabled and proxy_dir is not None:
        _prepare_png_dir(proxy_dir)
    frames: List[np.ndarray] = []
    sibr_lines: List[str] = []
    bg_sens_means: List[float] = []
    bg_sens_ratios: List[float] = []

    for i, src_cam in enumerate(cams):
        _, c2w = _camera_mats(src_cam)
        right = _normalize(c2w[:3, 0])
        up = _normalize(c2w[:3, 1])
        forward = _normalize(c2w[:3, 2])
        c = c2w[:3, 3].copy()
        local = float(nn_dist[i])
        theta = 2.0 * math.pi * (float(i) / float(max(1, n_use)))
        jitter = rng.uniform(-0.15, 0.15)
        delta = local * shift_lat * ((math.cos(theta + jitter) * right) + (math.sin(theta - jitter) * forward))
        delta += local * shift_up * (math.sin(2.0 * theta + jitter) * up)
        c_new = c + delta

        if lookat_blend > 0.0:
            c2w_look = _build_c2w_lookat(c_new, center_scene, up)
            R_old = c2w[:3, :3]
            R_new = (1.0 - lookat_blend) * R_old + lookat_blend * c2w_look[:3, :3]
            R_new = _orthonormalize(R_new.astype(np.float32))
            c2w_new = np.eye(4, dtype=np.float32)
            c2w_new[:3, :3] = R_new
            c2w_new[:3, 3] = c_new.astype(np.float32)
        else:
            c2w_new = c2w.astype(np.float32)
            c2w_new[:3, 3] = c_new.astype(np.float32)

        w2c_new = np.linalg.inv(c2w_new)
        width = int(src_cam.image_width)
        height = int(src_cam.image_height)
        fovy = float(src_cam.FoVy)
        fovx = float(src_cam.FoVx)
        znear = float(src_cam.znear)
        zfar = float(src_cam.zfar)
        proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).to(device)
        world_view = torch.tensor(w2c_new, dtype=torch.float32, device=device).transpose(0, 1)
        full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full_proj)

        _render_save_pair(
            cam, gaussians, pipe, bg_color, out_dir, f"{i:04d}.png", frames,
            proxy_enabled, proxy_dir, proxy_override_color, int(bg),
            bg_sens_means=bg_sens_means, bg_sens_ratios=bg_sens_ratios,
            sibr_lookat_lines=sibr_lines, sibr_cam_name=f"{i:04d}"
        )

    meta = {
        "mode": "test_offset",
        "num_frames_target": int(n),
        "num_frames_selected": int(n_use),
        "num_test_cameras": int(n_total),
        "shift_lat": float(shift_lat),
        "shift_up": float(shift_up),
        "lookat_blend": float(lookat_blend),
        "seed": int(seed),
        "bg_sensitivity_thr": float(8.0 / 255.0),
        "bg_sensitivity_means": bg_sens_means,
        "bg_sensitivity_ratios": bg_sens_ratios,
    }
    return frames, meta, sibr_lines


def _render_mode_grid72(scene: Scene, gaussians: GaussianModel, pipe, device: torch.device, bg: int, out_dir: Path,
                        near_scale: float, far_scale: float, far_blend: float,
                        far_cover_pad: float, far_cam_mult: float,
                        azimuth_count: int, pitch_list: List[float], dist_factors: List[float], include_topdown: bool,
                        jitter: bool, pos_jitter: float, ang_jitter_deg: float, seed: int,
                        proxy_enabled: bool = False, proxy_dir: Optional[Path] = None,
                        proxy_override_color: Optional[torch.Tensor] = None) -> Tuple[List[np.ndarray], Dict[str, Any], List[str]]:
    cams = scene.getTestCameras() or scene.getTrainCameras()
    if not cams:
        raise RuntimeError("No cameras found to derive grid72 path.")
    ref_cam = cams[0]
    centers, center_scene, up_guess, cam_d50, cam_d90 = _camera_scene_reference(cams)
    geom_center, _geom_radius = _scene_center_radius(scene, gaussians)
    render_center = geom_center.astype(np.float32) if np.all(np.isfinite(geom_center)) else center_scene
    # Derive image-plane axes from real camera rotations to lock roll to dataset convention.
    # This avoids relying on scene_up for camera roll, which can be ambiguous for aerial scenes.
    ref_ups = []
    ref_rights = []
    ref_forwards = []
    for c in cams:
        _, c2w_ref = _camera_mats(c)
        ref_ups.append(_normalize(c2w_ref[:3, 1].astype(np.float32)))
        ref_rights.append(_normalize(c2w_ref[:3, 0].astype(np.float32)))
        ref_forwards.append(_normalize(c2w_ref[:3, 2].astype(np.float32)))
    ref_ups_np = np.stack(ref_ups, axis=0).astype(np.float32)
    ref_rights_np = np.stack(ref_rights, axis=0).astype(np.float32)
    ref_forwards_np = np.stack(ref_forwards, axis=0).astype(np.float32)
    _, ref_c2w = _camera_mats(ref_cam)
    up_ref_seed = _normalize(ref_c2w[:3, 1].astype(np.float32))
    right_ref_seed = _normalize(ref_c2w[:3, 0].astype(np.float32))
    up_ref = _aligned_axis_mean(ref_ups)
    right_ref = _aligned_axis_mean(ref_rights)
    if float(np.linalg.norm(up_ref)) < 1e-8:
        up_ref = up_ref_seed
    if float(np.linalg.norm(right_ref)) < 1e-8:
        right_ref = right_ref_seed
    if float(np.dot(up_ref, up_ref_seed)) < 0.0:
        up_ref = -up_ref
    if float(np.dot(right_ref, right_ref_seed)) < 0.0:
        right_ref = -right_ref
    azimuth_count = max(1, int(azimuth_count))
    pitches = sorted({float(np.clip(p, 5.0, 89.9)) for p in pitch_list})
    if not pitches:
        pitches = [30.0, 60.0]
    factors = sorted({float(max(0.0, f)) for f in dist_factors})
    if not factors:
        factors = [0.0, 0.5, 1.0]
    far_blend = float(np.clip(far_blend, 0.1, 1.0))
    far_cover_pad = max(1.0, float(far_cover_pad))
    far_cam_mult = max(1.0, float(far_cam_mult))

    near_d = max(0.05, near_scale * cam_d50)
    width = int(ref_cam.image_width)
    height = int(ref_cam.image_height)
    fovy = float(ref_cam.FoVy)
    fovx = float(ref_cam.FoVx)
    # Strict viewpoint locking: derive distances only from camera geometry.
    far_cover_raw = max(far_scale * cam_d50, cam_d90 * 1.05)
    far_cover_d = min(far_cover_raw * far_cover_pad, cam_d90 * far_cam_mult)
    far_cover_d = max(far_cover_d, near_d + 1e-3)
    far_anchor = max(near_d + far_blend * (far_cover_d - near_d), near_d + 1e-3)
    # If user requests factors beyond 1.0, switch to cover-span mapping so that
    # larger factors can genuinely extend farther (not stuck in compressed anchor span).
    use_cover_span = any(float(f) > 1.0 for f in factors)
    span = (far_cover_d - near_d) if use_cover_span else (far_anchor - near_d)
    dists = [near_d + f * span for f in factors]

    # Build horizontal basis from camera spread projected to the plane.
    centered = centers - render_center[None, :]
    spread = centered - np.dot(centered, up_guess)[:, None] * up_guess[None, :]
    right_ref_h = _project_to_plane(right_ref, up_guess)
    if float(np.linalg.norm(right_ref_h)) >= 1e-8:
        h1 = right_ref_h.copy()
    elif spread.shape[0] >= 3:
        _, _, vt_h = np.linalg.svd(spread, full_matrices=False)
        h1 = _normalize(vt_h[0].astype(np.float32))
    else:
        h1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(h1, up_guess))) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(tmp, up_guess))) > 0.95:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        h1 = _normalize(tmp - float(np.dot(tmp, up_guess)) * up_guess)
    if float(np.linalg.norm(right_ref_h)) >= 1e-8:
        h2_try = _normalize(np.cross(up_guess, h1))
        if float(np.dot(h2_try, right_ref_h)) < 0.0:
            h1 = -h1
    h2 = _normalize(np.cross(up_guess, h1))

    # Use observed camera azimuths to avoid generating mostly out-of-coverage novel views.
    planar_x = np.dot(spread, h1)
    planar_y = np.dot(spread, h2)
    az_all = (np.degrees(np.arctan2(planar_y, planar_x)) + 360.0) % 360.0
    az_all = np.sort(az_all)
    if az_all.size >= azimuth_count:
        pick = np.linspace(0, az_all.size - 1, num=azimuth_count)
        pick = np.round(pick).astype(np.int32)
        azimuths = [float(az_all[int(i)]) for i in pick]
    else:
        azimuths = [k * (360.0 / float(azimuth_count)) for k in range(azimuth_count)]
    azimuths = sorted(azimuths)

    znear = float(ref_cam.znear)
    zfar = float(ref_cam.zfar)
    zfar_novel = max(zfar, far_anchor * 2.5, cam_d90 * 2.0)
    proj = getProjectionMatrix(znear=znear, zfar=zfar_novel, fovX=fovx, fovY=fovy).transpose(0, 1).to(device)
    bg_color = torch.tensor([1, 1, 1] if bg == 1 else [0, 0, 0], dtype=torch.float32, device=device)

    rng = np.random.default_rng(int(seed))
    _prepare_png_dir(out_dir)
    if proxy_enabled and proxy_dir is not None:
        _prepare_png_dir(proxy_dir)
    frames: List[np.ndarray] = []
    sibr_lines: List[str] = []
    bg_sens_means: List[float] = []
    bg_sens_ratios: List[float] = []
    idx = 0
    roll_flip_count = 0
    for azi_i, azi in enumerate(azimuths):
        for pit_i, pitch in enumerate(pitches):
            for dist_i, dist in enumerate(dists):
                a = float(azi)
                p = float(pitch)
                d = float(dist)
                if jitter:
                    a += float(rng.uniform(-ang_jitter_deg, ang_jitter_deg))
                    p += float(rng.uniform(-ang_jitter_deg, ang_jitter_deg))
                    p = float(np.clip(p, 20.0, 89.9))
                    d *= (1.0 + float(rng.uniform(-pos_jitter, pos_jitter)))
                    d = max(0.05, d)
                ar = math.radians(a)
                pr = math.radians(p)
                horiz = (math.cos(ar) * h1) + (math.sin(ar) * h2)
                dir_center_to_cam = (math.cos(pr) * horiz) + (math.sin(pr) * up_guess)
                dir_center_to_cam = _normalize(dir_center_to_cam)
                cam_pos = render_center + d * dir_center_to_cam
                look_dir = _normalize(render_center - cam_pos)
                forward_scores = np.dot(ref_forwards_np, look_dir)
                best_cam_idx = int(np.argmax(forward_scores))
                right_local = ref_rights_np[best_cam_idx]
                up_local = ref_ups_np[best_cam_idx]
                right_hint = _project_to_plane(right_local, look_dir)
                if float(np.linalg.norm(right_hint)) < 1e-8:
                    right_hint = _project_to_plane(right_ref, look_dir)
                if float(np.linalg.norm(right_hint)) >= 1e-8:
                    c2w_new = _build_c2w_from_forward_right(cam_pos, look_dir, right_hint)
                else:
                    up_hint = _project_to_plane(up_local, look_dir)
                    if float(np.linalg.norm(up_hint)) < 1e-8:
                        up_hint = _project_to_plane(up_ref, look_dir)
                    if float(np.linalg.norm(up_hint)) < 1e-8:
                        up_hint = _project_to_plane(up_guess, look_dir)
                    c2w_new = _build_c2w_lookat(cam_pos, render_center, up_hint)
                # Keep image-plane orientation globally consistent across the grid.
                novel_right = _project_to_plane(c2w_new[:3, 0], look_dir)
                ref_right = _project_to_plane(right_local, look_dir)
                if float(np.linalg.norm(ref_right)) < 1e-8:
                    ref_right = _project_to_plane(right_ref, look_dir)
                if float(np.linalg.norm(ref_right)) >= 1e-8 and float(np.dot(novel_right, ref_right)) < 0.0:
                    c2w_new[:3, 0] *= -1.0
                    c2w_new[:3, 1] *= -1.0
                    roll_flip_count += 1
                w2c_new = np.linalg.inv(c2w_new)

                world_view = torch.tensor(w2c_new, dtype=torch.float32, device=device).transpose(0, 1)
                full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
                cam = MiniCam(width, height, fovy, fovx, znear, zfar_novel, world_view, full_proj)

                azi_tag = int(round((a % 360.0)))
                pit_tag = int(round(p))
                # Naming encodes deterministic inspection order:
                # direction index -> pitch index -> distance index.
                file_name = (
                    f"{idx:04d}_dir{azi_i:02d}_pit{pit_i:02d}_dst{dist_i:02d}"
                    f"_a{azi_tag:03d}_p{pit_tag:02d}_de{d:.2f}.png"
                )
                _render_save_pair(
                    cam, gaussians, pipe, bg_color, out_dir, file_name, frames,
                    proxy_enabled, proxy_dir, proxy_override_color, int(bg),
                    bg_sens_means=bg_sens_means, bg_sens_ratios=bg_sens_ratios,
                    sibr_lookat_lines=sibr_lines, sibr_cam_name=Path(file_name).stem,
                )
                idx += 1

    if include_topdown:
        # Top-down at one azimuth, but for all configured distances.
        top_dir = up_guess
        top_dir = _normalize(top_dir)
        if float(np.linalg.norm(top_dir)) < 1e-8:
            top_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        # Fix top-down orientation using a single deterministic azimuth reference.
        top_az = float(azimuths[0]) if len(azimuths) > 0 else 0.0
        top_ar = math.radians(top_az)
        top_forward = _normalize(-top_dir)
        top_right_hint = _project_to_plane(right_ref, top_forward)
        if float(np.linalg.norm(top_right_hint)) < 1e-8:
            top_right_hint = _normalize((math.cos(top_ar) * h1) + (math.sin(top_ar) * h2))
        for dist_i, top_dist in enumerate(dists):
            top_cam_pos = render_center + top_dist * top_dir
            c2w_new = _build_c2w_from_forward_right(top_cam_pos, top_forward, top_right_hint)
            w2c_new = np.linalg.inv(c2w_new)

            world_view = torch.tensor(w2c_new, dtype=torch.float32, device=device).transpose(0, 1)
            full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
            cam = MiniCam(width, height, fovy, fovx, znear, zfar_novel, world_view, full_proj)

            file_name = f"{idx:04d}_topdown_a{int(round(top_az))%360:03d}_dst{dist_i:02d}_de{top_dist:.2f}.png"
            _render_save_pair(
                cam, gaussians, pipe, bg_color, out_dir, file_name, frames,
                proxy_enabled, proxy_dir, proxy_override_color, int(bg),
                bg_sens_means=bg_sens_means, bg_sens_ratios=bg_sens_ratios,
                sibr_lookat_lines=sibr_lines, sibr_cam_name=Path(file_name).stem,
            )
            idx += 1

    meta = {
        "mode": "grid",
        "num_frames_target": int(len(azimuths) * len(pitches) * len(dists) + (len(dists) if include_topdown else 0)),
        "num_frames_selected": int(len(frames)),
        "scene_center": [float(center_scene[0]), float(center_scene[1]), float(center_scene[2])],
        "render_center": [float(render_center[0]), float(render_center[1]), float(render_center[2])],
        "far_cover_distance": float(far_cover_d),
        "near_distance": float(near_d),
        "far_distance": float((near_d + 1.0 * span)),
        "distances": [float(d) for d in dists],
        "distance_factors": [float(f) for f in factors],
        "up_guess": [float(up_guess[0]), float(up_guess[1]), float(up_guess[2])],
        "up_ref": [float(up_ref[0]), float(up_ref[1]), float(up_ref[2])],
        "camera_distance_p50": float(cam_d50),
        "camera_distance_p90": float(cam_d90),
        "zfar_ref": float(zfar),
        "zfar_novel": float(zfar_novel),
        "azimuth_strategy": "observed_from_test_cameras" if az_all.size >= azimuth_count else "uniform_360_fallback",
        "azimuths_deg": [float(a) for a in azimuths],
        "azimuth_count": int(azimuth_count),
        "pitches_deg": [float(p) for p in pitches],
        "include_topdown": bool(include_topdown),
        "topdown_azimuth_deg": float(top_az) if include_topdown and len(azimuths) > 0 else 0.0,
        "ordering": "direction_then_pitch_then_distance_then_topdown",
        "distance_policy": "strict_locked: near=near_scale*p50; cover_far=camera_only; if any factor>1 use cover span else anchor span; d=near+factor*span",
        "distance_span_mode": "cover" if use_cover_span else "anchor",
        "roll_flip_count": int(roll_flip_count),
        "near_scale": float(near_scale),
        "far_scale": float(far_scale),
        "far_blend": float(far_blend),
        "far_cover_pad": float(far_cover_pad),
        "far_cam_mult": float(far_cam_mult),
        "jitter": bool(jitter),
        "pos_jitter": float(pos_jitter),
        "ang_jitter_deg": float(ang_jitter_deg),
        "seed": int(seed),
        "bg_sensitivity_thr": float(8.0 / 255.0),
        "bg_sensitivity_means": bg_sens_means,
        "bg_sensitivity_ratios": bg_sens_ratios,
    }
    return frames, meta, sibr_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel-view metrics parameters")
    model = ModelParams(parser, sentinel=True)
    pipe = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--N", type=int, default=60, help="Number of novel views (default: 60; mode=test_offset and N<=0 uses all test cams)")
    parser.add_argument("--mode", type=str, default="grid72", choices=["orbit", "test_offset", "grid72"],
                        help="Novel-view generation mode (default: grid72)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for jittered modes")
    parser.add_argument("--bg", type=int, default=0, choices=[0, 1], help="Background color: 0=black, 1=white")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Render device for novel-view evaluation (default: cuda)")
    parser.add_argument("--edge_thr", type=float, default=0.1, help="Edge threshold for spike score (default: 0.1)")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory (default: <model>/novel_views)")
    # test-offset mode
    parser.add_argument("--test_shift_lat", type=float, default=0.15, help="test_offset lateral shift factor wrt local camera spacing")
    parser.add_argument("--test_shift_up", type=float, default=0.03, help="test_offset vertical shift factor wrt local camera spacing")
    parser.add_argument("--test_lookat_blend", type=float, default=0.35, help="test_offset look-at center blend in [0,1]")
    # grid72 mode (configurable grid)
    parser.add_argument("--grid_near_scale", type=float, default=0.6, help="grid72 near distance scale * camera_distance_p50")
    parser.add_argument("--grid_far_scale", type=float, default=1.0, help="grid72 far distance scale * camera_distance_p50")
    parser.add_argument("--grid_far_blend", type=float, default=0.4, help="Blend factor from near->cover_far for practical far distance (default: 0.4)")
    parser.add_argument("--grid_far_cover_pad", type=float, default=1.08, help="Padding on FOV cover distance (default: 1.08)")
    parser.add_argument("--grid_far_cam_mult", type=float, default=2.0, help="Upper cap multiplier over camera_distance_p90 for far cover (default: 2.0)")
    parser.add_argument("--grid_azimuth_count", type=int, default=8, help="Number of azimuth directions (default: 8)")
    parser.add_argument("--grid_pitch_list", type=str, default="15,30,60", help="Comma-separated pitch degrees, e.g. 15,30,60 (default: 15,30,60)")
    parser.add_argument("--grid_distance_factors", type=str, default="0.5,1,1.5", help="Comma-separated factors >=0 between near and far, e.g. 0.5,1,1.5")
    parser.add_argument("--grid_no_topdown", action="store_true", default=False, help="Disable final top-down frame")
    parser.add_argument("--grid_jitter", action="store_true", default=False, help="Enable small random jitter in grid72")
    parser.add_argument("--grid_pos_jitter", type=float, default=0.05, help="grid72 distance jitter ratio")
    parser.add_argument("--grid_ang_jitter_deg", type=float, default=2.0, help="grid72 angle jitter in degrees")
    parser.add_argument("--clean_use_local_flicker", type=_str2bool, nargs="?", const=True, default=True,
                        help="Use local flicker (same dir/pit) in SGF clean score when available (default: True)")
    parser.add_argument("--structure_noise_penalty_w", type=float, default=0.0,
                        help="Optional weight for non-air isotropic noise penalty in structure score (default: 0.0/off)")
    parser.add_argument("--dump_ellipsoid_proxy", action="store_true", default=False,
                        help="Dump per-view ellipsoid proxy images with exact same camera poses (default: off)")
    parser.add_argument("--ellipsoid_proxy_dir", type=str, default="",
                        help="Output directory for ellipsoid proxy images (default: <out_dir>_ellip)")
    parser.add_argument("--dump_sibr_ellipsoid", type=_str2bool, nargs="?", const=True, default=False,
                        help="Dump SIBR offline ellipsoid renders with the exact same camera path (default: off)")
    parser.add_argument("--sibr_exe", type=str, default="",
                        help="Path to SIBR_gaussianViewer_app(.exe); auto-detected when empty")
    parser.add_argument("--sibr_out_dir", type=str, default="",
                        help="Output dir for SIBR ellipsoid renders (default: <out_dir>_ellip_sibr)")
    parser.add_argument("--sibr_device", type=int, default=0,
                        help="CUDA device index for SIBR viewer (default: 0)")
    parser.add_argument("--sibr_gaussian_mode", type=str, default="ellipsoids",
                        choices=["ellipsoids", "splats", "points"],
                        help="SIBR gaussian render mode for offline dump (default: ellipsoids)")
    parser.add_argument("--sibr_keep_path_file", action="store_true", default=False,
                        help="Keep generated .lookat path file after SIBR render (default: off)")
    args = get_combined_args(parser)

    safe_state(False)
    eval_device = _resolve_torch_device(getattr(args, "device", "cuda"))

    dataset = model.extract(args)
    pipeline = pipe.extract(args)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if args.mode == "orbit":
            out_dir = Path(args.model_path) / "novel_views"
        elif args.mode == "test_offset":
            out_dir = Path(args.model_path) / "novel_views_test_offset"
        else:
            out_dir = Path(args.model_path) / "novel_views_grid"

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussian_count = int(gaussians.get_xyz.shape[0]) if gaussians.get_xyz is not None else 0
    scale_outlier_ratio = 0.0
    scale_outlier_ratio_s20 = 0.0
    opacity_low_cov_ratio = 0.0
    opacity_low_cov_ratio_o01 = 0.0
    with torch.no_grad():
        try:
            if gaussians.get_scaling is not None and gaussians.get_scaling.numel() > 0:
                smax = torch.max(gaussians.get_scaling, dim=1).values
                scale_outlier_ratio = float(torch.mean((smax > 10.0).float()).item())
                scale_outlier_ratio_s20 = float(torch.mean((smax > 20.0).float()).item())
        except Exception:
            pass
        try:
            if gaussians.get_opacity is not None and gaussians.get_opacity.numel() > 0:
                op = gaussians.get_opacity.view(-1)
                opacity_low_cov_ratio = float(torch.mean((op < 0.05).float()).item())
                opacity_low_cov_ratio_o01 = float(torch.mean((op < 0.10).float()).item())
        except Exception:
            pass
    proxy_enabled = bool(args.dump_ellipsoid_proxy)
    if args.ellipsoid_proxy_dir:
        proxy_out_dir = Path(args.ellipsoid_proxy_dir)
    else:
        proxy_out_dir = out_dir.parent / f"{out_dir.name}_ellip"
    proxy_override_color = _build_ellipsoid_proxy_override_color(gaussians) if proxy_enabled else None
    sibr_lines: List[str] = []

    render_t0 = time.perf_counter()
    with torch.no_grad():
        if args.mode == "orbit":
            frames, mode_meta, sibr_lines = _render_mode_orbit(
                scene, gaussians, pipeline, eval_device, args.bg, args.N, out_dir,
                proxy_enabled=proxy_enabled, proxy_dir=proxy_out_dir, proxy_override_color=proxy_override_color
            )
        elif args.mode == "test_offset":
            frames, mode_meta, sibr_lines = _render_mode_test_offset(
                scene, gaussians, pipeline, eval_device, args.bg, args.N, out_dir,
                shift_lat=float(args.test_shift_lat),
                shift_up=float(args.test_shift_up),
                lookat_blend=float(np.clip(args.test_lookat_blend, 0.0, 1.0)),
                seed=int(args.seed),
                proxy_enabled=proxy_enabled, proxy_dir=proxy_out_dir, proxy_override_color=proxy_override_color,
            )
        else:
            grid_pitches = _parse_float_list(args.grid_pitch_list, [15.0, 30.0, 60.0], clip_min=5.0, clip_max=89.9)
            grid_dist_factors = _parse_float_list(args.grid_distance_factors, [0.5, 1.0, 1.5], clip_min=0.0, clip_max=None)
            frames, mode_meta, sibr_lines = _render_mode_grid72(
                scene, gaussians, pipeline, eval_device, args.bg, out_dir,
                near_scale=float(args.grid_near_scale),
                far_scale=float(args.grid_far_scale),
                far_blend=float(args.grid_far_blend),
                far_cover_pad=float(args.grid_far_cover_pad),
                far_cam_mult=float(args.grid_far_cam_mult),
                azimuth_count=int(args.grid_azimuth_count),
                pitch_list=grid_pitches,
                dist_factors=grid_dist_factors,
                include_topdown=not bool(args.grid_no_topdown),
                jitter=bool(args.grid_jitter),
                pos_jitter=float(max(0.0, args.grid_pos_jitter)),
                ang_jitter_deg=float(max(0.0, args.grid_ang_jitter_deg)),
                seed=int(args.seed),
                proxy_enabled=proxy_enabled, proxy_dir=proxy_out_dir, proxy_override_color=proxy_override_color,
            )
    render_total_s = float(max(0.0, time.perf_counter() - render_t0))
    render_per_frame_s = float(render_total_s / max(1, len(frames)))

    frame_tags = _collect_frame_tags(out_dir, len(frames))
    frame_names = _collect_frame_names(out_dir, len(frames))
    sibr_enabled = bool(getattr(args, "dump_sibr_ellipsoid", False))
    sibr_status = "disabled"
    sibr_out_dir = Path(args.sibr_out_dir) if str(getattr(args, "sibr_out_dir", "")).strip() else (out_dir.parent / f"{out_dir.name}_ellip_sibr")
    sibr_frame_src = 0
    if sibr_enabled:
        if len(sibr_lines) == 0:
            sibr_status = "skipped_no_cameras"
        else:
            sibr_exe = _find_sibr_viewer_exe(str(getattr(args, "sibr_exe", "")))
            if sibr_exe is None:
                sibr_status = "skipped_no_sibr_exe"
                print("[WARN] dump_sibr_ellipsoid=True but SIBR_gaussianViewer_app not found; skipped.")
            else:
                src_path = Path(str(getattr(dataset, "source_path", getattr(args, "source_path", ""))))
                h = int(frames[0].shape[0]) if frames else 0
                w = int(frames[0].shape[1]) if frames else 0
                iter_loaded = int(scene.loaded_iter) if getattr(scene, "loaded_iter", None) is not None else None
                path_file = out_dir / "_novel_views_grid.lookat"
                cand_root = out_dir / "_sibr_cands"
                if cand_root.exists():
                    shutil.rmtree(cand_root, ignore_errors=True)
                cand_root.mkdir(parents=True, exist_ok=True)
                candidates = [
                    ("fwd_pos", list(sibr_lines)),
                    ("fwd_neg", _invert_lookat_forward(sibr_lines)),
                ]
                best: Optional[Dict[str, Any]] = None
                best_err = float("inf")
                best_msg = "no_valid_candidate"
                for cand_name, cand_lines in candidates:
                    cand_path = cand_root / f"_novel_views_grid_{cand_name}.lookat"
                    cand_out = cand_root / cand_name
                    cand_out.mkdir(parents=True, exist_ok=True)
                    sibr_lines_warmup = [cand_lines[0]] + list(cand_lines)
                    _write_lookat_file(cand_path, sibr_lines_warmup)
                    ok, msg, used_mode_arg, used_offscreen = _run_sibr_offline(
                        sibr_exe=sibr_exe,
                        model_path=Path(str(args.model_path)),
                        source_path=src_path,
                        path_file=cand_path,
                        out_dir=cand_out,
                        width=w,
                        height=h,
                        device_id=int(getattr(args, "sibr_device", 0)),
                        gaussian_mode=str(getattr(args, "sibr_gaussian_mode", "ellipsoids")),
                        iteration=iter_loaded,
                    )
                    if not ok:
                        best_msg = msg
                        continue
                    src_cnt, tgt_cnt = _rename_sibr_frames_to_match(cand_out, frame_names)
                    transforms = ["none", "vflip", "hflip", "vhflip"]
                    trans_scores = {t: _score_sibr_alignment(out_dir, cand_out, frame_names, t) for t in transforms}
                    trans_best = min(trans_scores, key=trans_scores.get)
                    err = float(trans_scores[trans_best])
                    if err < best_err:
                        best_err = err
                        best = {
                            "cand_name": cand_name,
                            "cand_out": cand_out,
                            "src_cnt": int(src_cnt),
                            "tgt_cnt": int(tgt_cnt),
                            "transform": str(trans_best),
                            "used_mode_arg": bool(used_mode_arg),
                            "used_offscreen": bool(used_offscreen),
                            "err": float(err),
                        }
                if best is None:
                    sibr_status = f"failed_{best_msg}"
                    print(f"[WARN] SIBR ellipsoid dump failed: {best_msg}")
                else:
                    if sibr_out_dir.exists():
                        shutil.rmtree(sibr_out_dir, ignore_errors=True)
                    shutil.copytree(best["cand_out"], sibr_out_dir)
                    _apply_sibr_transform_inplace(sibr_out_dir, best["transform"])
                    kept_cnt = len(list(sibr_out_dir.glob("*.png")))
                    sibr_frame_src = int(best["src_cnt"])
                    mode_tag = "mode_arg" if best["used_mode_arg"] else "mode_default"
                    offscreen_tag = "offscreen" if best["used_offscreen"] else "windowed"
                    ori_tag = f"{best['cand_name']}_{best['transform']}"
                    base = f"ok_{offscreen_tag}_{mode_tag}_{ori_tag}_e{best['err']:.4f}"
                    sibr_status = base if kept_cnt == int(best["tgt_cnt"]) else f"{base}_count_mismatch_src{best['src_cnt']}_kept{kept_cnt}_tgt{best['tgt_cnt']}"
                if not bool(getattr(args, "sibr_keep_path_file", False)):
                    try:
                        for p in cand_root.glob("*.lookat"):
                            p.unlink()
                        if path_file.exists():
                            path_file.unlink()
                    except OSError:
                        pass
    bg_leaks: List[float] = []
    spike_scores_all: List[float] = []
    spike_scores_air: List[float] = []
    spike_scores_air_thr: List[float] = []
    tex_tenengrad: List[float] = []
    tex_grad_p90: List[float] = []
    tex_edge_density: List[float] = []
    edge_anisotropy: List[float] = []
    air_mask_ratios: List[float] = []
    air_edge_means: List[float] = []
    air_hf_means: List[float] = []
    air_blob_ratios: List[float] = []
    air_fog_ratios: List[float] = []
    air_artifact_scores: List[float] = []
    nonair_noise_penalties: List[float] = []
    grays: List[np.ndarray] = []

    for img in frames:
        gray = _rgb_to_gray(img)
        grays.append(gray)
        edge_mag = _sobel_mag(gray)
        ten, gp90, edens = _texture_metrics(gray)
        ean = _edge_anisotropy(gray)
        air_ratio, air_edge, air_hf, air_blob, air_fog, air_score = _air_cleanliness_metrics(gray, edge_mag)
        air_mask = _air_mask_from_render(gray, edge_mag)
        bg_leaks.append(_bg_leak_ratio(img, args.bg))
        spike_scores_all.append(_spike_score(edge_mag, thr=args.edge_thr))
        if int(np.sum(air_mask)) > 0:
            air_thr = max(float(args.edge_thr) * 0.35, float(np.percentile(edge_mag[air_mask], 85.0)))
        else:
            air_thr = float(args.edge_thr)
        spike_scores_air_thr.append(float(air_thr))
        spike_scores_air.append(_spike_score(edge_mag, thr=air_thr, valid_mask=air_mask))
        tex_tenengrad.append(ten)
        tex_grad_p90.append(gp90)
        tex_edge_density.append(edens)
        edge_anisotropy.append(ean)
        air_mask_ratios.append(air_ratio)
        air_edge_means.append(air_edge)
        air_hf_means.append(air_hf)
        air_blob_ratios.append(air_blob)
        air_fog_ratios.append(air_fog)
        air_artifact_scores.append(air_score)
        nonair_noise_penalties.append(_nonair_noise_penalty(gray, air_mask))

    flickers_global: List[float] = []
    for i in range(1, len(grays)):
        flickers_global.append(float(np.mean(np.abs(grays[i] - grays[i - 1]))))
    flickers_local, flick_local_mean, flick_local_p90 = _local_flicker_stats(grays, frame_names)
    flick_global_mean = float(np.mean(flickers_global)) if flickers_global else 0.0
    flick_global_p90 = float(np.percentile(flickers_global, 90.0)) if flickers_global else 0.0

    bg_sens_means = mode_meta.pop("bg_sensitivity_means", [])
    bg_sens_ratios = mode_meta.pop("bg_sensitivity_ratios", [])
    bg_sens_thr = float(mode_meta.get("bg_sensitivity_thr", 8.0 / 255.0))

    results = {
        "mode": str(args.mode),
        "num_frames": int(len(frames)),
        "gaussian_count": int(gaussian_count),
        "ScaleOutlierRatio": float(scale_outlier_ratio),
        "ScaleOutlierRatio_s20": float(scale_outlier_ratio_s20),
        "OpacityLowCoverageRatio": float(opacity_low_cov_ratio),
        "OpacityLowCoverageRatio_o10": float(opacity_low_cov_ratio_o01),
        "RenderTimeTotal_s": float(render_total_s),
        "RenderTimePerFrame_s": float(render_per_frame_s),
        "scale_outlier_ratio": float(scale_outlier_ratio),
        "opacity_low_coverage_ratio": float(opacity_low_cov_ratio),
        "render_time_per_frame_s": float(render_per_frame_s),
        "bg": int(args.bg),
        "edge_thr": float(args.edge_thr),
        "BgLeakRatio_mean": float(np.mean(bg_leaks)) if bg_leaks else 0.0,
        "SpikeScore_mean": float(np.mean(spike_scores_all)) if spike_scores_all else 0.0,
        "SpikeScore_all_mean": float(np.mean(spike_scores_all)) if spike_scores_all else 0.0,
        "SpikeScore_air_mean": float(np.mean(spike_scores_air)) if spike_scores_air else 0.0,
        "SpikeScore_air_thr_mean": float(np.mean(spike_scores_air_thr)) if spike_scores_air_thr else float(args.edge_thr),
        "BgSensitivity_mean": float(np.mean(bg_sens_means)) if bg_sens_means else 0.0,
        "BgSensitivityRatio": float(np.mean(bg_sens_ratios)) if bg_sens_ratios else 0.0,
        "BgSensitivity_thr": float(bg_sens_thr),
        "TemporalFlicker_mean": float(flick_global_mean),  # legacy alias
        "TemporalFlicker_global_mean": float(flick_global_mean),
        "TemporalFlicker_global_p90": float(flick_global_p90),
        "TemporalFlicker_local_mean": float(flick_local_mean),
        "TemporalFlicker_local_p90": float(flick_local_p90),
        "TextureTenengrad_mean": float(np.mean(tex_tenengrad)) if tex_tenengrad else 0.0,
        "TextureGradP90_mean": float(np.mean(tex_grad_p90)) if tex_grad_p90 else 0.0,
        "TextureEdgeDensityAdaptive_mean": float(np.mean(tex_edge_density)) if tex_edge_density else 0.0,
        "EdgeAnisotropy_mean": float(np.mean(edge_anisotropy)) if edge_anisotropy else 0.0,
        "AirMaskRatio_mean": float(np.mean(air_mask_ratios)) if air_mask_ratios else 0.0,
        "AirEdgeMean_mean": float(np.mean(air_edge_means)) if air_edge_means else 0.0,
        "AirHFMean_mean": float(np.mean(air_hf_means)) if air_hf_means else 0.0,
        "AirBlobRatio_mean": float(np.mean(air_blob_ratios)) if air_blob_ratios else 0.0,
        "AirFogRatio_mean": float(np.mean(air_fog_ratios)) if air_fog_ratios else 0.0,
        "AirArtifactScore_mean": float(np.mean(air_artifact_scores)) if air_artifact_scores else 0.0,
        "NonAirNoisePenalty_mean": float(np.mean(nonair_noise_penalties)) if nonair_noise_penalties else 0.0,
        "out_dir": str(out_dir),
        "ellipsoid_proxy_enabled": bool(proxy_enabled),
        "ellipsoid_proxy_out_dir": str(proxy_out_dir) if proxy_enabled else "",
        "ellipsoid_proxy_opacity_mode": "opaque" if proxy_enabled else "off",
        "sibr_ellipsoid_enabled": bool(sibr_enabled),
        "sibr_ellipsoid_status": str(sibr_status),
        "sibr_ellipsoid_out_dir": str(sibr_out_dir) if sibr_enabled else "",
        "sibr_ellipsoid_source_frame_count": int(sibr_frame_src),
        "sibr_ellipsoid_mode": str(getattr(args, "sibr_gaussian_mode", "ellipsoids")) if sibr_enabled else "",
    }
    if bg_sens_means:
        results["BgSensitivity_p50"] = float(np.percentile(bg_sens_means, 50.0))
    if bg_sens_ratios:
        results["BgSensitivityRatio_p50"] = float(np.percentile(bg_sens_ratios, 50.0))
    if flickers_global:
        results["TemporalFlicker_global_p50"] = float(np.percentile(flickers_global, 50.0))
    if flickers_local:
        results["TemporalFlicker_local_p50"] = float(np.percentile(flickers_local, 50.0))
    results.update(mode_meta)
    if "distance_factors" in mode_meta and isinstance(mode_meta["distance_factors"], list):
        for i, v in enumerate(mode_meta["distance_factors"]):
            try:
                results[f"DistanceFactor_d{i:02d}"] = float(v)
            except (TypeError, ValueError):
                continue
    _write_bucket_means(results, "TextureTenengrad", tex_tenengrad, frame_tags)
    _write_bucket_means(results, "TextureGradP90", tex_grad_p90, frame_tags)
    _write_bucket_means(results, "TextureEdgeDensityAdaptive", tex_edge_density, frame_tags)
    _write_bucket_means(results, "EdgeAnisotropy", edge_anisotropy, frame_tags)
    _write_bucket_means(results, "BgLeakRatio", bg_leaks, frame_tags)
    _write_bucket_means(results, "SpikeScore", spike_scores_all, frame_tags)
    _write_bucket_means(results, "SpikeScore_all", spike_scores_all, frame_tags)
    _write_bucket_means(results, "SpikeScore_air", spike_scores_air, frame_tags)
    if len(bg_sens_means) == len(frame_tags):
        _write_bucket_means(results, "BgSensitivity", bg_sens_means, frame_tags)
    if len(bg_sens_ratios) == len(frame_tags):
        _write_bucket_means(results, "BgSensitivityRatio", bg_sens_ratios, frame_tags)
    _write_bucket_means(results, "AirEdgeMean", air_edge_means, frame_tags)
    _write_bucket_means(results, "AirHFMean", air_hf_means, frame_tags)
    _write_bucket_means(results, "AirBlobRatio", air_blob_ratios, frame_tags)
    _write_bucket_means(results, "AirFogRatio", air_fog_ratios, frame_tags)
    _write_bucket_means(results, "AirArtifactScore", air_artifact_scores, frame_tags)
    _write_bucket_means(results, "NonAirNoisePenalty", nonair_noise_penalties, frame_tags)

    _append_mean_p90(results, "SpikeScore_all", spike_scores_all)
    _append_mean_p90(results, "SpikeScore_air", spike_scores_air)
    _append_mean_p90(results, "AirArtifactScore", air_artifact_scores)
    _append_mean_p90(results, "BgSensitivity", bg_sens_means)
    _append_mean_p90(results, "BgSensitivityRatio", bg_sens_ratios)

    # SGF-oriented novel-view quality summary:
    # - StructureNearScore: emphasize near/mid clarity.
    # - CleanFarScore: emphasize far-field artifact suppression.
    # - NovelQualityScore: balanced summary (higher is better).
    tex_pairs = _bucket_pairs(results, "TextureTenengrad")
    ani_pairs = _bucket_pairs(results, "EdgeAnisotropy")
    air_pairs = _bucket_pairs(results, "AirArtifactScore")
    spike_pairs = _bucket_pairs(results, "SpikeScore_air")
    bg_pairs = _bucket_pairs(results, "BgSensitivityRatio")
    bg_leak_pairs = _bucket_pairs(results, "BgLeakRatio")
    noise_pairs = _bucket_pairs(results, "NonAirNoisePenalty")

    tex_near2 = _head_mean(tex_pairs, 2, float(np.mean(tex_tenengrad)) if tex_tenengrad else 0.0)
    ani_near2 = _head_mean(ani_pairs, 2, float(np.mean(edge_anisotropy)) if edge_anisotropy else 0.0)
    air_far2 = _tail_mean(air_pairs, 2, float(np.mean(air_artifact_scores)) if air_artifact_scores else 0.0)
    spike_far2 = _tail_mean(spike_pairs, 2, float(np.mean(spike_scores_air)) if spike_scores_air else 0.0)
    bg_far2 = _tail_mean(bg_pairs, 2, float(np.mean(bg_sens_ratios)) if bg_sens_ratios else 0.0)
    bg_leak_far2 = _tail_mean(bg_leak_pairs, 2, float(np.mean(bg_leaks)) if bg_leaks else 0.0)
    noise_near2 = _head_mean(noise_pairs, 2, float(np.mean(nonair_noise_penalties)) if nonair_noise_penalties else 0.0)
    flick_use_local = bool(getattr(args, "clean_use_local_flicker", True)) and len(flickers_local) > 0
    flick_mean = float(flick_local_mean if flick_use_local else flick_global_mean)

    # Score shaping to [0,1] with fixed transforms (cross-experiment comparable on same dataset).
    s_tex = 1.0 - math.exp(-0.04 * max(0.0, tex_near2))
    s_ani = float(np.clip((ani_near2 - 0.10) / 0.45, 0.0, 1.0))
    structure_near_score = float(np.clip(0.75 * s_tex + 0.25 * s_ani, 0.0, 1.0))
    noise_w = max(0.0, float(getattr(args, "structure_noise_penalty_w", 0.0)))
    if noise_w > 0.0:
        # Penalty is optional and off by default.
        noise_factor = math.exp(-15.0 * noise_w * max(0.0, noise_near2))
        structure_near_score = float(np.clip(structure_near_score * noise_factor, 0.0, 1.0))

    c_air = math.exp(-2.4 * max(0.0, air_far2))
    c_spike = math.exp(-120.0 * max(0.0, spike_far2))
    c_bg = math.exp(-10.0 * max(0.0, bg_far2))
    c_bg_leak = math.exp(-30.0 * max(0.0, bg_leak_far2))
    c_flick = math.exp(-3.0 * max(0.0, flick_mean))
    clean_far_score = float(np.clip(0.50 * c_air + 0.20 * c_spike + 0.15 * c_bg + 0.05 * c_bg_leak + 0.10 * c_flick, 0.0, 1.0))

    novel_quality_score = float(np.clip(0.60 * structure_near_score + 0.40 * clean_far_score, 0.0, 1.0))

    results["SGF_TextureNear2_mean"] = float(tex_near2)
    results["SGF_EdgeAnisotropyNear2_mean"] = float(ani_near2)
    results["SGF_AirArtifactFar2_mean"] = float(air_far2)
    results["SGF_SpikeFar2_mean"] = float(spike_far2)
    results["SGF_BgLeakFar2_mean"] = float(bg_far2)
    results["SGF_BgLeakLegacyFar2_mean"] = float(bg_leak_far2)
    results["SGF_NonAirNoiseNear2_mean"] = float(noise_near2)
    results["SGF_CleanFlicker_source"] = "local" if flick_use_local else "global"
    results["SGF_CleanFlicker_mean"] = float(flick_mean)
    results["SGF_StructureNearScore"] = structure_near_score
    results["SGF_CleanFarScore"] = clean_far_score
    results["SGF_NovelQualityScore"] = novel_quality_score

    if args.mode == "orbit":
        out_json = out_dir / "novel_view_metrics.json"
    elif args.mode == "grid72":
        out_json = out_dir / "novel_view_metrics_grid.json"
    else:
        out_json = out_dir / f"novel_view_metrics_{args.mode}.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
