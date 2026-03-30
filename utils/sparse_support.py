"""Sparse COLMAP support utilities.

This module is designed for the "Sparse-Support" family of constraints:
- Parse COLMAP sparse reconstruction outputs.
- Provide lightweight spatial queries over COLMAP points.

Notes
-----
* Dependency-light: requires numpy. Torch is optional.
* Safe to import even if COLMAP outputs are missing; functions raise
  FileNotFoundError with clear messages when called.
* API is kept stable to allow backward-compatible wiring into training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

    def _query_np(self, query_xyz, max_voxel_radius: int = 2, return_index: bool = True):
        """Backward-compatible alias for the old private method name."""
        return self.query(query_xyz, max_voxel_radius=max_voxel_radius, return_index=return_index)


# -----------------------------
# COLMAP I/O helpers
# -----------------------------


def _first_existing(paths: Sequence[Union[str, Path]]) -> Optional[Path]:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None


def _looks_like_colmap_model_dir(d: Path) -> bool:
    """Whether a directory contains COLMAP model files."""
    if not d.exists() or not d.is_dir():
        return False
    for stem in ("cameras", "images", "points3D"):
        if (d / f"{stem}.bin").exists() or (d / f"{stem}.txt").exists():
            return True
    return False


def resolve_colmap_model_dir(sparse_root: Union[str, Path]) -> Path:
    """Resolve a COLMAP *model directory* that contains cameras/images/points3D.

    Accepts either:
      - a COLMAP model directory directly (contains points3D.bin/txt), or
      - a COLMAP sparse root directory (contains subdirs like sparse/0), or
      - a dataset root directory that contains a 'sparse' directory.

    Returns
    -------
    Path
        Directory containing COLMAP model files.

    Raises
    ------
    FileNotFoundError
        If no COLMAP model dir is found.
    """
    root = Path(sparse_root)
    if not root.exists():
        raise FileNotFoundError(f"COLMAP path does not exist: {root}")

    # Case 1: already a model dir.
    if _looks_like_colmap_model_dir(root):
        return root

    # Case 2: common structure: <root>/sparse/0
    candidates: List[Path] = []
    # If root itself is 'sparse', try numeric subfolders.
    if root.name.lower() == "sparse":
        candidates.extend([p for p in root.iterdir() if p.is_dir()])
    # If root contains 'sparse', try there.
    if (root / "sparse").exists():
        candidates.extend([p for p in (root / "sparse").iterdir() if p.is_dir()])

    # Also accept <root>/colmap/sparse/0
    if (root / "colmap" / "sparse").exists():
        candidates.extend([p for p in (root / "colmap" / "sparse").iterdir() if p.is_dir()])

    # Sort to prefer '0' then smallest index.
    def _key(p: Path) -> Tuple[int, str]:
        try:
            return (int(p.name), p.name)
        except Exception:
            return (10**9, p.name)

    for c in sorted(candidates, key=_key):
        if _looks_like_colmap_model_dir(c):
            return c

    raise FileNotFoundError(
        "Could not locate a COLMAP model directory under: "
        f"{root}. Expected files like points3D.bin or points3D.txt."
    )


@dataclass
class SparsePointCloud:
    xyz: np.ndarray  # (N,3) float32
    error: Optional[np.ndarray] = None  # (N,) float32
    track_len: Optional[np.ndarray] = None  # (N,) int32

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float32)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N,3), got {self.xyz.shape}")
        if self.error is not None:
            self.error = np.asarray(self.error, dtype=np.float32).reshape(-1)
        if self.track_len is not None:
            self.track_len = np.asarray(self.track_len, dtype=np.int32).reshape(-1)


def load_colmap_points3D(model_dir: Union[str, Path]) -> SparsePointCloud:
    """Load COLMAP points3D from a model directory."""
    model_dir = Path(model_dir)

    from utils.read_write_model import (  # local import to keep module import-safe
        read_points3D_binary,
        read_points3D_text,
    )

    p_bin = model_dir / "points3D.bin"
    p_txt = model_dir / "points3D.txt"
    if p_bin.exists():
        pts = read_points3D_binary(str(p_bin))
    elif p_txt.exists():
        pts = read_points3D_text(str(p_txt))
    else:
        raise FileNotFoundError(f"points3D.bin/txt not found in: {model_dir}")

    if len(pts) == 0:
        return SparsePointCloud(xyz=np.zeros((0, 3), dtype=np.float32))

    xyz = np.stack([p.xyz for p in pts.values()], axis=0).astype(np.float32)
    err = np.array([p.error for p in pts.values()], dtype=np.float32)
    tlen = np.array([len(p.image_ids) for p in pts.values()], dtype=np.int32)
    return SparsePointCloud(xyz=xyz, error=err, track_len=tlen)


def load_colmap_camera_centers(model_dir: Union[str, Path]) -> np.ndarray:
    """Load camera centers (world) from COLMAP images.bin/txt.

    Returns
    -------
    np.ndarray
        (M,3) float32 camera centers.
    """
    model_dir = Path(model_dir)

    from utils.read_write_model import (  # local import
        qvec2rotmat,
        read_images_binary,
        read_images_text,
    )

    i_bin = model_dir / "images.bin"
    i_txt = model_dir / "images.txt"
    if i_bin.exists():
        imgs = read_images_binary(str(i_bin))
    elif i_txt.exists():
        imgs = read_images_text(str(i_txt))
    else:
        raise FileNotFoundError(f"images.bin/txt not found in: {model_dir}")

    centers: List[np.ndarray] = []
    for im in imgs.values():
        R = qvec2rotmat(im.qvec)
        t = np.asarray(im.tvec, dtype=np.float64).reshape(3)
        c = (-R.T @ t).astype(np.float32)
        centers.append(c)

    if not centers:
        return np.zeros((0, 3), dtype=np.float32)

    return np.stack(centers, axis=0).astype(np.float32)


def robust_aabb(xyz: np.ndarray, quantile: float = 0.005, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a robust AABB by trimming outliers by quantile.

    Parameters
    ----------
    xyz : (N,3)
    quantile : float
        Trim both tails per axis. 0.005 means keep [0.5%, 99.5%].
    margin : float
        Expand bounds by this amount (in world units).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.size == 0:
        mn = np.zeros((3,), dtype=np.float32)
        mx = np.zeros((3,), dtype=np.float32)
        return mn, mx

    q0 = float(np.clip(quantile, 0.0, 0.49))
    lo = np.quantile(xyz, q0, axis=0).astype(np.float32)
    hi = np.quantile(xyz, 1.0 - q0, axis=0).astype(np.float32)
    if margin != 0.0:
        m = float(margin)
        lo -= m
        hi += m
    return lo, hi


# -----------------------------
# Voxel-hash nearest-neighbor (small-batch)
# -----------------------------


def _voxel_index(xyz: np.ndarray, voxel: float) -> np.ndarray:
    return np.floor(xyz / float(voxel)).astype(np.int64)


class VoxelHashNN:
    """
    Lightweight sparse-support nearest-neighbor index on a voxel grid.

    - Build: voxelize support points and keep ONE representative per voxel (centroid).
    - Query: search neighbor voxels (default radius=1 => 27 neighbors), return nearest centroid distance.

    Notes
    -----
    * Query path is pure PyTorch (supports CUDA) and avoids CPU/Numpy round-trips.
    * Public interface is backward-compatible with the previous implementation:
        - __init__(points_xyz, voxel_size)
        - query(query_xyz, max_voxel_radius=2, return_index=True)
        - query_torch(query_xyz, max_voxel_radius=2) -> torch.Tensor distances
    """

    _PACK_BITS = 21  # 3*21 = 63 bits
    _PACK_OFF  = 1 << (_PACK_BITS - 1)  # 2^20

    def __init__(self, points_xyz, voxel_size: float):
        if voxel_size is None:
            raise ValueError("voxel_size must be set for VoxelHashNN")
        voxel_size = float(voxel_size)
        if not (voxel_size > 0.0):
            raise ValueError(f"voxel_size must be >0, got {voxel_size}")

        self.voxel_size = voxel_size
        self.device = None  # set in build/to()

        # Keep a lightweight CPU copy for debugging/backward expectations.
        self.points_xyz = None
        try:
            import numpy as _np
            if isinstance(points_xyz, _np.ndarray):
                self.points_xyz = points_xyz.astype(_np.float32, copy=False)
        except Exception:
            pass

        # Core tensors (may live on CPU or CUDA)
        self._keys_sorted = None          # (M,) int64
        self._coords_sorted = None        # (M,3) int64
        self._centroids_sorted = None     # (M,3) float32

        # Cache neighbor offsets per radius (CPU), moved on-demand
        self._offset_cache_cpu = {}   # r -> (K,3) int64 on CPU
        self._offset_cache_dev = {}   # (r, device_str) -> tensor on device

        self._build(points_xyz)

    # ------------------------ internal helpers ------------------------

    @staticmethod
    def _as_torch_xyz(x):
        import torch
        if isinstance(x, torch.Tensor):
            return x
        # numpy / list / tuple
        return torch.as_tensor(x)

    @classmethod
    def _pack_key(cls, coords_ijk):
        """
        Pack int64 voxel coords (..,3) into a single int64 key.
        Collision-free as long as each axis is within [-2^20, 2^20-1].
        """
        import torch
        c = coords_ijk.to(torch.int64)
        off = cls._PACK_OFF
        # We avoid raising in hot paths; out-of-range will still pack but may collide.
        x = c[..., 0] + off
        y = c[..., 1] + off
        z = c[..., 2] + off
        return (x << (2 * cls._PACK_BITS)) | (y << cls._PACK_BITS) | z

    def _get_offsets(self, r: int, device):
        import torch
        r = int(r)
        if r < 0:
            r = 0

        if r not in self._offset_cache_cpu:
            rng = torch.arange(-r, r + 1, dtype=torch.int64)
            # cartesian_prod returns (K,3)
            off = torch.cartesian_prod(rng, rng, rng).contiguous()
            self._offset_cache_cpu[r] = off

        dev_key = (r, str(device))
        if dev_key not in self._offset_cache_dev:
            self._offset_cache_dev[dev_key] = self._offset_cache_cpu[r].to(device=device, non_blocking=True)
        return self._offset_cache_dev[dev_key]

    def _build(self, points_xyz):
        """
        Build voxel centroids and sorted hash keys.
        """
        import torch

        pts = self._as_torch_xyz(points_xyz)
        if pts.numel() == 0:
            # Empty index
            self.device = pts.device
            self._keys_sorted = torch.empty((0,), dtype=torch.int64, device=self.device)
            self._coords_sorted = torch.empty((0, 3), dtype=torch.int64, device=self.device)
            self._centroids_sorted = torch.empty((0, 3), dtype=torch.float32, device=self.device)
            return

        if pts.ndim != 2 or pts.shape[-1] != 3:
            raise ValueError(f"VoxelHashNN expects support xyz shape (N,3), got {tuple(pts.shape)}")

        pts = pts.to(dtype=torch.float32)
        self.device = pts.device

        # voxelize
        coords = torch.floor(pts / float(self.voxel_size)).to(torch.int64)  # (N,3)

        # unique voxels + inverse map
        uniq_coords, inv = torch.unique(coords, dim=0, return_inverse=True)
        m = uniq_coords.shape[0]
        if m == 0:
            self._keys_sorted = torch.empty((0,), dtype=torch.int64, device=self.device)
            self._coords_sorted = torch.empty((0, 3), dtype=torch.int64, device=self.device)
            self._centroids_sorted = torch.empty((0, 3), dtype=torch.float32, device=self.device)
            return

        # centroid per voxel (in metric xyz, not ijk)
        sums = torch.zeros((m, 3), dtype=torch.float32, device=self.device)
        ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=self.device)
        counts = torch.zeros((m, 1), dtype=torch.float32, device=self.device)

        sums.scatter_add_(0, inv[:, None].expand(-1, 3), pts)
        counts.scatter_add_(0, inv[:, None], ones)
        centroids = sums / torch.clamp(counts, min=1.0)

        # sort by packed key for O(log M) lookup
        keys = self._pack_key(uniq_coords)
        sort_idx = torch.argsort(keys)
        self._keys_sorted = keys[sort_idx].contiguous()
        self._coords_sorted = uniq_coords[sort_idx].contiguous()
        self._centroids_sorted = centroids[sort_idx].contiguous()

    # ------------------------ public API ------------------------

    def to(self, device):
        """
        Move the index to a given device (cpu/cuda).
        """
        import torch
        device = torch.device(device)
        if self._keys_sorted is None:
            self.device = device
            return self

        self._keys_sorted = self._keys_sorted.to(device=device, non_blocking=True)
        self._coords_sorted = self._coords_sorted.to(device=device, non_blocking=True)
        self._centroids_sorted = self._centroids_sorted.to(device=device, non_blocking=True)
        self.device = device

        # offsets cache will be re-materialized on-demand
        self._offset_cache_dev = {}
        return self

    def query_torch(self, query_xyz, max_voxel_radius: int = 2):
        """
        Query nearest centroid distance for each query point.

        Returns
        -------
        d : torch.Tensor, shape (Q,), float32
            Euclidean distance to nearest voxel centroid in neighbor voxels.
            If no neighbor voxel exists, distance is +inf.
        """
        import torch
        q = self._as_torch_xyz(query_xyz)

        if q.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=q.device)

        if q.ndim != 2 or q.shape[-1] != 3:
            raise ValueError(f"query_xyz must have shape (Q,3), got {tuple(q.shape)}")

        if self._keys_sorted is None or self._keys_sorted.numel() == 0:
            return torch.full((q.shape[0],), float("inf"), dtype=torch.float32, device=q.device)

        # Ensure index on same device as query
        if self.device is None or self._keys_sorted.device != q.device:
            # non-blocking move when possible; caller should preferably call .to() once.
            self.to(q.device)

        q = q.to(dtype=torch.float32)
        q_vox = torch.floor(q / float(self.voxel_size)).to(torch.int64)

        r = int(max_voxel_radius)
        if r < 0:
            r = 0
        offsets = self._get_offsets(r, q.device)  # (K,3)
        k = offsets.shape[0]

        # neighbor coords and keys
        neigh_coords = q_vox[:, None, :] + offsets[None, :, :]          # (Q,K,3)
        neigh_keys = self._pack_key(neigh_coords).reshape(-1)           # (Q*K,)

        keys_sorted = self._keys_sorted
        m = keys_sorted.numel()

        # searchsorted
        pos = torch.searchsorted(keys_sorted, neigh_keys)               # (Q*K,)
        pos_clamped = torch.clamp(pos, 0, max(m - 1, 0))

        hit = (pos < m) & (keys_sorted[pos_clamped] == neigh_keys)
        # Extra safety for rare packing collisions: verify coords equality
        if hit.any():
            coords_ref = self._coords_sorted[pos_clamped]
            coords_q = neigh_coords.reshape(-1, 3)
            hit = hit & (coords_ref == coords_q).all(dim=-1)

        # gather centroids and compute distances
        cent = self._centroids_sorted[pos_clamped]                      # (Q*K,3)
        q_rep = q.repeat_interleave(k, dim=0)                            # (Q*K,3)
        dist2 = (cent - q_rep).pow(2).sum(dim=-1)                        # (Q*K,)
        dist2 = torch.where(hit, dist2, torch.full_like(dist2, float("inf")))

        d2_min, _ = dist2.view(q.shape[0], k).min(dim=1)
        return torch.sqrt(d2_min)

    def query(self, query_xyz, max_voxel_radius: int = 2, return_index: bool = True):
        """
        Numpy-friendly wrapper.

        Parameters
        ----------
        query_xyz : np.ndarray or torch.Tensor, shape (Q,3)
        max_voxel_radius : int
        return_index : bool

        Returns
        -------
        d : np.ndarray, shape (Q,), float32
        idx : np.ndarray, shape (Q,), int64   (only if return_index=True)
              Index refers to the *voxel-centroid* list (after packing+sorting).
              If no hit, idx is -1.
        """
        import torch
        q = self._as_torch_xyz(query_xyz)
        if q.ndim == 1 and q.numel() == 3:
            q = q.view(1, 3)
        if q.ndim != 2 or q.shape[-1] != 3:
            raise ValueError(f"query_xyz must have shape (Q,3), got {tuple(q.shape)}")

        if self._keys_sorted is None or self._keys_sorted.numel() == 0:
            d = torch.full((q.shape[0],), float("inf"), dtype=torch.float32, device=q.device)
            if return_index:
                idx = torch.full((q.shape[0],), -1, dtype=torch.int64, device=q.device)
                return d.to("cpu").numpy(), idx.to("cpu").numpy()
            return d.to("cpu").numpy()

        # Use the same vectorized path, but also return argmin indices.
        if self.device is None or self._keys_sorted.device != q.device:
            self.to(q.device)

        qf = q.to(dtype=torch.float32)
        q_vox = torch.floor(qf / float(self.voxel_size)).to(torch.int64)

        r = int(max_voxel_radius)
        if r < 0:
            r = 0
        offsets = self._get_offsets(r, q.device)
        k = offsets.shape[0]

        neigh_coords = q_vox[:, None, :] + offsets[None, :, :]
        neigh_keys = self._pack_key(neigh_coords).reshape(-1)

        keys_sorted = self._keys_sorted
        m = keys_sorted.numel()
        pos = torch.searchsorted(keys_sorted, neigh_keys)
        pos_clamped = torch.clamp(pos, 0, max(m - 1, 0))
        hit = (pos < m) & (keys_sorted[pos_clamped] == neigh_keys)
        if hit.any():
            coords_ref = self._coords_sorted[pos_clamped]
            coords_q = neigh_coords.reshape(-1, 3)
            hit = hit & (coords_ref == coords_q).all(dim=-1)

        cent = self._centroids_sorted[pos_clamped]
        q_rep = qf.repeat_interleave(k, dim=0)
        dist2 = (cent - q_rep).pow(2).sum(dim=-1)
        dist2 = torch.where(hit, dist2, torch.full_like(dist2, float("inf")))

        dist2_view = dist2.view(q.shape[0], k)
        d2_min, argmin = dist2_view.min(dim=1)
        d = torch.sqrt(d2_min)

        if not return_index:
            return d.to("cpu").numpy()

        # Map argmin to centroid index; if all inf => -1
        hit_view = hit.view(q.shape[0], k)
        any_hit = hit_view.any(dim=1)
        # centroid index in sorted arrays:
        idx = pos_clamped.view(q.shape[0], k).gather(1, argmin[:, None]).squeeze(1)
        idx = torch.where(any_hit, idx, torch.full_like(idx, -1, dtype=torch.int64))

        return d.to("cpu").numpy(), idx.to("cpu").numpy()


# -----------------------------
# Convenience entrypoints for the pipeline
# -----------------------------


def build_sparse_support(
    colmap_path: Union[str, Path],
    *,
    voxel_size: float = 1.0,
    max_voxel_radius: int = 2,
    max_points: Optional[int] = None,
    error_prune_quantile: Optional[float] = 0.99,
    tracklen_min: int = 3,
) -> Tuple[SparsePointCloud, VoxelHashNN, Tuple[np.ndarray, np.ndarray]]:
    """High-level helper: load sparse points, filter, build NN index, compute AABB.

    Returns
    -------
    spc : SparsePointCloud
    index : VoxelHashNN
    aabb : (lo, hi)
    """
    model_dir = resolve_colmap_model_dir(colmap_path)
    spc = load_colmap_points3D(model_dir)

    xyz = spc.xyz
    if xyz.size == 0:
        idx = VoxelHashNN(xyz, voxel_size=voxel_size)
        aabb = robust_aabb(xyz)
        return spc, idx, aabb

    keep = np.ones((xyz.shape[0],), dtype=bool)

    if spc.track_len is not None and tracklen_min > 1:
        keep &= (spc.track_len >= int(tracklen_min))

    if spc.error is not None and error_prune_quantile is not None:
        q = float(np.clip(error_prune_quantile, 0.0, 1.0))
        thr = float(np.quantile(spc.error, q))
        keep &= (spc.error <= thr)

    xyz_f = xyz[keep]
    err_f = spc.error[keep] if spc.error is not None else None
    tlen_f = spc.track_len[keep] if spc.track_len is not None else None

    if max_points is not None and xyz_f.shape[0] > int(max_points):
        # Prefer longer tracks; if track_len absent, random sample.
        if tlen_f is not None:
            order = np.argsort(-tlen_f)  # desc
            order = order[: int(max_points)]
        else:
            rng = np.random.default_rng(0)
            order = rng.choice(xyz_f.shape[0], size=int(max_points), replace=False)
        xyz_f = xyz_f[order]
        if err_f is not None:
            err_f = err_f[order]
        if tlen_f is not None:
            tlen_f = tlen_f[order]

    spc_f = SparsePointCloud(xyz=xyz_f, error=err_f, track_len=tlen_f)
    aabb = robust_aabb(xyz_f, quantile=0.005, margin=0.0)

    index = VoxelHashNN(xyz_f, voxel_size=float(voxel_size))

    # Store query defaults as attributes (purely convenience; not relied on).
    index.default_max_voxel_radius = int(max_voxel_radius)  # type: ignore[attr-defined]

    return spc_f, index, aabb


def format_aabb(aabb: Tuple[np.ndarray, np.ndarray]) -> str:
    lo, hi = aabb
    lo = np.asarray(lo).reshape(3)
    hi = np.asarray(hi).reshape(3)
    return f"lo={lo.tolist()} hi={hi.tolist()}"

