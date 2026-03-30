#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, XXXX
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from typing import Optional, Tuple, Any
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # --- Sparse Support (optional; default disabled) ---
        self._ss_aabb_lo: Optional["torch.Tensor"] = None
        self._ss_aabb_hi: Optional["torch.Tensor"] = None
        self._ss_index: Any = None
        self._ss_nn_dist_thr: Optional[float] = None
        self._ss_enabled: bool = False
        self._ss_logged_config: bool = False
        # Optional SS refinement knobs (all default-off for backward compatibility)
        self._ss_adaptive_nn: bool = False
        self._ss_adaptive_alpha: float = 1.0
        self._ss_adaptive_beta: float = 0.0
        self._ss_adaptive_max_scale: float = 1.5
        self._ss_trim_tail_pct: float = 0.0
        self._ss_drop_small_islands: int = 0
        self._ss_island_radius: Optional[float] = None

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)



    # ============================================================
    # Sparse Support public API (optional; backward-compatible)
    # ============================================================
    def clear_sparse_support(self):
        """Disable sparse-support gating and clear cached support."""
        self._ss_aabb_lo = None
        self._ss_aabb_hi = None
        self._ss_index = None
        self._ss_nn_dist_thr = None
        self._ss_enabled = False

    def set_sparse_support(
        self,
        aabb=None,
        index=None,
        nn_dist_thr=None,
        adaptive_nn: bool = False,
        adaptive_alpha: float = 1.0,
        adaptive_beta: float = 0.0,
        adaptive_max_scale: float = 1.5,
        trim_tail_pct: float = 0.0,
        drop_small_islands: int = 0,
        island_radius: Optional[float] = None,
    ):
        """Enable optional sparse-support gating.

        Args:
            aabb: Optional tuple (lo, hi), each array-like shape (3,). If None, AABB gating is disabled.
            index: Optional VoxelHashNN-like object with .to(device) and .query_torch(xyz)->dist.
            nn_dist_thr: Optional float threshold for nearest-support distance gating.
            adaptive_nn: If True, relax NN threshold using local spacing proxy (default off).
            adaptive_alpha/beta/max_scale: adaptive threshold controls.
            trim_tail_pct: Optional tail-trim percent in [0,100), removes farthest kept points after NN.
            drop_small_islands: Optional minimum island size (points). <=0 disables.
            island_radius: Optional voxel radius used by island grouping. None => 2*voxel_size or 2.0.
        """
        # Clear if nothing provided
        if aabb is None and index is None and nn_dist_thr is None:
            self.clear_sparse_support()
            return
        # Default to disabled unless we successfully configure support.
        self._ss_enabled = False

        device = self.get_xyz.device if hasattr(self, 'get_xyz') else None

        # AABB
        if aabb is not None:
            if not (isinstance(aabb, (tuple, list)) and len(aabb) == 2):
                raise ValueError('set_sparse_support: aabb must be (lo, hi) or None')
            lo, hi = aabb
            import torch
            lo_t = torch.as_tensor(lo, dtype=torch.float32, device=device)
            hi_t = torch.as_tensor(hi, dtype=torch.float32, device=device)
            if lo_t.numel() != 3 or hi_t.numel() != 3:
                raise ValueError('set_sparse_support: aabb lo/hi must have 3 elements')
            self._ss_aabb_lo = lo_t.view(3)
            self._ss_aabb_hi = hi_t.view(3)
        else:
            self._ss_aabb_lo = None
            self._ss_aabb_hi = None

        # Index
        self._ss_index = index
        if self._ss_index is not None and hasattr(self._ss_index, 'to') and device is not None:
            # Move once; NEVER move in hot path
            self._ss_index = self._ss_index.to(device)

        # NN threshold
        self._ss_nn_dist_thr = float(nn_dist_thr) if nn_dist_thr is not None else None
        # Optional refinement knobs (all default-off)
        self._ss_adaptive_nn = bool(adaptive_nn)
        self._ss_adaptive_alpha = float(adaptive_alpha)
        self._ss_adaptive_beta = float(adaptive_beta)
        self._ss_adaptive_max_scale = max(float(adaptive_max_scale), 1.0)
        trim_tail_pct_f = float(trim_tail_pct)
        self._ss_trim_tail_pct = min(max(trim_tail_pct_f, 0.0), 99.999)
        self._ss_drop_small_islands = max(int(drop_small_islands), 0)
        self._ss_island_radius = None if island_radius is None else max(float(island_radius), 1e-6)
        # Mark enabled only when support configuration is valid.
        self._ss_enabled = self._ss_is_enabled()
        # One-time configuration log (only when support is being set).
        if not getattr(self, "_ss_logged_config", False):
            has_aabb = (self._ss_aabb_lo is not None and self._ss_aabb_hi is not None)
            has_index = (self._ss_index is not None)
            nn_thr = self._ss_nn_dist_thr
            aabb_msg = ""
            if has_aabb:
                try:
                    lo = self._ss_aabb_lo.detach().cpu().tolist()
                    hi = self._ss_aabb_hi.detach().cpu().tolist()
                    aabb_msg = f" aabb_lo={lo} aabb_hi={hi}"
                except Exception:
                    aabb_msg = ""
            extra_msg = (
                f" adaptive_nn={self._ss_adaptive_nn}"
                f" trim_tail_pct={self._ss_trim_tail_pct}"
                f" drop_small_islands={self._ss_drop_small_islands}"
            )
            print(f"[INFO] SparseSupport configured: enabled={self._ss_enabled} has_aabb={has_aabb} has_index={has_index} nn_thr={nn_thr}{aabb_msg}{extra_msg}")
            self._ss_logged_config = True

    def _ss_is_enabled(self) -> bool:
        return (self._ss_aabb_lo is not None and self._ss_aabb_hi is not None) or (self._ss_index is not None and self._ss_nn_dist_thr is not None)

    def _ss_query_nn_d1_d2(self, xyz_query):
        """Query nearest/second-nearest support-centroid distances for adaptive NN.

        Returns (d1, d2) in world units. If d2 is unavailable, values are +inf.
        Falls back to query_torch-only when internal index fields are unavailable.
        """
        import torch
        idx = self._ss_index
        if idx is None:
            inf = torch.full((xyz_query.shape[0],), float("inf"), dtype=torch.float32, device=xyz_query.device)
            return inf, inf

        # Fallback: only nearest distance
        if not (hasattr(idx, "_keys_sorted") and hasattr(idx, "_coords_sorted") and hasattr(idx, "_centroids_sorted")
                and hasattr(idx, "_pack_key") and hasattr(idx, "_get_offsets") and hasattr(idx, "to")):
            d1 = idx.query_torch(xyz_query)
            d2 = torch.full_like(d1, float("inf"))
            return d1, d2

        # Mirror VoxelHashNN query_torch internals and keep top-2 distances.
        q = idx._as_torch_xyz(xyz_query)
        if q.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.float32, device=q.device),
                torch.empty((0,), dtype=torch.float32, device=q.device),
            )
        if idx.device is None or idx._keys_sorted.device != q.device:
            idx.to(q.device)
        q = q.to(dtype=torch.float32)
        q_vox = torch.floor(q / float(idx.voxel_size)).to(torch.int64)
        r = int(getattr(idx, "default_max_voxel_radius", 2))
        offsets = idx._get_offsets(r, q.device)
        k = offsets.shape[0]
        neigh_coords = q_vox[:, None, :] + offsets[None, :, :]
        neigh_keys = idx._pack_key(neigh_coords).reshape(-1)
        keys_sorted = idx._keys_sorted
        m = keys_sorted.numel()
        pos = torch.searchsorted(keys_sorted, neigh_keys)
        pos_clamped = torch.clamp(pos, 0, max(m - 1, 0))
        hit = (pos < m) & (keys_sorted[pos_clamped] == neigh_keys)
        if hit.any():
            coords_ref = idx._coords_sorted[pos_clamped]
            coords_q = neigh_coords.reshape(-1, 3)
            hit = hit & (coords_ref == coords_q).all(dim=-1)
        cent = idx._centroids_sorted[pos_clamped]
        q_rep = q.repeat_interleave(k, dim=0)
        dist2 = (cent - q_rep).pow(2).sum(dim=-1)
        inf2 = torch.full_like(dist2, float("inf"))
        dist2 = torch.where(hit, dist2, inf2).view(q.shape[0], k)
        d2_smallest, _ = torch.topk(dist2, k=min(2, k), dim=1, largest=False)
        d1 = torch.sqrt(d2_smallest[:, 0])
        if d2_smallest.shape[1] > 1:
            d2 = torch.sqrt(d2_smallest[:, 1])
        else:
            d2 = torch.full_like(d1, float("inf"))
        return d1, d2

    def _ss_filter_small_islands(self, xyz_keep):
        """Return boolean keep mask for xyz_keep by dropping tiny voxel islands."""
        import torch
        if self._ss_drop_small_islands <= 0 or xyz_keep is None or xyz_keep.numel() == 0:
            return torch.ones((0 if xyz_keep is None else xyz_keep.shape[0],), dtype=torch.bool, device=xyz_keep.device if xyz_keep is not None else "cpu")

        voxel = self._ss_island_radius
        if voxel is None:
            voxel = 2.0
            if self._ss_index is not None and hasattr(self._ss_index, "voxel_size"):
                try:
                    voxel = max(2.0 * float(self._ss_index.voxel_size), 1e-6)
                except Exception:
                    voxel = 2.0
        else:
            voxel = max(float(voxel), 1e-6)

        # CPU path: one-shot prune stage, robust and dependency-free.
        arr = torch.floor(xyz_keep.detach().cpu() / float(voxel)).to(torch.int64).numpy()
        if arr.shape[0] == 0:
            return torch.zeros((0,), dtype=torch.bool, device=xyz_keep.device)
        uniq, inv, counts = np.unique(arr, axis=0, return_inverse=True, return_counts=True)
        m = int(uniq.shape[0])
        if m == 0:
            return torch.zeros((arr.shape[0],), dtype=torch.bool, device=xyz_keep.device)
        coord2id = {tuple(uniq[i].tolist()): i for i in range(m)}
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((dx, dy, dz))

        comp = np.full((m,), -1, dtype=np.int32)
        comp_sizes = []
        cid = 0
        for i in range(m):
            if comp[i] >= 0:
                continue
            stack = [i]
            comp[i] = cid
            size_pts = 0
            while stack:
                u = stack.pop()
                size_pts += int(counts[u])
                ux, uy, uz = uniq[u]
                for dx, dy, dz in neighbors:
                    vid = coord2id.get((int(ux + dx), int(uy + dy), int(uz + dz)))
                    if vid is None or comp[vid] >= 0:
                        continue
                    comp[vid] = cid
                    stack.append(vid)
            comp_sizes.append(size_pts)
            cid += 1

        comp_sizes = np.asarray(comp_sizes, dtype=np.int64)
        keep_comp = comp_sizes >= int(self._ss_drop_small_islands)
        keep_np = keep_comp[comp[inv]]
        return torch.from_numpy(keep_np).to(device=xyz_keep.device, dtype=torch.bool)

    def _ss_gate_selected_mask(self, selected_mask, xyz_all):
        """Filter a boolean selected_mask using AABB and optional NN distance.

        This is called only in densify, NOT in the per-iteration hot path.
        """
        import torch
        if selected_mask is None:
            return selected_mask
        if not self._ss_is_enabled():
            return selected_mask
        if xyz_all is None or xyz_all.numel() == 0:
            return selected_mask
        if xyz_all.dim() != 2 or xyz_all.size(-1) != 3:
            raise ValueError('SparseSupport gating expects xyz (N,3)')
        if selected_mask.numel() != xyz_all.size(0):
            raise ValueError('SparseSupport gating: mask length must match xyz')

        idx = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            self._ss_last_gate_stats = {
                "before": 0,
                "after_aabb": 0,
                "after_nn": 0,
                "rejected_aabb": 0,
                "rejected_nn": 0,
                "nn_ms": None,
            }
            return selected_mask

        xyz_sel = xyz_all[idx]
        keep = torch.ones((xyz_sel.size(0),), dtype=torch.bool, device=xyz_sel.device)
        before = int(idx.numel())

        # AABB gating
        if self._ss_aabb_lo is not None and self._ss_aabb_hi is not None:
            lo = self._ss_aabb_lo
            hi = self._ss_aabb_hi
            if lo.device != xyz_sel.device:
                lo = lo.to(xyz_sel.device)
                hi = hi.to(xyz_sel.device)
            keep = keep & (xyz_sel >= lo).all(dim=-1) & (xyz_sel <= hi).all(dim=-1)
        after_aabb = int(keep.sum().item())

        # Optional NN distance gating
        nn_ms = None
        after_tail = after_aabb
        rejected_tail = 0
        after_island = after_aabb
        rejected_island = 0
        if self._ss_index is not None and self._ss_nn_dist_thr is not None:
            if keep.any():
                xyz_keep = xyz_sel[keep]
                # query_torch must be device-native; forbid cpu/numpy conversions inside index
                import time
                t0 = time.perf_counter()
                if self._ss_adaptive_nn:
                    dist, dist2 = self._ss_query_nn_d1_d2(xyz_keep)
                else:
                    dist = self._ss_index.query_torch(xyz_keep)
                    dist2 = None
                t1 = time.perf_counter()
                nn_ms = (t1 - t0) * 1000.0
                base_thr = float(self._ss_nn_dist_thr)
                if self._ss_adaptive_nn and dist2 is not None:
                    # Local spacing proxy: second-nearest support centroid distance.
                    # Keep backward compatibility by applying only when adaptive_nn is enabled.
                    thr_local = self._ss_adaptive_alpha * dist2 + self._ss_adaptive_beta
                    thr_min = torch.full_like(dist, base_thr)
                    thr_max = torch.full_like(dist, base_thr * self._ss_adaptive_max_scale)
                    thr_eff = torch.where(torch.isfinite(thr_local), torch.clamp(thr_local, min=thr_min, max=thr_max), thr_min)
                    keep2 = dist <= thr_eff
                else:
                    keep2 = dist <= base_thr

                # Optional tail trim: drop farthest survivors by distance percentile.
                if self._ss_trim_tail_pct > 0.0 and keep2.any():
                    kept_dist = dist[keep2]
                    q = torch.quantile(kept_dist, 1.0 - (self._ss_trim_tail_pct / 100.0))
                    keep2 = keep2 & (dist <= q)
                after_tail = int(keep2.sum().item())
                rejected_tail = after_aabb - after_tail

                # Optional small-island pruning on kept points.
                if self._ss_drop_small_islands > 0 and keep2.any():
                    island_keep = self._ss_filter_small_islands(xyz_keep[keep2])
                    tmp2 = keep2.clone()
                    kidx = torch.nonzero(keep2, as_tuple=False).squeeze(-1)
                    tmp2[kidx] = island_keep
                    keep2 = tmp2
                after_island = int(keep2.sum().item())
                rejected_island = after_tail - after_island

                # scatter back
                tmp = keep.clone()
                tmp_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
                tmp[tmp_idx] = keep2
                keep = tmp
            else:
                # nothing passes AABB
                pass
        after_nn = int(keep.sum().item())
        self._ss_last_gate_stats = {
            "before": before,
            "after_aabb": after_aabb,
            "after_nn": after_nn,
            "rejected_aabb": before - after_aabb,
            "rejected_nn": after_aabb - after_nn,
            "after_tail": after_tail,
            "rejected_tail": rejected_tail,
            "after_island": after_island,
            "rejected_island": rejected_island,
            "nn_ms": nn_ms,
        }

        # write back
        out = selected_mask.clone()
        out[idx] = keep
        return out

    def prune_outside_sparse_support(self):
        """Prune gaussians outside configured sparse support."""
        if not getattr(self, '_ss_enabled', False) or not self._ss_is_enabled():
            return None
        import torch
        xyz = self.get_xyz
        if xyz is None or xyz.numel() == 0:
            return (0, 0)
        selected_mask = torch.ones((xyz.size(0),), dtype=torch.bool, device=xyz.device)
        keep_mask = self._ss_gate_selected_mask(selected_mask, xyz)
        if keep_mask is None:
            return None
        before = int(keep_mask.numel())
        after = int(keep_mask.sum().item())
        if after >= before:
            return (before, after)
        prune_mask = ~keep_mask
        reset_tmp_radii = False
        if not hasattr(self, "tmp_radii") or self.tmp_radii is None:
            self.tmp_radii = torch.zeros((xyz.size(0),), device=xyz.device)
            reset_tmp_radii = True
        self.prune_points(prune_mask)
        if reset_tmp_radii:
            self.tmp_radii = None
        return (before, after)

    def clamp_scaling_max_(self, max_scale: float):
        """Clamp log-space scaling to a maximum scale value.

        Returns (clamped_gauss, total, before_smax, after_smax).
        """
        import math
        import torch
        try:
            max_scale_f = float(max_scale)
        except Exception:
            return (0, 0, 0.0, 0.0)
        if max_scale_f <= 0.0:
            return (0, 0, 0.0, 0.0)
        if self._scaling is None or self._scaling.numel() == 0:
            return (0, 0, 0.0, 0.0)
        thr_log = math.log(max_scale_f)
        with torch.no_grad():
            smax_before = torch.max(self.get_scaling, dim=1).values
            before_smax = float(smax_before.max().item()) if smax_before.numel() > 0 else 0.0
            clamped_gauss = int((self._scaling.max(dim=1).values > thr_log).sum().item())
            total = int(self._scaling.shape[0])
            self._scaling.data.clamp_(max=thr_log)
            smax_after = torch.max(self.get_scaling, dim=1).values
            after_smax = float(smax_after.max().item()) if smax_after.numel() > 0 else 0.0
        return (clamped_gauss, total, before_smax, after_smax)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # SparseSupport gating (optional; default off)
        if getattr(self, '_ss_enabled', False):
            selected_pts_mask = self._ss_gate_selected_mask(selected_pts_mask, self.get_xyz)
            stats = getattr(self, "_ss_last_gate_stats", None)
            if stats is not None:
                rejected_aabb = int(stats.get("rejected_aabb", 0) or 0)
                rejected_nn = int(stats.get("rejected_nn", 0) or 0)
                if (rejected_aabb > 0 or rejected_nn > 0) and (not getattr(self, "_ss_logged_split", False)):
                    msg = (
                        f"[INFO] SparseSupport densify_split gating: "
                        f"before={stats.get('before')} after_aabb={stats.get('after_aabb')} after_nn={stats.get('after_nn')} "
                        f"rejected_aabb={rejected_aabb} rejected_nn={rejected_nn}"
                    )
                    nn_ms = stats.get("nn_ms", None)
                    if nn_ms is not None:
                        msg += f" nn_ms={nn_ms:.2f}"
                    print(msg)
                    self._ss_logged_split = True

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        
        # SparseSupport gating (optional; default off)
        if getattr(self, '_ss_enabled', False):
            selected_pts_mask = self._ss_gate_selected_mask(selected_pts_mask, self.get_xyz)
            stats = getattr(self, "_ss_last_gate_stats", None)
            if stats is not None:
                rejected_aabb = int(stats.get("rejected_aabb", 0) or 0)
                rejected_nn = int(stats.get("rejected_nn", 0) or 0)
                if (rejected_aabb > 0 or rejected_nn > 0) and (not getattr(self, "_ss_logged_clone", False)):
                    msg = (
                        f"[INFO] SparseSupport densify_clone gating: "
                        f"before={stats.get('before')} after_aabb={stats.get('after_aabb')} after_nn={stats.get('after_nn')} "
                        f"rejected_aabb={rejected_aabb} rejected_nn={rejected_nn}"
                    )
                    nn_ms = stats.get("nn_ms", None)
                    if nn_ms is not None:
                        msg += f" nn_ms={nn_ms:.2f}"
                    print(msg)
                    self._ss_logged_clone = True
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
