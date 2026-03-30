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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

# ============================
# Pseudo-color thermal support
# ============================

def _to_float01(x: torch.Tensor) -> torch.Tensor:
    """Convert input tensor to float32 in [0, 1] as robustly as possible."""
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                   torch.uint16 if hasattr(torch, "uint16") else torch.int32):
        x = x.to(torch.float32) / 255.0
        return x
    x = x.to(torch.float32)
    # Heuristic: if values look like 0..255 floats, scale down.
    with torch.no_grad():
        mx = float(x.max().item()) if x.numel() > 0 else 0.0
    if mx > 1.0 + 1e-3:
        x = x / 255.0
    return x


def pseudo_color_structure(x: torch.Tensor) -> torch.Tensor:
    """Extract a single-channel 'structure' map from a 3-channel pseudo-color thermal image.

    Input:
      - (3,H,W) or (B,3,H,W), uint8 or float.
    Output:
      - (1,H,W) or (B,1,H,W), float32 in [0,1] (approximately).
    """
    if x.dim() not in (3, 4):
        raise ValueError(f"pseudo_color_structure expects 3D/4D tensor, got shape={tuple(x.shape)}")
    if x.dim() == 3:
        c, h, w = x.shape
        if c != 3:
            raise ValueError(f"pseudo_color_structure expects C=3, got C={c}")
        xf = _to_float01(x).clamp(0.0, 1.0)
        v_max = xf.max(dim=0).values
        v_min = xf.min(dim=0).values
        s = 0.5 * (v_max + v_min)
        return s.unsqueeze(0)
    else:
        b, c, h, w = x.shape
        if c != 3:
            raise ValueError(f"pseudo_color_structure expects C=3, got C={c}")
        xf = _to_float01(x).clamp(0.0, 1.0)
        v_max = xf.max(dim=1).values
        v_min = xf.min(dim=1).values
        s = 0.5 * (v_max + v_min)
        return s.unsqueeze(1)


def _sobel_kernels(device: torch.device, dtype: torch.dtype):
    kx = torch.tensor([[1.0, 0.0, -1.0],
                       [2.0, 0.0, -2.0],
                       [1.0, 0.0, -1.0]], device=device, dtype=dtype) / 8.0
    ky = torch.tensor([[1.0, 2.0, 1.0],
                       [0.0, 0.0, 0.0],
                       [-1.0, -2.0, -1.0]], device=device, dtype=dtype) / 8.0
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def structure_grad_loss(pred: torch.Tensor,
                        gt: torch.Tensor,
                        mask: torch.Tensor = None,
                        normalize: bool = True,
                        eps: float = 1e-6) -> torch.Tensor:
    """Gradient-consistency loss on pseudo-color structure channel (Sobel magnitude).

    pred/gt: (3,H,W) or (B,3,H,W), uint8 or float.
    mask: optional (H,W), (1,H,W), (B,H,W) or (B,1,H,W). Non-zero keeps pixels.
    """
    sp = pseudo_color_structure(pred)
    sg = pseudo_color_structure(gt)

    # Ensure 4D for conv2d
    if sp.dim() == 3:
        sp4 = sp.unsqueeze(0)  # (1,1,H,W)
        sg4 = sg.unsqueeze(0)
    else:
        sp4 = sp  # (B,1,H,W)
        sg4 = sg

    kx, ky = _sobel_kernels(sp4.device, sp4.dtype)
    gx_p = F.conv2d(sp4, kx, padding=1)
    gy_p = F.conv2d(sp4, ky, padding=1)
    gx_g = F.conv2d(sg4, kx, padding=1)
    gy_g = F.conv2d(sg4, ky, padding=1)

    mag_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + eps)
    mag_g = torch.sqrt(gx_g * gx_g + gy_g * gy_g + eps)

    diff = torch.abs(mag_p - mag_g)

    if mask is not None:
        if not torch.is_tensor(mask):
            raise TypeError(f"mask must be torch.Tensor or None, got {type(mask)}")
        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif m.dim() == 3:
            # (1,H,W) or (B,H,W)
            if m.shape[0] == diff.shape[0] and m.shape[1] != 1:
                m = m.unsqueeze(1)  # (B,1,H,W)
            else:
                m = m.unsqueeze(0)  # (1,1,H,W) if (1,H,W)
        elif m.dim() == 4:
            # (B,1,H,W) or (B,C,H,W)
            if m.shape[1] != 1:
                m = m[:, :1, ...]
        else:
            raise ValueError(f"mask must be 2D/3D/4D, got shape={tuple(mask.shape)}")
        m = _to_float01(m).clamp(0.0, 1.0)
        # Broadcast if needed
        if m.shape[0] == 1 and diff.shape[0] > 1:
            m = m.expand(diff.shape[0], -1, -1, -1)
        diff = diff * m
        denom = m.mean().clamp_min(eps)
    else:
        denom = 1.0

    if normalize:
        scale = mag_g.mean().clamp_min(eps)
        diff = diff / scale

    return diff.mean() / denom
