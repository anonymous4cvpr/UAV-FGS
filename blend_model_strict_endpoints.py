# blend_model_strict_endpoints_v4.py
# ------------------------------------------------------------
# Blend two original 3DGS point_cloud.ply models into multiple alpha folders.
#
# v4 goals (requested):
#   1) One command can run multiple blend strategies (methods).
#   2) Only specify ONE output root; script creates subfolders per method.
#   3) Methods setting is simple/intuitive (no "name=...;preset=..." strings).
#   4) Do NOT change the blending core logic (lerp + optional dc_ycc fusion).
#
# Output layout:
#   <out_root>/<method>/<alpha>/point_cloud/iteration_<out_iter>/point_cloud.ply
#   <out_root>/<method>/<alpha>/cfg_args
#
# Example (compare recommended 5 strategies):
#   python blend_model_strict_endpoints_v4.py ^
#     --rgb_model_dir <RGB_MODEL_DIR> --rgb_iter 30000 ^
#     --t_model_dir   <THERMAL_MODEL_DIR> --t_iter 30000 ^
#     --alphas "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1" ^
#     --out_root <OUT_ROOT> --out_iter 30000 ^
#     --methods compare5 --clean_out --verify_endpoints
#
# ------------------------------------------------------------

import argparse
import re
import shutil
from pathlib import Path

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    raise SystemExit("Missing dependency: plyfile. Install with: pip install plyfile")


# -----------------------------
# Basic helpers
# -----------------------------
def norm_path(p: str) -> str:
    # avoid Windows escape/backslash surprises inside cfg_args
    return str(p).replace("\\", "/")


def parse_alphas(s: str):
    # Accept "0,0.25,1" or "0 0.25 1"
    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip() != ""]
    alphas = [float(p) for p in parts]
    for a in alphas:
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {a}")
    return alphas


def alpha_to_subdir(a: float) -> str:
    if a == 0.0:
        return "0"
    if a == 1.0:
        return "1"
    return str(a).rstrip("0").rstrip(".")


def load_vertex(p: Path):
    ply = PlyData.read(str(p))
    if "vertex" not in ply:
        raise ValueError(f"No vertex in {p}")
    return ply, ply["vertex"].data


def list_fields(names):
    f_dc = sorted([n for n in names if n.startswith("f_dc_")], key=lambda x: int(x.split("_")[-1]))
    f_rest = sorted([n for n in names if n.startswith("f_rest_")], key=lambda x: int(x.split("_")[-1]))
    sh = f_dc + f_rest
    return f_dc, f_rest, sh


def read_cfg_args(model_dir: Path) -> str:
    p = model_dir / "cfg_args"
    if not p.exists():
        raise FileNotFoundError(f"cfg_args not found: {p}")
    return p.read_text(encoding="utf-8", errors="ignore")


def rewrite_model_path(cfg_text: str, new_model_path: str):
    # only rewrite model_path; keep source_path/images untouched
    new_model_path = norm_path(new_model_path)

    def repl(m):
        q = m.group(2)
        return f"{m.group(1)}{q}{new_model_path}{q}"

    out, n = re.subn(r"(model_path\s*=\s*)(['\"])[^'\"]+(['\"])", repl, cfg_text, count=1)
    if n == 0:
        # fallback: append into Namespace(...)
        t = out.strip()
        if t.startswith("Namespace(") and t.endswith(")"):
            out = t[:-1] + f", model_path='{new_model_path}')"
        else:
            raise RuntimeError("cfg_args not Namespace(.) and model_path not found.")
    return out


def max_abs_diff(v_a, v_b, fields):
    m = 0.0
    for f in fields:
        a = np.asarray(v_a[f], dtype=np.float64)
        b = np.asarray(v_b[f], dtype=np.float64)
        m = max(m, float(np.max(np.abs(a - b))))
    return m


def safe_lerp(a, b, w, dtype):
    # Only blend float fields. For non-float, keep a.
    if np.issubdtype(dtype, np.floating):
        return ((1.0 - w) * a + w * b).astype(dtype, copy=False)
    return a.astype(dtype, copy=False)


# -----------------------------
# YCbCr helpers (for DC-only fusion)
# -----------------------------
def rgb_to_ycbcr(rgb):
    # rgb: (...,3)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = (B - Y) / 1.772
    Cr = (R - Y) / 1.402
    return Y, Cb, Cr


def ycbcr_to_rgb(Y, Cb, Cr):
    R = Y + 1.402 * Cr
    B = Y + 1.772 * Cb
    G = (Y - 0.299 * R - 0.114 * B) / 0.587
    return np.stack([R, G, B], axis=-1)


def robust_clip_like(out_rgb, ref_rgb, clip_percentile: float):
    """Clip out_rgb per-channel to the [p, 100-p] percentile range of ref_rgb."""
    p = float(clip_percentile)
    if p <= 0.0:
        return out_rgb

    o = out_rgb.reshape(-1, 3)
    r = ref_rgb.reshape(-1, 3)
    for c in range(3):
        lo, hi = np.percentile(r[:, c], [p, 100.0 - p])
        o[:, c] = np.clip(o[:, c], lo, hi)
    return o.reshape(out_rgb.shape)


def dc_ycc_blend(dc_rgb, dc_t, alpha, chroma_gain=1.0, dc_y_from="rgb", clip_percentile=0.5):
    """
    dc_rgb/dc_t: (N,3) in SH-DC coefficient space.

    - Y comes from {rgb, t, lerp} controlled by dc_y_from
    - Cb/Cr are blended: (1-a)*CbCr_rgb + a*(gain*CbCr_t)
    - Optionally robust-clip result to match dc_rgb channel range (percentiles).
    """
    a = float(alpha)
    gain = float(chroma_gain)

    Y_r, Cb_r, Cr_r = rgb_to_ycbcr(dc_rgb)
    Y_t, Cb_t, Cr_t = rgb_to_ycbcr(dc_t)

    if dc_y_from == "rgb":
        Y = Y_r
    elif dc_y_from == "t":
        Y = Y_t
    elif dc_y_from == "lerp":
        Y = (1.0 - a) * Y_r + a * Y_t
    else:
        raise ValueError(f"Unknown dc_y_from: {dc_y_from}")

    Cb = (1.0 - a) * Cb_r + a * (gain * Cb_t)
    Cr = (1.0 - a) * Cr_r + a * (gain * Cr_t)

    out = ycbcr_to_rgb(Y, Cb, Cr)
    out = robust_clip_like(out, dc_rgb, clip_percentile)
    return out


# -----------------------------
# Method definitions (simple + readable)
# -----------------------------
# Presets to ops (kept from v3 semantics)
PRESET_TO_OPS = {
    "sh_only": ["sh"],
    "sh_opacity": ["sh", "opacity"],
    "sh_opacity_scale": ["sh", "opacity", "scale"],
    "all_float": ["all_float"],
}


# Recommended comparison set
COMPARE5 = ["sh_only", "sh_opacity", "sh_opacity_scale", "sh_opacity_geom", "all_float"]


METHODS = {
    # Basic presets
    "sh_only": {"preset": "sh_only", "ops": []},
    "sh_opacity": {"preset": "sh_opacity", "ops": []},
    "sh_opacity_scale": {"preset": "sh_opacity_scale", "ops": []},
    "all_float": {"preset": "all_float", "ops": []},

    # Geometry-inclusive: appearance + xyz + scale + rotation
    "sh_opacity_geom": {"preset": "sh_opacity", "ops": ["xyz", "scale", "rot"]},

    # DC-only YCbCr fusion (optional but handy)
    "dc_ycc_only": {"preset": None, "ops": ["dc_ycc"]},
    "sh_opacity_dc_ycc": {"preset": "sh_opacity", "ops": ["dc_ycc"]},
}


def split_ops(op_list):
    # accept ["sh,opacity", "scale"] etc
    out = []
    for item in op_list:
        for t in str(item).split(","):
            t = t.strip()
            if t:
                out.append(t)
    return out


def fields_by_ops(*, names, v_rgb, f_dc, f_rest, ops):
    """Return (lerp_fields, use_dc_ycc)."""
    ops = [o.strip() for o in ops if o.strip()]
    use_dc_ycc = ("dc_ycc" in ops)

    lerp_fields = []

    def add_fields(fs):
        for f in fs:
            if f in names and f not in lerp_fields:
                lerp_fields.append(f)

    for op in ops:
        if op == "sh":
            add_fields(f_dc + f_rest)
        elif op == "dc":
            add_fields(f_dc)
        elif op == "rest":
            add_fields(f_rest)
        elif op == "opacity":
            if "opacity" in names:
                add_fields(["opacity"])
        elif op == "scale":
            add_fields([n for n in names if n.startswith("scale_")])
        elif op == "rot":
            add_fields([n for n in names if n.startswith("rot_")])
        elif op == "xyz":
            add_fields([n for n in ("x", "y", "z") if n in names])
        elif op == "all_float":
            for n in names:
                if np.issubdtype(v_rgb[n].dtype, np.floating):
                    add_fields([n])
        elif op == "dc_ycc":
            pass
        else:
            raise ValueError(
                f"Unknown op '{op}'. Supported ops: sh,dc,rest,opacity,scale,rot,xyz,all_float,dc_ycc."
            )

    # if dc_ycc is enabled, remove f_dc_* from lerp list to avoid double-writing
    if use_dc_ycc:
        lerp_fields = [f for f in lerp_fields if not f.startswith("f_dc_")]

    return lerp_fields, use_dc_ycc


def parse_methods(method_args):
    """Parse --methods.

    Accept:
      --methods compare5
      --methods sh_only sh_opacity all_float
      --methods "sh_only,sh_opacity,all_float"
    """
    if not method_args:
        return ["sh_only"]

    # flatten comma-separated
    tokens = []
    for m in method_args:
        for t in str(m).replace(" ", ",").split(","):
            t = t.strip()
            if t:
                tokens.append(t)

    expanded = []
    for t in tokens:
        if t == "compare5":
            expanded.extend(COMPARE5)
        else:
            expanded.append(t)

    # de-dup while preserving order
    seen = set()
    out = []
    for m in expanded:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def build_ops_for_method(method_id: str):
    if method_id not in METHODS:
        raise ValueError(
            f"Unknown method '{method_id}'. Available: {', '.join(sorted(METHODS.keys()))} and 'compare5'."
        )
    spec = METHODS[method_id]
    ops = []
    preset = spec.get("preset", None)
    if preset:
        if preset not in PRESET_TO_OPS:
            raise ValueError(f"Bad method preset '{preset}' for method '{method_id}'.")
        ops += PRESET_TO_OPS[preset]
    ops += spec.get("ops", [])
    ops = split_ops(ops)
    return ops


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Blend two 3DGS point_cloud.ply into multiple alpha folders. "
            "Supports multiple strategies via --methods (simplified in v4)."
        )
    )
    ap.add_argument("--rgb_model_dir", required=True)
    ap.add_argument("--rgb_iter", type=int, required=True)
    ap.add_argument("--t_model_dir", required=True)
    ap.add_argument("--t_iter", type=int, required=True)

    ap.add_argument("--alphas", required=True, help='Comma/space separated, e.g. "0,0.25,0.5,0.75,1"')
    ap.add_argument("--out_root", required=True, help="All methods will be stored under this root folder")
    ap.add_argument("--out_iter", type=int, required=True, help="write to point_cloud/iteration_{out_iter}")

    ap.add_argument(
        "--methods",
        nargs="*",
        default=["compare5"],
        help=(
            "Which strategies to run. Examples: --methods compare5  OR  --methods sh_only sh_opacity all_float. "
            f"Available: {', '.join(sorted(METHODS.keys()))} plus compare5."
        ),
    )

    ap.add_argument(
        "--endpoint_mode",
        choices=["blend", "copy"],
        default="blend",
        help=(
            "blend: alpha=0/1 are produced by the SAME rules as intermediate alphas (continuous, fair). "
            "copy: alpha=0 full copy RGB, alpha=1 full copy T (old strict endpoints)."
        ),
    )
    ap.add_argument(
        "--base",
        choices=["rgb", "t"],
        default="rgb",
        help="Base vertex table to start from before applying ops (for endpoint_mode=blend and intermediate alphas).",
    )

    ap.add_argument(
        "--cfg_template",
        choices=["auto", "rgb", "t", "base"],
        default="auto",
        help=(
            "Which cfg_args template to use. auto mimics old behavior for endpoint_mode=copy. "
            "base uses cfg of --base model for all alphas."
        ),
    )
    ap.add_argument(
        "--base_cfg",
        choices=["rgb", "t"],
        default="rgb",
        help="When cfg_template=auto and alpha is intermediate, use this template.",
    )

    # DC-only YCbCr fusion params (used only if a method includes dc_ycc)
    ap.add_argument("--chroma_gain", type=float, default=1.0, help="Scale T chroma when using dc_ycc.")
    ap.add_argument(
        "--dc_y_from",
        choices=["rgb", "t", "lerp"],
        default="lerp",
        help="Where DC luma Y comes from when using dc_ycc.",
    )
    ap.add_argument(
        "--clip_percentile",
        type=float,
        default=0.5,
        help="Robust clipping percentile for dc_ycc (0 disables).",
    )

    ap.add_argument("--clean_out", action="store_true")
    ap.add_argument("--verify_endpoints", action="store_true")
    ap.add_argument("--list_methods", action="store_true", help="Print all available method IDs then exit")

    args = ap.parse_args()

    if args.list_methods:
        print("Available methods:")
        for k in sorted(METHODS.keys()):
            print(" -", k)
        print("Groups:")
        print(" - compare5 ->", ", ".join(COMPARE5))
        return

    rgb_model = Path(args.rgb_model_dir)
    t_model = Path(args.t_model_dir)
    out_root = Path(args.out_root)
    alphas = parse_alphas(args.alphas)
    method_ids = parse_methods(args.methods)

    if args.clean_out and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rgb_ply = rgb_model / "point_cloud" / f"iteration_{args.rgb_iter}" / "point_cloud.ply"
    t_ply = t_model / "point_cloud" / f"iteration_{args.t_iter}" / "point_cloud.ply"
    if not rgb_ply.exists():
        raise FileNotFoundError(rgb_ply)
    if not t_ply.exists():
        raise FileNotFoundError(t_ply)

    print("RGB PLY:", rgb_ply)
    print("T   PLY:", t_ply)
    print("OUTROOT:", out_root)
    print("methods:", method_ids)
    print("endpoint_mode:", args.endpoint_mode, "| base:", args.base, "| out_iter:", args.out_iter)

    ply_rgb, v_rgb = load_vertex(rgb_ply)
    ply_t, v_t = load_vertex(t_ply)

    if len(v_rgb) != len(v_t):
        raise RuntimeError(f"Vertex count mismatch: RGB={len(v_rgb)} vs T={len(v_t)}")
    if list(v_rgb.dtype.names) != list(v_t.dtype.names):
        raise RuntimeError("PLY schema mismatch. Field names/order must match exactly.")

    names = list(v_rgb.dtype.names)
    f_dc, f_rest, _ = list_fields(names)

    # cfg templates (shared)
    cfg_rgb = read_cfg_args(rgb_model)
    cfg_t = read_cfg_args(t_model)
    cfg_base = cfg_rgb if args.base == "rgb" else cfg_t
    cfg_mid = cfg_rgb if args.base_cfg == "rgb" else cfg_t

    def pick_cfg(alpha, out_dir):
        if args.cfg_template == "rgb":
            return rewrite_model_path(cfg_rgb, out_dir)
        if args.cfg_template == "t":
            return rewrite_model_path(cfg_t, out_dir)
        if args.cfg_template == "base":
            return rewrite_model_path(cfg_base, out_dir)

        # auto
        if args.endpoint_mode == "copy":
            if alpha == 0.0:
                return rewrite_model_path(cfg_rgb, out_dir)
            if alpha == 1.0:
                return rewrite_model_path(cfg_t, out_dir)
            return rewrite_model_path(cfg_mid, out_dir)

        # endpoint_mode=blend
        return rewrite_model_path(cfg_base, out_dir)

    # Choose base vertex table for blend-mode outputs
    v_base = v_rgb if args.base == "rgb" else v_t

    # Pre-extract DC arrays once (used by any method that enables dc_ycc)
    if len(f_dc) == 3:
        dc_rgb = np.stack([np.asarray(v_rgb[n], dtype=np.float32) for n in f_dc], axis=1)
        dc_t = np.stack([np.asarray(v_t[n], dtype=np.float32) for n in f_dc], axis=1)
    else:
        dc_rgb, dc_t = None, None

    for method_id in method_ids:
        method_root = out_root / method_id
        method_root.mkdir(parents=True, exist_ok=True)

        ops = build_ops_for_method(method_id)
        lerp_fields, use_dc_ycc = fields_by_ops(
            names=names,
            v_rgb=v_rgb,
            f_dc=f_dc,
            f_rest=f_rest,
            ops=ops,
        )

        print("\n=============================")
        print("[METHOD]", method_id)
        print("ops      :", ops)
        print("lerp_cnt :", len(lerp_fields))
        if use_dc_ycc:
            print("dc_ycc   : enabled (f_dc_* written by YCbCr fusion, not linear lerp)")
            if dc_rgb is None:
                raise RuntimeError(f"Method '{method_id}' uses dc_ycc but f_dc fields are not 3: {f_dc}")

        for a in alphas:
            sub = alpha_to_subdir(a)
            out_dir = method_root / sub
            pc_dir = out_dir / "point_cloud" / f"iteration_{args.out_iter}"
            pc_dir.mkdir(parents=True, exist_ok=True)
            out_ply = pc_dir / "point_cloud.ply"

            # --- Endpoints behavior ---
            if args.endpoint_mode == "copy" and a in (0.0, 1.0):
                v_out = (v_rgb if a == 0.0 else v_t).copy()
            else:
                # --- Continuous behavior: start from base and apply selected ops for ANY alpha ---
                v_out = v_base.copy()

                # linear-lerp fields
                for f in lerp_fields:
                    v_out[f] = safe_lerp(
                        np.asarray(v_rgb[f]),
                        np.asarray(v_t[f]),
                        a,
                        v_out[f].dtype,
                    )

                # dc_ycc fusion (alpha-controlled)
                if use_dc_ycc:
                    dc_new = dc_ycc_blend(
                        dc_rgb,
                        dc_t,
                        a,
                        chroma_gain=args.chroma_gain,
                        dc_y_from=args.dc_y_from,
                        clip_percentile=args.clip_percentile,
                    )
                    for i, name in enumerate(f_dc):
                        v_out[name] = dc_new[:, i].astype(v_out[name].dtype, copy=False)

            # Write PLY (preserve ASCII/binary flag from RGB PLY)
            PlyData([PlyElement.describe(v_out, "vertex")], text=getattr(ply_rgb, "text", False)).write(str(out_ply))

            # Write cfg_args
            cfg = pick_cfg(a, norm_path(out_dir))
            (out_dir / "cfg_args").write_text(cfg, encoding="utf-8")

            print(f"[OK] {method_id} alpha={a:.3f} -> {out_dir}")

        # Verification per method
        if args.verify_endpoints:
            p0 = method_root / "0" / "point_cloud" / f"iteration_{args.out_iter}" / "point_cloud.ply"
            p1 = method_root / "1" / "point_cloud" / f"iteration_{args.out_iter}" / "point_cloud.ply"
            print("[VERIFY]", method_id)
            if p0.exists():
                _, v0 = load_vertex(p0)
                if args.endpoint_mode == "copy":
                    d0 = max_abs_diff(v0, v_rgb, names)
                    print("  alpha=0 copy: max|out-RGB| all fields =", d0)
                else:
                    d0 = max_abs_diff(v0, v_rgb, names) if args.base == "rgb" else max_abs_diff(v0, v_base, names)
                    print("  alpha=0 blend: max|out-BASE/RGB| all fields =", d0)

            if p1.exists():
                _, v1 = load_vertex(p1)
                if args.endpoint_mode == "copy":
                    d1 = max_abs_diff(v1, v_t, names)
                    print("  alpha=1 copy: max|out-T| all fields =", d1)
                else:
                    if lerp_fields:
                        d_blend = max_abs_diff(v1, v_t, lerp_fields)
                        print("  alpha=1 blend: max|out-T| LERP fields =", d_blend)

                    protected = set(lerp_fields)
                    if use_dc_ycc:
                        protected |= set(f_dc)
                    remain = [n for n in names if n not in protected]
                    if remain:
                        d_base = max_abs_diff(v1, v_base, remain)
                        print("  alpha=1 blend: max|out-BASE| NON-lerp fields =", d_base)

                    if use_dc_ycc:
                        d_dc_vs_t = max_abs_diff(v1, v_t, f_dc)
                        d_dc_vs_rgb = max_abs_diff(v1, v_rgb, f_dc)
                        print("  dc_ycc note: max|DC_out - DC_T| =", d_dc_vs_t, "| max|DC_out - DC_RGB| =", d_dc_vs_rgb)

    print("\nDone.")
    print(f"Example open: SIBR_gaussianViewer_app.exe -m \"{out_root / method_ids[0] / '1'}\" --iteration {args.out_iter}")


if __name__ == "__main__":
    main()
