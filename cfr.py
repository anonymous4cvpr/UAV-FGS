# -*- coding: utf-8 -*-
"""
cfr.py

Preprocess raw RGB-T UAV image pairs into reconstruction-ready paired views.
This module estimates overlap-consistent crops, updates camera intrinsics after
cropping and resizing, and optionally preserves selected EXIF/XMP metadata
required by downstream reconstruction tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import tempfile

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError(
        "OpenCV is required. Install: pip install opencv-python-headless "
        "(or pip install opencv-python)"
    ) from e

from cfr_features import build_structure
from cfr_quality import grad_ncc as quality_grad_ncc, edge_f1 as quality_edge_f1, combine_quality as quality_combine

try:
    from PIL import Image, ImageOps
except Exception as e:
    raise RuntimeError("Pillow is required. Install: pip install pillow") from e

try:
    import piexif  # type: ignore
    HAS_PIEXIF = True
except Exception:
    HAS_PIEXIF = False


# ----------------------------
# Logging
# ----------------------------

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.log_path, "w", encoding="utf-8", errors="ignore")

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass

    def log(self, level: str, msg: str) -> None:
        line = f"{now_ts()} [{level}] {msg}"
        print(line)
        self.fp.write(line + "\n")
        self.fp.flush()


# ----------------------------
# FS utils
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def stem_key(path: Path) -> str:
    return path.stem.strip()


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, tuple) and len(x) == 2:
            num, den = x
            denf = float(den)
            if denf == 0:
                return None
            return float(num) / denf
        return float(x)
    except Exception:
        return None


# ----------------------------
# Image conversions
# ----------------------------

def exif_transposed_pil(img_path: Path) -> Image.Image:
    """
    璇诲叆鍥惧儚骞舵妸 EXIF Orientation 搴旂敤鍒板儚绱狅紙寰楀埌鈥滄鍚戔€濆儚绱狅級銆?
    鍚庣画淇濆瓨鏃跺繀椤诲己鍒?Orientation=1锛岄伩鍏嶆煡鐪嬪櫒浜屾鏃嬭浆瀵艰嚧鈥滃€掑浘鈥濄€?
    """
    im = Image.open(img_path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode == "L":
        arr = np.array(img, dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil_rgb(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ----------------------------
# EXIF read (robust)
# ----------------------------

@dataclass
class ExifLensInfo:
    focal_35mm: Optional[float] = None        # EXIF: FocalLengthIn35mmFilm (piexif field name)
    focal_mm: Optional[float] = None          # EXIF: FocalLength
    digital_zoom: Optional[float] = None      # EXIF: DigitalZoomRatio

    def effective_focal_35mm(self) -> Optional[float]:
        if self.focal_35mm is None:
            return None
        dz = self.digital_zoom if (self.digital_zoom is not None and self.digital_zoom > 0) else 1.0
        return float(self.focal_35mm) * float(dz)


def _get_src_exif_bytes(src_path: Path) -> Optional[bytes]:
    try:
        with Image.open(src_path) as im:
            b = im.info.get("exif", None)
            if b:
                return b
            ex = im.getexif()
            if ex and len(ex) > 0:
                return ex.tobytes()
    except Exception:
        pass
    return None


def read_exif_lens_info(src_path: Path) -> ExifLensInfo:
    info = ExifLensInfo()
    b = _get_src_exif_bytes(src_path)
    if b is None:
        return info

    if HAS_PIEXIF:
        try:
            ex = piexif.load(b)
            f35 = ex.get("Exif", {}).get(piexif.ExifIFD.FocalLengthIn35mmFilm, None)
            fl = ex.get("Exif", {}).get(piexif.ExifIFD.FocalLength, None)
            dz = ex.get("Exif", {}).get(piexif.ExifIFD.DigitalZoomRatio, None)
            info.focal_35mm = safe_float(f35)
            info.focal_mm = safe_float(fl)
            info.digital_zoom = safe_float(dz)
            return info
        except Exception:
            pass

    # fallback (PIL tag IDs)
    try:
        with Image.open(src_path) as im:
            ex = im.getexif()
            info.focal_mm = safe_float(ex.get(37386, None))      # FocalLength
            info.digital_zoom = safe_float(ex.get(41988, None))  # DigitalZoomRatio
            info.focal_35mm = safe_float(ex.get(41989, None))    # FocalLengthIn35mmFilm
    except Exception:
        pass
    return info


def compute_zoom_ratio(rgb_exif: ExifLensInfo, th_exif: ExifLensInfo) -> Optional[float]:
    a = th_exif.effective_focal_35mm()
    b = rgb_exif.effective_focal_35mm()
    if a is not None and b is not None and b > 0:
        return float(a) / float(b)

    a2 = th_exif.focal_mm
    b2 = rgb_exif.focal_mm
    if a2 is not None and b2 is not None and b2 > 0:
        return float(a2) / float(b2)
    return None


# ----------------------------
# exiftool helpers
# ----------------------------

@lru_cache(maxsize=1)
def find_exiftool() -> Optional[str]:
    exe = shutil.which("exiftool") or shutil.which("exiftool.exe")
    return exe


def run_exiftool(cmd: List[str], logger: Logger) -> bool:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if out:
            logger.log("INFO", f"[exiftool] {out}")
        if err:
            logger.log("WARN", f"[exiftool] {err}")
        return p.returncode == 0
    except Exception as e:
        logger.log("WARN", f"[exiftool] failed to run: {e}")
        return False


def _compute_zoom_updates_for_crop(
    src_rgb_path: Path,
    zoom_factor: Optional[float],
) -> Tuple[Optional[float], Optional[int]]:
    """
    Convert crop-zoom to EXIF zoom-equivalent metadata updates:
      - DigitalZoomRatio = base_DZ * zoom_factor
      - FocalLengthIn35mmFormat = base_35mm_equiv * zoom_factor

    Physical focal length (FocalLength) is intentionally not modified.
    """
    if zoom_factor is None or zoom_factor <= 0:
        return None, None

    src = read_exif_lens_info(src_rgb_path)
    base_dz = src.digital_zoom if (src.digital_zoom is not None and src.digital_zoom > 0) else 1.0
    upd_dz = float(base_dz) * float(zoom_factor)

    upd_f35 = None
    if src.focal_35mm is not None and src.focal_35mm > 0:
        upd_f35 = int(round(float(src.focal_35mm) * float(zoom_factor)))

    return upd_dz, upd_f35


# --- JPEG XMP packet helpers (v9) ---

_XMP_STD_HEADER = b"XXXX\x00"
_XMP_EXT_HEADER = b"XXXX\x00"

def _extract_xmp_packet_from_jpeg(jpeg_path: Path) -> Optional[bytes]:
    """Extract the main XMP packet (RDF/XML) from a JPEG APP1 segment.

    Returns the raw XML bytes (without the APP1 header) or None if not present / not JPEG.
    """
    try:
        with open(jpeg_path, "rb") as fp:
            if fp.read(2) != b"\xFF\xD8":  # SOI
                return None
            while True:
                marker = fp.read(2)
                if len(marker) < 2:
                    return None
                # Scan to next 0xFF
                if marker[0] != 0xFF:
                    continue
                # Start of Scan or End of Image -> no more metadata segments
                if marker in (b"\xFF\xDA", b"\xFF\xD9"):
                    return None
                # Some markers have no length (RST, TEM, etc.) but APPn do.
                len_bytes = fp.read(2)
                if len(len_bytes) < 2:
                    return None
                seg_len = int.from_bytes(len_bytes, "big")
                if seg_len < 2:
                    return None
                seg = fp.read(seg_len - 2)
                if marker == b"\xFF\xE1":  # APP1
                    if seg.startswith(_XMP_STD_HEADER):
                        return seg[len(_XMP_STD_HEADER):]
                    # NOTE: Extended XMP is chunked; we don't reconstruct it here.
                    # Fall back to normal tag-copy if needed.
                    if seg.startswith(_XMP_EXT_HEADER):
                        # present but not reconstructed
                        return None
    except Exception:
        return None


def _write_temp_xmp(xmp_xml: bytes) -> Path:
    """Write XMP XML bytes to a temp file and return its path (caller deletes)."""
    tmp = tempfile.NamedTemporaryFile(prefix="cfrv9_xmp_", suffix=".xmp", delete=False)
    try:
        tmp.write(xmp_xml)
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def _exiftool_copy_then_patch(
    exiftool: str,
    src_rgb_path: Path,
    dst_path: Path,
    out_w: int,
    out_h: int,
    upd_dz: Optional[float],
    upd_f35: Optional[float],
    k_mode: str,
    logger: Logger,
) -> bool:
    """v9: One-pass copy+patch with raw XMP packet injection.

    - Copy all metadata with -all:all -unsafe
    - Strip preview/thumbnail/MPF/IFD1
    - Inject raw XMP packet from source JPEG (best effort)
    - Patch size + (DigitalZoomRatio, FocalLengthIn35mmFormat)
    - Force Orientation=1 in both IFD0 and XMP-tiff
    """
    xmp_tmp: Optional[Path] = None
    try:
        xmp_xml = _extract_xmp_packet_from_jpeg(src_rgb_path)
        if xmp_xml:
            xmp_tmp = _write_temp_xmp(xmp_xml)

        do_size_update = (k_mode != "no_kupdate")
        cmd: List[str] = [
            exiftool,
            "-overwrite_original",
            "-P",
            "-n",
            "-m",
            "-q",
            "-q",
            "-TagsFromFile", str(src_rgb_path),
            "-all:all",
            "-xmp:all",
            "-unsafe",
            "-icc_profile:all",

            # strip thumbnails / previews that may confuse downstream tools
            "-MPF:all=",
            "-IFD1:all=",
            "-PreviewImage=",
            "-ThumbnailImage=",
            "-JpgFromRaw=",
        ]
        if do_size_update:
            # size fields (keep v8 policy: do NOT write PixelXDimension/YDimension)
            cmd += [
                f"-IFD0:ImageWidth={int(out_w)}",
                f"-IFD0:ImageHeight={int(out_h)}",
                f"-ExifIFD:ExifImageWidth={int(out_w)}",
                f"-ExifIFD:ExifImageHeight={int(out_h)}",
            ]

        if upd_dz is not None:
            cmd.append(f"-ExifIFD:DigitalZoomRatio={float(upd_dz):.10f}")
        if upd_f35 is not None:
            cmd.append(f"-ExifIFD:FocalLengthIn35mmFormat={int(round(upd_f35))}")

        # raw XMP packet (best effort; placed before orientation patch)
        if xmp_tmp is not None:
            cmd.append(f"-XMP<={str(xmp_tmp)}")

        # final orientation normalization (must be last)
        cmd += [
            "-Orientation=1",
            "-IFD0:Orientation=1",
            "-XMP-tiff:Orientation=1",
            str(dst_path),
        ]

        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            logger.log("WARN", f"exiftool failed rc={r.returncode} dst={dst_path.name} err={r.stderr.strip()[:400]}")
            return False
        return True
    except Exception as e:
        logger.log("WARN", f"exiftool exception dst={dst_path.name}: {e}")
        return False
    finally:
        if xmp_tmp is not None:
            try:
                xmp_tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                try:
                    xmp_tmp.unlink()
                except Exception:
                    pass
def save_with_metadata(
    src_rgb_path: Path,
    dst_path: Path,
    out_bgr: np.ndarray,
    out_w: int,
    out_h: int,
    zoom_factor: Optional[float],
    k_mode: str,
    logger: Logger,
) -> None:
    """
    淇濆瓨鍍忕礌 + 灏藉彲鑳藉畬鏁翠繚鐣欏厓鏁版嵁锛堝惈 GPS / XMP-drone-dji 绛夛級
    """
    ensure_dir(dst_path.parent)

    # 1) write pixels first
    pil = bgr_to_pil_rgb(out_bgr)
    save_kwargs = dict(format="JPEG", quality=95, optimize=True, subsampling=2)
    pil.save(str(dst_path), **save_kwargs)

    k_mode = str(k_mode).strip().lower()
    if k_mode not in ("full", "no_kupdate", "naive_k"):
        k_mode = "full"
    do_zoom_update = (k_mode == "full")

    exiftool = find_exiftool()
    if exiftool:
        upd_dz, upd_f35 = _compute_zoom_updates_for_crop(src_rgb_path, zoom_factor) if do_zoom_update else (None, None)
        ok = _exiftool_copy_then_patch(
            exiftool=exiftool,
            src_rgb_path=src_rgb_path,
            dst_path=dst_path,
            out_w=out_w,
            out_h=out_h,
            upd_dz=upd_dz,
            upd_f35=upd_f35,
            k_mode=k_mode,
            logger=logger,
        )
        if ok:
            return
        logger.log("WARN", "exiftool copy/patch failed -> fallback to piexif (XMP/drone-dji may be lost).")

    # fallback: piexif (NOTE: will NOT preserve XMP blocks)
    b = _get_src_exif_bytes(src_rgb_path)
    if not (HAS_PIEXIF and b):
        return

    try:
        ex = piexif.load(b)

        # orientation + size
        try:
            ex["0th"][piexif.ImageIFD.Orientation] = 1
            if k_mode != "no_kupdate":
                ex["0th"][piexif.ImageIFD.ImageWidth] = int(out_w)
                ex["0th"][piexif.ImageIFD.ImageLength] = int(out_h)
        except Exception:
            pass
        try:
            if k_mode != "no_kupdate":
                ex["Exif"][piexif.ExifIFD.ExifImageWidth] = int(out_w)
                ex["Exif"][piexif.ExifIFD.ExifImageHeight] = int(out_h)
        except Exception:
            pass

        # remove thumbnails
        ex["thumbnail"] = None

        # update zoom + f35
        upd_dz, upd_f35 = _compute_zoom_updates_for_crop(src_rgb_path, zoom_factor) if do_zoom_update else (None, None)

        if upd_dz is not None:
            try:
                den = 1000000
                num = int(round(float(upd_dz) * den))
                g = int(np.gcd(num, den))
                ex["Exif"][piexif.ExifIFD.DigitalZoomRatio] = (num // g, den // g)
            except Exception:
                pass

        if upd_f35 is not None:
            try:
                ex["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(upd_f35)
            except Exception:
                pass

        exif_bytes = piexif.dump(ex)
        pil.save(str(dst_path), **save_kwargs, exif=exif_bytes)
    except Exception as e:
        logger.log("WARN", f"piexif fallback failed: {e}")

# ----------------------------
# Debug helpers (v9)
# ----------------------------

def _exiftool_dump_json(exiftool: str, img_path: Path, out_json: Path, logger: Logger) -> Optional[Dict]:
    """Dump full metadata as JSON (single image). Returns the first object dict.

    Robust to empty/None stdout (avoid NoneType write errors).
    """
    try:
        cmd = [exiftool, "-j", "-G1", "-a", "-u", "-s", "-n", str(img_path)]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        stdout = (r.stdout or "")
        stderr = (r.stderr or "").strip()
        if r.returncode != 0:
            logger.log("WARN", f"exiftool dump failed rc={r.returncode} file={img_path.name} err={stderr[:300]}")
            return None
        if not stdout.strip():
            logger.log("WARN", f"exiftool dump empty stdout file={img_path.name} err={stderr[:200]}")
            return None
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(stdout, encoding="utf-8", errors="ignore")
        arr = json.loads(stdout)
        if isinstance(arr, list) and arr:
            return arr[0]
        return None
    except Exception as e:
        logger.log("WARN", f"exiftool dump exception file={img_path.name}: {e}")
        return None


def _diff_meta_dict(src: Dict, dst: Dict) -> Dict:
    """Compute a lightweight diff between two exiftool JSON dicts."""
    ignore_prefix = ("File:", "System:", "ExifTool:", "Composite:")
    ignore_keys = {"SourceFile", "FileName", "Directory", "FileSize", "FileModifyDate", "FileAccessDate", "FileInodeChangeDate"}
    def norm(d: Dict) -> Dict:
        out = {}
        for k,v in d.items():
            if k in ignore_keys:
                continue
            if any(k.startswith(p) for p in ignore_prefix):
                continue
            out[k]=v
        return out

    a = norm(src)
    b = norm(dst)
    a_keys=set(a.keys()); b_keys=set(b.keys())
    missing = sorted(a_keys - b_keys)
    extra = sorted(b_keys - a_keys)
    changed = []
    for k in sorted(a_keys & b_keys):
        if a.get(k) != b.get(k):
            changed.append({"key": k, "src": a.get(k), "dst": b.get(k)})
    return {
        "missing_in_dst_count": len(missing),
        "extra_in_dst_count": len(extra),
        "changed_count": len(changed),
        "missing_in_dst": missing[:200],
        "extra_in_dst": extra[:200],
        "changed_sample": changed[:200],
    }


def _write_exif_audit_jsonl(
    exiftool: str,
    scan_dir: Path,
    out_jsonl: Path,
    kind: str,
    expected_w: int,
    expected_h: int,
    logger: Logger
) -> None:
    """Scan a directory with exiftool once and write per-file audit lines.

    Audit focuses on correctness for COLMAP:
    - Orientation normalized (IFD0 + optionally XMP-tiff)
    - Width/Height fields match expected (thermal size)
    - DigitalZoomRatio / FocalLengthIn35mmFormat present (recorded for review)
    """
    try:
        cmd = [
            exiftool,
            "-j", "-G1", "-a", "-u", "-s", "-n",
            "-r",
            "-Orientation",
            "-IFD0:Orientation",
            "-XMP-tiff:Orientation",
            "-IFD0:ImageWidth",
            "-IFD0:ImageHeight",
            "-ExifIFD:ExifImageWidth",
            "-ExifIFD:ExifImageHeight",
            "-ExifIFD:DigitalZoomRatio",
            "-ExifIFD:FocalLengthIn35mmFormat",
            "-GPSLatitude", "-GPSLongitude",
            "-XMP-drone-dji:UTCAtExposure",
            "-XMP-drone-dji:DroneModel",
            "-XMP-drone-dji:CameraSerialNumber",
            str(scan_dir),
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            logger.log("WARN", f"exiftool audit failed rc={r.returncode} dir={scan_dir} err={(r.stderr or '').strip()[:300]}")
            return
        stdout = (r.stdout or "").strip()
        if not stdout:
            logger.log("WARN", f"exiftool audit empty stdout dir={scan_dir}")
            return
        arr = json.loads(stdout)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "a", encoding="utf-8", errors="ignore") as fp:
            for obj in arr:
                if not isinstance(obj, dict):
                    continue

                problems: List[str] = []

                o1 = obj.get("Orientation") or obj.get("IFD0:Orientation")
                o2 = obj.get("XMP-tiff:Orientation")

                if o1 not in (None, 1, "1"):
                    problems.append("orientation_not_1")
                if o2 not in (None, 1, "1"):
                    problems.append("xmp_orientation_not_1")

                w0 = obj.get("IFD0:ImageWidth")
                h0 = obj.get("IFD0:ImageHeight")
                we = obj.get("ExifIFD:ExifImageWidth")
                he = obj.get("ExifIFD:ExifImageHeight")

                # size checks: expect both groups match thermal size
                if w0 is not None and int(w0) != int(expected_w):
                    problems.append("ifd0_width_mismatch")
                if h0 is not None and int(h0) != int(expected_h):
                    problems.append("ifd0_height_mismatch")
                if we is not None and int(we) != int(expected_w):
                    problems.append("exif_width_mismatch")
                if he is not None and int(he) != int(expected_h):
                    problems.append("exif_height_mismatch")

                fp.write(json.dumps({
                    "kind": kind,
                    "expected_size": {"w": int(expected_w), "h": int(expected_h)},
                    "SourceFile": obj.get("SourceFile"),
                    "Orientation": o1,
                    "XMP-tiff:Orientation": o2,
                    "IFD0:ImageWidth": w0,
                    "IFD0:ImageHeight": h0,
                    "ExifIFD:ExifImageWidth": we,
                    "ExifIFD:ExifImageHeight": he,
                    "ExifIFD:DigitalZoomRatio": obj.get("ExifIFD:DigitalZoomRatio"),
                    "ExifIFD:FocalLengthIn35mmFormat": obj.get("ExifIFD:FocalLengthIn35mmFormat"),
                    "GPSLatitude": obj.get("GPSLatitude"),
                    "GPSLongitude": obj.get("GPSLongitude"),
                    "XMP-drone-dji:UTCAtExposure": obj.get("XMP-drone-dji:UTCAtExposure"),
                    "XMP-drone-dji:DroneModel": obj.get("XMP-drone-dji:DroneModel"),
                    "XMP-drone-dji:CameraSerialNumber": obj.get("XMP-drone-dji:CameraSerialNumber"),
                    "problems": problems,
                }, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.log("WARN", f"exif audit exception dir={scan_dir}: {e}")

# ----------------------------
# Matching / crop logic (淇濇寔涓嶅彉)
# ----------------------------

# ----------------------------

def downscale_max(bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr, 1.0
    s = max_side / float(max(h, w))
    nw, nh = int(round(w * s)), int(round(h * s))
    ds = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return ds, s


def to_gray_for_match(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def build_sobel_mag_f32(bgr: np.ndarray) -> np.ndarray:
    return build_structure(bgr, mode="sobel_mag")


def _mat33_from_affine(aff: np.ndarray) -> np.ndarray:
    if aff.shape == (3, 3):
        return aff.astype(np.float32)
    if aff.shape == (2, 3):
        out = np.eye(3, dtype=np.float32)
        out[:2, :3] = aff.astype(np.float32)
        return out
    raise ValueError(f"Invalid affine shape: {aff.shape}")


def _normalize_homography(H: np.ndarray) -> np.ndarray:
    H = H.astype(np.float32)
    if abs(float(H[2, 2])) > 1e-8:
        H = H / float(H[2, 2])
    return H


def _warp_gray_with_h(gray_f32: np.ndarray, H: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    H33 = _mat33_from_affine(H)
    return cv2.warpPerspective(gray_f32, H33, out_size, flags=cv2.INTER_LINEAR)


def _sample_grid_points(w: int, h: int, grid: int = 5) -> np.ndarray:
    xs = np.linspace(0.0, float(w - 1), grid, dtype=np.float32)
    ys = np.linspace(0.0, float(h - 1), grid, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)
    return pts.astype(np.float32)


def fit_sfm_safe_similarity_from_h(
    H: np.ndarray,
    rgb_w: int,
    rgb_h: int,
    out_w: int,
    out_h: int,
    theta_deg: float,
) -> Tuple[np.ndarray, float, float, float]:
    theta = math.radians(float(theta_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    pts = _sample_grid_points(rgb_w, rgb_h, grid=5).reshape(-1, 1, 2)
    try:
        pts_t = cv2.perspectiveTransform(pts, _mat33_from_affine(H))
    except Exception:
        pts_t = pts.copy()

    A = []
    b = []
    for i in range(pts.shape[0]):
        x, y = float(pts[i, 0, 0]), float(pts[i, 0, 1])
        xt, yt = float(pts_t[i, 0, 0]), float(pts_t[i, 0, 1])
        u = cos_t * x - sin_t * y
        v = sin_t * x + cos_t * y
        A.append([u, 1.0, 0.0])
        b.append(xt)
        A.append([v, 0.0, 1.0])
        b.append(yt)

    A = np.array(A, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        s = float(sol[0])
        tx = float(sol[1])
        ty = float(sol[2])
    except Exception:
        s, tx, ty = 1.0, 0.0, 0.0

    if not (math.isfinite(s) and math.isfinite(tx) and math.isfinite(ty)) or s <= 1e-8:
        s, tx, ty = 1.0, 0.0, 0.0

    S = np.array([
        [s * cos_t, -s * sin_t, tx],
        [s * sin_t,  s * cos_t, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return S, s, tx, ty


def crop_box_from_similarity(
    s: float,
    tx: float,
    ty: float,
    out_w: int,
    out_h: int,
) -> Tuple[int, int, int, int]:
    if s <= 1e-8 or not math.isfinite(s):
        return 0, 0, out_w, out_h
    crop_w = float(out_w) / float(s)
    crop_h = float(out_h) / float(s)
    x0 = -float(tx) / float(s)
    y0 = -float(ty) / float(s)
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def crop_box_from_fov(
    rgb_w: int, rgb_h: int, th_w: int, th_h: int,
    fov_frac_w: float,
    cx: float, cy: float
) -> Tuple[int, int, int, int]:
    aspect_th = th_w / float(th_h)
    crop_w = rgb_w * float(fov_frac_w)
    crop_h = crop_w / aspect_th

    if crop_h > rgb_h:
        crop_h = rgb_h * float(fov_frac_w)
        crop_w = crop_h * aspect_th

    crop_w = max(2.0, min(float(rgb_w), crop_w))
    crop_h = max(2.0, min(float(rgb_h), crop_h))

    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    x1 = int(round(x0 + crop_w))
    y1 = int(round(y0 + crop_h))
    return x0, y0, x1, y1


def homography_from_crop_box(
    box: Tuple[int, int, int, int],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop_w = max(1.0, float(x1 - x0))
    crop_h = max(1.0, float(y1 - y0))
    sx = float(out_w) / crop_w
    sy = float(out_h) / crop_h
    H = np.array([
        [sx, 0.0, -sx * float(x0)],
        [0.0, sy, -sy * float(y0)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return H


def crop_with_pad(bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = bgr.shape[:2]
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        bgr = cv2.copyMakeBorder(bgr, pad_t, pad_b, pad_l, pad_r,
                                 borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t
    x0 = max(0, min(x0, bgr.shape[1] - 1))
    y0 = max(0, min(y0, bgr.shape[0] - 1))
    x1 = max(x0 + 1, min(x1, bgr.shape[1]))
    y1 = max(y0 + 1, min(y1, bgr.shape[0]))
    return bgr[y0:y1, x0:x1]


def safe_crop_with_pad(
    bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    max_pad: int,
    max_pixels: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    x0, y0, x1, y1 = box
    h, w = bgr.shape[:2]
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    if max(pad_l, pad_r, pad_t, pad_b) > max_pad:
        return None, "pad_too_large"
    padded_w = w + pad_l + pad_r
    padded_h = h + pad_t + pad_b
    if padded_w * padded_h > max_pixels:
        return None, "pad_area_too_large"
    try:
        if pad_l or pad_t or pad_r or pad_b:
            bgr = cv2.copyMakeBorder(bgr, pad_t, pad_b, pad_l, pad_r,
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            x0 += pad_l
            x1 += pad_l
            y0 += pad_t
            y1 += pad_t
        x0 = max(0, min(x0, bgr.shape[1] - 1))
        y0 = max(0, min(y0, bgr.shape[0] - 1))
        x1 = max(x0 + 1, min(x1, bgr.shape[1]))
        y1 = max(y0 + 1, min(y1, bgr.shape[0]))
        return bgr[y0:y1, x0:x1], None
    except cv2.error as e:
        msg = str(e).splitlines()[0] if str(e).splitlines() else str(e)
        return None, f"cv2_error: {msg}"


def ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    af -= af.mean()
    bf -= bf.mean()
    denom = (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-6)
    return float((af * bf).sum() / denom)


@dataclass
class FitResult:
    ok: bool
    reason: str
    fov_frac: Optional[float] = None
    match_score: Optional[float] = None
    ncc: Optional[float] = None
    cx_off_frac: Optional[float] = None
    cy_off_frac: Optional[float] = None


@dataclass
class EccResult:
    ok: bool
    reason: str
    h_raw: Optional[np.ndarray] = None
    ecc_score: Optional[float] = None
    warp_residual: Optional[np.ndarray] = None


@dataclass
class RunningStat:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_v: Optional[float] = None
    max_v: Optional[float] = None

    def add(self, x: Optional[float]) -> None:
        if x is None:
            return
        try:
            v = float(x)
        except Exception:
            return
        if not math.isfinite(v):
            return
        self.n += 1
        if self.min_v is None or v < self.min_v:
            self.min_v = v
        if self.max_v is None or v > self.max_v:
            self.max_v = v
        delta = v - self.mean
        self.mean += delta / self.n
        delta2 = v - self.mean
        self.m2 += delta * delta2

    def as_dict(self) -> Dict[str, Optional[float]]:
        if self.n <= 1:
            std = 0.0 if self.n == 1 else None
        else:
            std = float(math.sqrt(self.m2 / float(self.n)))
        return {
            "count": int(self.n),
            "mean": float(self.mean) if self.n > 0 else None,
            "std": std,
            "min": float(self.min_v) if self.min_v is not None else None,
            "max": float(self.max_v) if self.max_v is not None else None,
        }


def select_even_indices(total: int, max_count: int) -> List[int]:
    if total <= 0:
        return []
    if max_count <= 0 or max_count >= total:
        return list(range(total))
    xs = np.linspace(0, total - 1, max_count)
    idx = sorted({int(round(x)) for x in xs})
    return idx


def _project_corners(H: np.ndarray, w: int, h: int) -> Optional[np.ndarray]:
    pts = np.array([[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]],
                   dtype=np.float32).reshape(-1, 1, 2)
    try:
        pts_t = cv2.perspectiveTransform(pts, _mat33_from_affine(H))
    except Exception:
        return None
    arr = pts_t.reshape(-1, 2)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _quality_for_h(
    rgb_bgr: np.ndarray,
    th_bgr: np.ndarray,
    H: np.ndarray,
    structure_mode: str,
) -> Tuple[Optional[float], Optional[float]]:
    try:
        th_struct = build_structure(th_bgr, mode=structure_mode)
        rgb_struct = build_structure(rgb_bgr, mode=structure_mode)
        th_h, th_w = th_bgr.shape[:2]
        rgb_warp = _warp_gray_with_h(rgb_struct, H, (th_w, th_h))
        grad_ncc = quality_grad_ncc(th_struct, rgb_warp)
        edge_f1 = quality_edge_f1(th_struct, rgb_warp)
        return grad_ncc, edge_f1
    except Exception:
        return None, None


def _phase_corr_shift(
    a: np.ndarray,
    b: np.ndarray,
    max_shift: float = 3.0,
    min_resp: float = 0.02,
    roi_margin_frac: float = 0.10,
) -> Optional[Tuple[float, float, float]]:
    try:
        h, w = a.shape[:2]
        mx = int(round(w * float(roi_margin_frac)))
        my = int(round(h * float(roi_margin_frac)))
        if mx * 2 >= w or my * 2 >= h:
            return None

        a_roi = a[my:h - my, mx:w - mx].astype(np.float32)
        b_roi = b[my:h - my, mx:w - mx].astype(np.float32)
        if a_roi.size <= 0 or b_roi.size <= 0:
            return None

        win = cv2.createHanningWindow((a_roi.shape[1], a_roi.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a_roi, b_roi, win)
        if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(resp)):
            return None
        if float(resp) < float(min_resp):
            return None
        if abs(float(dx)) > float(max_shift) or abs(float(dy)) > float(max_shift):
            return None
        return float(dx), float(dy), float(resp)
    except Exception:
        return None


def _select_global_h_by_corner_median(cands: List[Dict[str, object]], rgb_w: int, rgb_h: int) -> Optional[Dict[str, object]]:
    corner_list = []
    for c in cands:
        H = c.get("H")
        if H is None:
            continue
        corners = _project_corners(H, rgb_w, rgb_h)
        if corners is None:
            continue
        c["corners"] = corners
        corner_list.append(corners)
    if not corner_list:
        return None
    median_c = np.median(np.stack(corner_list, axis=0), axis=0)
    usable = []
    for c in cands:
        corners = c.get("corners")
        if corners is None:
            continue
        err = float(np.linalg.norm(corners - median_c))
        c["corner_err"] = err
        usable.append(c)
    if not usable:
        return None
    def _score(item: Dict[str, object]) -> Tuple[float, float]:
        err = float(item.get("corner_err", 1e9))
        q = item.get("q_total")
        try:
            qv = float(q)  # type: ignore[arg-type]
        except Exception:
            qv = -1e9
        return (err, -qv)
    usable.sort(key=_score)
    return usable[0]


def estimate_fit_on_pair(
    rgb_bgr: np.ndarray,
    th_bgr: np.ndarray,
    th_size: Tuple[int, int],
    fov_candidates: List[float],
    rgb_max_side: int = 900,
    th_max_side: int = 600,
) -> FitResult:
    th_w, th_h = th_size
    rgb_h, rgb_w = rgb_bgr.shape[:2]

    rgb_ds, s_rgb = downscale_max(rgb_bgr, rgb_max_side)
    th_ds, _ = downscale_max(th_bgr, th_max_side)

    rgb_edge = to_gray_for_match(rgb_ds)
    th_edge = to_gray_for_match(th_ds)

    rgb_ds_h, rgb_ds_w = rgb_ds.shape[:2]
    th_ds_h, th_ds_w = th_ds.shape[:2]
    aspect_th = th_ds_w / float(th_ds_h)

    best = None  # (score, fov, x0, y0, tw, thh)
    for f in fov_candidates:
        tw = int(round(rgb_ds_w * float(f)))
        thh = int(round(tw / aspect_th))
        if thh > rgb_ds_h:
            thh = int(round(rgb_ds_h * float(f)))
            tw = int(round(thh * aspect_th))

        if tw < 40 or thh < 40:
            continue
        if tw >= rgb_ds_w or thh >= rgb_ds_h:
            continue

        tpl = cv2.resize(th_edge, (tw, thh), interpolation=cv2.INTER_AREA)
        try:
            res = cv2.matchTemplate(rgb_edge, tpl, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
        except Exception:
            continue

        if best is None or float(maxv) > best[0]:
            best = (float(maxv), float(f), int(maxloc[0]), int(maxloc[1]), int(tw), int(thh))

    if best is None:
        return FitResult(ok=False, reason="no_valid_candidate")

    match_score, f_best, x0_ds, y0_ds, tw, thh = best

    x0 = int(round(x0_ds / s_rgb))
    y0 = int(round(y0_ds / s_rgb))
    x1 = int(round((x0_ds + tw) / s_rgb))
    y1 = int(round((y0_ds + thh) / s_rgb))

    crop = crop_with_pad(rgb_bgr, (x0, y0, x1, y1))
    crop_rs = cv2.resize(crop, (th_w, th_h), interpolation=cv2.INTER_AREA)
    crop_edge = to_gray_for_match(crop_rs)

    th_rs = cv2.resize(th_bgr, (th_w, th_h), interpolation=cv2.INTER_AREA)
    th_edge_rs = to_gray_for_match(th_rs)

    ncc = ncc_score(crop_edge, th_edge_rs)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    cx_off_frac = (cx - rgb_w / 2.0) / float(rgb_w)
    cy_off_frac = (cy - rgb_h / 2.0) / float(rgb_h)

    if not (match_score >= 0.02 and ncc >= 0.01):
        return FitResult(
            ok=False,
            reason=f"low_score(match={match_score:.3f},ncc={ncc:.3f})",
            fov_frac=f_best,
            match_score=match_score,
            ncc=ncc,
            cx_off_frac=cx_off_frac,
            cy_off_frac=cy_off_frac,
        )

    return FitResult(
        ok=True,
        reason="ok",
        fov_frac=f_best,
        match_score=match_score,
        ncc=ncc,
        cx_off_frac=cx_off_frac,
        cy_off_frac=cy_off_frac,
    )


def estimate_ecc_on_pair(
    rgb_bgr: np.ndarray,
    th_bgr: np.ndarray,
    H_init: np.ndarray,
    motion: str,
    ecc_iter: int,
    ecc_eps: float,
    structure_mode: str,
) -> EccResult:
    th_h, th_w = th_bgr.shape[:2]

    th_struct = build_structure(th_bgr, mode=structure_mode)
    rgb_struct = build_structure(rgb_bgr, mode=structure_mode)

    if float(np.std(th_struct)) < 1e-6 or float(np.std(rgb_struct)) < 1e-6:
        return EccResult(ok=False, reason="low_texture")

    def _run_ecc(template: np.ndarray, inp: np.ndarray, warp_init: np.ndarray, motion_type: int):
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(ecc_iter), float(ecc_eps))
        try:
            cc, warp = cv2.findTransformECC(template, inp, warp_init, motion_type, criteria)
        except cv2.error as e:
            msg = str(e).splitlines()[0] if str(e).splitlines() else str(e)
            return None, f"cv2_error: {msg}"
        except Exception as e:
            return None, f"error: {e}"
        return (float(cc), warp), None

    motion = motion.lower().strip()
    H_init33 = _mat33_from_affine(H_init)

    # 1) Direct ECC on original images with H_init as initialization
    if motion == "affine":
        init_aff = H_init33[:2, :].astype(np.float32)
        res, err = _run_ecc(th_struct, rgb_struct, init_aff, cv2.MOTION_AFFINE)
        if res is not None:
            cc, warp = res
            warp33 = _mat33_from_affine(warp)
            H_raw = _normalize_homography(warp33)
            return EccResult(ok=True, reason="ok", h_raw=H_raw, ecc_score=cc, warp_residual=warp33)
        last_err = err
    else:
        init_h = H_init33.astype(np.float32)
        res, err = _run_ecc(th_struct, rgb_struct, init_h, cv2.MOTION_HOMOGRAPHY)
        if res is not None:
            cc, warp = res
            warp33 = _mat33_from_affine(warp)
            H_raw = _normalize_homography(warp33)
            return EccResult(ok=True, reason="ok", h_raw=H_raw, ecc_score=cc, warp_residual=warp33)
        last_err = err

        # 2) Fallback: affine ECC if homography fails
        init_aff = H_init33[:2, :].astype(np.float32)
        res, err = _run_ecc(th_struct, rgb_struct, init_aff, cv2.MOTION_AFFINE)
        if res is not None:
            cc, warp = res
            warp33 = _mat33_from_affine(warp)
            H_raw = _normalize_homography(warp33)
            return EccResult(ok=True, reason="fallback_affine", h_raw=H_raw, ecc_score=cc, warp_residual=warp33)
        last_err = err

    # 3) Residual ECC on pre-warped image (legacy fallback)
    rgb_init = _warp_gray_with_h(rgb_struct, H_init33, (th_w, th_h))
    if motion == "affine":
        res, err = _run_ecc(th_struct, rgb_init, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE)
        if res is not None:
            cc, warp = res
            warp33 = _mat33_from_affine(warp)
            H_raw = _normalize_homography(warp33 @ H_init33)
            return EccResult(ok=True, reason="fallback_residual", h_raw=H_raw, ecc_score=cc, warp_residual=warp33)
        last_err = err
    else:
        res, err = _run_ecc(th_struct, rgb_init, np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY)
        if res is not None:
            cc, warp = res
            warp33 = _mat33_from_affine(warp)
            H_raw = _normalize_homography(warp33 @ H_init33)
            return EccResult(ok=True, reason="fallback_residual", h_raw=H_raw, ecc_score=cc, warp_residual=warp33)
        last_err = err

    return EccResult(ok=False, reason=last_err or "ecc_failed")


# ----------------------------
# Visualization / montage
# ----------------------------

def resize_keep_aspect_pad(bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= 0 or h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def put_text(bgr: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.9,
             color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 2) -> None:
    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def overlay_thermal_rgb(th_bgr: np.ndarray, rgb_bgr: np.ndarray, alpha_th: float = 0.60) -> np.ndarray:
    if th_bgr.shape[:2] != rgb_bgr.shape[:2]:
        rgb_bgr = cv2.resize(rgb_bgr, (th_bgr.shape[1], th_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    beta = 1.0 - alpha_th
    return cv2.addWeighted(th_bgr, alpha_th, rgb_bgr, beta, 0.0)


def draw_boxes_on_vis(
    vis_bgr: np.ndarray,
    fit_box: Tuple[int, int, int, int],
    exif_box: Optional[Tuple[int, int, int, int]],
    ecc_box: Optional[Tuple[int, int, int, int]],
    dual_box: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    out = vis_bgr.copy()
    x0, y0, x1, y1 = fit_box
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 8)  # FIT red
    if exif_box is not None:
        gx0, gy0, gx1, gy1 = exif_box
        cv2.rectangle(out, (gx0, gy0), (gx1, gy1), (0, 255, 0), 8)  # EXIF green
    if dual_box is not None:
        dx0, dy0, dx1, dy1 = dual_box
        cv2.rectangle(out, (dx0, dy0), (dx1, dy1), (255, 255, 0), 8)  # DUAL cyan
    return out


def make_comparison_montage(
    vis_boxes_bgr: np.ndarray,
    th_bgr: np.ndarray,
    img_exif_bgr: Optional[np.ndarray],
    img_fit_bgr: np.ndarray,
    cell_w: int,
    cell_h: int,
    img_ecc_bgr: Optional[np.ndarray] = None,
    dual_th_bgr: Optional[np.ndarray] = None,
    img_dual_bgr: Optional[np.ndarray] = None,
    force_dual: bool = False,
) -> np.ndarray:
    def blank_cell(label: Optional[str] = None) -> np.ndarray:
        cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        if label:
            put_text(cell, label, (20, 50), 0.8)
        return cell

    vis_cell = resize_keep_aspect_pad(vis_boxes_bgr, cell_w, cell_h)
    label = "vis (red=FIT, green=EXIF)"
    if force_dual or (dual_th_bgr is not None) or (img_dual_bgr is not None):
        label = "vis (red=FIT, green=EXIF, cyan=DUAL)"
    put_text(vis_cell, label, (20, 50), 0.70)

    ir_cell = resize_keep_aspect_pad(th_bgr, cell_w, cell_h)
    put_text(ir_cell, "ir", (20, 50), 1.0)

    exif_cell = blank_cell("image-exif (none)")
    if img_exif_bgr is not None:
        exif_cell = resize_keep_aspect_pad(img_exif_bgr, cell_w, cell_h)
        put_text(exif_cell, "image-exif", (20, 50), 1.0)

    fit_cell = resize_keep_aspect_pad(img_fit_bgr, cell_w, cell_h)
    put_text(fit_cell, "image-fit", (20, 50), 1.0)

    dual_th_cell = blank_cell("ir-dual (none)")
    if dual_th_bgr is not None:
        dual_th_cell = resize_keep_aspect_pad(dual_th_bgr, cell_w, cell_h)
        put_text(dual_th_cell, "ir-dual", (20, 50), 1.0)

    dual_rgb_cell = blank_cell("image-dual (none)")
    if img_dual_bgr is not None:
        dual_rgb_cell = resize_keep_aspect_pad(img_dual_bgr, cell_w, cell_h)
        put_text(dual_rgb_cell, "image-dual", (20, 50), 1.0)

    ov_exif_cell = blank_cell("overlay-exif (none)")
    if img_exif_bgr is not None:
        ov_exif = overlay_thermal_rgb(th_bgr, img_exif_bgr, alpha_th=0.60)
        ov_exif_cell = resize_keep_aspect_pad(ov_exif, cell_w, cell_h)
        put_text(ov_exif_cell, "overlay-exif", (20, 50), 1.0)

    ov_fit = overlay_thermal_rgb(th_bgr, img_fit_bgr, alpha_th=0.60)
    ov_fit_cell = resize_keep_aspect_pad(ov_fit, cell_w, cell_h)
    put_text(ov_fit_cell, "overlay-fit", (20, 50), 1.0)

    ov_dual_cell = blank_cell("overlay-dual (none)")
    if (dual_th_bgr is not None) and (img_dual_bgr is not None):
        ov_dual = overlay_thermal_rgb(dual_th_bgr, img_dual_bgr, alpha_th=0.60)
        ov_dual_cell = resize_keep_aspect_pad(ov_dual, cell_w, cell_h)
        put_text(ov_dual_cell, "overlay-dual", (20, 50), 1.0)

    if not (force_dual or (dual_th_bgr is not None) or (img_dual_bgr is not None)):
        # Default comparison view keeps fit/exif only and hides dual placeholders.
        row1 = np.concatenate([vis_cell, ir_cell, exif_cell], axis=1)
        row2 = np.concatenate([fit_cell, ov_exif_cell, ov_fit_cell], axis=1)
        return np.concatenate([row1, row2], axis=0)

    row1 = np.concatenate([vis_cell, ir_cell, dual_th_cell], axis=1)
    row2 = np.concatenate([exif_cell, fit_cell, dual_rgb_cell], axis=1)
    row3 = np.concatenate([ov_exif_cell, ov_fit_cell, ov_dual_cell], axis=1)
    return np.concatenate([row1, row2, row3], axis=0)


# ----------------------------
# Pairing / IO
# ----------------------------

@dataclass
class PairItem:
    stem: str
    rgb_path: Path
    th_path: Path


def find_pairs(rgb_dir: Path, th_dir: Path, logger: Logger) -> List[PairItem]:
    rgb_files = [p for p in rgb_dir.glob("*") if p.is_file() and is_image_file(p)]
    th_files = [p for p in th_dir.glob("*") if p.is_file() and is_image_file(p)]
    rgb_map: Dict[str, Path] = {stem_key(p): p for p in rgb_files}

    pairs: List[PairItem] = []
    for p in th_files:
        k = stem_key(p)
        if k in rgb_map:
            pairs.append(PairItem(stem=k, rgb_path=rgb_map[k], th_path=p))

    pairs.sort(key=lambda x: x.stem)
    logger.log("INFO", f"Found matched pairs={len(pairs)}")
    return pairs


def load_pair_images(pair: PairItem) -> Tuple[np.ndarray, np.ndarray]:
    rgb_pil = exif_transposed_pil(pair.rgb_path)
    th_pil = exif_transposed_pil(pair.th_path)
    return pil_to_bgr(rgb_pil), pil_to_bgr(th_pil)


def save_png(path: Path, bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), bgr)


def robust_median(vals: List[float]) -> float:
    return float(np.median(np.array(vals, dtype=np.float32)))


def robust_mean(vals: List[float]) -> float:
    return float(np.mean(np.array(vals, dtype=np.float32)))


def aggregate_fit_vals(vals: List[float], mode: str) -> float:
    mode = str(mode).strip().lower()
    if mode == "mean":
        return robust_mean(vals)
    return robust_median(vals)


def _stable_rand01(tag: str, seed: int) -> float:
    b = f"{seed}|{tag}".encode("utf-8", errors="ignore")
    h = hashlib.md5(b).hexdigest()[:8]
    return (int(h, 16) % 1000000) / 1000000.0


def exif_perturb_zoom_ratio(
    zr: Optional[float],
    pair_stem: str,
    *,
    noise_pct: float,
    missing: bool,
    seed: int,
) -> Optional[float]:
    if zr is None or zr <= 0:
        return None
    if missing:
        return None
    p = float(noise_pct)
    if p <= 0:
        return float(zr)
    u = _stable_rand01(pair_stem, seed)
    scale = 1.0 + ((2.0 * u) - 1.0) * (p / 100.0)
    scale = max(0.05, scale)
    return float(zr) * float(scale)


def output_rgb_filename(pair: PairItem) -> str:
    suf = pair.rgb_path.suffix.lower()
    if suf in {".jpg", ".jpeg"}:
        return pair.rgb_path.name
    return f"{pair.stem}.jpg"


def output_th_filename(pair: PairItem) -> str:
    return pair.th_path.name



# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", type=str, required=True)
    ap.add_argument("--th_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # performance / UX
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--comparison", action="store_true",
                    help="If set, output comparison montages (slower).")

    # user-requested defaults
    ap.add_argument("--align", type=str, default="both", choices=["exif", "fit", "ecc", "both", "all", "raw"],
                    help="Which aligned outputs to generate: exif/fit/ecc/both/all/raw (default both).")
    ap.add_argument("--stage", type=str, default="both", choices=["fit", "apply", "both"],
                    help="Which stages to run: fit/apply/both (default both).")
    ap.add_argument(
        "--fit_k_mode",
        type=str,
        default="full",
        choices=["full", "no_kupdate", "naive_k"],
        help="FIT metadata mode: full/no_kupdate/naive_k (default: full).",
    )
    ap.add_argument(
        "--fit_agg_mode",
        type=str,
        default="median",
        choices=["median", "mean", "per_pair"],
        help="FIT aggregation mode: median/mean/per_pair (default: median).",
    )
    ap.add_argument(
        "--exif_noise_pct",
        type=float,
        default=0.0,
        help="Optional EXIF zoom perturbation percent (default: 0.0).",
    )
    ap.add_argument(
        "--exif_missing",
        action="store_true",
        help="Force EXIF zoom missing (for robustness stress tests).",
    )
    ap.add_argument(
        "--exif_noise_seed",
        type=int,
        default=0,
        help="Seed for deterministic EXIF perturbation (default: 0).",
    )

    # ECC / SfM-safe options (default OFF; only used when align includes ecc)
    ap.add_argument("--ecc_motion", type=str, default="homography", choices=["homography", "affine"],
                    help="ECC motion model for estimation (default: homography; only for estimation).")
    ap.add_argument("--ecc_iter", type=int, default=80, help="ECC max iterations (default: 80).")
    ap.add_argument("--ecc_eps", type=float, default=1e-6, help="ECC convergence epsilon (default: 1e-6).")
    ap.add_argument("--ecc_init", type=str, default="auto", choices=["auto", "fit", "exif", "center"],
                    help="ECC initialization source: auto/fit/exif/center (default: auto).")
    ap.add_argument("--structure_mode", type=str, default="sobel_mag",
                    choices=["sobel_mag", "rank_census_lite"],
                    help="Structure representation for ECC/quality (default: sobel_mag).")
    ap.add_argument("--ecc_q_min", type=float, default=None,
                    help="If set, mark frames with q_total < threshold as low_quality (default: None).")
    ap.add_argument("--dual", action="store_true",
                    help="Enable dual-output strategy: RGB SfM-safe + thermal aligned (default: off).")
    ap.add_argument("--dual_frames", type=int, default=60,
                    help="Frames used to estimate global H for dual-output (default: 60).")
    ap.add_argument("--dual_q_min", type=float, default=0.05,
                    help="Quality threshold for global H candidates (default: 0.05).")
    ap.add_argument("--sfm_allow_rot", action="store_true",
                    help="Allow a global small rotation for SfM-safe output (default: off).")
    ap.add_argument("--sfm_rot_deg", type=float, default=0.0,
                    help="Global rotation angle in degrees when --sfm_allow_rot is set (default: 0).")
    ap.add_argument("--sfm_max_rot_deg", type=float, default=2.0,
                    help="Clamp |rotation| <= this value in degrees (default: 2).")

    args = ap.parse_args()
    if float(args.exif_noise_pct) < 0.0:
        ap.error("--exif_noise_pct must be >= 0")

    rgb_dir = Path(args.rgb_dir)
    th_dir = Path(args.th_dir)
    out_dir = Path(args.out_dir)
    suffix = out_dir.name

    # flags
    want_raw = (args.align == "raw")
    want_comp = bool(args.comparison and not want_raw)
    want_fit = args.align in ("fit", "both", "all")
    want_exif = args.align in ("exif", "both", "all")
    want_ecc = args.align in ("ecc", "all")
    do_fit = args.stage in ("fit", "both")
    do_apply = args.stage in ("apply", "both")
    fit_k_mode = str(args.fit_k_mode).strip().lower()
    fit_agg_mode = str(args.fit_agg_mode).strip().lower()
    exif_noise_pct = float(args.exif_noise_pct)
    exif_missing = bool(args.exif_missing)
    exif_noise_seed = int(args.exif_noise_seed)

    ecc_motion = str(args.ecc_motion).strip().lower()
    ecc_init = str(args.ecc_init).strip().lower()
    structure_mode = str(args.structure_mode).strip().lower()
    ecc_q_min = args.ecc_q_min
    want_dual = bool(args.dual)
    dual_frames = int(args.dual_frames)
    dual_q_min = float(args.dual_q_min)
    allow_rot = bool(args.sfm_allow_rot)
    theta_req = float(args.sfm_rot_deg)
    theta_deg = theta_req if allow_rot else 0.0
    max_theta = float(args.sfm_max_rot_deg)
    if allow_rot and abs(theta_deg) > max_theta:
        theta_deg = max(-max_theta, min(max_theta, theta_deg))

    dual_struct_mode = structure_mode
    if want_dual and structure_mode == "sobel_mag":
        dual_struct_mode = "rank_census_lite"

    # comparison needs boxes; boxes need fit model
    need_fit_model = want_fit or want_comp or want_dual or (want_ecc and args.ecc_init in ("fit", "auto"))
    need_exif_probe = want_exif or need_fit_model or (want_ecc and args.ecc_init in ("exif", "auto"))  # exif can help seed fov candidates

    # paths
    debug_dir = out_dir / "debug"
    model_dir = out_dir / "model"
    img_root = out_dir / "image"
    img_fit_dir = img_root / "image-fit"
    img_exif_dir = img_root / "image-exif"
    img_ecc_dir = img_root / "image-ecc"
    img_dual_dir = img_root / "image-dual"
    comp_dir = img_root / "comparison"
    sidecar_ecc_dir = out_dir / "sidecar" / "ecc"
    sidecar_dual_dir = out_dir / "sidecar" / "dual"
    th_root = out_dir / "thermal"
    th_dual_dir = th_root / "thermal-dual"
    model_dual_path = model_dir / "model_dual.json"

    ensure_dir(debug_dir)
    ensure_dir(model_dir)

    logger = Logger(debug_dir / f"run-{suffix}.log")
    exif_audit_path = debug_dir / f"exif_audit-{suffix}.jsonl"
    per_image_path = debug_dir / f"per_image-{suffix}.jsonl"
    summary_path = debug_dir / f"summary-{suffix}.json"

    logger.log("INFO", "RUN START")
    logger.log("INFO", f"rgb_dir={rgb_dir}")
    logger.log("INFO", f"th_dir={th_dir}")
    logger.log("INFO", f"out_dir={out_dir}")
    logger.log("INFO", f"stage={args.stage} align={args.align} comparison={want_comp}")
    logger.log("INFO", f"piexif_installed={HAS_PIEXIF}")
    logger.log("INFO", f"exiftool_found={bool(find_exiftool())}")
    logger.log("INFO", f"[FIT] agg_mode={fit_agg_mode} k_mode={fit_k_mode}")
    if exif_missing or exif_noise_pct > 0.0:
        logger.log(
            "INFO",
            f"[EXIF_STRESS] missing={exif_missing} noise_pct={exif_noise_pct:.3f} seed={exif_noise_seed}",
        )
    if want_raw and args.comparison:
        logger.log("WARN", "[RAW] comparison output is ignored in --align raw mode.")
    if want_ecc:
        if allow_rot and theta_req != theta_deg:
            logger.log("WARN", f"[ECC] sfm_rot_deg clamped from {theta_req:.3f} to {theta_deg:.3f} (max={max_theta:.3f})")
        logger.log("INFO", f"[ECC] motion={ecc_motion} init={ecc_init} struct={structure_mode} "
                           f"iter={int(args.ecc_iter)} eps={float(args.ecc_eps):.2e} "
                           f"sfm_rot_deg={'0.0 (locked)' if not allow_rot else f'{theta_deg:.3f}'} "
                           f"q_min={'None' if ecc_q_min is None else ecc_q_min}")
    if want_dual:
        logger.log("INFO", f"[DUAL] enabled frames={dual_frames} q_min={dual_q_min} "
                           f"struct={dual_struct_mode} "
                           f"sfm_rot_deg={'0.0 (locked)' if not allow_rot else f'{theta_deg:.3f}'}")

    pairs_all = find_pairs(rgb_dir, th_dir, logger)
    pairs_found = len(pairs_all)
    if not pairs_all:
        logger.log("ERROR", "No matched pairs found. Check filenames by stem.")
        logger.close()
        return 2

    if want_raw:
        summary = {
            "align": args.align,
            "stage": args.stage,
            "comparison": bool(want_comp),
            "raw_mode": True,
            "counts": {
                "pairs_found": int(pairs_found),
                "pairs_processed": int(pairs_found),
                "fit_written": 0,
                "exif_written": 0,
                "ecc_written": 0,
                "dual_rgb_written": 0,
                "dual_th_written": 0,
            },
            "outputs": {
                "image_fit_dir": None,
                "image_exif_dir": None,
                "image_ecc_dir": None,
                "image_dual_dir": None,
                "thermal_dual_dir": None,
                "comparison_dir": None,
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.log("INFO", "[RAW] No aligned outputs requested. Original RGB should be used directly by the caller.")
        logger.log("INFO", f"[SUMMARY] summary={summary_path}")
        logger.log("INFO", "RUN END")
        logger.close()
        return 0

    pairs = pairs_all
    if args.samples is not None and args.samples > 0 and args.samples < len(pairs):
        rnd = random.Random(0)
        pairs = rnd.sample(pairs, args.samples)
        pairs.sort(key=lambda x: x.stem)
        logger.log("INFO", f"Processing SAMPLED pairs={len(pairs)} (--samples={args.samples})")
    else:
        logger.log("INFO", f"Processing ALL pairs={len(pairs)} (no --samples)")

    rgb0, th0 = load_pair_images(pairs[0])
    th_h0, th_w0 = th0.shape[:2]
    rgb_h0, rgb_w0 = rgb0.shape[:2]
    th_size = (th_w0, th_h0)
    logger.log("INFO", f"Thermal size={th_w0}x{th_h0}, RGB(example)={rgb_w0}x{rgb_h0}")

    # ---------------- EXIF probe (for exif output & fit initialization) ----------------
    exif_ok_count = 0
    exif_zoom_ratios: List[float] = []
    exif_fov_fracs_w: List[float] = []

    exif_usable = False
    zr_med: Optional[float] = None
    fov_exif_med: Optional[float] = None
    probe_n = min(30, len(pairs)) if need_exif_probe else 0

    if need_exif_probe:
        for i in range(probe_n):
            p = pairs[i]
            rgb_exif = read_exif_lens_info(p.rgb_path)
            th_exif = read_exif_lens_info(p.th_path)
            zr = compute_zoom_ratio(rgb_exif, th_exif)
            zr = exif_perturb_zoom_ratio(
                zr,
                p.stem,
                noise_pct=exif_noise_pct,
                missing=exif_missing,
                seed=exif_noise_seed,
            )
            if zr is None or zr <= 0:
                continue
            exif_ok_count += 1
            exif_zoom_ratios.append(zr)
            exif_fov_fracs_w.append(1.0 / zr)

        exif_usable = exif_ok_count >= max(3, int(0.2 * probe_n))
        if exif_usable:
            zr_med = robust_median(exif_zoom_ratios)
            fov_exif_med = robust_median(exif_fov_fracs_w)
            logger.log("INFO", f"[EXIF] usable=True probe_ok={exif_ok_count}/{probe_n} "
                               f"zoom_ratio(med)={zr_med:.4f} fov_frac_w(med)={fov_exif_med:.4f}")
        else:
            logger.log("WARN", f"[EXIF] usable=False probe_ok={exif_ok_count}/{probe_n}. "
                               f"image-exif will be skipped unless --align exif is requested.")
    else:
        logger.log("INFO", "[EXIF] probe skipped (not needed by current flags).")

    if want_exif and not exif_usable:
        logger.log("ERROR", "[EXIF] align=exif/both requested but EXIF zoom ratio is not usable on this dataset.")
        logger.close()
        return 3

    # ---------------- FIT estimation / load model ----------------
    fit_fov_med: float = 1.0
    fit_cx_med: float = 0.0
    fit_cy_med: float = 0.0
    fov_candidates: List[float] = []
    ok_fovs: List[float] = []
    ok_cxoffs: List[float] = []
    ok_cyoffs: List[float] = []

    model_fit_path = model_dir / "model_fit.json"
    model_exif_path: Optional[Path] = None

    t_fit = 0.0
    if need_fit_model:
        if do_fit:
            logger.log("INFO", "[FIT] Estimating global FIT model...")
            if exif_usable and fov_exif_med is not None:
                base = float(fov_exif_med)
                fov_candidates = sorted({max(0.15, min(0.95, base * k))
                                         for k in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]})
            else:
                fov_candidates = [round(x, 3) for x in np.linspace(0.20, 0.95, 16).tolist()]
            logger.log("INFO", f"[FIT] fov_candidates={fov_candidates}")

            t_fit0 = time.perf_counter()
            for idx, pair in enumerate(pairs, start=1):
                t0 = time.perf_counter()
                rgb_bgr, th_bgr = load_pair_images(pair)
                r = estimate_fit_on_pair(rgb_bgr, th_bgr, th_size=th_size, fov_candidates=fov_candidates)
                dt = time.perf_counter() - t0
                if r.ok and r.fov_frac is not None and r.cx_off_frac is not None and r.cy_off_frac is not None:
                    ok_fovs.append(float(r.fov_frac))
                    ok_cxoffs.append(float(r.cx_off_frac))
                    ok_cyoffs.append(float(r.cy_off_frac))
                    logger.log("INFO", f"[FIT] {pair.stem} ({idx}/{len(pairs)}) OK "
                                       f"fov={r.fov_frac:.3f} match={r.match_score:.3f} ncc={r.ncc:.3f} "
                                       f"cx_off={r.cx_off_frac:.5f} cy_off={r.cy_off_frac:.5f} ({dt:.2f}s)")
                else:
                    logger.log("INFO", f"[FIT] {pair.stem} ({idx}/{len(pairs)}) FAIL reason={r.reason} ({dt:.2f}s)")
            t_fit = time.perf_counter() - t_fit0

            if len(ok_fovs) < max(3, int(0.1 * len(pairs))):
                if exif_usable and fov_exif_med is not None:
                    fit_fov_med = float(fov_exif_med)
                    fit_cx_med = 0.0
                    fit_cy_med = 0.0
                    logger.log("WARN", f"[FIT] Too few OK fits. Fallback to EXIF fov={fit_fov_med:.4f}, offsets=0.")
                else:
                    fit_fov_med = 1.0
                    fit_cx_med = 0.0
                    fit_cy_med = 0.0
                    logger.log("ERROR", "[FIT] Too few OK fits and no EXIF. Fallback to fov=1.0.")
            else:
                agg_global = "median" if fit_agg_mode == "per_pair" else fit_agg_mode
                fit_fov_med = aggregate_fit_vals(ok_fovs, agg_global)
                fit_cx_med = aggregate_fit_vals(ok_cxoffs, agg_global)
                fit_cy_med = aggregate_fit_vals(ok_cyoffs, agg_global)

            logger.log("INFO", f"[FIT] Done. time={t_fit:.1f}s ok={len(ok_fovs)}/{len(pairs)} "
                               f"fit_fov_med={fit_fov_med:.4f} fit_cx_off_med={fit_cx_med:.5f} fit_cy_off_med={fit_cy_med:.5f}")

            model_fit = {
                "version": 10,
                "thermal_size": {"w": th_w0, "h": th_h0},
                "fit": {
                    "fov_frac_w": float(fit_fov_med),
                    "cx_off_frac": float(fit_cx_med),
                    "cy_off_frac": float(fit_cy_med),
                    "ok_count": int(len(ok_fovs)),
                    "total_count": int(len(pairs)),
                    "candidates": fov_candidates,
                    "agg_mode": fit_agg_mode,
                },
            }
            with open(model_fit_path, "w", encoding="utf-8") as f:
                json.dump(model_fit, f, ensure_ascii=False, indent=2)
        else:
            # load existing model_fit
            if not model_fit_path.exists():
                logger.log("ERROR", f"[FIT] stage=apply but model_fit.json not found at {model_fit_path}")
                logger.close()
                return 4
            try:
                model_fit = json.loads(model_fit_path.read_text(encoding="utf-8", errors="ignore"))
                fit_fov_med = float(model_fit["fit"]["fov_frac_w"])
                fit_cx_med = float(model_fit["fit"]["cx_off_frac"])
                fit_cy_med = float(model_fit["fit"]["cy_off_frac"])
                fit_agg_mode = str(model_fit.get("fit", {}).get("agg_mode", fit_agg_mode)).strip().lower()
                logger.log("INFO", f"[FIT] Loaded model_fit.json fov={fit_fov_med:.4f} cx_off={fit_cx_med:.5f} cy_off={fit_cy_med:.5f}")
            except Exception as e:
                logger.log("ERROR", f"[FIT] Failed to load model_fit.json: {e}")
                logger.close()
                return 5
    else:
        logger.log("INFO", "[FIT] Skipped (fit model not required by current flags).")

    # save model_exif when fitting and user wants exif/both
    if do_fit and want_exif and exif_usable and zr_med is not None and fov_exif_med is not None:
        model_exif = {
            "version": 10,
            "thermal_size": {"w": th_w0, "h": th_h0},
            "exif": {
                "zoom_ratio_med": float(zr_med),
                "fov_frac_w_med": float(fov_exif_med),
                "probe_ok": int(exif_ok_count),
                "probe_n": int(probe_n),
            },
        }
        model_exif_path = model_dir / "model_exif.json"
        with open(model_exif_path, "w", encoding="utf-8") as f:
            json.dump(model_exif, f, ensure_ascii=False, indent=2)

    # ---------------- DUAL global estimation (RGB SfM-safe + thermal aligned) ----------------
    dual_H_global = None
    dual_S = None
    dual_s = None
    dual_tx = None
    dual_ty = None
    dual_box_global = None
    dual_h_safe2thermal = None
    dual_h_thermal2safe = None
    dual_selected = None
    dual_stats_path = debug_dir / f"dual_stats-{suffix}.json"

    if want_dual and do_apply:
        logger.log("INFO", f"[DUAL] Estimating global H (samples={dual_frames}, q_min={dual_q_min}) ...")
        sample_indices = select_even_indices(len(pairs), dual_frames)
        dual_candidates_ok: List[Dict[str, object]] = []
        dual_candidates_any: List[Dict[str, object]] = []
        dual_stats = {
            "samples": int(len(sample_indices)),
            "candidates": 0,
            "q_total": RunningStat(),
            "grad_ncc": RunningStat(),
            "edge_f1": RunningStat(),
            "ecc_score": RunningStat(),
            "cands": [],
        }
        dual_min_ok = max(6, int(0.2 * len(sample_indices))) if len(sample_indices) > 0 else 0

        for s_idx, p_idx in enumerate(sample_indices, start=1):
            pair = pairs[p_idx]
            rgb_bgr, th_bgr = load_pair_images(pair)
            rgb_h, rgb_w = rgb_bgr.shape[:2]
            cx_fit = rgb_w / 2.0 + fit_cx_med * rgb_w
            cy_fit = rgb_h / 2.0 + fit_cy_med * rgb_h
            init_box = crop_box_from_fov(rgb_w, rgb_h, th_w0, th_h0, float(fit_fov_med), cx_fit, cy_fit)
            H_init = homography_from_crop_box(init_box, th_w0, th_h0)

            ecc_res = estimate_ecc_on_pair(
                rgb_bgr=rgb_bgr,
                th_bgr=th_bgr,
                H_init=H_init,
                motion=ecc_motion,
                ecc_iter=int(args.ecc_iter),
                ecc_eps=float(args.ecc_eps),
                structure_mode=dual_struct_mode,
            )

            ecc_ok = bool(ecc_res.ok and ecc_res.h_raw is not None)
            H_cand = ecc_res.h_raw if (ecc_ok and ecc_res.h_raw is not None) else H_init
            ecc_score = float(ecc_res.ecc_score) if (ecc_ok and ecc_res.ecc_score is not None) else None

            grad_ncc, edge_f1 = _quality_for_h(rgb_bgr, th_bgr, H_cand, dual_struct_mode)
            q_total = quality_combine(ecc_score, grad_ncc, edge_f1)

            dual_stats["q_total"].add(q_total)
            dual_stats["grad_ncc"].add(grad_ncc)
            dual_stats["edge_f1"].add(edge_f1)
            dual_stats["ecc_score"].add(ecc_score)

            info = {
                "stem": pair.stem,
                "ecc_ok": bool(ecc_ok),
                "ecc_reason": ecc_res.reason,
                "ecc_score": float(ecc_score) if ecc_score is not None else None,
                "grad_ncc": float(grad_ncc) if grad_ncc is not None else None,
                "edge_f1": float(edge_f1) if edge_f1 is not None else None,
                "q_total": float(q_total) if q_total is not None else None,
            }
            dual_stats["cands"].append(info)

            if q_total is not None and q_total >= float(dual_q_min):
                cand = {
                    "stem": pair.stem,
                    "H": H_cand,
                    "q_total": float(q_total),
                    "ecc_ok": bool(ecc_ok),
                    "ecc_score": float(ecc_score) if ecc_score is not None else None,
                    "grad_ncc": float(grad_ncc) if grad_ncc is not None else None,
                    "edge_f1": float(edge_f1) if edge_f1 is not None else None,
                }
                dual_candidates_any.append(cand)
                if ecc_ok:
                    dual_candidates_ok.append(cand)

            if s_idx % max(1, len(sample_indices) // 5) == 0:
                logger.log("INFO", f"[DUAL] sample {s_idx}/{len(sample_indices)} stem={pair.stem} "
                                   f"ecc_ok={ecc_ok} q_total={q_total if q_total is not None else 'None'}")

        use_candidates = dual_candidates_ok if len(dual_candidates_ok) >= dual_min_ok else dual_candidates_ok
        if len(use_candidates) < max(3, int(0.1 * len(sample_indices))):
            use_candidates = dual_candidates_any
        dual_stats["candidates"] = int(len(use_candidates))
        if len(dual_candidates_ok) >= dual_min_ok:
            logger.log("INFO", f"[DUAL] Using ECC-ok candidates: {len(dual_candidates_ok)}/{len(sample_indices)}")
        elif dual_candidates_ok:
            logger.log("WARN", f"[DUAL] Few ECC-ok candidates ({len(dual_candidates_ok)}). Using available ECC-ok set.")
        else:
            logger.log("WARN", "[DUAL] No ECC-ok candidates; falling back to any candidates (may resemble FIT).")

        best = _select_global_h_by_corner_median(use_candidates, rgb_w0, rgb_h0)
        if best is None and use_candidates:
            def _qkey(item: Dict[str, object]) -> float:
                q = item.get("q_total")
                try:
                    return float(q)  # type: ignore[arg-type]
                except Exception:
                    return -1e9
            use_candidates.sort(key=_qkey, reverse=True)
            best = use_candidates[0]

        if best is not None and best.get("H") is not None:
            dual_H_global = best.get("H")
            qv = best.get("q_total")
            ev = best.get("ecc_score")
            dual_selected = {
                "stem": best.get("stem"),
                "q_total": float(qv) if qv is not None else None,
                "ecc_score": float(ev) if ev is not None else None,
            }
            logger.log("INFO", f"[DUAL] Global H selected from stem={dual_selected.get('stem')} "
                               f"q_total={dual_selected.get('q_total')}")
        else:
            cx_fit0 = rgb_w0 / 2.0 + fit_cx_med * rgb_w0
            cy_fit0 = rgb_h0 / 2.0 + fit_cy_med * rgb_h0
            init_box0 = crop_box_from_fov(rgb_w0, rgb_h0, th_w0, th_h0, float(fit_fov_med), cx_fit0, cy_fit0)
            dual_H_global = homography_from_crop_box(init_box0, th_w0, th_h0)
            dual_selected = {"stem": None, "q_total": None, "ecc_score": None}
            logger.log("WARN", "[DUAL] No valid candidates; fallback to fit-derived H.")

        if dual_H_global is not None:
            S_dual, s_dual, tx_dual, ty_dual = fit_sfm_safe_similarity_from_h(
                dual_H_global, rgb_w0, rgb_h0, th_w0, th_h0, theta_deg if allow_rot else 0.0
            )
            dual_S = S_dual
            dual_s, dual_tx, dual_ty = float(s_dual), float(tx_dual), float(ty_dual)
            if not allow_rot or abs(theta_deg) <= 1e-6:
                dual_box_global = crop_box_from_similarity(dual_s, dual_tx, dual_ty, th_w0, th_h0)
            try:
                S_inv = np.linalg.inv(dual_S)
                dual_h_safe2thermal = _normalize_homography(_mat33_from_affine(dual_H_global) @ S_inv)
                dual_h_thermal2safe = np.linalg.inv(dual_h_safe2thermal)
            except Exception:
                dual_h_safe2thermal = None
                dual_h_thermal2safe = None

            try:
                dual_stats_out = {
                    "samples": dual_stats["samples"],
                    "candidates": dual_stats["candidates"],
                    "selected": dual_selected,
                    "q_total": dual_stats["q_total"].as_dict(),
                    "grad_ncc": dual_stats["grad_ncc"].as_dict(),
                    "edge_f1": dual_stats["edge_f1"].as_dict(),
                    "ecc_score": dual_stats["ecc_score"].as_dict(),
                    "cands": dual_stats["cands"],
                }
                dual_stats_path.write_text(json.dumps(dual_stats_out, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                logger.log("WARN", f"[DUAL] Failed to write dual_stats: {e}")

            try:
                dual_model = {
                    "version": 1,
                    "thermal_size": {"w": th_w0, "h": th_h0},
                    "rgb_size": {"w": rgb_w0, "h": rgb_h0},
                    "H_global": _mat33_from_affine(dual_H_global).tolist(),
                    "S_sfm_safe": _mat33_from_affine(dual_S).tolist() if dual_S is not None else None,
                    "H_safe2thermal": dual_h_safe2thermal.tolist() if dual_h_safe2thermal is not None else None,
                    "H_thermal2safe": dual_h_thermal2safe.tolist() if dual_h_thermal2safe is not None else None,
                    "sfm_safe": {"allow_rot": bool(allow_rot), "theta_deg": float(theta_deg)},
                    "dual_frames": int(dual_frames),
                    "dual_q_min": float(dual_q_min),
                    "structure_mode": structure_mode,
                    "selected": dual_selected,
                }
                model_dual_path.write_text(json.dumps(dual_model, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                logger.log("WARN", f"[DUAL] Failed to write model_dual: {e}")

    # ---------------- APPLY ----------------
    fit_written = 0
    exif_written = 0
    ecc_written = 0
    dual_rgb_written = 0
    dual_th_written = 0
    t_apply = 0.0

    if do_apply:
        logger.log("INFO", "[APPLY] Writing image-fit / image-exif / image-ecc / image-dual (optional) / comparison (optional) ...")

        if want_fit:
            ensure_dir(img_fit_dir)
        if want_exif and exif_usable:
            ensure_dir(img_exif_dir)
        if want_ecc:
            ensure_dir(img_ecc_dir)
            ensure_dir(sidecar_ecc_dir)
        if want_dual:
            ensure_dir(img_dual_dir)
            ensure_dir(th_dual_dir)
            ensure_dir(sidecar_dual_dir)
        if want_comp:
            ensure_dir(comp_dir)

        per_fp = open(per_image_path, "w", encoding="utf-8", errors="ignore")

        ecc_stats = {
            "count": 0,
            "ok": 0,
            "fail": 0,
            "reasons": {},
            "ecc_score": RunningStat(),
            "q_total": RunningStat(),
            "grad_ncc": RunningStat(),
            "edge_f1": RunningStat(),
            "worst_q": [],  # list of dicts
            "best_q": [],
            "fail_samples": [],
        }
        ecc_stats_path = debug_dir / f"ecc_stats-{suffix}.json"
        max_keep = 30

        cell_w, cell_h = th_w0, th_h0

        t_apply0 = time.perf_counter()
        for idx, pair in enumerate(pairs, start=1):
            t0 = time.perf_counter()
            rgb_bgr, th_bgr = load_pair_images(pair)
            rgb_h, rgb_w = rgb_bgr.shape[:2]

            out_name = output_rgb_filename(pair)

            # Decide which crops are needed
            need_fit_crop = want_fit or want_comp or want_ecc
            need_exif_crop = (want_exif or want_comp or want_ecc) and exif_usable

            fit_box = None
            img_fit = None
            zoom_fit = None

            if need_fit_crop:
                fit_fov_cur = float(fit_fov_med)
                fit_cx_cur = float(fit_cx_med)
                fit_cy_cur = float(fit_cy_med)
                if fit_agg_mode == "per_pair":
                    r_pair = estimate_fit_on_pair(rgb_bgr, th_bgr, th_size=th_size, fov_candidates=fov_candidates)
                    if r_pair.ok and r_pair.fov_frac is not None and r_pair.cx_off_frac is not None and r_pair.cy_off_frac is not None:
                        fit_fov_cur = float(r_pair.fov_frac)
                        fit_cx_cur = float(r_pair.cx_off_frac)
                        fit_cy_cur = float(r_pair.cy_off_frac)
                    else:
                        logger.log("WARN", f"[FIT] per_pair fallback to global on {pair.stem}: reason={r_pair.reason}")
                cx_fit = rgb_w / 2.0 + fit_cx_cur * rgb_w
                cy_fit = rgb_h / 2.0 + fit_cy_cur * rgb_h
                fit_box = crop_box_from_fov(rgb_w, rgb_h, th_w0, th_h0, fit_fov_cur, cx_fit, cy_fit)
                fit_crop_w = max(1, int(fit_box[2] - fit_box[0]))
                zoom_fit = float(rgb_w) / float(fit_crop_w)
                crop_fit = crop_with_pad(rgb_bgr, fit_box)
                img_fit = cv2.resize(crop_fit, (th_w0, th_h0), interpolation=cv2.INTER_AREA)

            exif_box = None
            img_exif = None
            zoom_exif = None
            if need_exif_crop:
                rgb_exif = read_exif_lens_info(pair.rgb_path)
                th_exif = read_exif_lens_info(pair.th_path)
                zr = compute_zoom_ratio(rgb_exif, th_exif)
                zr = exif_perturb_zoom_ratio(
                    zr,
                    pair.stem,
                    noise_pct=exif_noise_pct,
                    missing=exif_missing,
                    seed=exif_noise_seed,
                )
                if zr is not None and zr > 0:
                    fov_exif_w = 1.0 / zr
                    exif_box = crop_box_from_fov(rgb_w, rgb_h, th_w0, th_h0, float(fov_exif_w),
                                                 rgb_w / 2.0, rgb_h / 2.0)
                    exif_crop_w = max(1, int(exif_box[2] - exif_box[0]))
                    zoom_exif = float(rgb_w) / float(exif_crop_w)
                    crop_exif = crop_with_pad(rgb_bgr, exif_box)
                    img_exif = cv2.resize(crop_exif, (th_w0, th_h0), interpolation=cv2.INTER_AREA)

            # ECC (estimate H, but output SfM-safe crop+resize)
            img_ecc = None
            zoom_ecc = None
            ecc_score = None
            ecc_ok = False
            ecc_reason = None
            ecc_init_source = None
            ecc_h_init = None
            ecc_h_raw = None
            ecc_h_safe2thermal = None
            ecc_warp_residual = None
            ecc_s_matrix = None
            grad_ncc_score = None
            edge_f1_score = None
            q_total = None
            ecc_quality_ok = None
            ecc_box = None
            ecc_used_init = False
            ecc_fallback_reason = None
            ecc_q_init = None

            # DUAL outputs (global H + SfM-safe RGB + aligned thermal)
            img_dual = None
            th_dual = None
            zoom_dual = None
            dual_box = None
            dual_grad_ncc = None
            dual_edge_f1 = None
            dual_q_total = None
            dual_shift = None
            dual_shift_resp = None
            dual_shift_applied = False
            dual_invalid_ratio = None
            dual_fallback_reason = None
            dual_sidecar_path = None

            if want_ecc:
                init_box = None
                if ecc_init == "fit":
                    if fit_box is not None:
                        ecc_init_source = "fit"
                        init_box = fit_box
                    elif exif_box is not None:
                        ecc_init_source = "exif_fallback"
                        init_box = exif_box
                    else:
                        ecc_init_source = "center_fallback"
                elif ecc_init == "exif":
                    if exif_box is not None:
                        ecc_init_source = "exif"
                        init_box = exif_box
                    elif fit_box is not None:
                        ecc_init_source = "fit_fallback"
                        init_box = fit_box
                    else:
                        ecc_init_source = "center_fallback"
                elif ecc_init == "auto":
                    if fit_box is not None:
                        ecc_init_source = "fit"
                        init_box = fit_box
                    elif exif_box is not None:
                        ecc_init_source = "exif"
                        init_box = exif_box
                    else:
                        ecc_init_source = "center"
                else:
                    ecc_init_source = "center"

                if init_box is None:
                    init_box = crop_box_from_fov(rgb_w, rgb_h, th_w0, th_h0, 1.0, rgb_w / 2.0, rgb_h / 2.0)

                # Precompute init crop for potential fallback + quality comparison
                init_crop = None
                init_crop_reason = None
                max_pad = int(max(rgb_w, rgb_h) * 0.5)
                max_pixels = int(rgb_w * rgb_h * 6)
                init_crop, init_crop_reason = safe_crop_with_pad(rgb_bgr, init_box, max_pad, max_pixels)
                init_img = None
                if init_crop is not None:
                    init_img = cv2.resize(init_crop, (th_w0, th_h0), interpolation=cv2.INTER_AREA)
                    try:
                        struct_th_init = build_structure(th_bgr, mode=structure_mode)
                        struct_init = build_structure(init_img, mode=structure_mode)
                        ecc_q_init = quality_combine(
                            None,
                            quality_grad_ncc(struct_th_init, struct_init),
                            quality_edge_f1(struct_th_init, struct_init),
                        )
                    except Exception:
                        pass
                else:
                    # If init crop failed, fallback to safe center crop
                    init_box = crop_box_from_fov(rgb_w, rgb_h, th_w0, th_h0, 1.0, rgb_w / 2.0, rgb_h / 2.0)
                    init_crop, init_crop_reason = safe_crop_with_pad(rgb_bgr, init_box, max_pad, max_pixels)
                    if init_crop is not None:
                        init_img = cv2.resize(init_crop, (th_w0, th_h0), interpolation=cv2.INTER_AREA)

                ecc_h_init = homography_from_crop_box(init_box, th_w0, th_h0)

                ecc_res = estimate_ecc_on_pair(
                    rgb_bgr=rgb_bgr,
                    th_bgr=th_bgr,
                    H_init=ecc_h_init,
                    motion=ecc_motion,
                    ecc_iter=int(args.ecc_iter),
                    ecc_eps=float(args.ecc_eps),
                    structure_mode=structure_mode,
                )
                if ecc_res.ok and ecc_res.h_raw is not None:
                    ecc_ok = True
                    ecc_score = ecc_res.ecc_score
                    ecc_h_raw = ecc_res.h_raw
                    ecc_warp_residual = ecc_res.warp_residual
                else:
                    ecc_ok = False
                    ecc_reason = ecc_res.reason
                    ecc_h_raw = ecc_h_init

                H_for_s = ecc_h_raw if ecc_h_raw is not None else ecc_h_init
                S, s, tx, ty = fit_sfm_safe_similarity_from_h(
                    H_for_s, rgb_w, rgb_h, th_w0, th_h0, theta_deg if allow_rot else 0.0
                )
                ecc_s_matrix = S

                if allow_rot and abs(theta_deg) > 1e-6:
                    interp = cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA
                    img_ecc = cv2.warpAffine(
                        rgb_bgr,
                        S[:2],
                        (th_w0, th_h0),
                        flags=interp,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    crop_w = float(th_w0) / float(s) if s > 1e-8 else None
                else:
                    ecc_box = crop_box_from_similarity(s, tx, ty, th_w0, th_h0)
                    crop_ecc, crop_reason = safe_crop_with_pad(rgb_bgr, ecc_box, max_pad, max_pixels)
                    if crop_ecc is None:
                        ecc_ok = False
                        ecc_fallback_reason = f"ecc_crop_invalid:{crop_reason}"
                    else:
                        img_ecc = cv2.resize(crop_ecc, (th_w0, th_h0), interpolation=cv2.INTER_AREA)
                        crop_w = float(ecc_box[2] - ecc_box[0])

                if crop_w and crop_w > 1e-8:
                    zoom_ecc = float(rgb_w) / float(crop_w)

                try:
                    S_inv = np.linalg.inv(S)
                    ecc_h_safe2thermal = _normalize_homography(_mat33_from_affine(H_for_s) @ S_inv)
                except Exception:
                    ecc_h_safe2thermal = None

                try:
                    if img_ecc is not None:
                        struct_th = build_structure(th_bgr, mode=structure_mode)
                        struct_rgb = build_structure(img_ecc, mode=structure_mode)
                        grad_ncc_score = quality_grad_ncc(struct_th, struct_rgb)
                        edge_f1_score = quality_edge_f1(struct_th, struct_rgb)
                        q_total = quality_combine(ecc_score, grad_ncc_score, edge_f1_score)
                        if ecc_q_min is not None:
                            ecc_quality_ok = bool(q_total >= float(ecc_q_min))
                            if not ecc_quality_ok:
                                ecc_ok = False
                                ecc_reason = "low_quality"
                except Exception:
                    pass

                # If ECC output looks worse than init, fallback to init crop
                if init_img is not None:
                    if (q_total is None) or (ecc_q_init is not None and q_total + 1e-6 < float(ecc_q_init)):
                        img_ecc = init_img
                        ecc_box = init_box
                        ecc_used_init = True
                        if ecc_fallback_reason is None:
                            ecc_fallback_reason = "init_better"
                elif img_ecc is None:
                    # If ECC crop failed and init is unavailable, fall back to fit/exif image if present
                    if img_fit is not None:
                        img_ecc = img_fit
                        ecc_box = fit_box
                        ecc_used_init = True
                        if ecc_fallback_reason is None:
                            ecc_fallback_reason = "fallback_fit"
                    elif img_exif is not None:
                        img_ecc = img_exif
                        ecc_box = exif_box
                        ecc_used_init = True
                        if ecc_fallback_reason is None:
                            ecc_fallback_reason = "fallback_exif"

                ecc_stats["count"] += 1
                if ecc_ok:
                    ecc_stats["ok"] += 1
                else:
                    ecc_stats["fail"] += 1
                    if ecc_reason:
                        ecc_stats["reasons"][ecc_reason] = int(ecc_stats["reasons"].get(ecc_reason, 0)) + 1

                ecc_stats["ecc_score"].add(ecc_score)
                ecc_stats["q_total"].add(q_total)
                ecc_stats["grad_ncc"].add(grad_ncc_score)
                ecc_stats["edge_f1"].add(edge_f1_score)

                def _push_rank(lst, item, key, reverse=False):
                    lst.append(item)
                    lst.sort(key=key, reverse=reverse)
                    if len(lst) > max_keep:
                        lst.pop()

                if q_total is not None:
                    info = {
                        "stem": pair.stem,
                        "q_total": float(q_total),
                        "ecc_ok": bool(ecc_ok),
                        "ecc_score": float(ecc_score) if ecc_score is not None else None,
                        "grad_ncc": float(grad_ncc_score) if grad_ncc_score is not None else None,
                        "edge_f1": float(edge_f1_score) if edge_f1_score is not None else None,
                        "init": ecc_init_source,
                        "ecc_reason": ecc_reason,
                    }
                    _push_rank(ecc_stats["worst_q"], info, key=lambda x: x.get("q_total", 0.0), reverse=False)
                    _push_rank(ecc_stats["best_q"], info, key=lambda x: x.get("q_total", 0.0), reverse=True)

                if (not ecc_ok) and len(ecc_stats["fail_samples"]) < max_keep:
                    ecc_stats["fail_samples"].append({
                        "stem": pair.stem,
                        "ecc_reason": ecc_reason,
                        "init": ecc_init_source,
                        "ecc_score": float(ecc_score) if ecc_score is not None else None,
                        "fallback_reason": ecc_fallback_reason,
                    })

            # DUAL (global H -> SfM-safe RGB crop + aligned thermal)
            if want_dual and dual_S is not None and dual_H_global is not None:
                max_pad = int(max(rgb_w, rgb_h) * 0.5)
                max_pixels = int(rgb_w * rgb_h * 6)
                crop_w = None
                if allow_rot and abs(theta_deg) > 1e-6:
                    interp = cv2.INTER_LINEAR if (dual_s is not None and dual_s > 1.0) else cv2.INTER_AREA
                    img_dual = cv2.warpAffine(
                        rgb_bgr,
                        dual_S[:2],
                        (th_w0, th_h0),
                        flags=interp,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    if dual_s is not None and dual_s > 1e-8:
                        crop_w = float(th_w0) / float(dual_s)
                else:
                    if dual_s is not None and dual_tx is not None and dual_ty is not None:
                        dual_box = crop_box_from_similarity(dual_s, dual_tx, dual_ty, th_w0, th_h0)
                        crop_dual, crop_reason = safe_crop_with_pad(rgb_bgr, dual_box, max_pad, max_pixels)
                        if crop_dual is None:
                            dual_fallback_reason = f"dual_crop_invalid:{crop_reason}"
                        else:
                            img_dual = cv2.resize(crop_dual, (th_w0, th_h0), interpolation=cv2.INTER_AREA)
                            crop_w = float(dual_box[2] - dual_box[0])

                if img_dual is None:
                    if img_fit is not None:
                        img_dual = img_fit
                        dual_box = fit_box
                        zoom_dual = zoom_fit
                        if dual_fallback_reason is None:
                            dual_fallback_reason = "fallback_fit"
                    elif img_exif is not None:
                        img_dual = img_exif
                        dual_box = exif_box
                        zoom_dual = zoom_exif
                        if dual_fallback_reason is None:
                            dual_fallback_reason = "fallback_exif"

                if crop_w and crop_w > 1e-8 and zoom_dual is None:
                    zoom_dual = float(rgb_w) / float(crop_w)

                if dual_h_thermal2safe is not None:
                    try:
                        # Track valid region after warp, then inpaint invalid area to avoid black borders in thermal-dual.
                        valid_src = np.full((th_h0, th_w0), 255, dtype=np.uint8)
                        valid_mask = cv2.warpPerspective(
                            valid_src,
                            dual_h_thermal2safe,
                            (th_w0, th_h0),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        dual_invalid_ratio = float(np.mean(valid_mask <= 0))
                        th_dual = cv2.warpPerspective(
                            th_bgr,
                            dual_h_thermal2safe,
                            (th_w0, th_h0),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0),
                        )
                        if dual_invalid_ratio is not None and dual_invalid_ratio > 1e-6:
                            invalid_mask = (valid_mask <= 0).astype(np.uint8) * 255
                            th_dual = cv2.inpaint(th_dual, invalid_mask, 3, cv2.INPAINT_TELEA)
                    except Exception:
                        th_dual = None

                if img_dual is not None and th_dual is not None:
                    try:
                        struct_rgb = build_structure(img_dual, mode=dual_struct_mode)
                        struct_th = build_structure(th_dual, mode=dual_struct_mode)
                        dual_grad_ncc = quality_grad_ncc(struct_th, struct_rgb)
                        dual_edge_f1 = quality_edge_f1(struct_th, struct_rgb)
                        dual_q_total = quality_combine(None, dual_grad_ncc, dual_edge_f1)

                        shift = _phase_corr_shift(struct_rgb, struct_th, max_shift=3.0, min_resp=0.02)
                        if shift is not None:
                            dx, dy, resp = shift
                            dual_shift = (float(dx), float(dy))
                            dual_shift_resp = float(resp)
                            if abs(dx) > 0.2 or abs(dy) > 0.2:
                                M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
                                th_shift = cv2.warpAffine(
                                    th_dual,
                                    M,
                                    (th_w0, th_h0),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE,
                                )
                                struct_th_shift = build_structure(th_shift, mode=dual_struct_mode)
                                grad_ncc_shift = quality_grad_ncc(struct_th_shift, struct_rgb)
                                edge_f1_shift = quality_edge_f1(struct_th_shift, struct_rgb)
                                q_shift = quality_combine(None, grad_ncc_shift, edge_f1_shift)
                                if q_shift >= dual_q_total + 1e-4:
                                    th_dual = th_shift
                                    dual_grad_ncc = grad_ncc_shift
                                    dual_edge_f1 = edge_f1_shift
                                    dual_q_total = q_shift
                                    dual_shift_applied = True
                    except Exception:
                        pass

            out_fit_path = None
            out_exif_path = None
            out_ecc_path = None
            ecc_sidecar_path = None
            out_dual_rgb_path = None
            out_dual_th_path = None

            # Save fit output
            if want_fit and img_fit is not None:
                out_fit_path = img_fit_dir / out_name
                save_with_metadata(pair.rgb_path, out_fit_path, img_fit, th_w0, th_h0, zoom_fit, fit_k_mode, logger)
                fit_written += 1

            # Save exif output
            if want_exif and img_exif is not None and exif_usable:
                out_exif_path = img_exif_dir / out_name
                save_with_metadata(pair.rgb_path, out_exif_path, img_exif, th_w0, th_h0, zoom_exif, fit_k_mode, logger)
                exif_written += 1

            # Save ecc output + sidecar
            if want_ecc and img_ecc is not None:
                out_ecc_path = img_ecc_dir / out_name
                save_with_metadata(pair.rgb_path, out_ecc_path, img_ecc, th_w0, th_h0, zoom_ecc, fit_k_mode, logger)
                ecc_written += 1

                ecc_sidecar_path = sidecar_ecc_dir / f"{pair.stem}.json"
                sidecar = {
                    "stem": pair.stem,
                    "ecc_ok": bool(ecc_ok),
                    "ecc_reason": ecc_reason,
                    "ecc_used_init": bool(ecc_used_init),
                    "ecc_fallback_reason": ecc_fallback_reason,
                    "ecc_q_init": float(ecc_q_init) if ecc_q_init is not None else None,
                    "ecc_score": float(ecc_score) if ecc_score is not None else None,
                    "ecc_motion": ecc_motion,
                    "ecc_iter": int(args.ecc_iter),
                    "ecc_eps": float(args.ecc_eps),
                    "structure_mode": structure_mode,
                    "init_source": ecc_init_source,
                    "H_init": _mat33_from_affine(ecc_h_init).tolist() if ecc_h_init is not None else None,
                    "H_raw": _mat33_from_affine(ecc_h_raw).tolist() if ecc_h_raw is not None else None,
                    "H_residual": _mat33_from_affine(ecc_warp_residual).tolist() if ecc_warp_residual is not None else None,
                    "S_sfm_safe": _mat33_from_affine(ecc_s_matrix).tolist() if ecc_s_matrix is not None else None,
                    "sfm_safe": {
                        "allow_rot": bool(allow_rot),
                        "theta_deg": float(theta_deg),
                    },
                    "H_safe2thermal": ecc_h_safe2thermal.tolist() if ecc_h_safe2thermal is not None else None,
                    "quality": {
                        "grad_ncc": float(grad_ncc_score) if grad_ncc_score is not None else None,
                        "edge_f1": float(edge_f1_score) if edge_f1_score is not None else None,
                        "q_total": float(q_total) if q_total is not None else None,
                        "q_min": float(ecc_q_min) if ecc_q_min is not None else None,
                        "quality_ok": bool(ecc_quality_ok) if ecc_quality_ok is not None else None,
                    },
                }
                ecc_sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

            # Save dual outputs + sidecar
            if want_dual and img_dual is not None:
                out_dual_rgb_path = img_dual_dir / out_name
                save_with_metadata(pair.rgb_path, out_dual_rgb_path, img_dual, th_w0, th_h0, zoom_dual, fit_k_mode, logger)
                dual_rgb_written += 1
            if want_dual and th_dual is not None:
                out_dual_th_path = th_dual_dir / output_th_filename(pair)
                save_png(out_dual_th_path, th_dual)
                dual_th_written += 1
            if want_dual and (img_dual is not None or th_dual is not None):
                dual_sidecar_path = sidecar_dual_dir / f"{pair.stem}.json"
                dual_sidecar = {
                    "stem": pair.stem,
                    "dual_ok": bool(img_dual is not None and th_dual is not None),
                    "dual_fallback_reason": dual_fallback_reason,
                    "structure_mode": dual_struct_mode,
                    "H_global": _mat33_from_affine(dual_H_global).tolist() if dual_H_global is not None else None,
                    "S_sfm_safe": _mat33_from_affine(dual_S).tolist() if dual_S is not None else None,
                    "H_safe2thermal": dual_h_safe2thermal.tolist() if dual_h_safe2thermal is not None else None,
                    "H_thermal2safe": dual_h_thermal2safe.tolist() if dual_h_thermal2safe is not None else None,
                    "sfm_safe": {"allow_rot": bool(allow_rot), "theta_deg": float(theta_deg)},
                    "dual_shift": {
                        "dx": float(dual_shift[0]) if dual_shift is not None else None,
                        "dy": float(dual_shift[1]) if dual_shift is not None else None,
                        "resp": float(dual_shift_resp) if dual_shift_resp is not None else None,
                        "applied": bool(dual_shift_applied),
                    },
                    "warp_invalid_ratio": float(dual_invalid_ratio) if dual_invalid_ratio is not None else None,
                    "quality": {
                        "grad_ncc": float(dual_grad_ncc) if dual_grad_ncc is not None else None,
                        "edge_f1": float(dual_edge_f1) if dual_edge_f1 is not None else None,
                        "q_total": float(dual_q_total) if dual_q_total is not None else None,
                    },
                }
                dual_sidecar_path.write_text(json.dumps(dual_sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

            # comparison montage (optional)
            cmp_path = None
            if want_comp:
                # vis boxes needs fit_box at least
                if fit_box is None:
                    # fall back to full-frame box if fit was not computed for some reason
                    fit_box = (0, 0, rgb_w, rgb_h)
                vis_boxes = draw_boxes_on_vis(
                    rgb_bgr,
                    fit_box,
                    exif_box,
                    None,
                    dual_box if want_dual else None,
                )
                # montage expects img_fit; if missing, reuse img_exif or blank
                if img_fit is None:
                    img_fit = img_exif if img_exif is not None else cv2.resize(rgb_bgr, (th_w0, th_h0), interpolation=cv2.INTER_AREA)
                cmp_img = make_comparison_montage(
                    vis_boxes,
                    th_bgr,
                    img_exif,
                    img_fit,
                    cell_w,
                    cell_h,
                    None,
                    dual_th_bgr=th_dual if want_dual else None,
                    img_dual_bgr=img_dual if want_dual else None,
                    force_dual=bool(want_dual),
                )
                cmp_path = comp_dir / f"{pair.stem}.png"
                save_png(cmp_path, cmp_img)

            exp_dz_fit, exp_f35_fit = (None, None)
            exp_dz_exif, exp_f35_exif = (None, None)
            exp_dz_ecc, exp_f35_ecc = (None, None)
            exp_dz_dual, exp_f35_dual = (None, None)
            if zoom_fit is not None:
                exp_dz_fit, exp_f35_fit = _compute_zoom_updates_for_crop(pair.rgb_path, zoom_fit)
            if zoom_exif is not None:
                exp_dz_exif, exp_f35_exif = _compute_zoom_updates_for_crop(pair.rgb_path, zoom_exif)
            if zoom_ecc is not None:
                exp_dz_ecc, exp_f35_ecc = _compute_zoom_updates_for_crop(pair.rgb_path, zoom_ecc)
            if zoom_dual is not None:
                exp_dz_dual, exp_f35_dual = _compute_zoom_updates_for_crop(pair.rgb_path, zoom_dual)

            dt = time.perf_counter() - t0
            per_fp.write(json.dumps({
                "stem": pair.stem,
                "idx": idx,
                "total": len(pairs),
                "time_sec": round(dt, 4),
                "fit_out": str(out_fit_path) if out_fit_path else None,
                "exif_out": str(out_exif_path) if out_exif_path else None,
                "ecc_out": str(out_ecc_path) if out_ecc_path else None,
                "comparison": str(cmp_path) if cmp_path else None,
                "comparison_overlay_ecc": None,
                "ecc_sidecar": str(ecc_sidecar_path) if ecc_sidecar_path else None,
                "ecc_score": float(ecc_score) if ecc_score is not None else None,
                "ecc_ok": bool(ecc_ok) if want_ecc else None,
                "ecc_init": ecc_init_source if want_ecc else None,
                "ecc_motion": ecc_motion if want_ecc else None,
                "ecc_reason": ecc_reason if want_ecc else None,
                "ecc_used_init": bool(ecc_used_init) if want_ecc else None,
                "ecc_fallback_reason": ecc_fallback_reason if want_ecc else None,
                "ecc_q_init": float(ecc_q_init) if ecc_q_init is not None else None,
                "ecc_grad_ncc": float(grad_ncc_score) if grad_ncc_score is not None else None,
                "ecc_edge_f1": float(edge_f1_score) if edge_f1_score is not None else None,
                "ecc_q_total": float(q_total) if q_total is not None else None,
                "ecc_quality_ok": bool(ecc_quality_ok) if ecc_quality_ok is not None else None,
                "dual_rgb_out": str(out_dual_rgb_path) if out_dual_rgb_path else None,
                "dual_th_out": str(out_dual_th_path) if out_dual_th_path else None,
                "dual_sidecar": str(dual_sidecar_path) if dual_sidecar_path else None,
                "dual_grad_ncc": float(dual_grad_ncc) if dual_grad_ncc is not None else None,
                "dual_edge_f1": float(dual_edge_f1) if dual_edge_f1 is not None else None,
                "dual_q_total": float(dual_q_total) if dual_q_total is not None else None,
                "dual_shift_dx": float(dual_shift[0]) if dual_shift is not None else None,
                "dual_shift_dy": float(dual_shift[1]) if dual_shift is not None else None,
                "dual_shift_resp": float(dual_shift_resp) if dual_shift_resp is not None else None,
                "dual_shift_applied": bool(dual_shift_applied),
                "dual_warp_invalid_ratio": float(dual_invalid_ratio) if dual_invalid_ratio is not None else None,
                "zoom_fit": float(zoom_fit) if zoom_fit is not None else None,
                "zoom_exif": float(zoom_exif) if zoom_exif is not None else None,
                "zoom_ecc": float(zoom_ecc) if zoom_ecc is not None else None,
                "zoom_dual": float(zoom_dual) if zoom_dual is not None else None,
                "expected_dzoom_fit": float(exp_dz_fit) if exp_dz_fit is not None else None,
                "expected_f35_fit": int(exp_f35_fit) if exp_f35_fit is not None else None,
                "expected_dzoom_exif": float(exp_dz_exif) if exp_dz_exif is not None else None,
                "expected_f35_exif": int(exp_f35_exif) if exp_f35_exif is not None else None,
                "expected_dzoom_ecc": float(exp_dz_ecc) if exp_dz_ecc is not None else None,
                "expected_f35_ecc": int(exp_f35_ecc) if exp_f35_ecc is not None else None,
                "expected_dzoom_dual": float(exp_dz_dual) if exp_dz_dual is not None else None,
                "expected_f35_dual": int(exp_f35_dual) if exp_f35_dual is not None else None,
            }, ensure_ascii=False) + "\n")
            per_fp.flush()

            logger.log("INFO", f"[IMG] {pair.stem} ({idx}/{len(pairs)}) "
                               f"fit_out={'YES' if out_fit_path else 'NO'} "
                               f"exif_out={'YES' if out_exif_path else 'NO'} "
                               f"ecc_out={'YES' if out_ecc_path else 'NO'} "
                               f"dual_rgb={'YES' if out_dual_rgb_path else 'NO'} "
                               f"dual_th={'YES' if out_dual_th_path else 'NO'} "
                               f"cmp={'YES' if cmp_path else 'NO'} ({dt:.2f}s)")

        t_apply = time.perf_counter() - t_apply0
        per_fp.close()

        if want_ecc:
            try:
                ecc_stats_out = {
                    "count": int(ecc_stats["count"]),
                    "ok": int(ecc_stats["ok"]),
                    "fail": int(ecc_stats["fail"]),
                    "reasons": ecc_stats["reasons"],
                    "ecc_score": ecc_stats["ecc_score"].as_dict(),
                    "q_total": ecc_stats["q_total"].as_dict(),
                    "grad_ncc": ecc_stats["grad_ncc"].as_dict(),
                    "edge_f1": ecc_stats["edge_f1"].as_dict(),
                    "worst_q": ecc_stats["worst_q"],
                    "best_q": ecc_stats["best_q"],
                    "fail_samples": ecc_stats["fail_samples"],
                }
                ecc_stats_path.write_text(json.dumps(ecc_stats_out, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                logger.log("WARN", f"[ECC] Failed to write ecc_stats: {e}")

        # ---- v10 debug: exif audit + sample dump/diff ----
        exiftool = find_exiftool()
        if exiftool:
            try:
                exif_audit_path.write_text("", encoding="utf-8")
            except Exception:
                pass

            if want_fit and img_fit_dir.exists():
                _write_exif_audit_jsonl(exiftool, img_fit_dir, exif_audit_path, kind="fit",
                                        expected_w=th_w0, expected_h=th_h0, logger=logger)
            if want_exif and exif_usable and img_exif_dir.exists():
                _write_exif_audit_jsonl(exiftool, img_exif_dir, exif_audit_path, kind="exif",
                                        expected_w=th_w0, expected_h=th_h0, logger=logger)
            if want_ecc and img_ecc_dir.exists():
                _write_exif_audit_jsonl(exiftool, img_ecc_dir, exif_audit_path, kind="ecc",
                                        expected_w=th_w0, expected_h=th_h0, logger=logger)
            if want_dual and img_dual_dir.exists():
                _write_exif_audit_jsonl(exiftool, img_dual_dir, exif_audit_path, kind="dual",
                                        expected_w=th_w0, expected_h=th_h0, logger=logger)

            # dump + diff only for the first pair (keep debug light)
            if pairs:
                first = pairs[0]
                out_name0 = output_rgb_filename(first)
                src0 = first.rgb_path
                dst_fit0 = (img_fit_dir / out_name0) if (want_fit and (img_fit_dir / out_name0).exists()) else None
                dst_exif0 = (img_exif_dir / out_name0) if (want_exif and (img_exif_dir / out_name0).exists()) else None

                src_meta = _exiftool_dump_json(exiftool, src0, debug_dir / f"exif_src-{suffix}.json", logger)

                dst_fit_meta = None
                if dst_fit0 is not None:
                    dst_fit_meta = _exiftool_dump_json(exiftool, dst_fit0, debug_dir / f"exif_dst_fit-{suffix}.json", logger)

                dst_exif_meta = None
                if dst_exif0 is not None:
                    dst_exif_meta = _exiftool_dump_json(exiftool, dst_exif0, debug_dir / f"exif_dst_exif-{suffix}.json", logger)

                if src_meta is not None and dst_fit_meta is not None:
                    diff = _diff_meta_dict(src_meta, dst_fit_meta)
                    (debug_dir / f"exif_diff-{suffix}.json").write_text(
                        json.dumps(diff, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                        errors="ignore",
                    )
        else:
            logger.log("WARN", "exiftool not found; exif_audit/exif_dump/exif_diff skipped.")

    else:
        logger.log("INFO", "[APPLY] Skipped (stage=fit).")

    # summary
    summary = {
        "version": 10,
        "args": {
            "rgb_dir": str(rgb_dir),
            "th_dir": str(th_dir),
            "out_dir": str(out_dir),
            "samples": args.samples,
            "comparison": bool(want_comp),
            "align": args.align,
            "stage": args.stage,
            "fit_k_mode": fit_k_mode,
            "fit_agg_mode": fit_agg_mode,
            "exif_noise_pct": float(exif_noise_pct),
            "exif_missing": bool(exif_missing),
            "exif_noise_seed": int(exif_noise_seed),
            "ecc_motion": ecc_motion if want_ecc else None,
            "ecc_init": ecc_init if want_ecc else None,
            "structure_mode": structure_mode if (want_ecc or want_dual) else None,
            "ecc_iter": int(args.ecc_iter) if want_ecc else None,
            "ecc_eps": float(args.ecc_eps) if want_ecc else None,
            "ecc_q_min": float(ecc_q_min) if (want_ecc and ecc_q_min is not None) else None,
            "sfm_allow_rot": bool(allow_rot) if (want_ecc or want_dual) else None,
            "sfm_rot_deg": float(theta_deg) if (want_ecc or want_dual) else None,
            "sfm_max_rot_deg": float(max_theta) if (want_ecc or want_dual) else None,
            "dual": bool(want_dual),
            "dual_frames": int(dual_frames) if want_dual else None,
            "dual_q_min": float(dual_q_min) if want_dual else None,
        },
        "flags": {
            "want_fit": bool(want_fit),
            "want_exif": bool(want_exif),
            "want_ecc": bool(want_ecc),
            "want_dual": bool(want_dual),
            "do_fit": bool(do_fit),
            "do_apply": bool(do_apply),
            "exif_usable": bool(exif_usable),
        },
        "paths": {
            "out_dir": str(out_dir),
            "debug_dir": str(debug_dir),
            "model_fit": str(model_fit_path) if (need_fit_model and model_fit_path.exists()) else None,
            "model_exif": str(model_exif_path) if model_exif_path else None,
            "model_dual": str(model_dual_path) if (want_dual and model_dual_path.exists()) else None,
            "image_fit_dir": str(img_fit_dir) if (do_apply and want_fit) else None,
            "image_exif_dir": str(img_exif_dir) if (do_apply and want_exif and exif_usable) else None,
            "image_ecc_dir": str(img_ecc_dir) if (do_apply and want_ecc) else None,
            "image_dual_dir": str(img_dual_dir) if (do_apply and want_dual) else None,
            "thermal_dual_dir": str(th_dual_dir) if (do_apply and want_dual) else None,
            "sidecar_ecc_dir": str(sidecar_ecc_dir) if (do_apply and want_ecc) else None,
            "sidecar_dual_dir": str(sidecar_dual_dir) if (do_apply and want_dual) else None,
            "comparison_dir": str(comp_dir) if (do_apply and want_comp) else None,
            "run_log": str(debug_dir / f"run-{suffix}.log"),
            "per_image_jsonl": str(per_image_path) if do_apply else None,
            "exif_audit_jsonl": str(exif_audit_path) if do_apply else None,
            "ecc_stats_json": str(ecc_stats_path) if (do_apply and want_ecc) else None,
            "dual_stats_json": str(dual_stats_path) if (do_apply and want_dual and dual_stats_path.exists()) else None,
        },
        "counts": {
            "pairs_found": int(pairs_found),
            "pairs_processed": int(len(pairs)),
            "fit_written": int(fit_written),
            "exif_written": int(exif_written),
            "ecc_written": int(ecc_written),
            "dual_rgb_written": int(dual_rgb_written),
            "dual_th_written": int(dual_th_written),
        },
        "timing": {"fit_sec": round(float(t_fit), 3), "apply_sec": round(float(t_apply), 3)},
        "exif_probe": {
            "usable": bool(exif_usable),
            "probe_ok": int(exif_ok_count),
            "probe_n": int(probe_n),
            "noise_pct": float(exif_noise_pct),
            "missing": bool(exif_missing),
            "noise_seed": int(exif_noise_seed),
        },
        "metadata_policy": {
            "read_pixels": "apply EXIF orientation to pixels via ImageOps.exif_transpose",
            "write_orientation": "force Orientation=1 (IFD0 + XMP-tiff) using 1-pass exiftool after raw XMP injection",
            "sizes": "write IFD0:ImageWidth/Height and ExifIFD:ExifImageWidth/Height only (no PixelXDimension/YDimension)",
            "digital_zoom": "DigitalZoomRatio *= zoom_factor",
            "f35mm": "FocalLengthIn35mmFormat *= zoom_factor (correct ExifTool tag name: Format)",
            "fit_k_mode": fit_k_mode,
            "xmp": "copy -xmp:all and inject raw XMP packet from source JPEG (APP1) via exiftool -XMP<= (best effort for DJI namespaces)",
            "strip": "remove MPF/IFD1/Preview/Thumbnail blocks after crop/resize",
            "audit": "write debug/exif_audit-<suffix>.jsonl for orientation/size correctness + key GPS/XMP fields",
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.log("INFO", "[SUMMARY] ------------------------------")
    logger.log("INFO", f"[SUMMARY] pairs_found={pairs_found} pairs_processed={len(pairs)} "
                       f"fit_written={fit_written} exif_written={exif_written} ecc_written={ecc_written} "
                       f"dual_rgb_written={dual_rgb_written} dual_th_written={dual_th_written}")
    if need_fit_model and model_fit_path.exists():
        logger.log("INFO", f"[SUMMARY] model_fit={model_fit_path}")
    if model_exif_path:
        logger.log("INFO", f"[SUMMARY] model_exif={model_exif_path}")
    logger.log("INFO", f"[SUMMARY] summary={summary_path}")
    logger.log("INFO", "RUN END")
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
