#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_uavfgs.py

Prepare COLMAP- and 3DGS-compatible inputs for UAV-FGS.
This script converts standardized RGB-T data into the camera, database, and
layout artifacts expected by the downstream reconstruction pipeline and exports
pose-prior records for GPS-enabled runs.
"""

import argparse
import json
import os
import math
import re
import sqlite3
import struct
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np


def log_info(msg: str) -> None:
    print(f"INFO: {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"WARNING: {msg}", flush=True)


def log_err(msg: str) -> None:
    print(f"ERROR: {msg}", flush=True)


def _resolve_executable(exe: str) -> str:
    exe_expanded = os.path.expandvars(exe)
    if os.path.isabs(exe_expanded) and os.path.exists(exe_expanded):
        return exe_expanded

    candidates = [exe_expanded]
    if os.name == "nt":
        base = exe_expanded
        if not base.lower().endswith((".exe", ".cmd", ".bat")):
            candidates = [base, base + ".exe", base + ".cmd", base + ".bat"]

    for v in candidates:
        p = shutil.which(v)
        if p:
            return p
    return exe_expanded


def _should_use_shell(resolved_exe: str) -> bool:
    if os.name != "nt":
        return False
    low = resolved_exe.lower()
    return low.endswith(".bat") or low.endswith(".cmd")


def run_cmd(cmd_list, cwd=None):
    if not cmd_list:
        raise ValueError("Empty command list")

    resolved0 = _resolve_executable(cmd_list[0])
    use_shell = _should_use_shell(resolved0)

    if use_shell:
        cmd_str = " ".join([f'"{x}"' if (" " in str(x)) else str(x) for x in [resolved0] + cmd_list[1:]])
        log_info("Running (shell): " + cmd_str)
        subprocess.run(cmd_str, cwd=cwd, check=True, shell=True)
    else:
        cmd = [resolved0] + cmd_list[1:]
        log_info("Running: " + " ".join([str(x) for x in cmd]))
        subprocess.run(cmd, cwd=cwd, check=True)


def split_args(s: str):
    return s.strip().split() if s else []



def replace_alignment_max_error(arg_list, value):
    """Replace/insert --alignment_max_error in a tokenized arg list."""
    out = []
    i = 0
    found = False
    while i < len(arg_list):
        a = arg_list[i]
        if a.startswith("--alignment_max_error="):
            out.append(f"--alignment_max_error={value}")
            found = True
            i += 1
            continue
        if a == "--alignment_max_error":
            out.append(a)
            # replace next token if present, else append
            if i + 1 < len(arg_list) and not arg_list[i + 1].startswith("--"):
                out.append(str(value))
                i += 2
            else:
                out.append(str(value))
                i += 1
            found = True
            continue
        out.append(a)
        i += 1
    if not found:
        out += ["--alignment_max_error", str(value)]
    return out


def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def _read_c_string(f) -> str:
    chars = []
    while True:
        c = f.read(1)
        if not c or c == b"\x00":
            break
        chars.append(c)
    return b"".join(chars).decode("utf-8", errors="replace")


def read_num_registered_images(model_dir: Path) -> int:
    images_bin = model_dir / "images.bin"
    images_txt = model_dir / "images.txt"
    if images_bin.exists():
        with images_bin.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        return int(n)
    if images_txt.exists():
        n = 0
        with images_txt.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 10 and parts[0].isdigit():
                    n += 1
        return n
    return -1


def read_num_points3d(model_dir: Path) -> int:
    points_bin = model_dir / "points3D.bin"
    points_txt = model_dir / "points3D.txt"
    if points_bin.exists():
        with points_bin.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        return int(n)
    if points_txt.exists():
        n = 0
        with points_txt.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                n += 1
        return n
    return -1


def read_image_names(model_dir: Path):
    names = []
    images_txt = model_dir / "images.txt"
    images_bin = model_dir / "images.bin"

    if images_txt.exists():
        with images_txt.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 10 and parts[0].isdigit():
                    names.append(parts[9])
        return names

    if images_bin.exists():
        with images_bin.open("rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                f.read(4)
                f.read(8 * 7)
                f.read(4)
                name = _read_c_string(f)
                names.append(name)
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                f.read(num_points2d * (8 + 8 + 8))
        return names

    return names


def select_best_sparse_model(sparse_root: Path) -> Path:
    if not sparse_root.exists():
        raise FileNotFoundError(f"sparse root not found: {sparse_root}")

    candidates = []
    for p in sparse_root.iterdir():
        if not p.is_dir():
            continue
        if not re.fullmatch(r"\d+", p.name):
            continue
        reg = read_num_registered_images(p)
        pts = read_num_points3d(p)
        if reg < 0 and pts < 0:
            continue
        candidates.append((reg, pts, p))

    if not candidates:
        raise RuntimeError(f"No valid sparse models under: {sparse_root}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[0][2]

    log_info(f"Sparse models found under {sparse_root}:")
    for reg, pts, p in candidates[:30]:
        log_info(f"  model={p.name} | registered={reg} | points3D={pts}")
    log_info(f"Selected best sparse model: {best}")
    return best


def exiftool_extract_gps(input_dir: Path, exiftool_exe: str):
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    exiftool_path = _resolve_executable(exiftool_exe)

    cmd = [
        exiftool_path,
        "-q", "-q",
        "-json",
        "-n",
        # Ask exiftool for all groups so PNG EXIF/XMP GPS tags are not dropped.
        "-G",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSAltitude",
        "-EXIF:GPSLatitude",
        "-EXIF:GPSLongitude",
        "-EXIF:GPSAltitude",
        "-XMP:GPSLatitude",
        "-XMP:GPSLongitude",
        "-XMP:GPSAltitude",
        "-Composite:GPSLatitude",
        "-Composite:GPSLongitude",
        "-Composite:GPSAltitude",
        "-XMP-drone-dji:AbsoluteAltitude",
        "-r",
        "-ext", "jpg",
        "-ext", "jpeg",
        "-ext", "JPG",
        "-ext", "JPEG",
        "-ext", "png",
        "-ext", "PNG",
        str(input_dir),
    ]

    log_info("Extracting GPS from images via exiftool (may take a bit)...")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    text = p.stdout.decode("utf-8", errors="replace")

    records = []
    if text.strip():
        try:
            records = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('[')
            end = text.rfind(']')
            if start >= 0 and end > start:
                try:
                    records = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    log_warn("Exiftool output is not valid JSON after bracket recovery; treat as no GPS.")
                    records = []
            else:
                log_warn("Exiftool output has no JSON payload; treat as no GPS.")
                records = []
    else:
        log_warn("Exiftool returned empty output; treat as no GPS.")

    gps = {}
    def _pick_tag(rec: Dict, names: List[str]):
        # 1) exact key
        for n in names:
            if n in rec and rec[n] is not None:
                return rec[n]
        # 2) key suffix match (handles keys like "EXIF:GPSLatitude", "XMP:GPSLatitude")
        rec_items = list(rec.items())
        for n in names:
            n_low = n.lower()
            for k, v in rec_items:
                if v is None:
                    continue
                ks = str(k).lower()
                if ks == n_low or ks.endswith(":" + n_low):
                    return v
        return None

    for r in records:
        src = r.get("SourceFile", "")
        base = os.path.basename(src)
        lat = _pick_tag(r, ["GPSLatitude"])
        lon = _pick_tag(r, ["GPSLongitude"])
        alt = _pick_tag(r, ["GPSAltitude"])
        if alt is None:
            alt = _pick_tag(r, ["AbsoluteAltitude", "RelativeAltitude"])
        if lat is None or lon is None or alt is None:
            continue
        if not (is_number(lat) and is_number(lon) and is_number(alt)):
            continue
        gps[base] = (float(lat), float(lon), float(alt))

    log_info(f"EXIF GPS entries found: {len(gps)} (keyed by basename)")
    return gps


def ensure_pose_priors_table(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pose_priors'")
    if cur.fetchone():
        return
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pose_priors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            position BLOB,
            coordinate_system INTEGER NOT NULL,
            position_covariance BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
        """
    )
    con.commit()


def get_pose_priors_schema(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("PRAGMA table_info(pose_priors)")
    rows = cur.fetchall()
    return [r[1] for r in rows]


def populate_pose_priors_from_exif(
    database_path: Path,
    input_dir: Path,
    exiftool_exe: str,
    wgs84_code: int,
    prior_position_std_m: Optional[float] = None,
    swap_latlon: bool = False,
):
    """
    Populate COLMAP pose_priors from EXIF GPS.

    Notes:
    - Newer COLMAP versions store pose priors in a dedicated `pose_priors` table.
    - For GPS, COLMAP expects WGS84 coordinate system code (in your build it's 0).
    - `position_covariance` should be NULL or finite values. Writing NaNs can make
      downstream alignment fail.

    prior_position_std_m:
      If provided, we write a diagonal covariance:
        [std_lat_deg^2, std_lon_deg^2, std_alt_m^2]
      where std_lat_deg/std_lon_deg are meters converted to degrees at that latitude.
      If None, position_covariance is left NULL.
    """
    gps = exiftool_extract_gps(input_dir, exiftool_exe=exiftool_exe)

    con = sqlite3.connect(str(database_path))
    try:
        ensure_pose_priors_table(con)
        cols = get_pose_priors_schema(con)
        expected = ["image_id", "position", "coordinate_system", "position_covariance"]
        if cols != expected:
            raise RuntimeError(f"pose_priors columns unexpected: {cols}")

        cur = con.cursor()
        cur.execute("SELECT image_id, name FROM images")
        rows = cur.fetchall()

        # Map by basename (works if basenames are unique). Also keep a duplicate check.
        name2id = {}
        dup_bases = set()
        for image_id, name in rows:
            base = os.path.basename(name)
            if base in name2id:
                dup_bases.add(base)
            name2id[base] = image_id
        if dup_bases:
            log_warn(
                f"Duplicate basenames detected in DB images table (showing up to 5): {list(sorted(dup_bases))[:5]}. "
                "Basename-based EXIF matching may be ambiguous."
            )

        inserted = 0
        matched = 0

        # Precompute some sanity stats.
        lat_list, lon_list, alt_list = [], [], []

        for base, image_id in name2id.items():
            if base not in gps:
                continue
            matched += 1
            lat, lon, alt = gps[base]

            if swap_latlon:
                lat, lon = lon, lat  # experimental switch if needed

            # position is stored as 3 float64 values (lat_deg, lon_deg, alt_m)
            pos = np.asarray([lat, lon, alt], dtype=np.float64)

            # Optional covariance (NULL by default).
            cov_blob = None
            if prior_position_std_m is not None and prior_position_std_m > 0:
                # Convert meters -> degrees at this latitude (rough WGS84 approximation).
                meters_per_deg_lat = 111320.0
                meters_per_deg_lon = 111320.0 * max(1e-6, math.cos(math.radians(lat)))
                std_lat_deg = prior_position_std_m / meters_per_deg_lat
                std_lon_deg = prior_position_std_m / meters_per_deg_lon
                cov = np.diag([std_lat_deg**2, std_lon_deg**2, float(prior_position_std_m) ** 2]).astype(np.float64)
                cov_blob = cov.tobytes()

            cur.execute(
                "INSERT OR REPLACE INTO pose_priors(image_id, position, coordinate_system, position_covariance) "
                "VALUES (?, ?, ?, ?)",
                (int(image_id), pos.tobytes(), int(wgs84_code), cov_blob),
            )
            inserted += 1

            lat_list.append(pos[0])
            lon_list.append(pos[1])
            alt_list.append(pos[2])

        con.commit()
        log_info(f"Pose priors populated: inserted={inserted}, matched_images_in_db={matched}, db_images_total={len(rows)}")

        cur.execute("SELECT COUNT(*) FROM pose_priors")
        cnt = cur.fetchone()[0]
        log_info(f"pose_priors rows in DB: {cnt}")

        cur.execute("SELECT MIN(coordinate_system), MAX(coordinate_system) FROM pose_priors")
        mn, mx = cur.fetchone()
        log_info(f"pose_priors.coordinate_system range: {mn}..{mx}")

        if lat_list and lon_list and alt_list:
            log_info(
                "pose_priors position ranges (from inserted rows): "
                f"lat[{min(lat_list):.8f},{max(lat_list):.8f}] "
                f"lon[{min(lon_list):.8f},{max(lon_list):.8f}] "
                f"alt[{min(alt_list):.3f},{max(alt_list):.3f}]"
            )
            # Basic plausibility checks (WGS84 degrees)
            if not (-90 <= min(lat_list) <= 90 and -90 <= max(lat_list) <= 90):
                log_warn("Latitude range looks suspicious (outside [-90, 90]).")
            if not (-180 <= min(lon_list) <= 180 and -180 <= max(lon_list) <= 180):
                log_warn("Longitude range looks suspicious (outside [-180, 180]).")

    finally:
        con.close()

    return gps


def sanity_check_overlap(model_dir: Path, gps_by_basename: dict):
    model_names = read_image_names(model_dir)
    if not model_names:
        log_warn("Could not read image names from selected model; skip overlap check.")
        return
    bases = [os.path.basename(n) for n in model_names]
    overlap = sum(1 for b in bases if b in gps_by_basename)
    log_info(f"Selected model images: {len(bases)}; images-with-GPS(overlap by basename): {overlap}")
    if overlap < 3:
        log_warn("Overlap < 3; model_aligner likely fails (min_common_images default is 3).")

def ensure_3dgs_sparse_layout(root: Path) -> Path:
    """Normalize COLMAP output layout for 3DGS.

    3DGS loaders typically expect:
      <root>/images
      <root>/sparse/0/{cameras,images,points3D}.{bin|txt}

    COLMAP may instead write the model directly under <root>/sparse (no /0)
    or under a different numeric subfolder. This function makes sure that
    <root>/sparse/0 exists and contains the model files.
    """

    sparse = root / "sparse"
    if not sparse.exists():
        raise FileNotFoundError(f"COLMAP undistorted sparse folder not found: {sparse}")

    model0 = sparse / "0"

    def has_any_model_files(p: Path) -> bool:
        for fn in (
            "cameras.bin",
            "images.bin",
            "points3D.bin",
            "cameras.txt",
            "images.txt",
            "points3D.txt",
            "rigs.bin",
            "frames.bin",
        ):
            if (p / fn).exists():
                return True
        return False

    # Case 1: already in sparse/0
    if model0.is_dir() and has_any_model_files(model0):
        return model0

    # Case 2: model files are directly under sparse/
    if has_any_model_files(sparse):
        model0.mkdir(parents=True, exist_ok=True)
        for child in list(sparse.iterdir()):
            if child.name == "0":
                continue
            if child.is_file():
                dst = model0 / child.name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(child), str(dst))
        return model0

    # Case 3: numeric subfolders (but no /0)
    subdirs = [p for p in sparse.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]
    if subdirs:
        candidates = [p for p in subdirs if has_any_model_files(p)] or subdirs

        def score(p: Path) -> int:
            s = 0
            pts_bin = p / "points3D.bin"
            imgs_bin = p / "images.bin"
            pts_txt = p / "points3D.txt"
            if pts_bin.exists():
                s += int(pts_bin.stat().st_size)
            if imgs_bin.exists():
                s += int(imgs_bin.stat().st_size) // 10
            if pts_txt.exists():
                s += int(pts_txt.stat().st_size) // 50
            return s

        best = max(candidates, key=score)
        if best.name != "0":
            if model0.exists():
                shutil.rmtree(model0)
            shutil.copytree(best, model0)
        return model0

    raise RuntimeError(f"Could not locate any COLMAP model under: {sparse}")


def export_model_as_txt(colmap_exe: str, model_dir: Path) -> None:
    """Export cameras/images/points to TXT for 3DGS fallback readers."""
    if (model_dir / "cameras.txt").exists() and (model_dir / "images.txt").exists() and (model_dir / "points3D.txt").exists():
        return

    log_info(f"Exporting COLMAP model as TXT for 3DGS compatibility: {model_dir}")
    try:
        run_cmd([
            colmap_exe,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(model_dir),
            "--output_type",
            "TXT",
        ])
    except Exception as e:
        log_warn(f"model_converter failed (will continue): {e}")

def ensure_camera_models_supported_for_3dgs(colmap_exe: str, model_dir: Path) -> None:
    """3DGS loaders often only accept PINHOLE / SIMPLE_PINHOLE camera models.
    Some COLMAP outputs (or previous conversions) may keep e.g. SIMPLE_RADIAL/OPENCV models.
    This function:
      1) Ensures cameras.txt exists (via model_converter -> TXT).
      2) Rewrites cameras.txt to PINHOLE or SIMPLE_PINHOLE when possible by dropping distortion params.
      3) Regenerates cameras.bin/images.bin/points3D.bin from the (possibly) edited TXT model.
    """
    export_model_as_txt(colmap_exe, model_dir)
    cam_txt = model_dir / "cameras.txt"
    if not cam_txt.exists():
        log_warn(f"cameras.txt not found under {model_dir}; cannot enforce camera model.")
        return

    txt = cam_txt.read_text(encoding="utf-8", errors="replace").splitlines()
    changed = False
    new_lines: List[str] = []
    conversions = []

    def fmt_params(ps: List[float]) -> List[str]:
        # use repr to preserve enough precision without trailing '+'
        return [repr(float(p)) for p in ps]

    for line in txt:
        s = line.strip()
        if not s or s.startswith("#"):
            new_lines.append(line)
            continue

        parts = s.split()
        if len(parts) < 5:
            new_lines.append(line)
            continue

        cam_id, model, w, h = parts[0], parts[1], parts[2], parts[3]
        # params may include leading '+'; float() handles it
        try:
            params = [float(x) for x in parts[4:]]
        except Exception:
            new_lines.append(line)
            continue

        if model in ("PINHOLE", "SIMPLE_PINHOLE"):
            new_lines.append(line)
            continue

        model_new = None
        params_new: List[float] = []

        # Most models start with f cx cy (SIMPLE_*) or fx fy cx cy (OPENCV-like)
        if model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE", "FOV"):
            if len(params) >= 3:
                f, cx, cy = params[0], params[1], params[2]
                model_new = "SIMPLE_PINHOLE"
                params_new = [f, cx, cy]
        elif model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                model_new = "PINHOLE"
                params_new = [fx, fy, cx, cy]
        else:
            # Fallback heuristic:
            # - if >=4 params, treat as fx fy cx cy -> PINHOLE
            # - else if >=3 params, treat as f cx cy -> SIMPLE_PINHOLE
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                model_new = "PINHOLE"
                params_new = [fx, fy, cx, cy]
            elif len(params) >= 3:
                f, cx, cy = params[0], params[1], params[2]
                model_new = "SIMPLE_PINHOLE"
                params_new = [f, cx, cy]

        if model_new is None:
            # keep original line if we can't safely convert
            new_lines.append(line)
            continue

        changed = True
        conversions.append(f"{cam_id}:{model}->{model_new}")
        new_line = " ".join([cam_id, model_new, w, h] + fmt_params(params_new))
        new_lines.append(new_line)

    if changed:
        cam_txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        log_info(f"Adjusted camera models for 3DGS: {', '.join(conversions[:8])}{' ...' if len(conversions)>8 else ''}")

        # Regenerate BIN from TXT so 3DGS (which prefers .bin) reads the supported model.
        try:
            run_cmd([
                colmap_exe,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(model_dir),
                "--output_type",
                "BIN",
            ])
        except Exception as e:
            log_warn(f"Failed to regenerate BIN model after camera-model fix (will keep TXT): {e}")
    else:
        log_info("Camera models already supported (PINHOLE/SIMPLE_PINHOLE); no conversion needed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True, help="Dataset root (contains input/)")
    parser.add_argument("--colmap_executable", default="colmap", help="Path/name of colmap executable")
    parser.add_argument("--exiftool_executable", default="exiftool", help="Path/name of exiftool executable")
    parser.add_argument(
        "--wgs84_code", type=int, default=0,
        help="Integer code for PosePrior coordinate_system=WGS84. Your COLMAP error showed WGS84==0."
    )

    parser.add_argument(
        "--prior_position_std_m", type=float, default=None,
        help="If set, write pose_priors.position_covariance as a diagonal covariance based on this std-dev in meters. \nDefault: NULL covariance (recommended)."
    )
    parser.add_argument(
        "--swap_latlon", action="store_true",
        help="Swap lat/lon when writing pose priors (debug option if alignment fails due to ordering)."
    )


    parser.add_argument("--camera", default="SIMPLE_RADIAL")
    parser.add_argument("--matching", default="spatial", choices=["spatial", "exhaustive", "sequential", "vocab_tree"])
    parser.add_argument("--matcher_args", default="")
    parser.add_argument("--mapper_multiple_models", type=int, default=1)
    parser.add_argument("--min_model_size", type=int, default=10)
    parser.add_argument("--init_min_num_inliers", type=int, default=100)
    parser.add_argument("--abs_pose_min_num_inliers", type=int, default=30)

    parser.add_argument("--use_model_aligner", action="store_true")
    parser.add_argument("--model_aligner_args", default="")

    parser.add_argument("--resize", action="store_true")

    args = parser.parse_args()

    root = Path(args.source_path)
    input_dir = root / "input"
    distorted_dir = root / "distorted"
    sparse_root = distorted_dir / "sparse"
    db_path = distorted_dir / "database.db"
    sparse_aligned = distorted_dir / "sparse_aligned"

    distorted_dir.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)

    colmap_exe = args.colmap_executable

    run_cmd([
        colmap_exe, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(input_dir),
        "--ImageReader.camera_model", str(args.camera),
        "--ImageReader.single_camera", "1",
    ])

    if args.matching == "spatial":
        run_cmd([colmap_exe, "spatial_matcher", "--database_path", str(db_path)] + split_args(args.matcher_args))
    elif args.matching == "exhaustive":
        run_cmd([colmap_exe, "exhaustive_matcher", "--database_path", str(db_path)] + split_args(args.matcher_args))
    elif args.matching == "sequential":
        run_cmd([colmap_exe, "sequential_matcher", "--database_path", str(db_path)] + split_args(args.matcher_args))
    else:
        run_cmd([colmap_exe, "vocab_tree_matcher", "--database_path", str(db_path)] + split_args(args.matcher_args))

    run_cmd([
        colmap_exe, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(input_dir),
        "--output_path", str(sparse_root),
        "--Mapper.multiple_models", str(args.mapper_multiple_models),
        "--Mapper.min_model_size", str(args.min_model_size),
        "--Mapper.init_min_num_inliers", str(args.init_min_num_inliers),
        "--Mapper.abs_pose_min_num_inliers", str(args.abs_pose_min_num_inliers),
    ])

    best_model = select_best_sparse_model(sparse_root)

    gps_map = populate_pose_priors_from_exif(db_path, input_dir, exiftool_exe=args.exiftool_executable, wgs84_code=args.wgs84_code, prior_position_std_m=args.prior_position_std_m, swap_latlon=args.swap_latlon)
    sanity_check_overlap(best_model, gps_map)

    aligned_model_for_undistort = best_model
    
    if args.use_model_aligner:
        sparse_aligned.mkdir(parents=True, exist_ok=True)
        base_cmd = [
            colmap_exe, "model_aligner",
            "--input_path", str(best_model),
            "--output_path", str(sparse_aligned),
            "--database_path", str(db_path),
        ]
        user_tokens = split_args(args.model_aligner_args)
    
        # 1) Try user-provided args first
        log_info("Running model_aligner:")
        try:
            run_cmd(base_cmd + user_tokens)
        except subprocess.CalledProcessError as e:
            # Common failure mode when pose priors covariance is invalid or the error threshold is too strict.
            log_warn(f"model_aligner failed (exit={getattr(e, 'returncode', None)}). Will retry with relaxed alignment_max_error.")
    
            # 2) Retry with alignment_max_error=0 (often treated as 'no robust threshold')
            retry_vals = [0, 100, 300, 1000]
            success = False
            last_err = e
            for v in retry_vals:
                tokens2 = replace_alignment_max_error(user_tokens, v)
                log_info(f"Retrying model_aligner with --alignment_max_error={v}")
                try:
                    run_cmd(base_cmd + tokens2)
                    success = True
                    break
                except subprocess.CalledProcessError as e2:
                    last_err = e2
                    continue
            if not success:
                raise last_err
    
        aligned_model_for_undistort = sparse_aligned

    run_cmd([
        colmap_exe, "image_undistorter",
        "--image_path", str(input_dir),
        "--input_path", str(aligned_model_for_undistort),
        "--output_path", str(root),
        "--output_type", "COLMAP",
    ])

    # Make the output compatible with 3DGS (expects <root>/sparse/0).
    try:
        model0 = ensure_3dgs_sparse_layout(root)
        export_model_as_txt(colmap_exe, model0)
        ensure_camera_models_supported_for_3dgs(colmap_exe, model0)
        export_model_as_txt(colmap_exe, model0)  # keep TXT after possible BIN regen
        log_info(f"3DGS layout: images={root/'images'} | sparse_model={model0}")
    except Exception as e:
        log_warn(f"Failed to normalize/export sparse model for 3DGS: {e}")

    if args.resize:
        try:
            from PIL import Image
        except Exception as e:
            log_warn(f"PIL not available, skip resize. ({e})")
            return

        images_dir = root / "images"
        if not images_dir.exists():
            log_warn(f"images dir not found after undistort: {images_dir}; skip resize.")
            return

        scales = [2, 4, 8]
        for s in scales:
            (root / f"images_{s}").mkdir(parents=True, exist_ok=True)

        img_files = []
        for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
            img_files.extend(images_dir.glob(ext))

        log_info(f"Resizing {len(img_files)} undistorted images...")
        for fp in img_files:
            try:
                im = Image.open(fp)
                w, h = im.size
                for s in scales:
                    ow, oh = max(1, w // s), max(1, h // s)
                    im_s = im.resize((ow, oh), resample=Image.BILINEAR)
                    out_path = root / f"images_{s}" / fp.name
                    im_s.save(out_path)
            except Exception as e:
                log_warn(f"Failed to resize {fp}: {e}")

    log_info("Done.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        log_err(f"Command failed: {e}")
        sys.exit(e.returncode)
    except Exception as e:
        log_err(str(e))
        raise

