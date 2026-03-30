import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# Charts
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Optional SSIM
try:
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


def imread_any(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {path}")
    return img


def to_gray_float01(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale float32 in [0,1] with robust percentile normalization."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    lo = np.percentile(img, 1.0)
    hi = np.percentile(img, 99.0)
    if hi <= lo + 1e-6:
        lo = float(img.min())
        hi = float(img.max()) if img.max() > img.min() else (img.min() + 1.0)

    img = (img - lo) / (hi - lo)
    img = np.clip(img, 0.0, 1.0)
    return img


def sobel_mag(img01: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m = float(np.max(mag))
    if m > 1e-8:
        mag = mag / m
    return mag


def auto_canny(img01: np.ndarray) -> np.ndarray:
    img8 = (img01 * 255.0 + 0.5).astype(np.uint8)
    v = float(np.median(img8))
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(img8, lower, upper, L2gradient=True)
    return (edges > 0).astype(np.uint8)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    a -= float(a.mean())
    b -= float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / denom)


def mutual_information(a01: np.ndarray, b01: np.ndarray, bins: int = 64) -> Tuple[float, float]:
    """
    Returns (MI, NMI).
    NMI definition: (H(a)+H(b))/H(a,b)
    """
    a = np.clip(a01, 0.0, 1.0)
    b = np.clip(b01, 0.0, 1.0)
    a_bin = np.floor(a * (bins - 1)).astype(np.int32)
    b_bin = np.floor(b * (bins - 1)).astype(np.int32)

    joint = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(joint, (a_bin.reshape(-1), b_bin.reshape(-1)), 1.0)

    joint /= float(joint.sum() + 1e-12)
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)

    eps = 1e-12
    Ha = -np.sum(pa * np.log(pa + eps))
    Hb = -np.sum(pb * np.log(pb + eps))
    Hab = -np.sum(joint * np.log(joint + eps))

    mi = Ha + Hb - Hab
    nmi = (Ha + Hb) / (Hab + eps)
    return float(mi), float(nmi)


def dice_and_f1(edge_a: np.ndarray, edge_b: np.ndarray) -> Tuple[float, float]:
    a = edge_a.reshape(-1).astype(np.uint8)
    b = edge_b.reshape(-1).astype(np.uint8)

    tp = int(np.sum((a == 1) & (b == 1)))
    fp = int(np.sum((a == 1) & (b == 0)))
    fn = int(np.sum((a == 0) & (b == 1)))

    denom_d = (2 * tp + fp + fn)
    dice = (2 * tp) / denom_d if denom_d > 0 else float("nan")

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else float("nan")
    return float(dice), float(f1)


def safe_float(x: float) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


@dataclass
class Metrics:
    mi: Optional[float]
    nmi: Optional[float]
    grad_ncc: Optional[float]
    edge_dice: Optional[float]
    edge_f1: Optional[float]
    grad_ssim: Optional[float]


def compute_metrics(rgb_path: str, th_path: str, bins: int, do_ssim: bool) -> Metrics:
    rgb = imread_any(rgb_path)
    th = imread_any(th_path)

    rgb01 = to_gray_float01(rgb)
    th01 = to_gray_float01(th)

    if rgb01.shape != th01.shape:
        th01 = cv2.resize(th01, (rgb01.shape[1], rgb01.shape[0]), interpolation=cv2.INTER_LINEAR)

    mi, nmi_v = mutual_information(rgb01, th01, bins=bins)

    rgb_g = sobel_mag(rgb01)
    th_g = sobel_mag(th01)
    grad_ncc_v = ncc(rgb_g, th_g)

    rgb_e = auto_canny(rgb_g)
    th_e = auto_canny(th_g)
    dice_v, f1_v = dice_and_f1(rgb_e, th_e)

    grad_ssim_v = None
    if do_ssim:
        if not HAS_SKIMAGE:
            raise RuntimeError("scikit-image not available; install scikit-image or disable --ssim.")
        grad_ssim_v = float(sk_ssim(rgb_g, th_g, data_range=1.0))

    return Metrics(
        mi=safe_float(mi),
        nmi=safe_float(nmi_v),
        grad_ncc=safe_float(grad_ncc_v),
        edge_dice=safe_float(dice_v),
        edge_f1=safe_float(f1_v),
        grad_ssim=safe_float(grad_ssim_v) if do_ssim else None,
    )


def list_images(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    files = []
    for fn in os.listdir(dir_path):
        if fn.lower().endswith(exts):
            files.append(fn)
    files.sort()
    return files


def stem_key(fn: str) -> str:
    return os.path.splitext(fn)[0].lower()


def mean_std(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None, None
    arr = np.array(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def tqdm_wrap(iterable, total: int, desc: str):
    if HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)
    # fallback: no tqdm
    return iterable


def eval_one_candidate(tag: str, rgb_dir: str, th_map: Dict[str, str], args) -> Dict:
    rgb_files = list_images(rgb_dir, args.exts)

    # build matched list first (so progress total is correct)
    matched = []
    for fn in rgb_files[:: args.stride]:
        k = stem_key(fn)
        if k in th_map:
            matched.append(fn)

    if args.max_images and args.max_images > 0:
        matched = matched[: args.max_images]

    rows = []
    it = tqdm_wrap(matched, total=len(matched), desc=f"[{tag}]")
    for fn in it:
        rgb_path = os.path.join(rgb_dir, fn)
        th_path = th_map[stem_key(fn)]
        try:
            m = compute_metrics(rgb_path, th_path, bins=args.bins, do_ssim=args.ssim)
            row = {
                "file": fn,
                "mi": m.mi,
                "nmi": m.nmi,
                "grad_ncc": m.grad_ncc,
                "edge_dice": m.edge_dice,
                "edge_f1": m.edge_f1,
            }
            if args.ssim:
                row["grad_ssim"] = m.grad_ssim
            rows.append(row)
        except Exception as e:
            rows.append({"file": fn, "error": str(e)})

    metrics_keys = ["mi", "nmi", "grad_ncc", "edge_dice", "edge_f1"] + (["grad_ssim"] if args.ssim else [])
    summary = {"tag": tag, "rgb_dir": rgb_dir, "count": 0, "mean": {}, "std": {}}

    ok_rows = [r for r in rows if "mi" in r and r.get("nmi") is not None]
    summary["count"] = len(ok_rows)

    for k in metrics_keys:
        vals = [r.get(k) for r in ok_rows]
        mu, sd = mean_std(vals)
        summary["mean"][k] = mu
        summary["std"][k] = sd

    return {"summary": summary, "rows": rows}


def write_csv(path: str, rows: List[Dict]):
    # stable header union
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_bar(metric: str, candidates: List[Dict], out_png: str):
    tags = [c["tag"] for c in candidates]
    means = [c["mean"].get(metric) for c in candidates]
    stds = [c["std"].get(metric) for c in candidates]

    # Replace None with nan for plotting (matplotlib will skip poorly; we handle by 0 and warn in title)
    y = np.array([v if v is not None else np.nan for v in means], dtype=np.float64)
    e = np.array([v if v is not None else 0.0 for v in stds], dtype=np.float64)

    fig = plt.figure(figsize=(max(6, 1.2 * len(tags)), 4))
    ax = fig.add_subplot(111)

    x = np.arange(len(tags))
    ax.bar(x, np.nan_to_num(y, nan=0.0), yerr=e, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} (mean ± std)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate cropped RGB vs thermal (reference) without RGB GT using multimodal similarity metrics + progress bar + bar charts."
    )
    ap.add_argument("--th_dir", required=True, help="Thermal directory (reference/GT coordinate frame)")
    ap.add_argument("--rgb_dir", action="append", required=True, help="Cropped RGB directory (repeatable for multiple candidates)")
    ap.add_argument("--tag", action="append", help="Optional tag for each --rgb_dir (same count). If omitted, uses dir basename.")
    ap.add_argument("--out_dir", required=True, help="Output directory for reports")
    ap.add_argument("--bins", type=int, default=64, help="Histogram bins for MI/NMI (default 64)")
    ap.add_argument("--ssim", action="store_true", help="Also compute SSIM on gradient magnitude (requires scikit-image)")
    ap.add_argument("--max_images", type=int, default=0, help="Limit number of evaluated pairs per candidate (0=all)")
    ap.add_argument("--stride", type=int, default=1, help="Evaluate every N images (default 1=all)")
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".tif", ".tiff"], help="File extensions to consider")
    args = ap.parse_args()

    args.exts = tuple([e.lower() for e in args.exts])
    os.makedirs(args.out_dir, exist_ok=True)

    # thermal map by stem
    th_files = list_images(args.th_dir, args.exts)
    th_map = {stem_key(fn): os.path.join(args.th_dir, fn) for fn in th_files}

    tags = args.tag[:] if args.tag else []
    if tags and len(tags) != len(args.rgb_dir):
        raise ValueError("--tag count must match --rgb_dir count.")

    all_results = []
    for i, rgb_dir in enumerate(args.rgb_dir):
        tag = tags[i] if tags else os.path.basename(os.path.normpath(rgb_dir))
        print(f"\n=== Evaluating candidate: {tag} ===")
        res = eval_one_candidate(tag, rgb_dir, th_map, args)
        all_results.append(res)

        # per-image outputs
        out_csv = os.path.join(args.out_dir, f"per_image_{tag}.csv")
        out_json = os.path.join(args.out_dir, f"summary_{tag}.json")
        write_csv(out_csv, res["rows"])
        write_json(out_json, res["summary"])
        print(f"[OK] wrote {out_csv}")
        print(f"[OK] wrote {out_json}")

    candidates = [r["summary"] for r in all_results]
    combined = {"candidates": candidates}
    out_all_json = os.path.join(args.out_dir, "summary_all.json")
    write_json(out_all_json, combined)
    print(f"\n[OK] wrote {out_all_json}")

    # summary table csv
    metrics_keys = ["mi", "nmi", "grad_ncc", "edge_dice", "edge_f1"] + (["grad_ssim"] if args.ssim else [])
    summary_rows = []
    for c in candidates:
        row = {"tag": c["tag"], "count": c["count"], "rgb_dir": c["rgb_dir"]}
        for k in metrics_keys:
            row[f"{k}_mean"] = c["mean"].get(k)
            row[f"{k}_std"] = c["std"].get(k)
        summary_rows.append(row)

    out_all_csv = os.path.join(args.out_dir, "summary_all.csv")
    write_csv(out_all_csv, summary_rows)
    print(f"[OK] wrote {out_all_csv}")

    # bar charts per metric
    for k in metrics_keys:
        out_png = os.path.join(args.out_dir, f"bar_{k}.png")
        plot_bar(k, candidates, out_png)
        print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()
