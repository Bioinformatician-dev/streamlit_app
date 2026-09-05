"""
Lentil seed phenotyping pipeline.

Ported from `lentil_phenotyping_full_workflow.ipynb`. Runs the same
segmentation -> morphology -> colour -> coat-pattern -> mm-conversion
chain, but wraps detection so it works two ways:

  1. YOLO mode   - if a trained weights file (best.pt from the notebook)
                   is present at MODEL_PATH, use it for per-seed detection
                   and class labels (Black / Defective / Dotted / Marbled /
                   Spotted / Unspotted).
  2. Classical mode - no weights available: segment seeds directly from
                   the full image with adaptive thresholding + watershed,
                   so the tool is still useful out of the box. Seeds are
                   left unclassified ("Seed").

Everything downstream of "I have one crop containing one seed" is
identical in both modes.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

CLASS_NAMES = ['Black seeds', 'Defective', 'Dotted seeds',
               'Marbled seeds', 'Spotted seeds', 'Unspotted seeds']

PATTERN_RULES = dict(
    black_L_max=35.0,
    plain_coverage_max=0.02,
    dotted_count_min=12,
    marbled_largest=0.15,
    spot_min_separation=8.0,
)

CONF, IOU, MAX_DET = 0.10, 0.5, 300

THICKNESS_FROM_WIDTH = 0.62  # placeholder ratio, see notebook section 14


# --------------------------------------------------------------------------
# Model loading (optional)
# --------------------------------------------------------------------------
_YOLO_MODEL = None
_YOLO_TRIED = False


def get_yolo_model():
    """Lazily load the YOLO model if weights exist. Returns None otherwise."""
    global _YOLO_MODEL, _YOLO_TRIED
    if _YOLO_TRIED:
        return _YOLO_MODEL
    _YOLO_TRIED = True
    if MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO(str(MODEL_PATH))
        except Exception as e:
            print(f"[pipeline] Could not load YOLO weights ({e}); "
                  f"falling back to classical segmentation.")
            _YOLO_MODEL = None
    return _YOLO_MODEL


# --------------------------------------------------------------------------
# Panel 3 + 4A: per-seed segmentation & morphology (identical to notebook)
# --------------------------------------------------------------------------
def measure_seed(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    border = np.concatenate([th[0, :], th[-1, :], th[:, 0], th[:, -1]])
    if (border > 0).mean() > 0.5:
        th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    num, lab, stats, _ = cv2.connectedComponentsWithStats((th > 0).astype(np.uint8), 8)
    if num <= 1:
        return None
    h, w = th.shape
    cid = int(lab[h // 2, w // 2])
    if cid == 0:
        cid = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_area = stats[cid, cv2.CC_STAT_AREA]
    neighbour_frags = int((stats[1:, cv2.CC_STAT_AREA] > 0.25 * main_area).sum()) - 1

    mask = ((lab == cid).astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if area < 20 or perim == 0:
        return None

    (_, _), (w_r, h_r), _ = cv2.minAreaRect(cnt)
    length, width = max(w_r, h_r), min(w_r, h_r)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hp = hull.reshape(-1, 2).astype(float)
    feret_max = float(np.max(np.linalg.norm(hp[:, None, :] - hp[None, :, :], axis=-1)))

    if len(cnt) >= 5:
        (_, _), (e1, e2), e_ang = cv2.fitEllipse(cnt)
        ell_major, ell_minor, ell_angle = max(e1, e2), min(e1, e2), e_ang
    else:
        ell_major, ell_minor, ell_angle = length, width, np.nan

    ell_area_pred = math.pi / 4 * length * width

    return {
        "area_px2": area,
        "perimeter_px": perim,
        "length_px": length,
        "width_px": width,
        "feret_max_px": feret_max,
        "feret_min_px": width,
        "ellipse_major_px": ell_major,
        "ellipse_minor_px": ell_minor,
        "ellipse_angle": ell_angle,
        "radius_px": cv2.minEnclosingCircle(cnt)[1],
        "eq_radius_px": math.sqrt(area / math.pi),
        "roundness": 4 * math.pi * area / (perim ** 2),
        "aspect_ratio": length / width if width else np.nan,
        "solidity": area / hull_area if hull_area else np.nan,
        "area_ratio": area / ell_area_pred if ell_area_pred else np.nan,
        "neighbour_frags": neighbour_frags,
        "_mask": mask,
        "_contour": cnt,
    }


# --------------------------------------------------------------------------
# Panel 4B + 4C: colour and coat pattern (identical to notebook)
# --------------------------------------------------------------------------
def classify_pattern(f):
    R = PATTERN_RULES
    if f["mean_L"] < R["black_L_max"]:
        return "black"
    if f["largest_spot"] > R["marbled_largest"]:
        return "marbled"
    if f["spot_count"] == 0 or f["spot_coverage"] < R["plain_coverage_max"]:
        return "plain"
    if f["spot_count"] >= R["dotted_count_min"]:
        return "dotted"
    return "spotted"


def colour_and_pattern(crop_bgr, mask):
    core = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    m = (core if (core > 0).sum() >= 10 else mask).astype(bool)
    if m.sum() < 10:
        return {}

    bgr = crop_bgr[m]
    gray_full = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_full[m]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[m]

    lab_full = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(float)
    lab = lab_full[m]
    L, A, B = lab[:, 0] * 100 / 255, lab[:, 1] - 128, lab[:, 2] - 128

    out = {
        "mean_R": float(bgr[:, 2].mean()), "mean_G": float(bgr[:, 1].mean()),
        "mean_B": float(bgr[:, 0].mean()),
        "mean_L": float(L.mean()), "mean_a": float(A.mean()), "mean_b": float(B.mean()),
        "mean_hue": float(hsv[:, 0].mean()), "saturation": float(hsv[:, 1].mean()),
        "brightness": float(gray.mean()),
        "texture_std": float(gray.std()),
    }

    seed_px = gray.astype(np.uint8).reshape(-1, 1)
    thr, _ = cv2.threshold(seed_px, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark, light = gray[gray <= thr], gray[gray > thr]

    if len(dark) == 0 or len(light) == 0 or \
       (light.mean() - dark.mean()) < PATTERN_RULES["spot_min_separation"]:
        thr = -1

    spot_mask = np.zeros_like(gray_full, dtype=np.uint8)
    spot_mask[m] = (gray <= thr).astype(np.uint8) * 255
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    seed_area = float(m.sum())
    n_lab, lab_s, stats_s, _ = cv2.connectedComponentsWithStats((spot_mask > 0).astype(np.uint8), 8)
    areas = stats_s[1:, cv2.CC_STAT_AREA] if n_lab > 1 else np.array([])
    areas = areas[areas >= 3]

    if seed_area and areas.sum() / seed_area > 0.90:
        areas = np.array([])
        spot_mask[:] = 0

    spot_px = spot_mask.astype(bool) & m
    if spot_px.sum() >= 5 and (m & ~spot_px).sum() >= 5:
        s_lab = lab_full[spot_px]
        b_lab = lab_full[m & ~spot_px]
        d = np.array([(s_lab[:, 0].mean() - b_lab[:, 0].mean()) * 100 / 255,
                      s_lab[:, 1].mean() - b_lab[:, 1].mean(),
                      s_lab[:, 2].mean() - b_lab[:, 2].mean()])
        delta_e = float(np.sqrt((d ** 2).sum()))
    else:
        delta_e = 0.0

    out.update({
        "spot_count": int(len(areas)),
        "spot_coverage": float(areas.sum() / seed_area) if seed_area else np.nan,
        "largest_spot": float(areas.max() / seed_area) if len(areas) and seed_area else 0.0,
        "spot_contrast_dE": delta_e,
        "dark_fraction": float((gray < gray.mean() - gray.std()).mean()),
    })
    out["pattern_class"] = classify_pattern(out)
    return out


# --------------------------------------------------------------------------
# Classical (no-model) whole-image seed segmentation
# --------------------------------------------------------------------------
def classical_detect(img):
    """Segment all seeds in a full image without a trained detector.

    Adaptive Otsu + watershed splits touching seeds. Returns a list of
    (x1, y1, x2, y2) boxes, in the same shape analyze_image expects from
    YOLO boxes.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    border = np.concatenate([th[0, :], th[-1, :], th[:, 0], th[:, -1]])
    if (border > 0).mean() > 0.5:
        th = cv2.bitwise_not(th)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img.copy(), markers)

    boxes = []
    for lbl in np.unique(markers):
        if lbl <= 1:
            continue
        mask = np.uint8(markers == lbl) * 255
        area = int(mask.sum() / 255)
        if area < 40:
            continue
        ys, xs = np.where(mask > 0)
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = 3
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


# --------------------------------------------------------------------------
# Panel 2/3: run detection (YOLO if available, else classical) + measure
# --------------------------------------------------------------------------
def analyze_image(img, name, px_per_mm=None, conf=CONF, iou=IOU, max_det=MAX_DET):
    """Detect + segment + measure every seed in one loaded (BGR) image.

    Returns (rows: list[dict], annotated_bgr: np.ndarray, mode: str)
    """
    H, W = img.shape[:2]
    vis = img.copy()
    rows = []
    model = get_yolo_model()

    if model is not None:
        mode = "yolo"
        res = model.predict(img, imgsz=640, conf=conf, iou=iou, agnostic_nms=True,
                             max_det=max_det, verbose=False)[0]
        names = model.names
        detections = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = names[int(box.cls[0])]
            confv = float(box.conf[0])
            detections.append((x1, y1, x2, y2, cls_name, confv))
    else:
        mode = "classical"
        boxes = classical_detect(img)
        detections = [(x1, y1, x2, y2, "Seed", 1.0) for (x1, y1, x2, y2) in boxes]

    seed_id = 0
    for (x1, y1, x2, y2, cls_name, confv) in detections:
        pad = 4
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
        crop = img[cy1:cy2, cx1:cx2]

        m = measure_seed(crop)
        if m is None:
            continue
        mask, cnt = m.pop("_mask"), m.pop("_contour")
        cnt_g = cnt + np.array([cx1, cy1])

        row = {
            "image": name, "seed_id": seed_id,
            "class": cls_name, "confidence": confv,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "img_w": W, "img_h": H,
            **m, **colour_and_pattern(crop, mask),
        }
        rows.append(row)
        seed_id += 1

        cv2.drawContours(vis, [cnt_g], -1, (0, 255, 0), 1)
        cv2.putText(vis, cls_name[:5], (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.putText(vis, f"Seeds: {len(rows)}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = to_physical(df, px_per_mm)
    return df, vis, mode


# --------------------------------------------------------------------------
# Panel 5: pixels -> mm and derived physical properties
# --------------------------------------------------------------------------
def to_physical(df, px_per_mm=None):
    d = df.copy()
    mm_cols = ["length_mm", "width_mm", "thickness_mm", "area_mm2", "perimeter_mm",
               "gmd_mm", "amd_mm", "sphericity", "surface_area_mm2", "volume_mm3"]
    if not px_per_mm:
        for c in mm_cols:
            d[c] = np.nan
        return d

    d["length_mm"] = d["length_px"] / px_per_mm
    d["width_mm"] = d["width_px"] / px_per_mm
    d["area_mm2"] = d["area_px2"] / px_per_mm ** 2
    d["perimeter_mm"] = d["perimeter_px"] / px_per_mm
    d["thickness_mm"] = d["width_mm"] * THICKNESS_FROM_WIDTH

    L, W, T = d["length_mm"], d["width_mm"], d["thickness_mm"]
    d["gmd_mm"] = (L * W * T) ** (1 / 3)
    d["amd_mm"] = (L + W + T) / 3
    d["sphericity"] = d["gmd_mm"] / L
    d["surface_area_mm2"] = math.pi * d["gmd_mm"] ** 2
    d["volume_mm3"] = (math.pi / 6) * L * W * T
    return d


# --------------------------------------------------------------------------
# Summary statistics across all measured seeds
# --------------------------------------------------------------------------
def compute_statistics(df_seeds):
    shape_cols = [c for c in ["area_px2", "perimeter_px", "length_px", "width_px",
                               "feret_max_px", "roundness", "aspect_ratio",
                               "solidity", "area_ratio"] if c in df_seeds.columns]
    colour_cols = [c for c in ["mean_L", "mean_a", "mean_b", "mean_hue", "saturation",
                                "brightness", "texture_std", "spot_count",
                                "spot_coverage", "largest_spot", "spot_contrast_dE"]
                   if c in df_seeds.columns]
    mm_cols = [c for c in ["length_mm", "width_mm", "thickness_mm", "area_mm2",
                            "gmd_mm", "amd_mm", "sphericity", "surface_area_mm2",
                            "volume_mm3"] if c in df_seeds.columns
               and df_seeds[c].notna().any()]
    trait_cols = shape_cols + colour_cols + mm_cols
    if not trait_cols:
        return pd.DataFrame()
    stats = df_seeds[trait_cols].agg(["count", "mean", "std", "min", "median", "max"]).T
    stats["cv_pct"] = 100 * stats["std"] / stats["mean"]
    return stats.round(4)
