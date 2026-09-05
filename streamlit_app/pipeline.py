"""
Lentil Seed Phenotyping Pipeline
================================

Standalone backend for lentil seed image analysis.

Modes
-----
1. YOLO segmentation mode
   Uses models/best.pt when available.

2. Classical mode
   Falls back to thresholding + watershed when YOLO weights
   are unavailable.

Outputs
-------
For every detected seed:
- Detection/classification
- Bounding box
- Centroid
- Segmentation-based morphology
- Color measurements
- Spot/pattern measurements
- Optional physical measurements

Designed to be imported by a Streamlit app.

Example
-------
from pipeline import analyze_image

df, annotated, mode = analyze_image(
    image,
    "sample.jpg",
    px_per_mm=10
)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR /  "best.pt"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Black seeds",
    "Defective",
    "Dotted seeds",
    "Marbled seeds",
    "Spotted seeds",
    "Unspotted seeds",
]


# ============================================================
# PATTERN RULES
# ============================================================

PATTERN_RULES = {
    "black_L_max": 35.0,
    "plain_coverage_max": 0.02,
    "dotted_count_min": 12,
    "marbled_largest": 0.15,
    "spot_min_separation": 8.0,
}


# ============================================================
# YOLO SETTINGS
# ============================================================

CONF = 0.10
IOU = 0.50
MAX_DET = 300

YOLO_IMAGE_SIZE = 640


# ============================================================
# PHYSICAL MEASUREMENT SETTINGS
# ============================================================

# IMPORTANT:
# This is an assumption-based placeholder.
# Replace it with an experimentally validated ratio if you
# actually want reliable thickness/volume values.
THICKNESS_FROM_WIDTH = 0.62


# ============================================================
# GLOBAL YOLO MODEL CACHE
# ============================================================

_YOLO_MODEL = None
_YOLO_TRIED = False


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def reset_model_cache() -> None:
    """
    Reset cached YOLO model.

    Useful in Streamlit if the model path changes.
    """
    global _YOLO_MODEL, _YOLO_TRIED

    _YOLO_MODEL = None
    _YOLO_TRIED = False


def get_class_name(model, class_id: int) -> str:
    """
    Safely retrieve a class name from a YOLO model.
    """

    names = getattr(model, "names", {})

    if isinstance(names, dict):
        return str(names.get(class_id, f"Class {class_id}"))

    if isinstance(names, list):
        if 0 <= class_id < len(names):
            return str(names[class_id])

    return f"Class {class_id}"


def clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """
    Clip bounding box to image boundaries.
    """

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))

    return x1, y1, x2, y2


# ============================================================
# YOLO MODEL
# ============================================================

def get_yolo_model():
    """
    Load YOLO segmentation model if models/best.pt exists.

    Returns
    -------
    YOLO model or None
    """

    global _YOLO_MODEL
    global _YOLO_TRIED

    if _YOLO_TRIED:
        return _YOLO_MODEL

    _YOLO_TRIED = True

    if not MODEL_PATH.exists():
        print(
            f"[pipeline] YOLO weights not found at: "
            f"{MODEL_PATH}"
        )
        print("[pipeline] Using classical segmentation fallback.")
        return None

    try:
        from ultralytics import YOLO

        print(f"[pipeline] Loading YOLO model: {MODEL_PATH}")

        _YOLO_MODEL = YOLO(str(MODEL_PATH))

        print("[pipeline] YOLO model loaded successfully.")

    except Exception as e:

        print(
            "[pipeline] Could not load YOLO weights "
            f"({e}). Falling back to classical segmentation."
        )

        _YOLO_MODEL = None

    return _YOLO_MODEL


# ============================================================
# CLASSICAL SEGMENTATION
# ============================================================

def classical_detect(
    img: np.ndarray,
    min_area: int = 40,
) -> List[Dict[str, Any]]:
    """
    Detect individual seeds using thresholding + watershed.

    Used when YOLO weights are unavailable.

    Returns
    -------
    list of dictionaries containing:
        box
        mask
        confidence
        class
    """

    if img is None or img.size == 0:
        return []

    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # Otsu threshold
    _, th = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Determine whether foreground/background are inverted.
    border = np.concatenate(
        [
            th[0, :],
            th[-1, :],
            th[:, 0],
            th[:, -1],
        ]
    )

    if (border > 0).mean() > 0.5:
        th = cv2.bitwise_not(th)

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    opened = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2,
    )

    # Background
    sure_bg = cv2.dilate(
        opened,
        kernel,
        iterations=3,
    )

    # Distance transform
    dist = cv2.distanceTransform(
        opened,
        cv2.DIST_L2,
        5,
    )

    if dist.max() <= 0:
        return []

    # Foreground seeds
    _, sure_fg = cv2.threshold(
        dist,
        0.4 * dist.max(),
        255,
        0,
    )

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(
        sure_bg,
        sure_fg,
    )

    # Connected components
    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    # Watershed
    watershed_input = img.copy()

    markers = cv2.watershed(
        watershed_input,
        markers,
    )

    detections = []

    for label in np.unique(markers):

        if label <= 1:
            continue

        mask = (
            np.uint8(markers == label)
            * 255
        )

        area = int(
            np.sum(mask > 0)
        )

        if area < min_area:
            continue

        ys, xs = np.where(
            mask > 0
        )

        if len(xs) == 0:
            continue

        x1 = int(xs.min())
        x2 = int(xs.max()) + 1

        y1 = int(ys.min())
        y2 = int(ys.max()) + 1

        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class": "Seed",
                "confidence": 1.0,
                "mask": mask,
            }
        )

    return detections


# ============================================================
# MASK / CONTOUR HELPERS
# ============================================================

def contour_from_mask(
    mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Extract the largest contour from a binary mask.
    """

    if mask is None or mask.size == 0:
        return None

    binary = np.uint8(
        mask > 0
    ) * 255

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    return contour


def clean_mask(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Clean a segmentation mask.
    """

    mask = np.uint8(mask > 0) * 255

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return mask


# ============================================================
# MORPHOLOGICAL MEASUREMENTS
# ============================================================

def measure_seed(
    crop_bgr: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Measure a single seed from a cropped image.

    This function performs classical segmentation within
    the crop.

    Used as fallback when YOLO segmentation masks are
    unavailable.
    """

    if crop_bgr is None or crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    _, th = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    border = np.concatenate(
        [
            th[0, :],
            th[-1, :],
            th[:, 0],
            th[:, -1],
        ]
    )

    if (border > 0).mean() > 0.5:
        th = cv2.bitwise_not(th)

    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=2,
    )

    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        (th > 0).astype(np.uint8),
        8,
    )

    if num <= 1:
        return None

    h, w = th.shape

    # Try to select the component at crop center.
    cid = int(
        lab[h // 2, w // 2]
    )

    # If center is background, use largest component.
    if cid == 0:

        cid = (
            1
            + int(
                np.argmax(
                    stats[
                        1:,
                        cv2.CC_STAT_AREA,
                    ]
                )
            )
        )

    main_area = stats[
        cid,
        cv2.CC_STAT_AREA,
    ]

    # Count substantial neighbouring components.
    neighbour_frags = int(
        (
            stats[
                1:,
                cv2.CC_STAT_AREA,
            ]
            > 0.25 * main_area
        ).sum()
    ) - 1

    mask = (
        (lab == cid).astype(np.uint8)
        * 255
    )

    cnt = contour_from_mask(mask)

    if cnt is None:
        return None

    area = float(
        cv2.contourArea(cnt)
    )

    perim = float(
        cv2.arcLength(
            cnt,
            True,
        )
    )

    if area < 20 or perim == 0:
        return None

    return calculate_morphology(
        cnt,
        mask,
        neighbour_frags=neighbour_frags,
    )


def calculate_morphology(
    cnt: np.ndarray,
    mask: np.ndarray,
    neighbour_frags: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Calculate morphology from an already available contour/mask.

    This is the preferred function for YOLO segmentation.
    """

    if cnt is None:
        return None

    area = float(
        cv2.contourArea(cnt)
    )

    perim = float(
        cv2.arcLength(
            cnt,
            True,
        )
    )

    if area <= 0 or perim <= 0:
        return None

    # Minimum-area rectangle
    (_, _), (w_r, h_r), rect_angle = cv2.minAreaRect(cnt)

    length = float(
        max(w_r, h_r)
    )

    width = float(
        min(w_r, h_r)
    )

    # Convex hull
    hull = cv2.convexHull(cnt)

    hull_area = float(
        cv2.contourArea(hull)
    )

    # Feret diameter
    hp = hull.reshape(
        -1,
        2,
    ).astype(float)

    if len(hp) >= 2:

        # Avoid enormous memory use for extremely large masks.
        if len(hp) > 500:
            indices = np.linspace(
                0,
                len(hp) - 1,
                500,
            ).astype(int)

            hp = hp[indices]

        diff = (
            hp[:, None, :]
            - hp[None, :, :]
        )

        distances = np.sqrt(
            np.sum(
                diff ** 2,
                axis=-1,
            )
        )

        feret_max = float(
            distances.max()
        )

    else:
        feret_max = width

    feret_min = width

    # Ellipse
    if len(cnt) >= 5:

        try:

            (
                _,
                (e1, e2),
                e_ang,
            ) = cv2.fitEllipse(cnt)

            ell_major = float(
                max(e1, e2)
            )

            ell_minor = float(
                min(e1, e2)
            )

            ell_angle = float(e_ang)

        except Exception:

            ell_major = length
            ell_minor = width
            ell_angle = np.nan

    else:

        ell_major = length
        ell_minor = width
        ell_angle = np.nan

    # Equivalent ellipse area
    ell_area_pred = (
        math.pi
        / 4.0
        * length
        * width
    )

    # Enclosing circle
    (_, _), radius = cv2.minEnclosingCircle(
        cnt
    )

    radius = float(radius)

    # Equivalent radius
    eq_radius = math.sqrt(
        area / math.pi
    )

    # Roundness / circularity
    roundness = (
        4.0
        * math.pi
        * area
        / (perim ** 2)
    )

    # Aspect ratio
    aspect_ratio = (
        length / width
        if width > 0
        else np.nan
    )

    # Solidity
    solidity = (
        area / hull_area
        if hull_area > 0
        else np.nan
    )

    # Area ratio
    area_ratio = (
        area / ell_area_pred
        if ell_area_pred > 0
        else np.nan
    )

    # Centroid
    moments = cv2.moments(cnt)

    if moments["m00"] != 0:

        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )

    else:

        cx = np.nan
        cy = np.nan

    return {
        "area_px2": area,
        "perimeter_px": perim,

        "length_px": length,
        "width_px": width,

        "feret_max_px": feret_max,
        "feret_min_px": feret_min,

        "ellipse_major_px": ell_major,
        "ellipse_minor_px": ell_minor,
        "ellipse_angle": ell_angle,

        "rect_angle": rect_angle,

        "radius_px": radius,
        "eq_radius_px": eq_radius,

        "roundness": roundness,
        "circularity": roundness,

        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "area_ratio": area_ratio,

        "centroid_local_x": cx,
        "centroid_local_y": cy,

        "neighbour_frags": neighbour_frags,

        "_mask": mask,
        "_contour": cnt,
    }


# ============================================================
# COLOR + PATTERN ANALYSIS
# ============================================================

def classify_pattern(
    features: Dict[str, Any]
) -> str:
    """
    Rule-based seed surface pattern classification.
    """

    R = PATTERN_RULES

    mean_L = features.get(
        "mean_L",
        np.nan,
    )

    largest_spot = features.get(
        "largest_spot",
        0.0,
    )

    spot_count = features.get(
        "spot_count",
        0,
    )

    spot_coverage = features.get(
        "spot_coverage",
        0.0,
    )

    if (
        np.isfinite(mean_L)
        and mean_L < R["black_L_max"]
    ):
        return "black"

    if (
        largest_spot
        > R["marbled_largest"]
    ):
        return "marbled"

    if (
        spot_count == 0
        or spot_coverage
        < R["plain_coverage_max"]
    ):
        return "plain"

    if (
        spot_count
        >= R["dotted_count_min"]
    ):
        return "dotted"

    return "spotted"


def colour_and_pattern(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate color, texture and surface-pattern features.
    """

    if (
        crop_bgr is None
        or crop_bgr.size == 0
        or mask is None
    ):
        return {}

    mask = np.uint8(
        mask > 0
    ) * 255

    # Erode boundary slightly so background pixels do not
    # contaminate color measurements.
    core = cv2.erode(
        mask,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    if (
        np.sum(core > 0)
        >= 10
    ):
        m = core > 0
    else:
        m = mask > 0

    if m.sum() < 10:
        return {}

    # --------------------------------------------------------
    # BGR
    # --------------------------------------------------------

    bgr = crop_bgr[m]

    # --------------------------------------------------------
    # Gray
    # --------------------------------------------------------

    gray_full = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    gray = gray_full[m]

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv_full = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2HSV,
    )

    hsv = hsv_full[m]

    # --------------------------------------------------------
    # LAB
    # --------------------------------------------------------

    lab_full = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2LAB,
    ).astype(float)

    lab = lab_full[m]

    L = (
        lab[:, 0]
        * 100.0
        / 255.0
    )

    A = (
        lab[:, 1]
        - 128.0
    )

    B = (
        lab[:, 2]
        - 128.0
    )

    # --------------------------------------------------------
    # Base color statistics
    # --------------------------------------------------------

    out = {
        "mean_R": float(
            bgr[:, 2].mean()
        ),

        "mean_G": float(
            bgr[:, 1].mean()
        ),

        "mean_B": float(
            bgr[:, 0].mean()
        ),

        "std_R": float(
            bgr[:, 2].std()
        ),

        "std_G": float(
            bgr[:, 1].std()
        ),

        "std_B": float(
            bgr[:, 0].std()
        ),

        "mean_L": float(
            L.mean()
        ),

        "mean_a": float(
            A.mean()
        ),

        "mean_b": float(
            B.mean()
        ),

        "std_L": float(
            L.std()
        ),

        "std_a": float(
            A.std()
        ),

        "std_b": float(
            B.std()
        ),

        "mean_hue": float(
            hsv[:, 0].mean()
        ),

        "hue_std": float(
            hsv[:, 0].std()
        ),

        "saturation": float(
            hsv[:, 1].mean()
        ),

        "saturation_std": float(
            hsv[:, 1].std()
        ),

        "brightness": float(
            gray.mean()
        ),

        "texture_std": float(
            gray.std()
        ),
    }

    # ========================================================
    # SPOT DETECTION
    # ========================================================

    seed_px = (
        gray
        .astype(np.uint8)
        .reshape(-1, 1)
    )

    thr, _ = cv2.threshold(
        seed_px,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    thr = float(thr)

    dark = gray[
        gray <= thr
    ]

    light = gray[
        gray > thr
    ]

    # If there is not enough contrast, disable spot detection.
    if (
        len(dark) == 0
        or len(light) == 0
        or (
            light.mean()
            - dark.mean()
        )
        < PATTERN_RULES[
            "spot_min_separation"
        ]
    ):
        thr = -1

    spot_mask = np.zeros_like(
        gray_full,
        dtype=np.uint8,
    )

    if thr >= 0:

        spot_mask[m] = (
            gray <= thr
        ).astype(np.uint8) * 255

    spot_mask = cv2.morphologyEx(
        spot_mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    seed_area = float(
        m.sum()
    )

    # Connected components
    n_lab, lab_s, stats_s, _ = (
        cv2.connectedComponentsWithStats(
            (spot_mask > 0)
            .astype(np.uint8),
            8,
        )
    )

    if n_lab > 1:

        areas = stats_s[
            1:,
            cv2.CC_STAT_AREA,
        ]

    else:

        areas = np.array([])

    # Remove tiny noise
    areas = areas[
        areas >= 3
    ]

    # If almost the entire seed is detected as "spots",
    # assume thresholding failed.
    if (
        seed_area
        and areas.sum()
        / seed_area
        > 0.90
    ):

        areas = np.array([])

        spot_mask[:] = 0

    spot_px = (
        spot_mask.astype(bool)
        & m
    )

    # ========================================================
    # SPOT COLOR CONTRAST
    # ========================================================

    if (
        spot_px.sum() >= 5
        and (
            m
            & ~spot_px
        ).sum()
        >= 5
    ):

        s_lab = lab_full[
            spot_px
        ]

        b_lab = lab_full[
            m & ~spot_px
        ]

        d = np.array(
            [
                (
                    s_lab[:, 0].mean()
                    - b_lab[:, 0].mean()
                )
                * 100.0
                / 255.0,

                s_lab[:, 1].mean()
                - b_lab[:, 1].mean(),

                s_lab[:, 2].mean()
                - b_lab[:, 2].mean(),
            ]
        )

        delta_e = float(
            np.sqrt(
                (d ** 2).sum()
            )
        )

    else:

        delta_e = 0.0

    # ========================================================
    # PATTERN FEATURES
    # ========================================================

    spot_count = int(
        len(areas)
    )

    spot_coverage = (
        float(
            areas.sum()
            / seed_area
        )
        if seed_area
        else np.nan
    )

    largest_spot = (
        float(
            areas.max()
            / seed_area
        )
        if len(areas)
        and seed_area
        else 0.0
    )

    dark_fraction = float(
        (
            gray
            < gray.mean()
            - gray.std()
        ).mean()
    )

    out.update(
        {
            "spot_count": spot_count,

            "spot_coverage":
                spot_coverage,

            "largest_spot":
                largest_spot,

            "spot_contrast_dE":
                delta_e,

            "dark_fraction":
                dark_fraction,

            "pattern_class":
                classify_pattern(
                    {
                        **out,
                        "spot_count":
                            spot_count,
                        "spot_coverage":
                            spot_coverage,
                        "largest_spot":
                            largest_spot,
                    }
                ),
        }
    )

    return out


# ============================================================
# PHYSICAL MEASUREMENTS
# ============================================================

def to_physical(
    df: pd.DataFrame,
    px_per_mm: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convert pixel measurements into physical units.

    Parameters
    ----------
    df:
        Seed measurements.

    px_per_mm:
        Number of pixels corresponding to one millimeter.

    Notes
    -----
    Thickness is estimated using:

        thickness = width * THICKNESS_FROM_WIDTH

    This is only an assumption until experimentally validated.
    """

    d = df.copy()

    mm_cols = [
        "length_mm",
        "width_mm",
        "thickness_mm",
        "area_mm2",
        "perimeter_mm",
        "gmd_mm",
        "amd_mm",
        "sphericity",
        "surface_area_mm2",
        "volume_mm3",
    ]

    # No calibration supplied.
    if (
        px_per_mm is None
        or px_per_mm <= 0
    ):

        for c in mm_cols:
            d[c] = np.nan

        return d

    # --------------------------------------------------------
    # Linear measurements
    # --------------------------------------------------------

    d["length_mm"] = (
        d["length_px"]
        / px_per_mm
    )

    d["width_mm"] = (
        d["width_px"]
        / px_per_mm
    )

    d["feret_max_mm"] = (
        d["feret_max_px"]
        / px_per_mm
    )

    d["feret_min_mm"] = (
        d["feret_min_px"]
        / px_per_mm
    )

    d["ellipse_major_mm"] = (
        d["ellipse_major_px"]
        / px_per_mm
    )

    d["ellipse_minor_mm"] = (
        d["ellipse_minor_px"]
        / px_per_mm
    )

    d["radius_mm"] = (
        d["radius_px"]
        / px_per_mm
    )

    d["eq_radius_mm"] = (
        d["eq_radius_px"]
        / px_per_mm
    )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    d["area_mm2"] = (
        d["area_px2"]
        / (px_per_mm ** 2)
    )

    # --------------------------------------------------------
    # Perimeter
    # --------------------------------------------------------

    d["perimeter_mm"] = (
        d["perimeter_px"]
        / px_per_mm
    )

    # --------------------------------------------------------
    # Thickness estimate
    # --------------------------------------------------------

    d["thickness_mm"] = (
        d["width_mm"]
        * THICKNESS_FROM_WIDTH
    )

    # --------------------------------------------------------
    # Derived seed dimensions
    # --------------------------------------------------------

    L = d["length_mm"]
    W = d["width_mm"]
    T = d["thickness_mm"]

    # Geometric mean diameter
    d["gmd_mm"] = (
        L * W * T
    ) ** (1.0 / 3.0)

    # Arithmetic mean diameter
    d["amd_mm"] = (
        L + W + T
    ) / 3.0

    # Sphericity
    d["sphericity"] = (
        d["gmd_mm"]
        / L.replace(0, np.nan)
    )

    # Approximate surface area
    d["surface_area_mm2"] = (
        math.pi
        * d["gmd_mm"] ** 2
    )

    # Ellipsoid approximation
    d["volume_mm3"] = (
        math.pi
        / 6.0
        * L
        * W
        * T
    )

    return d


# ============================================================
# YOLO SEGMENTATION EXTRACTION
# ============================================================

def extract_yolo_detections(
    result,
) -> List[Dict[str, Any]]:
    """
    Extract detections from an Ultralytics YOLO result.

    If segmentation masks are available, they are returned.

    Returns dictionaries with:
        x1, y1, x2, y2
        class
        confidence
        mask
    """

    detections = []

    if result is None:
        return detections

    boxes = getattr(
        result,
        "boxes",
        None,
    )

    masks = getattr(
        result,
        "masks",
        None,
    )

    if boxes is None:
        return detections

    # --------------------------------------------------------
    # Mask data
    # --------------------------------------------------------

    mask_data = None

    if masks is not None:

        try:
            mask_data = (
                masks.data
                .detach()
                .cpu()
                .numpy()
            )

        except Exception:

            mask_data = None

    for i, box in enumerate(boxes):

        try:

            xyxy = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = map(
                int,
                xyxy,
            )

            class_id = int(
                box.cls[0]
                .detach()
                .cpu()
                .item()
            )

            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )

        except Exception:

            continue

        cls_name = get_class_name(
            result.names
            if hasattr(result, "names")
            else {},
            class_id,
        )

        # ----------------------------------------------------
        # Full-resolution segmentation mask
        # ----------------------------------------------------

        full_mask = None

        if (
            mask_data is not None
            and i < len(mask_data)
        ):

            small_mask = (
                mask_data[i] > 0.5
            ).astype(np.uint8) * 255

            # Ultralytics mask.data may be at the model
            # mask resolution. Resize to original image size.
            orig_shape = getattr(
                result,
                "orig_shape",
                None,
            )

            if (
                orig_shape is not None
                and len(orig_shape) >= 2
            ):

                orig_h = int(
                    orig_shape[0]
                )

                orig_w = int(
                    orig_shape[1]
                )

                full_mask = cv2.resize(
                    small_mask,
                    (
                        orig_w,
                        orig_h,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

            else:

                # Fallback to box-sized mask if original shape
                # cannot be determined.
                full_mask = small_mask

        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class": cls_name,
                "class_id": class_id,
                "confidence": confidence,
                "mask": full_mask,
            }
        )

    return detections


# ============================================================
# CREATE LOCAL CROP MASK
# ============================================================

def crop_full_mask(
    full_mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Optional[np.ndarray]:
    """
    Extract a segmentation mask corresponding to a crop.
    """

    if full_mask is None:
        return None

    H, W = full_mask.shape[:2]

    x1, y1, x2, y2 = clip_box(
        x1,
        y1,
        x2,
        y2,
        W,
        H,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return full_mask[
        y1:y2,
        x1:x2,
    ].copy()


# ============================================================
# MEASURE YOLO MASK
# ============================================================

def measure_yolo_mask(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Measure a seed directly from its YOLO segmentation mask.
    """

    if (
        crop_bgr is None
        or crop_bgr.size == 0
        or mask is None
    ):
        return None

    mask = clean_mask(mask)

    cnt = contour_from_mask(
        mask
    )

    if cnt is None:
        return None

    morphology = calculate_morphology(
        cnt,
        mask,
    )

    if morphology is None:
        return None

    return morphology


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(
    img: np.ndarray,
    name: str = "image",
    px_per_mm: Optional[float] = None,
    conf: float = CONF,
    iou: float = IOU,
    max_det: int = MAX_DET,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    str,
]:
    """
    Analyze one image.

    Parameters
    ----------
    img:
        BGR OpenCV image.

    name:
        Image filename/name.

    px_per_mm:
        Calibration factor.

    conf:
        YOLO confidence threshold.

    iou:
        YOLO IoU threshold.

    max_det:
        Maximum detections.

    Returns
    -------
    df:
        One row per seed.

    annotated:
        Annotated BGR image.

    mode:
        "yolo" or "classical"
    """

    if img is None or img.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    H, W = img.shape[:2]

    vis = img.copy()

    model = get_yolo_model()

    # ========================================================
    # DETECTION
    # ========================================================

    if model is not None:

        mode = "yolo"

        try:

            results = model.predict(
                source=img,
                imgsz=YOLO_IMAGE_SIZE,
                conf=conf,
                iou=iou,
                agnostic_nms=True,
                max_det=max_det,
                verbose=False,
            )

            if not results:
                detections = []

            else:

                detections = (
                    extract_yolo_detections(
                        results[0]
                    )
                )

        except Exception as e:

            print(
                "[pipeline] YOLO inference failed: "
                f"{e}"
            )

            print(
                "[pipeline] Falling back "
                "to classical segmentation."
            )

            mode = "classical"

            detections = (
                classical_detect(img)
            )

    else:

        mode = "classical"

        detections = (
            classical_detect(img)
        )

    # ========================================================
    # ANALYZE EACH SEED
    # ========================================================

    rows = []

    seed_id = 0

    for detection in detections:

        x1 = int(
            detection["x1"]
        )

        y1 = int(
            detection["y1"]
        )

        x2 = int(
            detection["x2"]
        )

        y2 = int(
            detection["y2"]
        )

        cls_name = str(
            detection.get(
                "class",
                "Seed",
            )
        )

        confidence = float(
            detection.get(
                "confidence",
                1.0,
            )
        )

        # ----------------------------------------------------
        # Clip box
        # ----------------------------------------------------

        x1, y1, x2, y2 = clip_box(
            x1,
            y1,
            x2,
            y2,
            W,
            H,
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        pad = 4

        cx1 = max(
            0,
            x1 - pad,
        )

        cy1 = max(
            0,
            y1 - pad,
        )

        cx2 = min(
            W,
            x2 + pad,
        )

        cy2 = min(
            H,
            y2 + pad,
        )

        crop = img[
            cy1:cy2,
            cx1:cx2,
        ]

        # ====================================================
        # SEGMENTATION MASK
        # ====================================================

        full_mask = detection.get(
            "mask"
        )

        local_mask = None

        if full_mask is not None:

            local_mask = crop_full_mask(
                full_mask,
                cx1,
                cy1,
                cx2,
                cy2,
            )

            if local_mask is not None:

                # Resize if needed.
                if (
                    local_mask.shape[:2]
                    != crop.shape[:2]
                ):

                    local_mask = cv2.resize(
                        local_mask,
                        (
                            crop.shape[1],
                            crop.shape[0],
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    )

                morphology = (
                    measure_yolo_mask(
                        crop,
                        local_mask,
                    )
                )

            else:

                morphology = None

        else:

            morphology = None

        # ====================================================
        # FALLBACK CROP SEGMENTATION
        # ====================================================

        if morphology is None:

            fallback = (
                measure_seed(crop)
            )

            if fallback is None:
                continue

            local_mask = fallback.pop(
                "_mask"
            )

            morphology = fallback

        else:

            # Remove internal objects from
            # morphology before DataFrame.
            local_mask = morphology.pop(
                "_mask"
            )

            morphology.pop(
                "_contour",
                None,
            )

        # ----------------------------------------------------
        # Contour
        # ----------------------------------------------------

        cnt = contour_from_mask(
            local_mask
        )

        if cnt is None:
            continue

        # ----------------------------------------------------
        # Global contour coordinates
        # ----------------------------------------------------

        cnt_g = (
            cnt.astype(np.int32)
            + np.array(
                [cx1, cy1],
                dtype=np.int32,
            )
        )

        # ----------------------------------------------------
        # Global centroid
        # ----------------------------------------------------

        M = cv2.moments(
            cnt_g
        )

        if M["m00"] != 0:

            centroid_x = (
                M["m10"]
                / M["m00"]
            )

            centroid_y = (
                M["m01"]
                / M["m00"]
            )

        else:

            centroid_x = (
                x1 + x2
            ) / 2.0

            centroid_y = (
                y1 + y2
            ) / 2.0

        # ====================================================
        # COLOR / PATTERN
        # ====================================================

        color_features = (
            colour_and_pattern(
                crop,
                local_mask,
            )
        )

        # ====================================================
        # ROW
        # ====================================================

        row = {
            "image": name,

            "seed_id": seed_id + 1,

            "class": cls_name,

            "confidence": confidence,

            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,

            "cx": float(
                centroid_x
            ),

            "cy": float(
                centroid_y
            ),

            "img_w": W,
            "img_h": H,

            "detection_mode": mode,

            **morphology,

            **color_features,
        }

        rows.append(row)

        # ====================================================
        # DRAW SEGMENTATION
        # ====================================================

        # Draw mask outline.
        cv2.drawContours(
            vis,
            [cnt_g],
            -1,
            (0, 255, 0),
            2,
        )

        # Bounding box
        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            1,
        )

        # ====================================================
        # SEED LABEL
        # ====================================================

        label = (
            f"#{seed_id + 1} "
            f"{cls_name}"
        )

        label_y = max(
            18,
            y1 - 5,
        )

        cv2.putText(
            vis,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # Confidence
        conf_text = (
            f"{confidence:.2f}"
        )

        cv2.putText(
            vis,
            conf_text,
            (
                x1,
                min(
                    H - 5,
                    y2 + 14,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

        seed_id += 1

    # ========================================================
    # IMAGE SUMMARY
    # ========================================================

    total_seeds = len(
        rows
    )

    cv2.rectangle(
        vis,
        (0, 0),
        (W, 35),
        (0, 0, 0),
        -1,
    )

    summary = (
        f"Seeds: {total_seeds} | "
        f"Mode: {mode.upper()}"
    )

    cv2.putText(
        vis,
        summary,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    if not df.empty:

        df = to_physical(
            df,
            px_per_mm,
        )

    return (
        df,
        vis,
        mode,
    )


# ============================================================
# MULTI-IMAGE ANALYSIS
# ============================================================

def analyze_images(
    images: List[Tuple[str, np.ndarray]],
    px_per_mm: Optional[float] = None,
    conf: float = CONF,
    iou: float = IOU,
    max_det: int = MAX_DET,
) -> Tuple[
    pd.DataFrame,
    Dict[str, np.ndarray],
    Dict[str, str],
]:
    """
    Analyze multiple images.

    Parameters
    ----------
    images:
        List of:
            (filename, BGR image)

    Returns
    -------
    all_df:
        Combined seed DataFrame.

    annotated_images:
        Dictionary filename -> annotated image.

    modes:
        Dictionary filename -> detection mode.
    """

    all_rows = []

    annotated_images = {}

    modes = {}

    for name, image in images:

        df, annotated, mode = (
            analyze_image(
                image,
                name=name,
                px_per_mm=px_per_mm,
                conf=conf,
                iou=iou,
                max_det=max_det,
            )
        )

        if not df.empty:

            all_rows.append(
                df
            )

        annotated_images[
            name
        ] = annotated

        modes[
            name
        ] = mode

    if all_rows:

        all_df = pd.concat(
            all_rows,
            ignore_index=True,
        )

    else:

        all_df = pd.DataFrame()

    return (
        all_df,
        annotated_images,
        modes,
    )


# ============================================================
# CLASS COUNTS
# ============================================================

def class_counts(
    df_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return seed counts by class.
    """

    if (
        df_seeds is None
        or df_seeds.empty
        or "class"
        not in df_seeds.columns
    ):

        return pd.DataFrame(
            columns=[
                "class",
                "count",
            ]
        )

    counts = (
        df_seeds[
            "class"
        ]
        .value_counts()
        .rename_axis(
            "class"
        )
        .reset_index(
            name="count"
        )
    )

    return counts


# ============================================================
# PATTERN COUNTS
# ============================================================

def pattern_counts(
    df_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return counts by rule-based pattern class.
    """

    if (
        df_seeds is None
        or df_seeds.empty
        or "pattern_class"
        not in df_seeds.columns
    ):

        return pd.DataFrame(
            columns=[
                "pattern_class",
                "count",
            ]
        )

    return (
        df_seeds[
            "pattern_class"
        ]
        .value_counts()
        .rename_axis(
            "pattern_class"
        )
        .reset_index(
            name="count"
        )
    )


# ============================================================
# IMAGE SUMMARY
# ============================================================

def image_summary(
    df_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create summary statistics for each image.
    """

    if (
        df_seeds is None
        or df_seeds.empty
    ):

        return pd.DataFrame()

    records = []

    for image_name, group in (
        df_seeds.groupby(
            "image",
            dropna=False,
        )
    ):

        record = {
            "image": image_name,

            "total_seeds": len(group),

            "mean_area_px2":
                group["area_px2"].mean()
                if "area_px2"
                in group
                else np.nan,

            "mean_length_px":
                group["length_px"].mean()
                if "length_px"
                in group
                else np.nan,

            "mean_width_px":
                group["width_px"].mean()
                if "width_px"
                in group
                else np.nan,

            "mean_roundness":
                group["roundness"].mean()
                if "roundness"
                in group
                else np.nan,

            "mean_aspect_ratio":
                group["aspect_ratio"].mean()
                if "aspect_ratio"
                in group
                else np.nan,
        }

        if "length_mm" in group:

            record[
                "mean_length_mm"
            ] = group[
                "length_mm"
            ].mean()

            record[
                "mean_width_mm"
            ] = group[
                "width_mm"
            ].mean()

            record[
                "mean_area_mm2"
            ] = group[
                "area_mm2"
            ].mean()

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# STATISTICS
# ============================================================

def compute_statistics(
    df_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for seed traits.

    Returns:
        count, mean, std, min, median, max, CV%
    """

    if (
        df_seeds is None
        or df_seeds.empty
    ):

        return pd.DataFrame()

    shape_cols = [
        c
        for c in [
            "area_px2",
            "perimeter_px",
            "length_px",
            "width_px",
            "feret_max_px",
            "feret_min_px",
            "roundness",
            "circularity",
            "aspect_ratio",
            "solidity",
            "area_ratio",
        ]
        if c in df_seeds.columns
    ]

    colour_cols = [
        c
        for c in [
            "mean_R",
            "mean_G",
            "mean_B",
            "mean_L",
            "mean_a",
            "mean_b",
            "mean_hue",
            "saturation",
            "brightness",
            "texture_std",
            "spot_count",
            "spot_coverage",
            "largest_spot",
            "spot_contrast_dE",
            "dark_fraction",
        ]
        if c in df_seeds.columns
    ]

    mm_cols = [
        c
        for c in [
            "length_mm",
            "width_mm",
            "thickness_mm",
            "area_mm2",
            "perimeter_mm",
            "gmd_mm",
            "amd_mm",
            "sphericity",
            "surface_area_mm2",
            "volume_mm3",
        ]
        if (
            c in df_seeds.columns
            and df_seeds[c].notna().any()
        )
    ]

    trait_cols = (
        shape_cols
        + colour_cols
        + mm_cols
    )

    if not trait_cols:
        return pd.DataFrame()

    stats = (
        df_seeds[
            trait_cols
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
        .T
    )

    # Avoid divide-by-zero.
    stats["cv_pct"] = np.where(
        stats["mean"].abs() > 1e-12,
        100.0
        * stats["std"]
        / stats["mean"].abs(),
        np.nan,
    )

    return stats.round(
        4
    )


# ============================================================
# CLASS-WISE STATISTICS
# ============================================================

def class_statistics(
    df_seeds: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate mean statistics for each seed class.
    """

    if (
        df_seeds is None
        or df_seeds.empty
        or "class"
        not in df_seeds.columns
    ):

        return pd.DataFrame()

    if columns is None:

        columns = [
            "area_px2",
            "length_px",
            "width_px",
            "roundness",
            "aspect_ratio",
            "solidity",
            "mean_L",
            "brightness",
            "texture_std",
            "spot_count",
            "spot_coverage",
            "length_mm",
            "width_mm",
            "area_mm2",
            "gmd_mm",
            "sphericity",
        ]

    columns = [
        c
        for c in columns
        if c in df_seeds.columns
    ]

    if not columns:
        return pd.DataFrame()

    return (
        df_seeds
        .groupby("class")[columns]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
            ]
        )
        .round(4)
    )


# ============================================================
# SPATIAL FEATURES
# ============================================================

def add_spatial_features(
    df_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add normalized spatial coordinates.

    Useful for spatial heatmaps.

    Adds:
        x_norm
        y_norm
        position_x_percent
        position_y_percent
    """

    d = df_seeds.copy()

    if d.empty:
        return d

    if (
        "cx" in d.columns
        and "img_w" in d.columns
    ):

        d["x_norm"] = (
            d["cx"]
            / d["img_w"].replace(
                0,
                np.nan,
            )
        )

        d[
            "position_x_percent"
        ] = (
            d["x_norm"]
            * 100.0
        )

    if (
        "cy" in d.columns
        and "img_h" in d.columns
    ):

        d["y_norm"] = (
            d["cy"]
            / d["img_h"].replace(
                0,
                np.nan,
            )
        )

        d[
            "position_y_percent"
        ] = (
            d["y_norm"]
            * 100.0
        )

    return d


# ============================================================
# HEATMAP GRID
# ============================================================

def create_spatial_grid(
    df_seeds: pd.DataFrame,
    value_column: Optional[str] = None,
    grid_size: int = 50,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Create a spatial grid for heatmaps.

    Parameters
    ----------
    df_seeds:
        Seed DataFrame.

    value_column:
        If None:
            creates seed-density grid.

        If supplied:
            averages that feature spatially.

    grid_size:
        Number of grid cells.

    Returns
    -------
    grid:
        Heatmap values.

    x_edges:
        X edges.

    y_edges:
        Y edges.
    """

    if (
        df_seeds is None
        or df_seeds.empty
        or "cx" not in df_seeds.columns
        or "cy" not in df_seeds.columns
    ):
        return None, None, None

    x = (
        df_seeds["cx"]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        df_seeds["cy"]
        .to_numpy(
            dtype=float
        )
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return None, None, None

    if (
        "img_w" in df_seeds.columns
        and "img_h" in df_seeds.columns
    ):

        img_w = float(
            df_seeds["img_w"]
            .iloc[0]
        )

        img_h = float(
            df_seeds["img_h"]
            .iloc[0]
        )

    else:

        img_w = float(
            np.max(x)
        )

        img_h = float(
            np.max(y)
        )

    x_edges = np.linspace(
        0,
        img_w,
        grid_size + 1,
    )

    y_edges = np.linspace(
        0,
        img_h,
        grid_size + 1,
    )

    # --------------------------------------------------------
    # Density heatmap
    # --------------------------------------------------------

    if (
        value_column is None
        or value_column
        not in df_seeds.columns
    ):

        grid, _, _ = np.histogram2d(
            y,
            x,
            bins=[
                y_edges,
                x_edges,
            ],
        )

        return (
            grid,
            x_edges,
            y_edges,
        )

    # --------------------------------------------------------
    # Feature heatmap
    # --------------------------------------------------------

    values = (
        df_seeds.loc[
            valid,
            value_column,
        ]
        .to_numpy(
            dtype=float
        )
    )

    value_valid = np.isfinite(
        values
    )

    x = x[value_valid]
    y = y[value_valid]
    values = values[value_valid]

    grid_sum, _, _ = np.histogram2d(
        y,
        x,
        bins=[
            y_edges,
            x_edges,
        ],
        weights=values,
    )

    grid_count, _, _ = np.histogram2d(
        y,
        x,
        bins=[
            y_edges,
            x_edges,
        ],
    )

    grid = np.divide(
        grid_sum,
        grid_count,
        out=np.full_like(
            grid_sum,
            np.nan,
        ),
        where=grid_count > 0,
    )

    return (
        grid,
        x_edges,
        y_edges,
    )


# ============================================================
# EXPORT HELPERS
# ============================================================

def dataframe_for_export(
    df_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove internal/private columns from export.

    Columns beginning with "_" are removed.
    """

    if df_seeds is None:
        return pd.DataFrame()

    d = df_seeds.copy()

    private_cols = [
        c
        for c in d.columns
        if c.startswith("_")
    ]

    if private_cols:

        d = d.drop(
            columns=private_cols,
            errors="ignore",
        )

    return d


def export_csv(
    df_seeds: pd.DataFrame,
    path: str | Path,
) -> None:
    """
    Save seed measurements as CSV.
    """

    d = dataframe_for_export(
        df_seeds
    )

    d.to_csv(
        path,
        index=False,
    )


def export_excel(
    df_seeds: pd.DataFrame,
    path: str | Path,
) -> None:
    """
    Save measurements to Excel.

    Includes:
        Seed Measurements
        Class Counts
        Pattern Counts
        Statistics
        Class Statistics
    """

    d = dataframe_for_export(
        df_seeds
    )

    counts = class_counts(
        df_seeds
    )

    patterns = pattern_counts(
        df_seeds
    )

    stats = compute_statistics(
        df_seeds
    )

    cls_stats = class_statistics(
        df_seeds
    )

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
    ) as writer:

        d.to_excel(
            writer,
            sheet_name="Seed Measurements",
            index=False,
        )

        counts.to_excel(
            writer,
            sheet_name="Class Counts",
            index=False,
        )

        patterns.to_excel(
            writer,
            sheet_name="Pattern Counts",
            index=False,
        )

        stats.to_excel(
            writer,
            sheet_name="Statistics",
        )

        cls_stats.to_excel(
            writer,
            sheet_name="Class Statistics",
        )


# ============================================================
# PIPELINE INFORMATION
# ============================================================

def get_pipeline_info() -> Dict[str, Any]:
    """
    Return useful information about the pipeline.
    """

    model_available = (
        MODEL_PATH.exists()
    )

    return {
        "model_path": str(
            MODEL_PATH
        ),

        "model_available":
            model_available,

        "classes":
            CLASS_NAMES.copy(),

        "num_classes":
            len(CLASS_NAMES),

        "confidence":
            CONF,

        "iou":
            IOU,

        "max_detections":
            MAX_DET,

        "thickness_ratio":
            THICKNESS_FROM_WIDTH,

        "physical_measurements":
            [
                "length_mm",
                "width_mm",
                "thickness_mm",
                "area_mm2",
                "perimeter_mm",
                "gmd_mm",
                "amd_mm",
                "sphericity",
                "surface_area_mm2",
                "volume_mm3",
            ],
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Lentil Seed Phenotyping Pipeline")
    print("=" * 60)

    info = get_pipeline_info()

    print(
        f"Model path: "
        f"{info['model_path']}"
    )

    print(
        f"YOLO available: "
        f"{info['model_available']}"
    )

    print(
        f"Classes: "
        f"{info['classes']}"
    )

    print(
        f"Number of classes: "
        f"{info['num_classes']}"
    )

    print(
        "\nPipeline ready."
    )
