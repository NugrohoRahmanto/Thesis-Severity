from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


ROOT_DIR = Path(__file__).resolve().parents[1]
ECG_CROP_CONFIG_PATH = ROOT_DIR / "data" / "patient" / "valid" / "ecg_pvc_crop_config.json"
NORMALIZATION_PATH = ROOT_DIR / "data" / "ecg-data-features" / "fusion-ready" / "clinical_normalization_params.json"
SEVERITY_MODEL_PATH = ROOT_DIR / "data" / "model-training-lightweight" / "final_lightweight_severity_ridge_model.npz"
YOLO_MODEL_OPTIONS = {
    "YOLO12 Nano": "runs/yolo_pvc/yolo12_nano_pvc/weights/best.pt",
    "YOLO12 Small": "runs/yolo_pvc/yolo12_small_pvc/weights/best.pt",
    "YOLO12 Medium": "runs/yolo_pvc/yolo12_medium_pvc/weights/best.pt",
}

REFERENCE_SIZE = (1085, 767)
TARGET_STANDARD_WIDTH = 384
LEADS_PER_HALF = 6
LEFT_LEAD_LABELS = ["I", "II", "III", "aVR", "aVL", "aVF"]
RIGHT_LEAD_LABELS = ["V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_ORDER = LEFT_LEAD_LABELS + RIGHT_LEAD_LABELS
LEAD_COLORS = {
    "I": (230, 25, 75),
    "II": (60, 60, 60),
    "III": (30, 80, 220),
    "aVR": (245, 170, 20),
    "aVL": (0, 170, 200),
    "aVF": (240, 40, 190),
    "V1": (160, 60, 60),
    "V2": (40, 150, 65),
    "V3": (40, 60, 170),
    "V4": (120, 100, 30),
    "V5": (20, 125, 145),
    "V6": (170, 55, 150),
}

PATIENT_REFERENCE_CROP_BOXES = {
    "P-00001": {"x1": 17, "y1": 143, "x2": 1065, "y2": 675},
    "P-00002": {"x1": 20, "y1": 135, "x2": 1070, "y2": 685},
    "P-00007": {"x1": 30, "y1": 150, "x2": 1040, "y2": 680},
    "P-00015": {"x1": 45, "y1": 138, "x2": 1070, "y2": 694},
}

CLINICAL_FIELDS = [
    ("risk_hipertensi", "Hypertension", 0.0, 0.0, 1.0, 1.0),
    ("risk_diabetes", "Diabetes", 0.0, 0.0, 1.0, 1.0),
    ("risk_cad", "CAD", 0.0, 0.0, 1.0, 1.0),
    ("syncope", "Syncope", 0.0, 0.0, 1.0, 1.0),
    ("coronary_catheterization", "Coronary catheterization", 0.0, 0.0, 2.0, 1.0),
    ("lvef", "LVEF (%)", 68.0, 0.0, 100.0, 0.1),
    ("lvidd", "LVIDd", 3.2, 0.0, 10.0, 0.01),
    ("tapse", "TAPSE", 2.4, 0.0, 6.0, 0.01),
    ("rv_dilatation", "RV dilatation", 1.0, 0.0, 1.0, 1.0),
    ("diastolic_function_grade", "Diastolic function grade", 0.0, 0.0, 4.0, 1.0),
    ("holter_pvc_percent", "Holter PVC (%)", 12.0, 0.0, 100.0, 0.1),
    ("run_vt", "Run VT", 0.0, 0.0, 1.0, 1.0),
]
YES_NO_OPTIONS = {"No": 0.0, "Yes": 1.0}
CORONARY_CATH_OPTIONS = {"Never DCA": 0.0, "Lesi K-": 1.0, "Lesi K+": 2.0}

BASELINE_THRESHOLD = 170
BASELINE_SMOOTH_KERNEL = 7
LEFT_SIGNAL_MARGIN = 26
RIGHT_SIGNAL_MARGIN = 20
EDGE_MARGIN = 6
BAND_VERTICAL_MARGIN = 2
INITIAL_LEAD_HALF_HEIGHT = 12
EXPAND_STEP = 2
MAX_EXPAND_ITER = 80
MAX_VERTICAL_EXTRA = 28
NEIGHBOR_BASELINE_GUARD = 2
TOUCH_MARGIN = 3
MIN_LEAD_HEIGHT = 24

OTSU_BLUR_KERNEL = (3, 3)
OTSU_SIGNAL_OFFSET = 4
SIGNAL_THRESHOLD_MIN = 120
SIGNAL_THRESHOLD_MAX = 220
CONNECTED_MIN_AREA = 3
CONNECTED_MIN_WIDTH = 2
CONNECTED_MIN_HEIGHT = 2
BASELINE_NEAR_MARGIN = 5
COMPONENT_MAX_GAP_TO_BASELINE = 8
MAIN_SIGNAL_BRIDGE_KERNEL = (7, 3)
MAIN_SIGNAL_MIN_SPAN_RATIO = 0.45
MAIN_SIGNAL_EDGE_MARGIN = 4
MAIN_SIGNAL_REPAIR_MAX_GAP = 4
MAIN_SIGNAL_REPAIR_MAX_Y_GAP = 4

ENABLE_BASELINE_PROTECTED_EROSION = True
BASELINE_EROSION_GUARD = 20
BASELINE_EROSION_KERNEL = (3, 3)
BASELINE_EROSION_ITERATIONS = 1
BASELINE_EROSION_MAX_AREA_LOSS_RATIO = 0.45
BASELINE_EROSION_MIN_SPAN_KEEP_RATIO = 0.70

RECONSTRUCTED_LEAD_WIDTH = 2500
RECONSTRUCTED_LEAD_HEIGHT = 640
YOLO_INPUT_SIZE = 640
RPEAK_MIN_DISTANCE_RATIO = 0.12
MAX_CYCLES_PER_LEAD = 8
CYCLE_LEFT_FIXED = 250
CYCLE_RIGHT_FIXED = 390
SIGNAL_ONLY_DARK_THRESHOLD = 230
SIGNAL_ONLY_MASK_DILATE_KERNEL = (3, 3)


st.set_page_config(page_title="ECG PVC Demo Inference", page_icon=":material/ecg:", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.28);
        padding: 0.75rem 0.9rem;
        border-radius: 0.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def validate_box(box: dict[str, int | float]) -> dict[str, int]:
    clean = {key: int(round(float(box[key]))) for key in ("x1", "y1", "x2", "y2")}
    if clean["x1"] >= clean["x2"] or clean["y1"] >= clean["y2"]:
        raise ValueError(f"Crop box tidak valid: {clean}")
    return clean


def clamp_box(box: dict[str, int | float], size: tuple[int, int]) -> dict[str, int]:
    width, height = size
    x1 = max(0, min(width - 1, int(round(float(box["x1"])))))
    y1 = max(0, min(height - 1, int(round(float(box["y1"])))))
    x2 = max(x1 + 1, min(width, int(round(float(box["x2"])))))
    y2 = max(y1 + 1, min(height, int(round(float(box["y2"])))))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def scale_box(box: dict[str, int | float], source_size: tuple[int, int], target_size: tuple[int, int]) -> dict[str, int]:
    source_w, source_h = source_size
    target_w, target_h = target_size
    scaled = {
        "x1": float(box["x1"]) * target_w / source_w,
        "y1": float(box["y1"]) * target_h / source_h,
        "x2": float(box["x2"]) * target_w / source_w,
        "y2": float(box["y2"]) * target_h / source_h,
    }
    return clamp_box(scaled, target_size)


def standard_size() -> tuple[int, int]:
    if ECG_CROP_CONFIG_PATH.exists():
        config = json.loads(ECG_CROP_CONFIG_PATH.read_text(encoding="utf-8"))
        if "standard_size" in config:
            return int(config["standard_size"]["width"]), int(config["standard_size"]["height"])
    scale = TARGET_STANDARD_WIDTH / REFERENCE_SIZE[0]
    return TARGET_STANDARD_WIDTH, int(round(REFERENCE_SIZE[1] * scale))


STANDARD_SIZE = standard_size()


def standardize_image(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize(STANDARD_SIZE, Image.Resampling.LANCZOS)


def point_from_standard_to_raw(point: dict[str, float], raw_size: tuple[int, int]) -> dict[str, int]:
    raw_w, raw_h = raw_size
    return {
        "x": int(round(float(point["x"]) * raw_w / STANDARD_SIZE[0])),
        "y": int(round(float(point["y"]) * raw_h / STANDARD_SIZE[1])),
    }


def box_from_standard_to_raw(box: dict[str, int | float], raw_size: tuple[int, int]) -> dict[str, int]:
    raw_w, raw_h = raw_size
    return clamp_box(
        {
            "x1": float(box["x1"]) * raw_w / STANDARD_SIZE[0],
            "y1": float(box["y1"]) * raw_h / STANDARD_SIZE[1],
            "x2": float(box["x2"]) * raw_w / STANDARD_SIZE[0],
            "y2": float(box["y2"]) * raw_h / STANDARD_SIZE[1],
        },
        raw_size,
    )


def smooth_1d(values: np.ndarray, kernel_size: int = BASELINE_SMOOTH_KERNEL) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    return np.convolve(values.astype(float), kernel, mode="same")


def estimate_baseline_in_band(gray_half: np.ndarray, y0: int, y1: int, x_left: int, x_right: int) -> dict[str, float]:
    roi = gray_half[y0:y1, x_left:gray_half.shape[1] - x_right]
    dark_mask = roi < BASELINE_THRESHOLD
    projection = smooth_1d(dark_mask.sum(axis=1))
    if projection.max() <= 0:
        return {"baseline_y": float((y0 + y1) // 2), "confidence": 0.0}
    baseline_y = y0 + int(np.argmax(projection))
    return {"baseline_y": float(baseline_y), "confidence": float(projection.max() / max(1, dark_mask.shape[1]))}


def estimate_12lead_baselines(crop_image: Image.Image) -> list[dict[str, Any]]:
    gray = np.array(crop_image.convert("L"))
    height, width = gray.shape
    split_x = width // 2
    band_height = height / LEADS_PER_HALF
    records: list[dict[str, Any]] = []
    for side, labels in (("left", LEFT_LEAD_LABELS), ("right", RIGHT_LEAD_LABELS)):
        half = gray[:, :split_x] if side == "left" else gray[:, split_x:]
        x_left = LEFT_SIGNAL_MARGIN if side == "left" else RIGHT_SIGNAL_MARGIN
        for index, label in enumerate(labels, start=1):
            band_y0 = int(round((index - 1) * band_height)) + BAND_VERTICAL_MARGIN
            band_y1 = int(round(index * band_height)) - BAND_VERTICAL_MARGIN
            baseline = estimate_baseline_in_band(half, band_y0, band_y1, x_left, EDGE_MARGIN)
            records.append(
                {
                    "lead_label": label,
                    "side": side,
                    "lead_index": index,
                    "band_y0": band_y0,
                    "band_y1": band_y1,
                    "baseline_y": int(round(baseline["baseline_y"])),
                    "confidence": float(baseline["confidence"]),
                }
            )
    return records


def expansion_limits_for_row(row: dict[str, Any], side_rows: list[dict[str, Any]], image_height: int) -> tuple[int, int]:
    index = int(row["lead_index"]) - 1
    baseline_y = int(row["baseline_y"])
    limit_y0 = max(0, int(row["band_y0"]) - MAX_VERTICAL_EXTRA)
    limit_y1 = min(image_height, int(row["band_y1"]) + MAX_VERTICAL_EXTRA)
    if index > 0:
        limit_y0 = max(limit_y0, int(side_rows[index - 1]["baseline_y"]) + NEIGHBOR_BASELINE_GUARD)
    if index < len(side_rows) - 1:
        limit_y1 = min(limit_y1, int(side_rows[index + 1]["baseline_y"]) - NEIGHBOR_BASELINE_GUARD)
    limit_y0 = min(limit_y0, baseline_y - 1)
    limit_y1 = max(limit_y1, baseline_y + 1)
    return limit_y0, limit_y1


def initial_lead_boxes(crop_image: Image.Image, baselines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    width, height = crop_image.size
    split_x = width // 2
    side_lookup = {
        side: sorted([row for row in baselines if row["side"] == side], key=lambda item: int(item["lead_index"]))
        for side in ("left", "right")
    }
    boxes = []
    for row in baselines:
        x0 = LEFT_SIGNAL_MARGIN if row["side"] == "left" else split_x + RIGHT_SIGNAL_MARGIN
        x1 = split_x - EDGE_MARGIN if row["side"] == "left" else width - EDGE_MARGIN
        limit_y0, limit_y1 = expansion_limits_for_row(row, side_lookup[row["side"]], height)
        baseline_y = int(row["baseline_y"])
        y0 = max(limit_y0, baseline_y - INITIAL_LEAD_HALF_HEIGHT)
        y1 = min(limit_y1, baseline_y + INITIAL_LEAD_HALF_HEIGHT + 1)
        if y1 - y0 < MIN_LEAD_HEIGHT:
            target = min(MIN_LEAD_HEIGHT, max(1, limit_y1 - limit_y0))
            y0 = max(limit_y0, baseline_y - target // 2)
            y1 = min(limit_y1, y0 + target)
            y0 = max(limit_y0, y1 - target)
        boxes.append({**row, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "limit_y0": limit_y0, "limit_y1": limit_y1})
    return boxes


def apply_otsu_blur(gray: np.ndarray) -> np.ndarray:
    if OTSU_BLUR_KERNEL[0] <= 1 or OTSU_BLUR_KERNEL[1] <= 1:
        return gray.copy()
    return cv2.GaussianBlur(gray, OTSU_BLUR_KERNEL, 0)


def connected_component_summary(mask: np.ndarray) -> dict[str, Any]:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for component_id in range(1, num_labels):
        x, y, width, height, area = stats[component_id]
        if area >= CONNECTED_MIN_AREA:
            components.append({"component_id": int(component_id), "x": int(x), "y": int(y), "width": int(width), "height": int(height), "area": int(area)})
    if not components:
        return {"count": 0, "x_span": 0, "area": 0}
    largest = max(components, key=lambda item: item["area"])
    widest = max(components, key=lambda item: item["width"])
    return {"count": len(components), "x_span": widest["width"], "area": largest["area"], "largest": largest, "widest": widest}


def remove_tiny_components(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary, dtype=np.uint8)
    for component_id in range(1, num_labels):
        x, y, width, height, area = stats[component_id]
        if area >= CONNECTED_MIN_AREA and width >= CONNECTED_MIN_WIDTH and height >= CONNECTED_MIN_HEIGHT:
            keep[labels == component_id] = 1
    return (keep * 255).astype(np.uint8)


def component_edge_point(labels: np.ndarray, component_id: int, side: str) -> tuple[int, int]:
    ys, xs = np.where(labels == component_id)
    target_x = xs.min() if side == "left" else xs.max()
    candidate_ys = ys[xs == target_x]
    return int(target_x), int(np.median(candidate_ys))


def repair_column_gaps(mask: np.ndarray) -> np.ndarray:
    result = (mask > 0).astype(np.uint8) * 255
    columns = np.flatnonzero(np.any(result > 0, axis=0))
    if len(columns) <= 1:
        return result.astype(np.uint8)
    for left_x, right_x in zip(columns[:-1], columns[1:]):
        gap = int(right_x - left_x)
        if gap <= 1 or gap > MAIN_SIGNAL_REPAIR_MAX_GAP:
            continue
        left_y = int(np.median(np.flatnonzero(result[:, left_x] > 0)))
        right_y = int(np.median(np.flatnonzero(result[:, right_x] > 0)))
        if abs(right_y - left_y) <= MAIN_SIGNAL_REPAIR_MAX_Y_GAP:
            cv2.line(result, (int(left_x), left_y), (int(right_x), right_y), 255, thickness=1)
    return result.astype(np.uint8)


def connect_components_by_x(mask: np.ndarray) -> np.ndarray:
    result = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((result > 0).astype(np.uint8), connectivity=8)
    component_ids = [component_id for component_id in range(1, num_labels) if stats[component_id, cv2.CC_STAT_AREA] >= CONNECTED_MIN_AREA]
    if len(component_ids) <= 1:
        return result.astype(np.uint8)
    component_ids.sort(key=lambda component_id: stats[component_id, cv2.CC_STAT_LEFT])
    for left_id, right_id in zip(component_ids[:-1], component_ids[1:]):
        x_left, y_left = component_edge_point(labels, left_id, side="right")
        x_right, y_right = component_edge_point(labels, right_id, side="left")
        if 0 < x_right - x_left <= MAIN_SIGNAL_REPAIR_MAX_GAP and abs(y_right - y_left) <= MAIN_SIGNAL_REPAIR_MAX_Y_GAP:
            cv2.line(result, (x_left, y_left), (x_right, y_right), 255, thickness=1)
    return result.astype(np.uint8)


def repair_main_signal_gaps(mask: np.ndarray) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return mask.astype(np.uint8)
    repaired = repair_column_gaps(mask)
    if connected_component_summary(repaired)["count"] == 1:
        return repaired.astype(np.uint8)
    repaired = connect_components_by_x(repaired)
    if connected_component_summary(repaired)["count"] == 1:
        return repaired.astype(np.uint8)
    binary = (repaired > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return (binary * 255).astype(np.uint8)
    widest_id = max(range(1, num_labels), key=lambda component_id: stats[component_id, cv2.CC_STAT_WIDTH])
    return np.where(labels == widest_id, 255, 0).astype(np.uint8)


def choose_main_component(mask: np.ndarray, baseline_local_y: int):
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return None, labels, stats, {"reason": "no_foreground"}
    image_height, image_width = binary.shape
    baseline_top = baseline_local_y - BASELINE_NEAR_MARGIN
    baseline_bottom = baseline_local_y + BASELINE_NEAR_MARGIN
    min_span = image_width * MAIN_SIGNAL_MIN_SPAN_RATIO
    best_id = None
    best_score = -1e18
    best_info: dict[str, Any] = {}
    fallback_id = None
    fallback_score = -1e18
    fallback_info: dict[str, Any] = {}
    for component_id in range(1, num_labels):
        x, y, width, height, area = stats[component_id]
        if area < CONNECTED_MIN_AREA:
            continue
        component_top = y
        component_bottom = y + height - 1
        intersects_baseline = component_top <= baseline_bottom and component_bottom >= baseline_top
        if component_bottom < baseline_top:
            baseline_distance = baseline_top - component_bottom
        elif component_top > baseline_bottom:
            baseline_distance = component_top - baseline_bottom
        else:
            baseline_distance = 0
        touches_left = x <= MAIN_SIGNAL_EDGE_MARGIN
        touches_right = (x + width) >= (image_width - MAIN_SIGNAL_EDGE_MARGIN)
        span_ratio = width / max(1, image_width)
        baseline_bonus = 200 if intersects_baseline else max(0, 80 - baseline_distance * 8)
        edge_bonus = (60 if touches_left else 0) + (60 if touches_right else 0)
        score = width * 4.0 + area * 0.04 + baseline_bonus + edge_bonus - baseline_distance * 10.0
        info = {
            "component_id": int(component_id),
            "width": int(width),
            "height": int(height),
            "area": int(area),
            "span_ratio": float(span_ratio),
            "baseline_distance": int(baseline_distance),
            "intersects_baseline": bool(intersects_baseline),
            "score": float(score),
        }
        if score > fallback_score:
            fallback_id, fallback_score, fallback_info = component_id, score, info
        if width >= min_span and baseline_distance <= COMPONENT_MAX_GAP_TO_BASELINE and score > best_score:
            best_id, best_score, best_info = component_id, score, info
    if best_id is not None:
        return best_id, labels, stats, best_info
    return fallback_id, labels, stats, fallback_info


def keep_single_main_signal_component(mask: np.ndarray, baseline_local_y: int) -> tuple[np.ndarray, dict[str, Any]]:
    binary = (mask > 0).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(binary, dtype=np.uint8), {"reason": "empty_otsu"}
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MAIN_SIGNAL_BRIDGE_KERNEL)
    bridge_mask = cv2.dilate(binary, bridge_kernel, iterations=1)
    component_id, labels, _, component_info = choose_main_component(bridge_mask, baseline_local_y)
    if component_id is None:
        return np.zeros_like(binary, dtype=np.uint8), {"reason": "no_component"}
    selected_otsu = np.where((labels == component_id) & (binary > 0), 255, 0).astype(np.uint8)
    selected_otsu = remove_tiny_components(selected_otsu)
    repaired = repair_main_signal_gaps(selected_otsu)
    component_info.update({"reason": "selected_otsu_main_signal", "component_count": int(connected_component_summary(repaired)["count"])})
    return repaired, component_info


def keep_components_touching_baseline_band(mask: np.ndarray, guard_top: int, guard_bottom: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary, dtype=np.uint8)
    for component_id in range(1, num_labels):
        x, y, width, height, area = stats[component_id]
        if area < CONNECTED_MIN_AREA:
            continue
        if int(y) < guard_bottom and int(y + height) > guard_top:
            keep[labels == component_id] = 1
    return (keep * 255).astype(np.uint8)


def baseline_protected_erosion(mask: np.ndarray, baseline_local_y: int) -> tuple[np.ndarray, dict[str, Any]]:
    binary = (mask > 0).astype(np.uint8)
    original = (binary * 255).astype(np.uint8)
    if not ENABLE_BASELINE_PROTECTED_EROSION or np.count_nonzero(binary) == 0:
        return original, {"used": False}
    height, _ = binary.shape
    baseline_y = int(np.clip(baseline_local_y, 0, height - 1))
    guard_top = max(0, baseline_y - BASELINE_EROSION_GUARD)
    guard_bottom = min(height, baseline_y + BASELINE_EROSION_GUARD + 1)
    protected_rows = np.zeros_like(binary, dtype=bool)
    protected_rows[guard_top:guard_bottom, :] = True
    protected_mask = np.where(protected_rows & (binary > 0), 255, 0).astype(np.uint8)
    far_mask = np.where((~protected_rows) & (binary > 0), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, BASELINE_EROSION_KERNEL)
    eroded_far = cv2.erode(far_mask, kernel, iterations=BASELINE_EROSION_ITERATIONS)
    candidate = cv2.bitwise_or(protected_mask, eroded_far)
    anchored = keep_components_touching_baseline_band(candidate, guard_top, guard_bottom)
    if np.count_nonzero(anchored) == 0:
        return original, {"used": False, "reason": "removed_all"}
    before_summary = connected_component_summary(original)
    after_summary = connected_component_summary(anchored)
    before_area = max(1, int(np.count_nonzero(original)))
    after_area = int(np.count_nonzero(anchored))
    area_loss_ratio = 1.0 - (after_area / before_area)
    before_span = max(1, int(before_summary.get("x_span", 1)))
    span_keep_ratio = int(after_summary.get("x_span", 0)) / before_span
    if area_loss_ratio > BASELINE_EROSION_MAX_AREA_LOSS_RATIO or span_keep_ratio < BASELINE_EROSION_MIN_SPAN_KEEP_RATIO:
        return original, {"used": False, "reason": "fallback_area_or_span_loss"}
    return anchored.astype(np.uint8), {"used": True, "area_loss_ratio": float(area_loss_ratio), "span_keep_ratio": float(span_keep_ratio)}


def preprocess_lead_mask(lead_image: Image.Image, baseline_local_y: int) -> dict[str, Any]:
    gray = cv2.cvtColor(np.array(lead_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = apply_otsu_blur(gray)
    otsu_threshold, raw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    signal_threshold = int(np.clip(otsu_threshold - OTSU_SIGNAL_OFFSET, SIGNAL_THRESHOLD_MIN, SIGNAL_THRESHOLD_MAX))
    otsu = np.where(blur < signal_threshold, 255, 0).astype(np.uint8)
    main_signal_mask, main_info = keep_single_main_signal_component(otsu, baseline_local_y)
    eroded_mask, erosion_info = baseline_protected_erosion(main_signal_mask, baseline_local_y)
    main_info["baseline_protected_erosion"] = erosion_info
    return {
        "gray": gray,
        "raw_otsu": raw_otsu,
        "otsu": otsu,
        "connected_mask": main_signal_mask,
        "expansion_touch_mask": main_signal_mask,
        "line_mask": eroded_mask,
        "main_component_info": main_info,
    }


def mask_touches_bounds(mask: np.ndarray) -> tuple[bool, bool]:
    if mask.size == 0 or np.count_nonzero(mask) == 0:
        return False, False
    return bool(np.any(mask[: TOUCH_MARGIN + 1, :] > 0)), bool(np.any(mask[-(TOUCH_MARGIN + 1) :, :] > 0))


def expand_box_until_signal_inside(crop_image: Image.Image, box: dict[str, Any]) -> dict[str, Any]:
    y0, y1 = int(box["y0"]), int(box["y1"])
    limit_y0, limit_y1 = int(box["limit_y0"]), int(box["limit_y1"])
    stages = None
    history = []
    for iteration in range(MAX_EXPAND_ITER + 1):
        lead_image = crop_image.crop((box["x0"], y0, box["x1"], y1))
        baseline_local_y = int(box["baseline_y"]) - y0
        stages = preprocess_lead_mask(lead_image, baseline_local_y)
        touch_top, touch_bottom = mask_touches_bounds(stages["expansion_touch_mask"])
        history.append({"iteration": iteration, "y0": y0, "y1": y1, "touch_top": touch_top, "touch_bottom": touch_bottom})
        expand_top = touch_top and y0 > limit_y0
        expand_bottom = touch_bottom and y1 < limit_y1
        if not expand_top and not expand_bottom:
            break
        y0 = max(limit_y0, y0 - EXPAND_STEP) if expand_top else y0
        y1 = min(limit_y1, y1 + EXPAND_STEP) if expand_bottom else y1
    return {**box, "y0": y0, "y1": y1, "height": y1 - y0, "history": history, "stages": stages}


def resize_mask_nearest(mask: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)


def make_signal_only_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    source = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    mask_binary = (mask > 0).astype(np.uint8)
    if SIGNAL_ONLY_MASK_DILATE_KERNEL[0] > 1 or SIGNAL_ONLY_MASK_DILATE_KERNEL[1] > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, SIGNAL_ONLY_MASK_DILATE_KERNEL)
        mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)
    signal_pixels = (mask_binary > 0) & (gray < SIGNAL_ONLY_DARK_THRESHOLD)
    signal_only = np.full_like(source, 255)
    signal_only[signal_pixels] = source[signal_pixels]
    signal_only[cv2.cvtColor(signal_only, cv2.COLOR_RGB2GRAY) >= SIGNAL_ONLY_DARK_THRESHOLD] = 255
    return Image.fromarray(signal_only)


def reconstruct_lead_to_height(crop_image: Image.Image, segment: dict[str, Any]) -> dict[str, Any]:
    lead_crop = crop_image.crop((segment["x0"], segment["y0"], segment["x1"], segment["y1"]))
    mask = segment["stages"]["line_mask"]
    new_size = (RECONSTRUCTED_LEAD_WIDTH, RECONSTRUCTED_LEAD_HEIGHT)
    reconstructed_crop = lead_crop.resize(new_size, Image.Resampling.LANCZOS)
    reconstructed_mask = resize_mask_nearest(mask, new_size)
    signal_only = make_signal_only_image(reconstructed_crop, reconstructed_mask)
    scale_x = RECONSTRUCTED_LEAD_WIDTH / max(1, lead_crop.width)
    scale_y = RECONSTRUCTED_LEAD_HEIGHT / max(1, lead_crop.height)
    baseline_y = int(round((int(segment["baseline_y"]) - int(segment["y0"])) * scale_y))
    return {
        "signal_only": signal_only,
        "mask": reconstructed_mask,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "baseline_y": int(np.clip(baseline_y, 0, RECONSTRUCTED_LEAD_HEIGHT - 1)),
        "width": RECONSTRUCTED_LEAD_WIDTH,
        "height": RECONSTRUCTED_LEAD_HEIGHT,
    }


def reconstruct_lead_from_raw(raw_image: Image.Image, crop_box_standard: dict[str, int], segment: dict[str, Any]) -> dict[str, Any]:
    raw_w, raw_h = raw_image.size
    sx, sy = raw_w / STANDARD_SIZE[0], raw_h / STANDARD_SIZE[1]
    raw_x0 = int(round((crop_box_standard["x1"] + segment["x0"]) * sx))
    raw_y0 = int(round((crop_box_standard["y1"] + segment["y0"]) * sy))
    raw_x1 = int(round((crop_box_standard["x1"] + segment["x1"]) * sx))
    raw_y1 = int(round((crop_box_standard["y1"] + segment["y1"]) * sy))
    raw_box = clamp_box({"x1": raw_x0, "y1": raw_y0, "x2": raw_x1, "y2": raw_y1}, raw_image.size)
    raw_lead_crop = raw_image.crop((raw_box["x1"], raw_box["y1"], raw_box["x2"], raw_box["y2"]))
    recon_raw_crop = raw_lead_crop.resize((RECONSTRUCTED_LEAD_WIDTH, RECONSTRUCTED_LEAD_HEIGHT), Image.Resampling.LANCZOS)
    recon_mask = resize_mask_nearest(segment["stages"]["line_mask"], (RECONSTRUCTED_LEAD_WIDTH, RECONSTRUCTED_LEAD_HEIGHT))
    return {
        "signal_only": make_signal_only_image(recon_raw_crop, recon_mask),
        "raw_box": raw_box,
        "width": RECONSTRUCTED_LEAD_WIDTH,
        "height": RECONSTRUCTED_LEAD_HEIGHT,
    }


def moving_average(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    return np.convolve(values.astype(float), kernel, mode="same")


def signal_trace_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    xs: list[int] = []
    ys: list[float] = []
    for x_coord in range(width):
        y_positions = np.flatnonzero(mask[:, x_coord] > 0)
        if len(y_positions) > 0:
            xs.append(x_coord)
            ys.append(float(np.median(y_positions)))
    if not xs:
        return np.arange(width), np.full(width, height / 2, dtype=float)
    full_x = np.arange(width)
    trace_y = np.interp(full_x, np.array(xs), np.array(ys))
    return full_x, trace_y


def find_local_peaks(values: np.ndarray, threshold: float, min_distance: int) -> list[int]:
    values = np.asarray(values, dtype=float)
    candidates = [
        idx
        for idx in range(1, len(values) - 1)
        if values[idx] >= threshold and values[idx] >= values[idx - 1] and values[idx] >= values[idx + 1]
    ]
    candidates.sort(key=lambda idx: values[idx], reverse=True)
    selected: list[int] = []
    for idx in candidates:
        if all(abs(idx - chosen) > min_distance for chosen in selected):
            selected.append(idx)
    return sorted(selected)


def detect_r_peaks_from_reconstruction(reconstructed: dict[str, Any]) -> dict[str, Any]:
    _, trace_y = signal_trace_from_mask(reconstructed["mask"])
    derivative = np.diff(trace_y, prepend=trace_y[0])
    energy = derivative ** 2
    smooth_energy = moving_average(energy, max(5, int(reconstructed["width"] * 0.012)))
    threshold = float(np.mean(smooth_energy) + 1.2 * np.std(smooth_energy))
    min_distance = max(20, int(reconstructed["width"] * RPEAK_MIN_DISTANCE_RATIO))
    energy_peaks = find_local_peaks(smooth_energy, threshold, min_distance)
    candidates = []
    for peak_x in energy_peaks:
        start = max(0, peak_x - 10)
        end = min(len(trace_y) - 1, peak_x + 10)
        local_trace = trace_y[start : end + 1]
        displacement = local_trace - float(reconstructed["baseline_y"])
        local_idx = int(np.argmax(np.abs(displacement)))
        amplitude = float(abs(displacement[local_idx]))
        if amplitude > 8:
            candidates.append({"x": int(start + local_idx), "y": int(round(local_trace[local_idx])), "amplitude": amplitude})
    candidates.sort(key=lambda item: item["amplitude"], reverse=True)
    final_peaks: list[dict[str, Any]] = []
    for candidate in candidates:
        if all(abs(candidate["x"] - selected["x"]) > min_distance for selected in final_peaks):
            final_peaks.append(candidate)
    return {"trace_y": trace_y, "peaks": sorted(final_peaks, key=lambda item: item["x"])[:MAX_CYCLES_PER_LEAD], "threshold": threshold}


def get_consensus_r_peaks(leads_peaks: dict[str, list[int]], pixel_tolerance: int = 25, min_votes: int = 3) -> list[int]:
    pooled = sorted((int(x), lead) for lead, peaks in leads_peaks.items() for x in peaks)
    if not pooled:
        return []
    clusters: list[list[tuple[int, str]]] = [[pooled[0]]]
    for item in pooled[1:]:
        if item[0] - clusters[-1][0][0] <= pixel_tolerance:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    candidates = []
    for cluster in clusters:
        if len({lead for _, lead in cluster}) >= min_votes:
            candidates.append({"x": int(np.median([x for x, _ in cluster])), "votes": len({lead for _, lead in cluster})})
    candidates.sort(key=lambda item: item["votes"], reverse=True)
    final_xs: list[int] = []
    for candidate in candidates:
        if all(abs(candidate["x"] - x) > 180 for x in final_xs):
            final_xs.append(candidate["x"])
    return sorted(final_xs)


def refine_peak_coordinates(consensus_xs: list[int], trace_y: np.ndarray, baseline_y: float, search_window: int = 70) -> list[dict[str, Any]]:
    refined = []
    for x_coord in consensus_xs:
        start = max(0, int(x_coord) - search_window)
        end = min(len(trace_y), int(x_coord) + search_window)
        local_trace = trace_y[start:end]
        if len(local_trace) == 0:
            refined.append({"x": int(x_coord), "y": int(baseline_y), "amplitude": 0.0})
            continue
        displacement = np.abs(local_trace - float(baseline_y))
        local_idx = int(np.argmax(displacement))
        best_x = start + local_idx
        refined.append({"x": int(best_x), "y": int(round(trace_y[best_x])), "amplitude": float(displacement[local_idx])})
    return refined


def cycle_windows_from_peaks(peaks: list[dict[str, Any]], image_width: int) -> list[dict[str, Any]]:
    windows = []
    target_width = CYCLE_LEFT_FIXED + CYCLE_RIGHT_FIXED
    for order, peak in enumerate(peaks, start=1):
        peak_x = int(peak["x"])
        x0 = max(0, peak_x - CYCLE_LEFT_FIXED)
        x1 = min(image_width, peak_x + CYCLE_RIGHT_FIXED)
        if x1 - x0 < target_width:
            if x0 == 0:
                x1 = min(image_width, target_width)
            elif x1 == image_width:
                x0 = max(0, image_width - target_width)
        windows.append({"order": order, "x0": int(x0), "x1": int(x1), "peak": peak})
    return windows


def yolo_letterbox_info(source_size: tuple[int, int]) -> dict[str, Any]:
    source_w, source_h = source_size
    scale = min(YOLO_INPUT_SIZE / max(1, source_w), YOLO_INPUT_SIZE / max(1, source_h))
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    return {
        "scale": float(scale),
        "pad_x": int((YOLO_INPUT_SIZE - resized_w) // 2),
        "pad_y": int((YOLO_INPUT_SIZE - resized_h) // 2),
        "resized_width": int(resized_w),
        "resized_height": int(resized_h),
    }


def yolo_square_from_reconstructed_cycle(reconstructed_signal: Image.Image, window: dict[str, Any]) -> tuple[Image.Image, dict[str, Any], Image.Image]:
    x0 = max(0, min(reconstructed_signal.width - 1, int(window["x0"])))
    x1 = max(x0 + 1, min(reconstructed_signal.width, int(window["x1"])))
    cycle_crop = reconstructed_signal.crop((x0, 0, x1, reconstructed_signal.height)).convert("RGB")
    letterbox = yolo_letterbox_info(cycle_crop.size)
    canvas = Image.new("RGB", (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), "white")
    resized = cycle_crop.resize((letterbox["resized_width"], letterbox["resized_height"]), Image.Resampling.LANCZOS)
    canvas.paste(resized, (letterbox["pad_x"], letterbox["pad_y"]))
    return canvas, letterbox, cycle_crop


def box_from_reconstructed_cycle_to_standard(crop_box_standard: dict[str, int], segment: dict[str, Any], reconstructed: dict[str, Any], x0_recon: int, x1_recon: int) -> dict[str, int]:
    return {
        "x1": int(round(crop_box_standard["x1"] + segment["x0"] + (x0_recon / float(reconstructed["scale_x"])))),
        "y1": int(round(crop_box_standard["y1"] + segment["y0"])),
        "x2": int(round(crop_box_standard["x1"] + segment["x0"] + (x1_recon / float(reconstructed["scale_x"])))),
        "y2": int(round(crop_box_standard["y1"] + segment["y1"])),
    }


def build_cycle_records(raw_image: Image.Image, crop_image: Image.Image, crop_box_standard: dict[str, int], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    temp_records = []
    leads_peaks_map: dict[str, list[int]] = {}
    for lead_order, segment in enumerate(segments, start=1):
        reconstructed = reconstruct_lead_to_height(crop_image, segment)
        detection = detect_r_peaks_from_reconstruction(reconstructed)
        leads_peaks_map[segment["lead_label"]] = [peak["x"] for peak in detection["peaks"]]
        temp_records.append({"lead_order": lead_order, "segment": segment, "reconstructed": reconstructed, "detection": detection})

    consensus_xs = get_consensus_r_peaks(leads_peaks_map, pixel_tolerance=25, min_votes=3)
    common_windows = cycle_windows_from_peaks([{"x": x} for x in consensus_xs], RECONSTRUCTED_LEAD_WIDTH)
    records = []
    for item in temp_records:
        segment = item["segment"]
        reconstructed = item["reconstructed"]
        recon_raw = reconstruct_lead_from_raw(raw_image, crop_box_standard, segment)
        _, trace_y = signal_trace_from_mask(reconstructed["mask"])
        refined_peaks = refine_peak_coordinates(consensus_xs, trace_y, reconstructed["baseline_y"], search_window=70)
        cycles = []
        for window, peak_data in zip(common_windows, refined_peaks):
            standard_box = box_from_reconstructed_cycle_to_standard(crop_box_standard, segment, reconstructed, int(window["x0"]), int(window["x1"]))
            raw_box = box_from_standard_to_raw(standard_box, raw_image.size)
            yolo_image, letterbox, cycle_crop = yolo_square_from_reconstructed_cycle(recon_raw["signal_only"], window)
            cycles.append(
                {
                    **window,
                    "peak": peak_data,
                    "yolo_image": yolo_image,
                    "cycle_crop": cycle_crop,
                    "yolo_letterbox": letterbox,
                    "standard_box": standard_box,
                    "raw_box": raw_box,
                    "cycle_id": f"{item['lead_order']:02d}_{segment['lead_label']}_cycle_{window['order']:02d}",
                }
            )
        records.append({**item, "recon_raw": recon_raw, "cycles": cycles, "refined_peaks": refined_peaks})
    return records


def reverse_letterbox_bbox(bbox: list[float], letterbox: dict[str, Any], source_size: tuple[int, int]) -> tuple[float, float, float, float]:
    bx1, by1, bx2, by2 = [float(value) for value in bbox]
    scale = float(letterbox["scale"])
    pad_x = float(letterbox["pad_x"])
    pad_y = float(letterbox["pad_y"])
    source_w, source_h = source_size
    x1 = np.clip((bx1 - pad_x) / scale, 0, source_w)
    y1 = np.clip((by1 - pad_y) / scale, 0, source_h)
    x2 = np.clip((bx2 - pad_x) / scale, 0, source_w)
    y2 = np.clip((by2 - pad_y) / scale, 0, source_h)
    return float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))


def map_yolo_bbox_to_raw(bbox: list[float], cycle: dict[str, Any], record: dict[str, Any], crop_box_standard: dict[str, int], raw_size: tuple[int, int]) -> list[float]:
    crop_w = int(cycle["x1"] - cycle["x0"])
    crop_h = RECONSTRUCTED_LEAD_HEIGHT
    cx1, cy1, cx2, cy2 = reverse_letterbox_bbox(bbox, cycle["yolo_letterbox"], (crop_w, crop_h))
    recon_x1 = float(cycle["x0"]) + cx1
    recon_y1 = cy1
    recon_x2 = float(cycle["x0"]) + cx2
    recon_y2 = cy2
    segment = record["segment"]
    reconstructed = record["reconstructed"]
    std_box = {
        "x1": crop_box_standard["x1"] + segment["x0"] + recon_x1 / float(reconstructed["scale_x"]),
        "y1": crop_box_standard["y1"] + segment["y0"] + recon_y1 / float(reconstructed["scale_y"]),
        "x2": crop_box_standard["x1"] + segment["x0"] + recon_x2 / float(reconstructed["scale_x"]),
        "y2": crop_box_standard["y1"] + segment["y0"] + recon_y2 / float(reconstructed["scale_y"]),
    }
    raw_box = box_from_standard_to_raw(std_box, raw_size)
    return [float(raw_box["x1"]), float(raw_box["y1"]), float(raw_box["x2"]), float(raw_box["y2"])]


@st.cache_resource(show_spinner=False)
def load_yolo_model(weight_path: str):
    if YOLO is None:
        return None
    path = Path(weight_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return None
    return YOLO(str(path))


def run_yolo_on_cycles(records: list[dict[str, Any]], crop_box_standard: dict[str, int], raw_size: tuple[int, int], weight_path: str, confidence_threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    model = load_yolo_model(weight_path)
    if model is None:
        return [], [], "missing_model"
    detections: list[dict[str, Any]] = []
    cycle_predictions: list[dict[str, Any]] = []
    for record in records:
        label = record["segment"]["lead_label"]
        for cycle in record["cycles"]:
            prediction_rows = model.predict(np.array(cycle["yolo_image"]), imgsz=YOLO_INPUT_SIZE, conf=confidence_threshold, verbose=False, device="cpu")
            best_detection: dict[str, Any] | None = None
            for prediction in prediction_rows:
                for detection_index, box in enumerate(prediction.boxes):
                    xyxy = [float(value) for value in box.xyxy[0].tolist()]
                    class_id = int(box.cls.item())
                    class_name = str(prediction.names.get(class_id, class_id))
                    confidence = float(box.conf.item())
                    raw_bbox = map_yolo_bbox_to_raw(xyxy, cycle, record, crop_box_standard, raw_size)
                    row = {
                        "cycle_id": cycle["cycle_id"],
                        "lead_label": label,
                        "cycle_order": int(cycle["order"]),
                        "detection_index": detection_index,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox_yolo": xyxy,
                        "bbox_raw": raw_bbox,
                    }
                    detections.append(row)
                    if best_detection is None or confidence > float(best_detection["confidence"]):
                        best_detection = row
            cycle_predictions.append(
                {
                    "cycle_id": cycle["cycle_id"],
                    "lead_label": label,
                    "cycle_order": int(cycle["order"]),
                    "best_class_name": best_detection["class_name"] if best_detection else "none",
                    "best_confidence": float(best_detection["confidence"]) if best_detection else 0.0,
                    "is_pvc": bool(best_detection and ("pvc" in str(best_detection["class_name"]).lower() or int(best_detection["class_id"]) == 1)),
                }
            )
    return detections, cycle_predictions, "yolo"


def load_image_upload(uploaded_file: Any) -> Image.Image:
    uploaded_file.seek(0)
    return ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")


def load_clinical_csv(uploaded_file: Any) -> tuple[str | None, dict[str, float]]:
    uploaded_file.seek(0)
    frame = pd.read_csv(uploaded_file)
    if frame.empty:
        raise ValueError("Clinical CSV is empty.")
    row = frame.iloc[0].to_dict()
    patient_id = str(row.get("patient_id", "")).strip() or None
    values = {}
    for field, _, default, *_ in CLINICAL_FIELDS:
        raw_value = row.get(field, default)
        numeric_value = pd.to_numeric(raw_value, errors="coerce")
        values[field] = float(default if pd.isna(numeric_value) else numeric_value)
    return patient_id, values


def clinical_form(defaults: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    cols = st.sidebar.columns(2)
    for index, (field, label, default, min_value, max_value, step) in enumerate(CLINICAL_FIELDS):
        with cols[index % 2]:
            current_value = float(defaults.get(field, default))
            if field in {"syncope", "rv_dilatation"}:
                option_labels = list(YES_NO_OPTIONS)
                selected_index = 1 if current_value >= 0.5 else 0
                selected = st.selectbox(label, option_labels, index=selected_index, key=f"clinical_{field}")
                values[field] = YES_NO_OPTIONS[selected]
            elif field == "coronary_catheterization":
                option_labels = list(CORONARY_CATH_OPTIONS)
                encoded_values = np.array(list(CORONARY_CATH_OPTIONS.values()), dtype=float)
                selected_index = int(np.argmin(np.abs(encoded_values - current_value)))
                selected = st.selectbox(label, option_labels, index=selected_index, key=f"clinical_{field}")
                values[field] = CORONARY_CATH_OPTIONS[selected]
            else:
                values[field] = float(
                    st.number_input(
                        label,
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=current_value,
                        step=float(step),
                        key=f"clinical_{field}",
                    )
                )
    return values


@st.cache_data(show_spinner=False)
def load_normalization_params() -> dict[str, dict[str, float]]:
    if not NORMALIZATION_PATH.exists():
        return {}
    return json.loads(NORMALIZATION_PATH.read_text(encoding="utf-8"))


def normalize_clinical_features(raw_values: dict[str, float]) -> dict[str, float]:
    params = load_normalization_params()
    normalized: dict[str, float] = {}
    for field, *_ in CLINICAL_FIELDS:
        value = float(raw_values.get(field, 0.0) or 0.0)
        if field not in params:
            normalized[field] = value
            continue
        min_value = float(params[field]["min"])
        max_value = float(params[field]["max"])
        normalized[field] = 0.0 if abs(max_value - min_value) < 1e-8 else float((value - min_value) / (max_value - min_value))
    return normalized


def severity_label_from_percent(percent: float) -> str:
    if percent < 5:
        return "none_or_minimal"
    if percent < 15:
        return "mild"
    if percent < 30:
        return "moderate"
    return "severe"


def inference_success_label(results: dict[str, Any] | None) -> str:
    if results is None:
        return "--"
    return "Success" if float(results.get("success_score", 0.0)) >= 1.0 else "Failed"


def average_pvc_per_lead(results: dict[str, Any] | None) -> str:
    if results is None:
        return "--"
    lead_counts = {lead: 0 for lead in LEAD_ORDER}
    for row in results.get("cycle_predictions", []):
        if bool(row.get("is_pvc")):
            lead_counts[str(row.get("lead_label"))] = lead_counts.get(str(row.get("lead_label")), 0) + 1
    return f"{(sum(lead_counts.values()) / max(1, len(LEAD_ORDER))):.2f}"


def cached_fusion_vector(patient_id: str | None, all_columns: list[str]) -> np.ndarray | None:
    if not patient_id:
        return None
    path = ROOT_DIR / "data" / "ecg-data-features" / patient_id / "fusion" / "fusion_vector.npy"
    if not path.exists():
        return None
    vector = np.load(path).astype(float)
    if len(vector) != len(all_columns):
        return None
    return vector


def predict_severity(
    clinical_values: dict[str, float],
    cycle_predictions: list[dict[str, Any]],
    patient_id: str | None,
    use_cached_fusion: bool,
) -> dict[str, Any]:
    if not SEVERITY_MODEL_PATH.exists():
        return {"severity_percent": 0.0, "severity_label": "model_missing", "source": "missing_model"}
    model = np.load(SEVERITY_MODEL_PATH, allow_pickle=True)
    all_columns = [str(column) for column in model["all_columns"].tolist()]
    selected_idx = model["selected_idx"].astype(int)
    weights = model["weights"].astype(float)
    mean = model["mean"].astype(float)
    std = model["std"].astype(float)
    cached_vector = cached_fusion_vector(patient_id, all_columns) if use_cached_fusion else None
    feature_values = {column: 0.0 for column in all_columns}
    if cached_vector is not None:
        feature_values.update({column: float(cached_vector[index]) for index, column in enumerate(all_columns)})
    feature_values.update(normalize_clinical_features(clinical_values))
    pvc_confidences = sorted(
        [float(row["best_confidence"]) for row in cycle_predictions if bool(row.get("is_pvc"))],
        reverse=True,
    )[:5]
    for rank in range(5):
        feature_values[f"pvc_conf_top_{rank + 1}"] = pvc_confidences[rank] if rank < len(pvc_confidences) else 0.0
    x_raw = np.array([feature_values.get(column, 0.0) for column in all_columns], dtype=float)
    x = (x_raw - mean) / np.where(std < 1e-8, 1.0, std)
    x_selected = x[selected_idx]
    severity_percent = float(np.clip(np.r_[1.0, x_selected] @ weights, 0.0, 100.0))
    return {
        "severity_percent": severity_percent,
        "severity_label": severity_label_from_percent(severity_percent),
        "source": str(model["model_name"]),
        "pvc_conf_top5": pvc_confidences,
        "note": (
            "Menggunakan cached fusion/ViT pasien valid, lalu clinical dan confidence PVC dari inference saat ini dioverride."
            if cached_vector is not None
            else "ViT embedding diisi 0 pada demo ini; confidence PVC dari YOLO tetap dipakai."
        ),
    }


def draw_crop_box_preview(image: Image.Image, crop_box: dict[str, int], label: str = "crop") -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    x1, y1, x2, y2 = crop_box["x1"], crop_box["y1"], crop_box["x2"], crop_box["y2"]
    draw.rectangle((x1, y1, x2, y2), outline=(230, 20, 20, 255), width=3)
    draw.text((x1 + 4, max(0, y1 - 18)), label, fill=(230, 20, 20, 255))
    return canvas


def draw_segmentation_overlay(crop_image: Image.Image, segments: list[dict[str, Any]]) -> Image.Image:
    canvas = np.array(crop_image.convert("RGB"))
    for segment in segments:
        color = np.array(LEAD_COLORS[segment["lead_label"]], dtype=np.float32)
        mask = segment["stages"]["line_mask"]
        ys, xs = np.where(mask > 0)
        yy = ys + int(segment["y0"])
        xx = xs + int(segment["x0"])
        valid = (yy >= 0) & (yy < canvas.shape[0]) & (xx >= 0) & (xx < canvas.shape[1])
        canvas[yy[valid], xx[valid]] = (canvas[yy[valid], xx[valid]].astype(np.float32) * 0.2 + color * 0.8).astype(np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for segment in segments:
        color = LEAD_COLORS[segment["lead_label"]]
        draw.rectangle((segment["x0"], segment["y0"], segment["x1"], segment["y1"]), outline=color, width=1)
        draw.line((segment["x0"], segment["baseline_y"], segment["x1"], segment["baseline_y"]), fill=color, width=1)
        draw.text((segment["x0"] + 2, max(0, segment["y0"] - 10)), segment["lead_label"], fill=color)
    return image


def draw_segmentation_mask_on_raw(
    raw_image: Image.Image,
    crop_box_standard: dict[str, int],
    segments: list[dict[str, Any]],
) -> Image.Image:
    canvas = np.array(raw_image.copy().convert("RGB"))
    raw_w, raw_h = raw_image.size
    scale_x = raw_w / STANDARD_SIZE[0]
    scale_y = raw_h / STANDARD_SIZE[1]

    for segment in segments:
        mask = segment["stages"]["line_mask"].astype(np.uint8)
        mask_h, mask_w = mask.shape
        raw_mask_w = max(1, int(round(mask_w * scale_x)))
        raw_mask_h = max(1, int(round(mask_h * scale_y)))
        raw_mask = cv2.resize(mask, (raw_mask_w, raw_mask_h), interpolation=cv2.INTER_NEAREST)

        x0 = int(round((crop_box_standard["x1"] + int(segment["x0"])) * scale_x))
        y0 = int(round((crop_box_standard["y1"] + int(segment["y0"])) * scale_y))
        color = np.array(LEAD_COLORS[segment["lead_label"]], dtype=np.float32)

        ys, xs = np.where(raw_mask > 0)
        yy = y0 + ys
        xx = x0 + xs
        valid = (yy >= 0) & (yy < canvas.shape[0]) & (xx >= 0) & (xx < canvas.shape[1])
        yy = yy[valid]
        xx = xx[valid]
        canvas[yy, xx] = (canvas[yy, xx].astype(np.float32) * 0.15 + color * 0.85).astype(np.uint8)

    return Image.fromarray(canvas)


def draw_cycle_windows_on_raw(raw_image: Image.Image, records: list[dict[str, Any]]) -> Image.Image:
    canvas = raw_image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for record in records:
        color = LEAD_COLORS[record["segment"]["lead_label"]]
        for cycle in record["cycles"]:
            box = cycle["raw_box"]
            draw.rectangle((box["x1"], box["y1"], box["x2"], box["y2"]), outline=color, width=3)
    return canvas


def draw_yolo_detections_on_raw(raw_image: Image.Image, detections: list[dict[str, Any]], only_pvc: bool = True) -> Image.Image:
    canvas = raw_image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for row in detections:
        is_pvc = "pvc" in str(row["class_name"]).lower() or int(row["class_id"]) == 1
        if only_pvc and not is_pvc:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in row["bbox_raw"]]
        color = (220, 38, 38) if is_pvc else (34, 197, 94)
        label = f"{row['lead_label']} {row['class_name']} {row['confidence']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle((x1, max(0, y1 - 24), x1 + (text_box[2] - text_box[0]) + 8, y1), fill=color)
        draw.text((x1 + 4, max(0, y1 - 22)), label, fill=(255, 255, 255), font=font)
    return canvas


def make_cycle_grid(records: list[dict[str, Any]], max_items: int = 24) -> Image.Image | None:
    items = [(record["segment"]["lead_label"], cycle) for record in records for cycle in record["cycles"]]
    if not items:
        return None
    thumb = 112
    cols = 6
    rows = int(np.ceil(min(len(items), max_items) / cols))
    canvas = Image.new("RGB", (cols * (thumb + 10) + 10, rows * (thumb + 30) + 10), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, cycle) in enumerate(items[:max_items]):
        row, col = divmod(idx, cols)
        x = 10 + col * (thumb + 10)
        y = 10 + row * (thumb + 30)
        draw.text((x, y), f"{label}-{cycle['order']}", fill=LEAD_COLORS[label])
        canvas.paste(cycle["yolo_image"].resize((thumb, thumb), Image.Resampling.NEAREST), (x, y + 18))
    return canvas


def run_pipeline(
    raw_image: Image.Image,
    crop_reference_box: dict[str, int],
    clinical_values: dict[str, float],
    yolo_weight_path: str,
    confidence_threshold: float,
    patient_id: str | None = None,
    use_cached_fusion: bool = True,
) -> dict[str, Any]:
    print("[Inference] Starting ECG inference pipeline.", flush=True)
    print(f"[Inference] Patient ID: {patient_id or '-'}", flush=True)
    print(f"[Inference] YOLO weight: {yolo_weight_path}", flush=True)
    print(f"[Inference] YOLO confidence threshold: {confidence_threshold:.2f}", flush=True)

    print("[Inference] Step 1/7 - Standardizing raw ECG image.", flush=True)
    standardized = standardize_image(raw_image)
    crop_box_standard = scale_box(crop_reference_box, REFERENCE_SIZE, STANDARD_SIZE)
    crop_image = standardized.crop((crop_box_standard["x1"], crop_box_standard["y1"], crop_box_standard["x2"], crop_box_standard["y2"]))

    print("[Inference] Step 2/7 - Estimating 12-lead baselines.", flush=True)
    baselines = estimate_12lead_baselines(crop_image)

    print("[Inference] Step 3/7 - Building dynamic lead boxes and masks.", flush=True)
    lead_boxes = initial_lead_boxes(crop_image, baselines)
    segments = [expand_box_until_signal_inside(crop_image, box) for box in lead_boxes]

    print("[Inference] Step 4/7 - Reconstructing raw-signal beats at 640x640.", flush=True)
    records = build_cycle_records(raw_image, crop_image, crop_box_standard, segments)

    print("[Inference] Step 5/7 - Running YOLO PVC detection.", flush=True)
    detections, cycle_predictions, yolo_source = run_yolo_on_cycles(records, crop_box_standard, raw_image.size, yolo_weight_path, confidence_threshold)

    print("[Inference] Step 6/7 - Estimating severity from clinical and PVC features.", flush=True)
    severity = predict_severity(clinical_values, cycle_predictions, patient_id, use_cached_fusion)
    cycle_count = sum(len(record["cycles"]) for record in records)
    success_score = 1.0 if cycle_count > 0 and yolo_source == "yolo" else 0.0
    pvc_count = sum(1 for row in cycle_predictions if row["is_pvc"])
    print(
        "[Inference] Step 7/7 - Completed. "
        f"beats={cycle_count}, detections={len(detections)}, pvc={pvc_count}, "
        f"status={'success' if success_score >= 1.0 else 'failed'}, "
        f"severity={severity['severity_percent']:.2f}% ({severity['severity_label']}).",
        flush=True,
    )
    return {
        "raw_image": raw_image,
        "standardized": standardized,
        "crop_box_standard": crop_box_standard,
        "crop_box_raw": box_from_standard_to_raw(crop_box_standard, raw_image.size),
        "crop_image": crop_image,
        "baselines": baselines,
        "segments": segments,
        "records": records,
        "detections": detections,
        "cycle_predictions": cycle_predictions,
        "severity": severity,
        "success_score": success_score,
        "yolo_source": yolo_source,
        "cycle_count": cycle_count,
    }


def patient_defaults_from_csv() -> dict[str, float]:
    example_path = ROOT_DIR / "examples" / "patient_01_clinical_features_raw_example.csv"
    if not example_path.exists():
        return {field: default for field, _, default, *_ in CLINICAL_FIELDS}
    row = pd.read_csv(example_path).iloc[0].to_dict()
    return {field: float(row.get(field, default) or 0.0) for field, _, default, *_ in CLINICAL_FIELDS}


def main() -> None:
    st.title("ECG PVC Demo Inference")
    st.caption("Demo pipeline follows the notebook order: raw ECG -> standard resize -> PVC crop -> lead segmentation -> raw-signal beat reconstruction at 640 -> YOLO -> lightweight severity fusion.")

    defaults = patient_defaults_from_csv()

    with st.sidebar:
        st.header("Input")
        image_upload = st.file_uploader("ECG raw image", type=["png", "jpg", "jpeg"])
        clinical_csv = st.file_uploader("Clinical 1D CSV", type=["csv"])
        st.divider()
        st.subheader("Manual 1D Data")
        clinical_values = defaults
        csv_patient_id = None
        if clinical_csv is not None:
            try:
                csv_patient_id, clinical_values = load_clinical_csv(clinical_csv)
                st.success("Clinical CSV loaded.")
            except Exception as exc:
                st.error(f"Failed to read clinical CSV: {exc}")
        patient_id = st.text_input("Patient ID", value=csv_patient_id or "P-00001")
        clinical_values = clinical_form(clinical_values)
        st.divider()
        st.subheader("Crop Preset")
        preset = st.selectbox("Coordinate preset", ["P-00001", "P-00002", "P-00007", "P-00015"], index=0)
        reference_box = dict(PATIENT_REFERENCE_CROP_BOXES[preset])
        reference_box["x1"] = st.slider("Left", 0, REFERENCE_SIZE[0] - 2, reference_box["x1"])
        reference_box["x2"] = st.slider("Right", 1, REFERENCE_SIZE[0], reference_box["x2"])
        reference_box["y1"] = st.slider("Top", 0, REFERENCE_SIZE[1] - 2, reference_box["y1"])
        reference_box["y2"] = st.slider("Bottom", 1, REFERENCE_SIZE[1], reference_box["y2"])
        reference_box = validate_box(reference_box)
        st.divider()
        st.subheader("Model")
        selected_yolo_model = st.selectbox("YOLO model", list(YOLO_MODEL_OPTIONS), index=0)
        yolo_weight_path = YOLO_MODEL_OPTIONS[selected_yolo_model]
        confidence_threshold = st.slider("YOLO confidence", 0.05, 0.95, 0.50, 0.05)
        use_cached_fusion = st.checkbox("Use cached ViT fusion when the patient ID is valid", value=True)

    if image_upload is None:
        st.info("Upload a raw ECG image to run the demo inference.")
        st.stop()

    raw_image = load_image_upload(image_upload)
    standardized = standardize_image(raw_image)
    crop_box_standard = scale_box(reference_box, REFERENCE_SIZE, STANDARD_SIZE)
    current_results = st.session_state.get("last_results")

    st.subheader("Main Inference Output")
    metric_cols = st.columns(4)
    if current_results is None:
        metric_cols[0].metric("Avg PVC per lead", "--")
        metric_cols[1].metric("PVC detected in all leads", "--")
        metric_cols[2].metric("Inference Status", "--")
        metric_cols[3].metric("Severity", "--")
    else:
        severity = current_results["severity"]
        metric_cols[0].metric("Avg PVC per lead", average_pvc_per_lead(current_results))
        metric_cols[1].metric("PVC detected in all leads", sum(1 for row in current_results["cycle_predictions"] if row["is_pvc"]))
        metric_cols[2].metric("Inference Status", inference_success_label(current_results))
        metric_cols[3].metric("Severity", f"{severity['severity_percent']:.2f}% ({severity['severity_label']})")
        st.caption(
            f"YOLO source: {current_results['yolo_source']} | "
            f"Severity source: {severity['source']} | {severity.get('note', '')}"
        )

    if st.button("Run inference", type="primary", width="stretch"):
        with st.spinner("Running ECG preprocessing, beat reconstruction at 640, and YOLO inference..."):
            st.session_state["last_results"] = run_pipeline(
                raw_image,
                reference_box,
                clinical_values,
                yolo_weight_path,
                confidence_threshold,
                patient_id=patient_id,
                use_cached_fusion=use_cached_fusion,
            )
        st.rerun()

    preview_cols = st.columns(3)
    with preview_cols[0]:
        st.subheader("Raw ECG")
        st.image(raw_image, width="stretch")
    with preview_cols[1]:
        st.subheader("Resized + Crop Box")
        st.image(draw_crop_box_preview(standardized, crop_box_standard, "PVC crop"), width="stretch")
    with preview_cols[2]:
        st.subheader("PVC Crop")
        st.image(standardized.crop((crop_box_standard["x1"], crop_box_standard["y1"], crop_box_standard["x2"], crop_box_standard["y2"])), width="stretch")

    with st.expander("Coordinate and Size Details", expanded=False):
        st.json(
            {
                "raw_size": raw_image.size,
                "standard_size": STANDARD_SIZE,
                "reference_crop_box": reference_box,
                "standard_crop_box": crop_box_standard,
                "raw_crop_box": box_from_standard_to_raw(crop_box_standard, raw_image.size),
            }
        )

    results = st.session_state.get("last_results")
    if results is None:
        st.stop()

    st.subheader("Lead Segmentation and Mask")
    st.image(
        draw_segmentation_mask_on_raw(results["raw_image"], results["crop_box_standard"], results["segments"]),
        width="stretch",
    )

    st.subheader("Beat YOLO Input 640 dari Rekonstruksi Raw Signal")
    cycle_grid = make_cycle_grid(results["records"])
    if cycle_grid is not None:
        st.image(cycle_grid, width="stretch")
    else:
        st.warning("No beat was generated.")

    image_cols = st.columns(2)
    with image_cols[0]:
        st.subheader("Raw ECG + Beat Boxes")
        st.image(draw_cycle_windows_on_raw(results["raw_image"], results["records"]), width="stretch")
    with image_cols[1]:
        st.subheader("Raw ECG + BBox PVC")
        st.image(draw_yolo_detections_on_raw(results["raw_image"], results["detections"], only_pvc=True), width="stretch")

    st.subheader("Detection Summary")
    if results["cycle_predictions"]:
        beat_predictions = pd.DataFrame(results["cycle_predictions"]).rename(
            columns={"cycle_id": "beat_id", "cycle_order": "beat_order"}
        )
        st.dataframe(beat_predictions, width="stretch", hide_index=True)
    if results["detections"]:
        flat_rows = []
        for row in results["detections"]:
            flat_rows.append(
                {
                    "beat_id": row["cycle_id"],
                    "lead_label": row["lead_label"],
                    "class_name": row["class_name"],
                    "confidence": row["confidence"],
                    "bbox_raw_x1": row["bbox_raw"][0],
                    "bbox_raw_y1": row["bbox_raw"][1],
                    "bbox_raw_x2": row["bbox_raw"][2],
                    "bbox_raw_y2": row["bbox_raw"][3],
                }
            )
        st.dataframe(pd.DataFrame(flat_rows), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
