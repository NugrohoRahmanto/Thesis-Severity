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
from PIL import Image, ImageDraw, ImageOps

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency path
    YOLO = None


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.json"
LEAD_ORDER = [
    "Lead_I",
    "Lead_II",
    "Lead_III",
    "Lead_aVR",
    "Lead_aVL",
    "Lead_aVF",
    "Lead_V1",
    "Lead_V2",
    "Lead_V3",
    "Lead_V4",
    "Lead_V5",
    "Lead_V6",
]
LEAD_DISPLAY = {
    "Lead_I": "I",
    "Lead_II": "II",
    "Lead_III": "III",
    "Lead_aVR": "aVR",
    "Lead_aVL": "aVL",
    "Lead_aVF": "aVF",
    "Lead_V1": "V1",
    "Lead_V2": "V2",
    "Lead_V3": "V3",
    "Lead_V4": "V4",
    "Lead_V5": "V5",
    "Lead_V6": "V6",
}
LEAD_COLORS = {
    "Lead_I": (239, 68, 68),
    "Lead_II": (249, 115, 22),
    "Lead_III": (234, 179, 8),
    "Lead_aVR": (34, 197, 94),
    "Lead_aVL": (14, 165, 233),
    "Lead_aVF": (168, 85, 247),
    "Lead_V1": (236, 72, 153),
    "Lead_V2": (244, 114, 182),
    "Lead_V3": (20, 184, 166),
    "Lead_V4": (59, 130, 246),
    "Lead_V5": (101, 163, 13),
    "Lead_V6": (244, 63, 94),
}
SEVERITY_THRESHOLDS = {"normal": 20.0, "mild": 40.0, "moderate": 70.0}


st.set_page_config(
    page_title="12-Lead ECG Inference Pipeline",
    page_icon="ECG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(36, 99, 235, 0.16), transparent 34%),
            radial-gradient(circle at bottom right, rgba(15, 23, 42, 0.96), rgba(2, 6, 23, 1));
        color: #e5eef8;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(7, 11, 23, 0.98), rgba(3, 8, 20, 0.98));
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 0.75rem 0.9rem;
        border-radius: 0.9rem;
        box-shadow: 0 10px 24px rgba(2, 6, 23, 0.22);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing configuration file: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def get_reference_size(config: dict[str, Any]) -> tuple[int, int]:
    size = config.get("reference_image_size", {})
    return int(size.get("width", 1085)), int(size.get("height", 767))


def get_initial_crop_box(config: dict[str, Any]) -> dict[str, int]:
    box = config.get("initial_crop_box", {})
    return {
        "x1": int(box.get("x1", 30)),
        "y1": int(box.get("y1", 120)),
        "x2": int(box.get("x2", 1040)),
        "y2": int(box.get("y2", 680)),
    }


def clamp_box(box: dict[str, int], width: int, height: int) -> dict[str, int]:
    x1 = max(0, min(int(box["x1"]), width - 1))
    x2 = max(x1 + 1, min(int(box["x2"]), width))
    y1 = max(0, min(int(box["y1"]), height - 1))
    y2 = max(y1 + 1, min(int(box["y2"]), height))
    return {"x1": x1, "x2": x2, "y1": y1, "y2": y2}


def overlay_mask_on_array(base: np.ndarray, mask: np.ndarray, x0: int, y0: int, color: tuple[int, int, int], alpha: float = 0.85) -> None:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return
    yy = y0 + ys
    xx = x0 + xs
    valid = (yy >= 0) & (yy < base.shape[0]) & (xx >= 0) & (xx < base.shape[1])
    yy = yy[valid]
    xx = xx[valid]
    color_arr = np.array(color, dtype=np.float32)
    base[yy, xx] = (base[yy, xx].astype(np.float32) * (1 - alpha) + color_arr * alpha).astype(np.uint8)


def scale_box(box: dict[str, int], src_size: tuple[int, int], dst_size: tuple[int, int]) -> dict[str, int]:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scale_x = dst_w / max(1, src_w)
    scale_y = dst_h / max(1, src_h)
    scaled = {
        "x1": int(round(box["x1"] * scale_x)),
        "x2": int(round(box["x2"] * scale_x)),
        "y1": int(round(box["y1"] * scale_y)),
        "y2": int(round(box["y2"] * scale_y)),
    }
    return clamp_box(scaled, dst_w, dst_h)


def image_from_matplotlib_figure(fig: plt.Figure) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def normalize_signal_matrix(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        raise ValueError("Signal input must be a 1D vector or a 2D matrix.")
    if 12 in data.shape:
        if data.shape[0] == 12:
            matrix = data
        elif data.shape[1] == 12:
            matrix = data.T
        else:
            matrix = data
    else:
        matrix = data if data.shape[0] <= data.shape[1] else data.T
    matrix = matrix.astype(np.float32, copy=False)
    if matrix.shape[0] < 12:
        pad_rows = 12 - matrix.shape[0]
        matrix = np.vstack([matrix, np.repeat(matrix[-1:], pad_rows, axis=0)])
    elif matrix.shape[0] > 12:
        matrix = matrix[:12]
    return matrix


def render_signal_matrix_to_image(signal_matrix: np.ndarray) -> Image.Image:
    matrix = normalize_signal_matrix(signal_matrix)
    sample_count = int(matrix.shape[1])
    if sample_count > 3000:
        indices = np.linspace(0, sample_count - 1, 3000).astype(int)
        matrix = matrix[:, indices]
        sample_count = matrix.shape[1]

    x_axis = np.arange(sample_count)
    fig, axes = plt.subplots(6, 2, figsize=(15, 10), sharex=True)
    fig.patch.set_facecolor("white")
    axes = np.asarray(axes)
    for idx, axis in enumerate(axes.flat):
        series = matrix[idx].astype(np.float32, copy=False)
        series = series - float(np.mean(series))
        std = float(np.std(series)) or 1.0
        series = series / std
        axis.plot(x_axis, series, color="black", linewidth=0.85)
        axis.set_title(LEAD_DISPLAY[LEAD_ORDER[idx]], fontsize=9)
        axis.axis("off")
    fig.tight_layout(pad=0.35)
    return image_from_matplotlib_figure(fig)


def load_signal_upload(uploaded_file: Any) -> np.ndarray:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        uploaded_file.seek(0)
        frame = pd.read_csv(uploaded_file)
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("CSV does not contain numeric columns.")
        values = numeric.to_numpy(dtype=np.float32)
    elif suffix == ".npy":
        uploaded_file.seek(0)
        values = np.load(uploaded_file, allow_pickle=False)
    else:
        raise ValueError("Unsupported signal file type.")
    return np.asarray(values)


def load_image_upload(uploaded_file: Any) -> Image.Image:
    uploaded_file.seek(0)
    return Image.open(uploaded_file).convert("RGB")


def otsu_mask(image: Image.Image, blur_kernel: int = 3, morph_kernel: int = 2) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_kernel = max(1, int(blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    morph_kernel = max(1, int(morph_kernel))
    kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    alpha_mask = Image.fromarray(mask).convert("L").resize(base.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    highlighted = Image.new("RGBA", base.size, (255, 0, 0, 110))
    overlay = Image.composite(highlighted, overlay, alpha_mask)
    return Image.alpha_composite(base, overlay).convert("RGB")


def signal_trace_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    trace_y = np.full(width, np.nan, dtype=np.float32)
    for x_coord in range(width):
        active_y = np.flatnonzero(mask[:, x_coord] > 0)
        if active_y.size:
            trace_y[x_coord] = float(np.median(active_y))
    valid = np.flatnonzero(~np.isnan(trace_y))
    if valid.size == 0:
        trace_y[:] = height / 2.0
    elif valid.size != width:
        trace_y = np.interp(np.arange(width), valid, trace_y[valid]).astype(np.float32)
    trace_x = np.arange(width, dtype=np.int32)
    return trace_x, trace_y


def render_trace_image(trace_x: np.ndarray, trace_y: np.ndarray, canvas_size: tuple[int, int], baseline_y: float) -> Image.Image:
    width, height = canvas_size[1], canvas_size[0]
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.line((0, baseline_y, width, baseline_y), fill=(180, 180, 180), width=1)
    points = list(zip(trace_x.astype(int).tolist(), np.clip(trace_y, 0, height - 1).astype(int).tolist()))
    if len(points) >= 2:
        draw.line(points, fill=(12, 74, 110), width=2)
    return canvas


def reconstruct_lead_to_height(segment: dict[str, Any]) -> dict[str, Any]:
    mask = np.asarray(segment["mask"], dtype=np.uint8)
    trace_x, trace_y = signal_trace_from_mask(mask)
    baseline_y = float(np.nanmedian(trace_y))
    signal_only = render_trace_image(trace_x, trace_y, mask.shape, baseline_y)
    return {
        "mask": mask,
        "trace_x": trace_x,
        "trace_y": trace_y,
        "baseline_y": baseline_y,
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "signal_only": signal_only,
    }


def detect_local_peak_candidates(trace_y: np.ndarray, baseline_y: float) -> list[dict[str, Any]]:
    displacement = np.abs(np.asarray(trace_y, dtype=np.float32) - float(baseline_y))
    if displacement.size == 0:
        return []
    threshold = max(float(np.percentile(displacement, 82)), float(displacement.mean() + displacement.std() * 0.5))
    candidates: list[dict[str, Any]] = []
    for idx in range(1, displacement.size - 1):
        center = displacement[idx]
        if center >= displacement[idx - 1] and center >= displacement[idx + 1] and center >= threshold:
            candidates.append({"x": int(idx), "y": int(round(trace_y[idx])), "amplitude": float(center)})
    if not candidates:
        peak_index = int(np.argmax(displacement))
        candidates.append({"x": peak_index, "y": int(round(trace_y[peak_index])), "amplitude": float(displacement[peak_index])})
    return candidates


def get_consensus_r_peaks(leads_peaks: dict[str, list[int]], pixel_tolerance: int = 25, min_votes: int = 3) -> list[int]:
    observations: list[tuple[int, str]] = []
    for lead_name, peak_list in leads_peaks.items():
        for peak_x in peak_list:
            observations.append((int(peak_x), lead_name))
    if not observations:
        return []
    observations.sort(key=lambda item: item[0])
    clusters: list[list[tuple[int, str]]] = [[observations[0]]]
    for x_coord, lead_name in observations[1:]:
        current_cluster = clusters[-1]
        cluster_center = float(np.median([point[0] for point in current_cluster]))
        if abs(x_coord - cluster_center) <= int(pixel_tolerance):
            current_cluster.append((x_coord, lead_name))
        else:
            clusters.append([(x_coord, lead_name)])
    consensus: list[int] = []
    for cluster in clusters:
        vote_count = len({lead_name for _, lead_name in cluster})
        if vote_count >= int(min_votes):
            consensus.append(int(round(float(np.median([x_coord for x_coord, _ in cluster])))))
    return sorted(set(consensus))


def refine_peak_coordinates(consensus_xs: list[int], trace_y: np.ndarray, baseline_y: float, search_window: int = 40) -> list[dict[str, Any]]:
    trace = np.asarray(trace_y, dtype=np.float32)
    if trace.size == 0:
        return []
    refined: list[dict[str, Any]] = []
    for consensus_x in consensus_xs:
        left = max(0, int(consensus_x) - int(search_window))
        right = min(trace.size - 1, int(consensus_x) + int(search_window))
        local_displacement = np.abs(trace[left : right + 1] - float(baseline_y))
        local_peak_index = int(np.argmax(local_displacement))
        best_x = left + local_peak_index
        refined.append(
            {
                "x": int(best_x),
                "y": int(round(float(trace[best_x]))),
                "amplitude": float(local_displacement[local_peak_index]),
            }
        )
    return refined


def offset_box(box: dict[str, int], offset_x: int, offset_y: int) -> dict[str, int]:
    return {
        "x1": int(box["x1"]) + int(offset_x),
        "x2": int(box["x2"]) + int(offset_x),
        "y1": int(box["y1"]) + int(offset_y),
        "y2": int(box["y2"]) + int(offset_y),
    }


def derive_lead_boxes(sheet_crop: Image.Image) -> dict[str, dict[str, int]]:
    width, height = sheet_crop.size
    split_x = width // 2
    band_height = height / 6.0
    lead_boxes: dict[str, dict[str, int]] = {}
    for index, lead_name in enumerate(LEAD_ORDER[:6]):
        y0 = int(round(index * band_height))
        y1 = int(round((index + 1) * band_height))
        lead_boxes[lead_name] = clamp_box(
            {"x1": 18, "x2": max(19, split_x - 6), "y1": y0 + 2, "y2": max(y0 + 3, y1 - 2)},
            width,
            height,
        )
    for index, lead_name in enumerate(LEAD_ORDER[6:]):
        y0 = int(round(index * band_height))
        y1 = int(round((index + 1) * band_height))
        lead_boxes[lead_name] = clamp_box(
            {"x1": min(width - 19, split_x + 6), "x2": width - 18, "y1": y0 + 2, "y2": max(y0 + 3, y1 - 2)},
            width,
            height,
        )
    return lead_boxes


def build_cycle_thumbnails(lead_crop: Image.Image, refined_peaks: list[dict[str, Any]], half_window: int) -> list[Image.Image]:
    thumbnails: list[Image.Image] = []
    for peak in refined_peaks[:4]:
        x_center = int(peak["x"])
        x1 = max(0, x_center - int(half_window))
        x2 = min(lead_crop.width, x_center + int(half_window))
        if x2 <= x1:
            continue
        snippet = lead_crop.crop((x1, 0, x2, lead_crop.height)).convert("RGB")
        framed = ImageOps.expand(snippet, border=2, fill=(220, 38, 38))
        thumbnails.append(framed)
    return thumbnails


def draw_peaks_on_image(image: Image.Image, peaks_by_lead: dict[str, list[dict[str, Any]]], lead_boxes: dict[str, dict[str, int]], display_boxes: bool = True) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for lead_name, peak_items in peaks_by_lead.items():
        box = lead_boxes[lead_name]
        if display_boxes:
            draw.rectangle((box["x1"], box["y1"], box["x2"], box["y2"]), outline=(251, 191, 36), width=3)
        for peak in peak_items:
            x_coord = box["x1"] + int(peak["x"])
            y_coord = box["y1"] + int(peak["y"])
            radius = 7
            draw.ellipse((x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius), fill=(239, 68, 68), outline=(255, 255, 255), width=2)
    return canvas


def draw_yolo_boxes(image: Image.Image, boxes: list[dict[str, Any]]) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])
        label = box.get("label", "bbox")
        conf = box.get("confidence")
        caption = f"{label} {conf:.2f}" if isinstance(conf, (float, int)) else label
        draw.rectangle((x1, y1, x2, y2), outline=(34, 197, 94), width=3)
        draw.rectangle((x1, max(0, y1 - 18), x1 + 8 * len(caption) + 8, y1), fill=(34, 197, 94))
        draw.text((x1 + 4, max(0, y1 - 16)), caption, fill=(0, 0, 0))
    return canvas


def predict_yolo_boxes(image: Image.Image, config: dict[str, Any], peaks_by_lead: dict[str, list[dict[str, Any]]], lead_boxes: dict[str, dict[str, int]]) -> tuple[list[dict[str, Any]], str]:
    model_path = ROOT_DIR / str(config.get("model_paths", {}).get("yolo_detector", ""))
    if YOLO is not None and model_path.exists():
        try:
            model = YOLO(str(model_path))
            predictions = model.predict(np.array(image), verbose=False)
            boxes: list[dict[str, Any]] = []
            for prediction in predictions:
                for box in prediction.boxes:
                    xyxy = box.xyxy[0].tolist()
                    boxes.append(
                        {
                            "x1": int(round(xyxy[0])),
                            "y1": int(round(xyxy[1])),
                            "x2": int(round(xyxy[2])),
                            "y2": int(round(xyxy[3])),
                            "label": prediction.names.get(int(box.cls.item()), "PVC"),
                            "confidence": float(box.conf.item()),
                        }
                    )
            if boxes:
                return boxes, "detector"
        except Exception:
            pass

    proxy_boxes: list[dict[str, Any]] = []
    half_window = int(config.get("signal_processing", {}).get("cycle_half_window_pixels", 70))
    for lead_name, peak_items in peaks_by_lead.items():
        box = lead_boxes[lead_name]
        for peak in peak_items[:2]:
            x_center = box["x1"] + int(peak["x"])
            y_center = box["y1"] + int(peak["y"])
            x1 = max(0, x_center - half_window)
            x2 = min(image.width, x_center + half_window)
            y1 = max(0, y_center - half_window)
            y2 = min(image.height, y_center + half_window)
            proxy_boxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": f"{LEAD_DISPLAY[lead_name]} proxy",
                    "confidence": min(0.99, 0.35 + peak["amplitude"] / 100.0),
                }
            )
    return proxy_boxes, "proxy"


def estimate_severity(results: dict[str, Any], config: dict[str, Any]) -> tuple[float, str, str]:
    peaks_by_lead = results["peaks_by_lead"]
    refined_peaks = results["refined_peaks"]
    total_peaks = len(refined_peaks)
    consensus_count = len(results["consensus_xs"])
    mask_ratios = [float(segment["mask"].mean() / 255.0) for segment in results["segments"]]
    mean_mask_ratio = float(np.mean(mask_ratios)) if mask_ratios else 0.0
    amplitude_values = [float(peak["amplitude"]) for peak in refined_peaks]
    mean_amplitude = float(np.mean(amplitude_values)) if amplitude_values else 0.0
    lead_peak_variability = float(np.std([len(peaks) for peaks in peaks_by_lead.values()])) if peaks_by_lead else 0.0

    severity_value = (
        total_peaks * 6.5
        + consensus_count * 7.0
        + min(30.0, mean_amplitude * 0.75)
        + min(20.0, mean_mask_ratio * 120.0)
        + min(12.0, lead_peak_variability * 4.0)
    )
    severity_value = float(np.clip(severity_value, 0.0, 100.0))
    thresholds = config.get("signal_processing", {}).get("classification_thresholds", SEVERITY_THRESHOLDS)
    if severity_value < float(thresholds.get("normal", 20.0)):
        severity_class = "Normal"
    elif severity_value < float(thresholds.get("mild", 40.0)):
        severity_class = "Mild"
    elif severity_value < float(thresholds.get("moderate", 70.0)):
        severity_class = "Moderate"
    else:
        severity_class = "Severe"
    return severity_value, severity_class, "heuristic"


def create_dashboard_state(initial_box: dict[str, int]) -> None:
    if "initial_crop_box" not in st.session_state:
        st.session_state.initial_crop_box = dict(initial_box)
    if "last_inference" not in st.session_state:
        st.session_state.last_inference = None


def reset_initial_crop_box(default_box: dict[str, int]) -> None:
    st.session_state.initial_crop_box = dict(default_box)
    st.session_state["sheet_x1"] = int(default_box["x1"])
    st.session_state["sheet_x2"] = int(default_box["x2"])
    st.session_state["sheet_y1"] = int(default_box["y1"])
    st.session_state["sheet_y2"] = int(default_box["y2"])


def draw_crop_box_preview(image: Image.Image, crop_box: dict[str, int]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    x1 = max(0, min(int(crop_box["x1"]), width - 1))
    x2 = max(x1 + 1, min(int(crop_box["x2"]), width))
    y1 = max(0, min(int(crop_box["y1"]), height - 1))
    y2 = max(y1 + 1, min(int(crop_box["y2"]), height))
    dim_color = (15, 23, 42, 120)
    accent_color = (59, 130, 246, 255)
    draw.rectangle((0, 0, width, y1), fill=dim_color)
    draw.rectangle((0, y2, width, height), fill=dim_color)
    draw.rectangle((0, y1, x1, y2), fill=dim_color)
    draw.rectangle((x2, y1, width, y2), fill=dim_color)
    draw.rectangle((x1, y1, x2, y2), outline=accent_color, width=4)
    label = f"crop box: x1={x1}, x2={x2}, y1={y1}, y2={y2}"
    text_y = max(0, y1 - 24)
    draw.rectangle((x1, text_y, min(width, x1 + 14 * len(label)), y1), fill=(15, 23, 42, 210))
    draw.text((x1 + 6, text_y + 4), label, fill=(226, 232, 240, 255))
    return canvas


def render_initial_crop_editor(default_box: dict[str, int], reference_size: tuple[int, int]) -> None:
    ref_w, ref_h = reference_size
    st.sidebar.subheader("Initial Crop Box")
    if st.sidebar.button("Reset to defaults", use_container_width=True):
        reset_initial_crop_box(default_box)
        st.rerun()

    box = st.session_state.initial_crop_box
    st.sidebar.caption("Geser slider untuk ubah crop awal gambar 12-lead.")
    x1 = st.sidebar.slider("x1", min_value=0, max_value=ref_w - 2, value=int(box["x1"]), key="sheet_x1")
    x2 = st.sidebar.slider("x2", min_value=1, max_value=ref_w, value=int(box["x2"]), key="sheet_x2")
    y1 = st.sidebar.slider("y1", min_value=0, max_value=ref_h - 2, value=int(box["y1"]), key="sheet_y1")
    y2 = st.sidebar.slider("y2", min_value=1, max_value=ref_h, value=int(box["y2"]), key="sheet_y2")
    st.session_state.initial_crop_box = clamp_box({"x1": int(x1), "x2": int(x2), "y1": int(y1), "y2": int(y2)}, ref_w, ref_h)


def render_crop_preview_section(image: Image.Image, crop_box: dict[str, int], reference_size: tuple[int, int]) -> None:
    st.subheader("Crop Preview")
    preview_box = scale_box(crop_box, reference_size, image.size)
    preview_box = clamp_box(preview_box, image.width, image.height)
    preview_image = draw_crop_box_preview(image, preview_box)
    st.image(preview_image, use_container_width=True)
    st.caption(
        f"Kotak crop ditampilkan di atas gambar input. Saat slider berubah, preview ini ikut bergeser dan ukuran kotaknya mengikuti x1/x2/y1/y2."
    )


def pipeline(image: Image.Image, config: dict[str, Any], reference_size: tuple[int, int]) -> dict[str, Any]:
    processed_image = image.convert("RGB")
    sheet_crop_box = scale_box(st.session_state.initial_crop_box, reference_size, processed_image.size)
    sheet_crop_box = clamp_box(sheet_crop_box, processed_image.width, processed_image.height)
    sheet_crop = processed_image.crop((sheet_crop_box["x1"], sheet_crop_box["y1"], sheet_crop_box["x2"], sheet_crop_box["y2"]))
    lead_boxes_local = derive_lead_boxes(sheet_crop)
    lead_boxes_global = {lead_name: offset_box(box, sheet_crop_box["x1"], sheet_crop_box["y1"]) for lead_name, box in lead_boxes_local.items()}
    processing = config.get("signal_processing", {})
    blur_kernel = int(processing.get("otsu_blur_kernel", 3))
    morph_kernel = int(processing.get("morph_kernel_size", 2))
    search_window = int(processing.get("search_window_pixels", 40))
    consensus_tolerance = int(processing.get("consensus_pixel_tolerance", 25))
    min_votes = int(processing.get("min_votes", 3))
    half_cycle_window = int(processing.get("cycle_half_window_pixels", 70))

    segments: list[dict[str, Any]] = []
    peaks_by_lead: dict[str, list[dict[str, Any]]] = {}
    for lead_name in LEAD_ORDER:
        box = lead_boxes_local[lead_name]
        lead_crop = sheet_crop.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
        mask = otsu_mask(lead_crop, blur_kernel=blur_kernel, morph_kernel=morph_kernel)
        reconstructed = reconstruct_lead_to_height({"mask": mask, "lead_label": lead_name})
        trace_peaks = detect_local_peak_candidates(reconstructed["trace_y"], reconstructed["baseline_y"])
        peaks_by_lead[lead_name] = [int(peak["x"]) for peak in trace_peaks]
        segments.append(
            {
                "lead_label": lead_name,
                "lead_crop": lead_crop,
                "mask": mask,
                "box": box,
                "reconstructed": reconstructed,
                "trace_peaks": trace_peaks,
            }
        )

    consensus_xs = get_consensus_r_peaks(peaks_by_lead, pixel_tolerance=consensus_tolerance, min_votes=min_votes)
    refined_peaks: list[dict[str, Any]] = []
    for segment in segments:
        reconstructed = segment["reconstructed"]
        refined_for_segment = refine_peak_coordinates(consensus_xs, reconstructed["trace_y"], reconstructed["baseline_y"], search_window=search_window)
        segment["refined_peaks"] = refined_for_segment
        refined_peaks.extend(refined_for_segment)

    cycle_thumbnails: list[dict[str, Any]] = []
    for segment in segments:
        thumbnails = build_cycle_thumbnails(segment["lead_crop"], segment["refined_peaks"], half_cycle_window)
        for index, thumbnail in enumerate(thumbnails, start=1):
            cycle_thumbnails.append({"lead_label": segment["lead_label"], "order": index, "image": thumbnail})

    overlay_canvas = processed_image.copy().convert("RGB")
    overlay_array = np.array(overlay_canvas)
    for segment in segments:
        local_box = lead_boxes_local[segment["lead_label"]]
        offset_x = sheet_crop_box["x1"] + local_box["x1"]
        offset_y = sheet_crop_box["y1"] + local_box["y1"]
        overlay_mask_on_array(overlay_array, segment["mask"], offset_x, offset_y, LEAD_COLORS[segment["lead_label"]])
    overlay_mask = Image.fromarray(overlay_array)
    peaks_overlay = draw_peaks_on_image(processed_image, {segment["lead_label"]: segment["refined_peaks"] for segment in segments}, lead_boxes_global)
    yolo_boxes, yolo_source = predict_yolo_boxes(processed_image, config, {segment["lead_label"]: segment["refined_peaks"] for segment in segments}, lead_boxes_global)
    yolo_overlay = draw_yolo_boxes(processed_image, yolo_boxes)
    severity_value, severity_class, severity_source = estimate_severity(
        {
            "segments": segments,
            "peaks_by_lead": {segment["lead_label"]: segment["refined_peaks"] for segment in segments},
            "consensus_xs": consensus_xs,
            "refined_peaks": refined_peaks,
        },
        config,
    )

    return {
        "status": "Successful",
        "severity_value": severity_value,
        "severity_class": severity_class,
        "severity_source": severity_source,
        "yolo_source": yolo_source,
        "processed_image": processed_image,
        "sheet_crop_box": sheet_crop_box,
        "sheet_crop": sheet_crop,
        "lead_boxes_local": lead_boxes_local,
        "lead_boxes_global": lead_boxes_global,
        "segments": segments,
        "peaks_by_lead": {segment["lead_label"]: segment["refined_peaks"] for segment in segments},
        "consensus_xs": consensus_xs,
        "refined_peaks": refined_peaks,
        "cycle_thumbnails": cycle_thumbnails,
        "mask_overlay": overlay_mask,
        "peaks_overlay": peaks_overlay,
        "yolo_boxes_overlay": yolo_overlay,
        "yolo_boxes": yolo_boxes,
        "half_cycle_window": half_cycle_window,
    }


def render_top_summary(slot: Any, results: dict[str, Any] | None) -> None:
    with slot.container():
        st.markdown("### Inference Summary")
        columns = st.columns(3)
        if results is None:
            columns[0].metric("Inference Status", "Waiting")
            columns[1].metric("Severity Estimation", "--")
            columns[2].metric("Severity Class", "--")
            st.caption("Upload ECG data, adjust the crop coordinates if needed, then press Run inference.")
            return
        columns[0].metric("Inference Status", results["status"])
        columns[1].metric("Severity Estimation", f"{results['severity_value']:.2f}")
        columns[2].metric("Severity Class", results["severity_class"])
        st.caption(
            f"Severity source: {results['severity_source']} | YOLO overlay source: {results['yolo_source']}"
        )


def render_cycle_grid(cycle_thumbnails: list[dict[str, Any]]) -> None:
    if not cycle_thumbnails:
        st.info("No cycle crops were generated from the current input.")
        return
    st.subheader("Cropped Single-Cycle Images")
    columns = st.columns(4)
    for index, entry in enumerate(cycle_thumbnails[:12]):
        with columns[index % 4]:
            st.image(entry["image"], caption=f"{LEAD_DISPLAY[entry['lead_label']]} cycle {entry['order']}", use_container_width=True)


def render_image_section(title: str, image: Image.Image, caption: str | None = None) -> None:
    st.subheader(title)
    st.image(image, caption=caption, use_container_width=True)


def main() -> None:
    try:
        config = load_config()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    reference_size = get_reference_size(config)
    initial_crop_box = get_initial_crop_box(config)
    create_dashboard_state(initial_crop_box)

    summary_slot = st.empty()
    render_top_summary(summary_slot, st.session_state.get("last_inference"))

    st.title("12-Lead ECG Deep Learning Inference")
    st.caption(
        "Dark-theme Streamlit pipeline for ECG image ingestion, bounding-box adjustment, Otsu masking, consensus R-peak extraction, and severity estimation."
    )

    with st.sidebar:
        st.header("Input Controls")
        image_upload = st.file_uploader("Upload 2D ECG Image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"], key="image_upload")
        signal_upload = st.file_uploader("Upload 1D Signal Data (.csv, .npy)", type=["csv", "npy"], key="signal_upload")
        st.markdown("---")
        st.caption("The image input takes priority. If only signal data is uploaded, a synthetic ECG sheet is rendered first.")
        render_initial_crop_editor(initial_crop_box, reference_size)

    working_image: Image.Image | None = None
    source_label = "none"
    signal_array: np.ndarray | None = None

    if image_upload is not None:
        working_image = load_image_upload(image_upload)
        source_label = "image"
    elif signal_upload is not None:
        try:
            signal_array = load_signal_upload(signal_upload)
            working_image = render_signal_matrix_to_image(signal_array)
            source_label = "signal"
        except Exception as exc:
            st.error(f"Unable to load signal file: {exc}")
            working_image = None
    else:
        st.info("Provide at least one input file to run the pipeline.")

    if working_image is not None:
        preview_cols = st.columns([1.0, 1.0])
        with preview_cols[0]:
            st.subheader("Input Preview")
            st.image(working_image, use_container_width=True)
        with preview_cols[1]:
            render_crop_preview_section(working_image, st.session_state.initial_crop_box, reference_size)

        st.subheader("Configuration Snapshot")
        st.write(
            {
                "source": source_label,
                "reference_size": reference_size,
                "initial_crop_box": st.session_state.initial_crop_box,
                "lightweight_model": config.get("model_paths", {}).get("lightweight_severity_model"),
                "yolo_detector": config.get("model_paths", {}).get("yolo_detector"),
            }
        )
        if signal_array is not None:
            st.write({"signal_shape": tuple(int(x) for x in signal_array.shape)})

    run_button = st.button("Run inference", type="primary", disabled=working_image is None, use_container_width=True)
    if run_button and working_image is not None:
        with st.spinner("Running ECG inference pipeline..."):
            st.session_state.last_inference = pipeline(working_image, config, reference_size)
            st.session_state.last_inference["input_source"] = source_label
        st.rerun()

    results = st.session_state.get("last_inference")
    render_top_summary(summary_slot, results)

    if results is None:
        st.subheader("Visual Diagnostics")
        st.info("Inference results will appear here after the pipeline completes.")
        return

    st.subheader("Visual Diagnostics")
    render_cycle_grid(results["cycle_thumbnails"])
    render_image_section("Initial Crop and Otsu Reconstruction Overlay", results["mask_overlay"], "Masks are derived from the full-sheet crop defined by the initial crop box.")
    render_image_section("Refined R-Peak Overlay", results["peaks_overlay"], "Red dots indicate the refined peak coordinates in the original image space.")
    render_image_section("Predicted YOLO Bounding Boxes", results["yolo_boxes_overlay"], "If a detector checkpoint is not available, the app falls back to a geometry-based proxy overlay.")

    st.subheader("Consensus Summary")
    consensus_frame = pd.DataFrame(
        {
            "lead": [LEAD_DISPLAY[lead] for lead in results["peaks_by_lead"].keys()],
            "peak_count": [len(peaks) for peaks in results["peaks_by_lead"].values()],
        }
    )
    st.dataframe(consensus_frame, use_container_width=True, hide_index=True)

    if results["yolo_boxes"]:
        st.subheader("Detected Bounding Boxes")
        st.dataframe(pd.DataFrame(results["yolo_boxes"]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()