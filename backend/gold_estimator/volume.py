from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Segmentation:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid_px: tuple[float, float]
    bottom_px: float
    top_px: float
    left_px: float
    right_px: float


@dataclass(frozen=True)
class SideInfo:
    angle_deg: float
    scale_cm: float
    segmentation: Segmentation


@dataclass(frozen=True)
class VolumeEstimate:
    volume_cm3: float
    method: str
    dimensions_cm: dict[str, float]
    diagnostics: dict[str, float | int | list[float]]
    visuals: dict[str, str]


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Use JPG or PNG images.")
    return img


def _png_data_url(img: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode debug image.")
    import base64

    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _mask_visual(mask: np.ndarray, title: str, scale: float | None = None) -> str:
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 220, 180), 3)
    cv2.putText(vis, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if scale is not None:
        cv2.putText(
            vis,
            f"scale {scale:.5f} cm/px",
            (18, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 255, 255),
            2,
        )
    return _png_data_url(vis)


def _side_height_visual(mask: np.ndarray, avg_height_cm: float, scale: float) -> str:
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 220, 180), 3)

    ys, xs = np.where(mask > 0)
    if len(xs) and len(ys):
        x_med = int(np.median(xs))
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        cv2.line(vis, (x_med, y_min), (x_med, y_max), (40, 40, 255), 4)
        cv2.putText(
            vis,
            f"weighted height {avg_height_cm:.2f} cm",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis,
            f"scale {scale:.5f} cm/px",
            (18, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 255, 255),
            2,
        )
    return _png_data_url(vis)


def detect_reference_square(
    img: np.ndarray, real_size_cm: float = 1.0
) -> tuple[float, float, tuple[int, int, int, int]]:
    """Detect the 1 cm dark square marker used by the notebooks."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, int, int, int, int, float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue

        aspect_ratio = w / h
        if not 0.75 <= aspect_ratio <= 1.25:
            continue

        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        if rw <= 0 or rh <= 0:
            continue

        px_size = (rw + rh) / 2.0
        score = area - 500 * abs(1 - aspect_ratio)
        candidates.append((score, x, y, w, h, px_size))

    if not candidates:
        raise ValueError(
            "Reference square not found. Place a dark 1 cm x 1 cm square marker in every photo."
        )

    _, x, y, w, h, px_size = max(candidates, key=lambda item: item[0])
    return real_size_cm / px_size, px_size, (x, y, w, h)


def compute_scale_from_ref_prototype(
    img: np.ndarray, real_size_cm: float = 1.0, threshold_value: int = 120
) -> tuple[float, int]:
    """Match prototype.ipynb reference-square detection for quick mode."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 500:
            continue

        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue

        _x, _y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h if h != 0 else 0
        if not 0.8 <= aspect_ratio <= 1.2:
            continue

        return real_size_cm / w, w

    raise ValueError("Square not detected. Try adjusting threshold_value.")


def _background_border_mask(shape: tuple[int, int], margin_ratio: float = 0.06) -> np.ndarray:
    h, w = shape
    margin = max(12, int(min(h, w) * margin_ratio))
    mask = np.zeros((h, w), dtype=bool)
    mask[:margin, :] = True
    mask[-margin:, :] = True
    mask[:, :margin] = True
    mask[:, -margin:] = True
    return mask


def _exclude_reference_region(
    shape: tuple[int, int],
    ref_bbox: tuple[int, int, int, int] | None = None,
    ref_square_size_px: int | None = None,
) -> np.ndarray:
    h, w = shape
    valid = np.ones((h, w), dtype=bool)
    if ref_bbox is None:
        return valid

    x, y, bw, bh = ref_bbox
    pad = int(max(bw, bh, ref_square_size_px or 0) * 0.35)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    valid[y0:y1, x0:x1] = False
    return valid


def _restore_background_holes(
    mask: np.ndarray,
    contour: np.ndarray,
    gray: np.ndarray,
    bg_mean: float,
    bg_similarity: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask, []

    hole_contours: list[np.ndarray] = []
    hierarchy = hierarchy[0]
    contour_idx = None
    for i, candidate in enumerate(contours):
        if candidate.shape == contour.shape and np.array_equal(candidate, contour):
            contour_idx = i
            break
    if contour_idx is None:
        contour_idx = max(
            range(len(contours)),
            key=lambda idx: cv2.contourArea(contours[idx]),
        )

    updated = mask.copy()
    for i, hinfo in enumerate(hierarchy):
        if hinfo[3] != contour_idx:
            continue

        inner = contours[i]
        inner_region_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(inner_region_mask, [inner], -1, 255, -1)
        inner_mean = cv2.mean(gray, mask=inner_region_mask)[0]
        ratio = inner_mean / bg_mean if bg_mean > 0 else 0

        if ratio >= bg_similarity:
            cv2.drawContours(updated, [inner], -1, 0, -1)
            hole_contours.append(inner)

    return updated, hole_contours


def segment_object_reflection_robust(
    img: np.ndarray,
    ref_square_size_px: int | None = None,
    ref_bbox: tuple[int, int, int, int] | None = None,
    bg_similarity: float = 0.80,
    allow_holes: bool = True,
    min_component_area: int = 400,
    edge_low: int = 35,
    edge_high: int = 120,
) -> dict:
    """Segment reflective or dark objects against paper backgrounds.

    The key idea is to stop relying on "dark object on light paper" thresholding.
    We instead:
    1) model the paper background from the image borders in LAB space,
    2) use chroma distance (`a/b`) as the primary, shadow-resistant cue,
    3) select the strongest non-reference component from that chroma mask,
    4) fall back to the general LAB+GrabCut path only when chroma is too weak,
       which protects darker neutral objects without letting shadows dominate
       reflective ones.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    border = _background_border_mask((h, w))
    valid = _exclude_reference_region((h, w), ref_bbox=ref_bbox, ref_square_size_px=ref_square_size_px)
    bg_pixels = lab[border & valid]
    if len(bg_pixels) == 0:
        raise ValueError("Not enough paper background pixels for reflective-object segmentation.")

    bg_median = np.median(bg_pixels, axis=0)
    delta_ab = np.linalg.norm(lab[:, :, 1:] - bg_median[1:], axis=2)
    bg_delta_ab = delta_ab[border & valid]
    chroma_threshold = max(float(np.percentile(bg_delta_ab, 99.5)), 5.0)

    chroma_mask = (delta_ab > chroma_threshold).astype(np.uint8) * 255
    chroma_mask = cv2.morphologyEx(chroma_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    chroma_mask = cv2.morphologyEx(chroma_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    chroma_mask[~valid] = 0

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(chroma_mask, connectivity=8)
    best_label = None
    best_score = -math.inf
    img_cx = w / 2.0
    img_cy = h / 2.0
    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label_id]

        border_touch = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        score = float(area)
        score -= 30000 if border_touch else 0
        score -= 0.0015 * ((cx - img_cx) ** 2 + (cy - img_cy) ** 2)

        if ref_square_size_px is not None:
            aspect = bw / bh if bh else 0
            size_gap = abs(bw - ref_square_size_px) + abs(bh - ref_square_size_px)
            if 0.8 <= aspect <= 1.2 and size_gap < ref_square_size_px * 0.35:
                score -= 25000

        if score > best_score:
            best_score = score
            best_label = label_id

    use_threshold_fallback = best_label is None
    if best_label is None:
        seed_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        seed_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
        seed_area = cv2.countNonZero(seed_mask)
        seed_mean_delta = float(delta_ab[seed_mask > 0].mean()) if seed_area > 0 else 0.0
        chroma_confidence = (seed_mean_delta - chroma_threshold) / (chroma_threshold + 1e-6)

        # Low-chroma matte objects are better served by the original threshold/
        # edge pipeline. This also prevents broad paper shadows from being
        # treated as object material when the chroma signal is weak.
        if seed_area < 0.003 * h * w or chroma_confidence < 0.35:
            use_threshold_fallback = True

    if use_threshold_fallback:
        fallback_mode = "auto" if allow_holes else "threshold"
        fallback = segment_object_prototype(
            img,
            ref_square_size_px=ref_square_size_px,
            ref_bbox=ref_bbox,
            bg_similarity=bg_similarity,
            close_iterations=3,
            edge_low=edge_low,
            edge_high=edge_high,
            segmentation_mode=fallback_mode,
            allow_holes=allow_holes,
            shadow_suppression=True,
            edge_support_min_ratio=0.10,
            min_component_area=min_component_area,
        )
        mask = fallback["mask"].copy()
    else:
        ys, xs = np.where(seed_mask > 0)
        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())

        pad_x = int((x1 - x0 + 1) * 0.25) + 20
        pad_y = int((y1 - y0 + 1) * 0.25) + 20
        x0 = max(0, x0 - pad_x)
        x1 = min(w - 1, x1 + pad_x)
        y0 = max(0, y0 - pad_y)
        y1 = min(h - 1, y1 + pad_y)

        crop = img[y0 : y1 + 1, x0 : x1 + 1].copy()
        crop_seed = seed_mask[y0 : y1 + 1, x0 : x1 + 1]
        gc_mask = np.full(crop.shape[:2], cv2.GC_PR_BGD, np.uint8)
        gc_mask[0, :] = cv2.GC_BGD
        gc_mask[-1, :] = cv2.GC_BGD
        gc_mask[:, 0] = cv2.GC_BGD
        gc_mask[:, -1] = cv2.GC_BGD

        sure_fg = cv2.erode(crop_seed, np.ones((7, 7), np.uint8), iterations=1)
        probable_fg = cv2.dilate(crop_seed, np.ones((11, 11), np.uint8), iterations=1)
        gc_mask[probable_fg > 0] = cv2.GC_PR_FGD
        gc_mask[sure_fg > 0] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

        crop_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        crop_mask = cv2.morphologyEx(crop_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        crop_mask = cv2.morphologyEx(crop_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

        # Keep the refined mask near the original chroma seed so soft shadows
        # that touch the object do not grow into the final contour.
        allowed_region = cv2.dilate(crop_seed, np.ones((35, 35), np.uint8), iterations=1)
        crop_mask = cv2.bitwise_and(crop_mask, allowed_region)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0 : y1 + 1, x0 : x1 + 1] = crop_mask

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        raise ValueError("Object mask is empty after reflective-object cleanup.")

    best_component = None
    best_component_score = -math.inf
    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label_id]

        border_touch = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        score = float(area)
        score -= 30000 if border_touch else 0
        score -= 0.0015 * ((cx - img_cx) ** 2 + (cy - img_cy) ** 2)

        if score > best_component_score:
            best_component_score = score
            best_component = label_id

    if best_component is None:
        raise ValueError("Could not isolate a valid object component after cleanup.")

    mask = np.where(labels == best_component, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Object mask is empty after reflective-object cleanup.")

    obj_contour = max(contours, key=cv2.contourArea)

    bg_mask = cv2.bitwise_not(mask)
    bg_mean = cv2.mean(gray, mask=bg_mask)[0]
    hole_contours: list[np.ndarray] = []
    if allow_holes:
        mask, hole_contours = _restore_background_holes(
            mask,
            obj_contour,
            gray,
            bg_mean=bg_mean,
            bg_similarity=bg_similarity,
        )

    return {
        "mask": mask,
        "contour": obj_contour,
        "holes": hole_contours,
        "hull": cv2.convexHull(obj_contour),
    }


def segment_object_prototype(
    img: np.ndarray,
    ref_square_size_px: int | None = None,
    ref_bbox: tuple[int, int, int, int] | None = None,
    bg_similarity: float = 0.80,
    close_iterations: int = 2,
    edge_low: int = 50,
    edge_high: int = 150,
    segmentation_mode: str = "auto",
    allow_holes: bool = True,
    shadow_suppression: bool = True,
    edge_support_min_ratio: float = 0.06,
    min_component_area: int = 250,
) -> dict:
    """Match prototype.ipynb object segmentation for quick mode."""
    if segmentation_mode == "robust":
        return segment_object_reflection_robust(
            img,
            ref_square_size_px=ref_square_size_px,
            ref_bbox=ref_bbox,
            bg_similarity=bg_similarity,
            allow_holes=allow_holes,
            min_component_area=min_component_area,
            edge_low=edge_low,
            edge_high=edge_high,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, edge_low, edge_high)
    kernel = np.ones((5, 5), np.uint8)
    edge_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th_adapt = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7,
    )
    th = cv2.bitwise_and(th_otsu, th_adapt)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    th_mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    if shadow_suppression and np.count_nonzero(th_mask) > 0:
        edge_support = cv2.dilate(edge_mask, np.ones((7, 7), np.uint8), iterations=1)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th_mask, connectivity=8)
        th_filtered = np.zeros_like(th_mask)

        for label in range(1, n_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_component_area:
                continue

            comp_mask = (labels == label).astype(np.uint8) * 255
            overlap = cv2.countNonZero(cv2.bitwise_and(comp_mask, edge_support))
            overlap_ratio = overlap / area

            if overlap_ratio >= edge_support_min_ratio:
                th_filtered[labels == label] = 255

        if np.count_nonzero(th_filtered) > 0:
            th_mask = th_filtered

    if segmentation_mode == "edge":
        work_mask = edge_mask
    elif segmentation_mode == "threshold":
        work_mask = th_mask
    else:
        work_mask = cv2.bitwise_or(edge_mask, th_mask)

    contours, hierarchy = cv2.findContours(work_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        raise ValueError("No contours found.")

    hierarchy = hierarchy[0]
    external_indices = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    if not external_indices:
        raise ValueError("No external contours found.")

    candidate_indices = []
    for i in external_indices:
        contour = contours[i]
        _x, _y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0

        is_reference_square = False
        if ref_square_size_px is not None:
            size_tolerance = ref_square_size_px * 0.2
            if (
                abs(w - ref_square_size_px) < size_tolerance
                and abs(h - ref_square_size_px) < size_tolerance
                and 0.8 <= aspect_ratio <= 1.2
            ):
                is_reference_square = True

        if not is_reference_square:
            candidate_indices.append(i)

    if not candidate_indices:
        candidate_indices = external_indices

    obj_idx = max(candidate_indices, key=lambda idx: cv2.contourArea(contours[idx]))
    obj_contour = contours[obj_idx]

    outer_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(outer_mask, [obj_contour], -1, 255, -1)
    bg_mask = cv2.bitwise_not(outer_mask)
    bg_mean = cv2.mean(gray, mask=bg_mask)[0]

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [obj_contour], -1, 255, -1)

    hole_contours = []
    if allow_holes:
        for i, h in enumerate(hierarchy):
            if h[3] != obj_idx:
                continue

            inner_contour = contours[i]
            inner_region_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(inner_region_mask, [inner_contour], -1, 255, -1)
            inner_mean = cv2.mean(gray, mask=inner_region_mask)[0]
            ratio = inner_mean / bg_mean if bg_mean > 0 else 0

            if ratio >= bg_similarity:
                cv2.drawContours(mask, [inner_contour], -1, 0, -1)
                hole_contours.append(inner_contour)

    return {
        "mask": mask,
        "contour": obj_contour,
        "holes": hole_contours,
        "hull": cv2.convexHull(obj_contour),
    }


def _prototype_top_visual(mask: np.ndarray) -> str:
    top_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, contour in enumerate(contours):
            color = (0, 255, 255) if hierarchy[i][3] == -1 else (255, 0, 255)
            cv2.drawContours(top_vis, [contour], -1, color, 2)
    return _png_data_url(top_vis)


def _prototype_side_visual(mask: np.ndarray, avg_height_cm: float) -> str:
    side_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(mask > 0)
    if len(xs) and len(ys):
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        x_med = int(np.median(xs))
        cv2.line(side_vis, (x_med, y_min), (x_med, y_max), (0, 0, 255), 2)
        cv2.putText(
            side_vis,
            f"Avg Height: {avg_height_cm:.2f} cm",
            (x_med + 8, max(y_min + 20, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return _png_data_url(side_vis)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def segment_object_general(
    img: np.ndarray, ref_bbox: tuple[int, int, int, int] | None = None
) -> Segmentation:
    h, w = img.shape[:2]
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    valid = np.ones((h, w), dtype=bool)
    if ref_bbox is not None:
        x, y, bw, bh = ref_bbox
        pad = int(max(bw, bh) * 0.25)
        valid[max(0, y - pad) : min(h, y + bh + pad), max(0, x - pad) : min(w, x + bw + pad)] = False

    border = np.zeros((h, w), dtype=bool)
    margin = max(10, int(0.06 * min(h, w)))
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True

    bg_pixels = lab_img[border & valid]
    if len(bg_pixels) == 0:
        raise ValueError("Not enough background pixels for segmentation.")

    bg_median = np.median(bg_pixels, axis=0)
    dist = np.linalg.norm(lab_img.astype(np.float32) - bg_median[None, None, :], axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, init = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if ref_bbox is not None:
        x, y, bw, bh = ref_bbox
        pad = int(max(bw, bh) * 0.2)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        init[y0:y1, x0:x1] = 0
    else:
        x0 = y0 = x1 = y1 = 0

    init = cv2.morphologyEx(init, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    init = cv2.morphologyEx(init, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc_mask[border] = cv2.GC_BGD
    sure_fg = cv2.erode(init, np.ones((7, 7), np.uint8), iterations=1)
    probable_fg = cv2.dilate(init, np.ones((9, 9), np.uint8), iterations=1)
    gc_mask[probable_fg > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD
    if ref_bbox is not None:
        gc_mask[y0:y1, x0:x1] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)

    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = _fill_holes(mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        raise ValueError("No segmented object found.")

    img_cx = w / 2
    img_cy = h / 2
    best_label = 1
    best_score = -math.inf
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label_id]
        border_touch = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        center_penalty = 0.002 * ((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
        score = area - (50000 if border_touch else 0) - center_penalty
        if score > best_score:
            best_score = score
            best_label = label_id

    mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Object mask is empty after cleanup.")

    return Segmentation(
        mask=mask,
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        centroid_px=(float(xs.mean()), float(ys.mean())),
        bottom_px=float(np.percentile(ys, 99.5)),
        top_px=float(np.percentile(ys, 0.5)),
        left_px=float(np.percentile(xs, 0.5)),
        right_px=float(np.percentile(xs, 99.5)),
    )


def estimate_quick_volume(top_img: np.ndarray, side_img: np.ndarray) -> VolumeEstimate:
    top_scale, top_ref_px = compute_scale_from_ref_prototype(top_img)
    side_scale, side_ref_px = compute_scale_from_ref_prototype(side_img)
    top_seg = segment_object_prototype(
        top_img,
        ref_square_size_px=top_ref_px,
        bg_similarity=0.75,
        close_iterations=2,
        segmentation_mode="auto",
        allow_holes=True,
        shadow_suppression=True,
    )
    side_seg = segment_object_prototype(
        side_img,
        ref_square_size_px=side_ref_px,
        bg_similarity=0.75,
        close_iterations=3,
        edge_low=35,
        edge_high=120,
        segmentation_mode="threshold",
        allow_holes=False,
        shadow_suppression=True,
        edge_support_min_ratio=0.10,
        min_component_area=400,
    )

    top_mask = top_seg["mask"]
    side_mask = side_seg["mask"]

    top_area_px = cv2.countNonZero(top_mask)
    if top_area_px == 0:
        raise ValueError("Top mask is empty.")

    top_area_cm2 = top_area_px * (top_scale**2)
    top_x = np.where(np.sum(top_mask > 0, axis=0) > 0)[0]
    side_x = np.where(np.sum(side_mask > 0, axis=0) > 0)[0]
    if len(top_x) == 0 or len(side_x) == 0:
        raise ValueError("Could not extract object width profiles from masks.")

    n_slices = 120
    u = np.linspace(0, 1, n_slices)
    top_cols = np.round(top_x[0] + u * (top_x[-1] - top_x[0])).astype(int)
    side_cols = np.round(side_x[0] + u * (side_x[-1] - side_x[0])).astype(int)

    area_profile_px = np.array(
        [np.count_nonzero(top_mask[:, x]) for x in top_cols], dtype=float
    )
    height_profile_px = []
    for x in side_cols:
        ys = np.where(side_mask[:, x] > 0)[0]
        height_profile_px.append((ys.max() - ys.min() + 1) if len(ys) else 0)
    height_profile_px_arr = np.array(height_profile_px, dtype=float)

    kernel = np.ones(5, dtype=float) / 5.0
    area_profile_px = np.convolve(area_profile_px, kernel, mode="same")
    height_profile_px_arr = np.convolve(height_profile_px_arr, kernel, mode="same")

    weights = area_profile_px / np.sum(area_profile_px)
    avg_height_cm = float(np.sum(weights * height_profile_px_arr * side_scale))
    volume_cm3 = float(top_area_cm2 * avg_height_cm)

    return VolumeEstimate(
        volume_cm3=volume_cm3,
        method="quick_two_view",
        dimensions_cm={
            "top_area_cm2": float(top_area_cm2),
            "avg_height_cm": avg_height_cm,
        },
        diagnostics={
            "top_scale_cm_per_px": float(top_scale),
            "side_scale_cm_per_px": float(side_scale),
            "top_ref_px": float(top_ref_px),
            "side_ref_px": float(side_ref_px),
            "top_area_px": int(top_area_px),
        },
        visuals={
            "top_mask": _prototype_top_visual(top_mask),
            "side_height_mask": _prototype_side_visual(side_mask, avg_height_cm),
        },
    )


def estimate_multiview_volume(
    top_img: np.ndarray,
    side_images: list[np.ndarray],
    angles_deg: list[float] | None = None,
    voxel_cm: float = 0.15,
    angle_offset_deg: float = 0.0,
) -> VolumeEstimate:
    if len(side_images) < 3:
        raise ValueError("Multiview mode needs at least 3 side images.")
    if angles_deg is None:
        step = 360.0 / len(side_images)
        angles_deg = [i * step for i in range(len(side_images))]
    if len(angles_deg) != len(side_images):
        raise ValueError("Number of side angles must match number of side images.")

    top_scale, top_ref_px, top_ref_bbox = detect_reference_square(top_img)
    top_seg = segment_object_general(top_img, top_ref_bbox)

    side_infos: list[SideInfo] = []
    ref_sizes: list[float] = []
    for img, angle in zip(side_images, angles_deg):
        scale, ref_px, ref_bbox = detect_reference_square(img)
        side_infos.append(SideInfo(angle, scale, segment_object_general(img, ref_bbox)))
        ref_sizes.append(ref_px)

    top_mask = top_seg.mask > 0
    top_h, top_w = top_mask.shape
    top_cx, top_cy = top_seg.centroid_px
    ys, xs = np.where(top_mask)

    x_min_cm = (xs.min() - top_cx) * top_scale
    x_max_cm = (xs.max() - top_cx) * top_scale
    y_min_cm = -(ys.max() - top_cy) * top_scale
    y_max_cm = -(ys.min() - top_cy) * top_scale

    z_max_cm = max(
        (side.segmentation.bottom_px - side.segmentation.top_px) * side.scale_cm
        for side in side_infos
    )

    pad = voxel_cm * 2
    x_vals = np.arange(x_min_cm - pad, x_max_cm + pad, voxel_cm)
    y_vals = np.arange(y_min_cm - pad, y_max_cm + pad, voxel_cm)
    z_vals = np.arange(0.0, z_max_cm + pad, voxel_cm)
    if len(x_vals) * len(y_vals) * len(z_vals) > 6_000_000:
        raise ValueError(
            "Voxel grid is too large. Use a larger voxel size or crop images closer to the object."
        )

    occ = np.ones((len(x_vals), len(y_vals), len(z_vals)), dtype=bool)
    xx, yy = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)), indexing="ij")
    u_top = np.round(top_cx + x_vals[xx] / top_scale).astype(int)
    v_top = np.round(top_cy - y_vals[yy] / top_scale).astype(int)
    valid_top = (u_top >= 0) & (u_top < top_w) & (v_top >= 0) & (v_top < top_h)
    in_top = np.zeros((len(x_vals), len(y_vals)), dtype=bool)
    in_top[valid_top] = top_mask[v_top[valid_top], u_top[valid_top]]
    occ[~in_top, :] = False

    all_cx = [
        (side.segmentation.left_px + side.segmentation.right_px) / 2.0
        for side in side_infos
    ]
    all_bottom = [side.segmentation.bottom_px for side in side_infos]
    global_cx = float(np.median(all_cx))
    global_bottom = float(np.median(all_bottom))

    x3, y3, z3 = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    for side in side_infos:
        angle = np.deg2rad(side.angle_deg + angle_offset_deg)
        c, s = np.cos(angle), np.sin(angle)
        mask = side.segmentation.mask > 0
        side_h, side_w = mask.shape
        u = np.round(global_cx + (x3 * c + y3 * s) / side.scale_cm).astype(int)
        v = np.round(global_bottom - z3 / side.scale_cm).astype(int)

        valid = (u >= 0) & (u < side_w) & (v >= 0) & (v < side_h)
        inside = np.zeros_like(occ)
        inside[valid] = mask[v[valid], u[valid]]
        occ &= inside
        if not occ.any():
            raise ValueError(
                f"Visual hull became empty after side image at {side.angle_deg:g} degrees. Check image order, angles, and segmentation."
            )

    filled = np.argwhere(occ)
    x_extent = (filled[:, 0].max() - filled[:, 0].min() + 1) * voxel_cm
    y_extent = (filled[:, 1].max() - filled[:, 1].min() + 1) * voxel_cm
    z_extent = (filled[:, 2].max() - filled[:, 2].min() + 1) * voxel_cm

    return VolumeEstimate(
        volume_cm3=float(occ.sum() * (voxel_cm**3)),
        method="multiview_visual_hull",
        dimensions_cm={
            "width_cm": float(x_extent),
            "depth_cm": float(y_extent),
            "height_cm": float(z_extent),
        },
        diagnostics={
            "top_scale_cm_per_px": float(top_scale),
            "top_ref_px": float(top_ref_px),
            "side_ref_px": [float(v) for v in ref_sizes],
            "side_count": len(side_images),
            "voxel_cm": float(voxel_cm),
        },
        visuals={
            "top_mask": _mask_visual(top_seg.mask, "top mask", top_scale),
        },
    )
