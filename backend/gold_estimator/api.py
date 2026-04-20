from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from .purity import KARAT_DENSITY_RANGES, density_to_karat_formula, estimate_purity
from .volume import (
    decode_image,
    estimate_multiview_volume,
    estimate_quick_volume,
)


app = FastAPI(title="Gold Purity Estimator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _png_data_url(img: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode graph image.")
    import base64

    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _density_graph(density: float, karat: float) -> str:
    width, height = 1000, 520
    margin_left, margin_right, margin_top, margin_bottom = 82, 32, 42, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x_min, x_max = 10.5, 20.0
    y_min, y_max = 8.0, 26.0

    img = np.full((height, width, 3), 255, dtype=np.uint8)

    def x_px(x: float) -> int:
        return int(margin_left + (x - x_min) / (x_max - x_min) * plot_w)

    def y_px(y: float) -> int:
        return int(margin_top + (y_max - y) / (y_max - y_min) * plot_h)

    # Grid and labels
    for x in range(11, 21):
        px = x_px(x)
        cv2.line(img, (px, margin_top), (px, margin_top + plot_h), (230, 233, 236), 1)
        cv2.putText(img, str(x), (px - 10, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 86, 94), 1)
    for y in [10, 14, 18, 20, 22, 24]:
        py = y_px(y)
        cv2.line(img, (margin_left, py), (margin_left + plot_w, py), (225, 229, 233), 1)
        cv2.putText(img, f"{y}K", (22, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 86, 94), 1)

    # Density range bands
    for karat_key, (d_min, d_max) in KARAT_DENSITY_RANGES.items():
        y = y_px(karat_key)
        cv2.rectangle(
            img,
            (x_px(d_min), y - 13),
            (x_px(d_max), y + 13),
            (222, 241, 238),
            -1,
        )
        cv2.rectangle(
            img,
            (x_px(d_min), y - 13),
            (x_px(d_max), y + 13),
            (30, 127, 100),
            1,
        )

    # Polynomial curve
    points = []
    for d in np.linspace(x_min, x_max, 400):
        k = density_to_karat_formula(float(d))
        if y_min <= k <= y_max:
            points.append((x_px(float(d)), y_px(float(k))))
    if len(points) > 1:
        cv2.polylines(img, [np.array(points, dtype=np.int32)], False, (168, 102, 18), 3)

    # Sample point
    clamped_karat = max(y_min, min(y_max, karat))
    sample_x = max(x_min, min(x_max, density))
    cx, cy = x_px(sample_x), y_px(clamped_karat)
    cv2.circle(img, (cx, cy), 11, (20, 20, 220), -1)
    cv2.circle(img, (cx, cy), 16, (20, 20, 220), 2)
    cv2.putText(
        img,
        f"sample: {density:.2f} g/cm3 -> {karat:.2f}K",
        (min(cx + 18, width - 350), max(cy - 18, 26)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 120),
        2,
    )

    # Axes and titles
    cv2.rectangle(img, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (35, 40, 45), 2)
    cv2.putText(img, "Density vs Karat", (margin_left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 24, 28), 2)
    cv2.putText(img, "Density (g/cm3)", (width // 2 - 95, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (35, 40, 45), 2)
    cv2.putText(img, "Polynomial curve + density ranges", (margin_left + 250, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 86, 94), 1)

    return _png_data_url(img)


def _response(weight_g: float, volume) -> dict:
    purity = estimate_purity(weight_g, volume.volume_cm3)
    return {
        "weight_g": weight_g,
        "volume": asdict(volume),
        "purity": asdict(purity),
        "karat_density_ranges": {
            str(karat): {"min": limits[0], "max": limits[1]}
            for karat, limits in KARAT_DENSITY_RANGES.items()
        },
        "equation": "K = 0.0089D^3 - 0.550D^2 + 12.5299D - 77.06",
        "visuals": {
            "density_graph": _density_graph(
                purity.density_g_cm3, purity.karat_formula
            )
        },
        "warning": (
            "Prototype estimate only. Results depend strongly on photo setup, marker detection, segmentation, and object alloy composition."
        ),
    }


@app.post("/estimate/quick")
async def estimate_quick(
    weight_g: Annotated[float, Form()],
    top_image: Annotated[UploadFile, File()],
    side_image: Annotated[UploadFile, File()],
) -> dict:
    try:
        top_img = decode_image(await top_image.read())
        side_img = decode_image(await side_image.read())
        volume = estimate_quick_volume(top_img, side_img)
        return _response(weight_g, volume)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Estimation failed: {exc}") from exc


@app.post("/estimate/multiview")
async def estimate_multiview(
    weight_g: Annotated[float, Form()],
    top_image: Annotated[UploadFile, File()],
    side_images: Annotated[list[UploadFile], File()],
    angles_deg: Annotated[str | None, Form()] = None,
    voxel_cm: Annotated[float, Form()] = 0.15,
) -> dict:
    try:
        parsed_angles = None
        if angles_deg:
            parsed_angles = [float(v) for v in json.loads(angles_deg)]

        top_img = decode_image(await top_image.read())
        side_imgs = [decode_image(await image.read()) for image in side_images]
        volume = estimate_multiview_volume(
            top_img,
            side_imgs,
            angles_deg=parsed_angles,
            voxel_cm=voxel_cm,
        )
        return _response(weight_g, volume)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail="angles_deg must be a JSON array of numbers."
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Estimation failed: {exc}") from exc
