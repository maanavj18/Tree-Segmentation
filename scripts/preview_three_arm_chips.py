"""Write preview PNGs of the three 100 m arms.

Each chip id has one figure: 2020 Pennington, the same 2019 window, and
the MillionTrees chip of that scene type. The MillionTrees panel is a
different site. Pixel size is left at the native GSD, so the panels are
the same ground window at different resolutions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
CHIPS = ROOT / "data" / "chips"
OUT = CHIPS / "previews"
ARMS = ("2020", "2019", "milliontrees")
SCENES = ("isolated", "street", "merged")


# EPSG:6527 uses US survey feet. MillionTrees chips have no CRS; their
# transform was written in metres.
TO_METRES = {
    "metre": 1.0,
    "meter": 1.0,
    "US survey foot": 1200 / 3937,
    "foot": 0.3048,
}


def load_rgb(path: Path) -> tuple[np.ndarray, float, float]:
    with rasterio.open(path) as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        unit = src.crs.linear_units if src.crs is not None else "metre"
        res_m = abs(float(src.res[0])) * TO_METRES[unit]
        ground_m = src.width * res_m
    if rgb.dtype != np.uint8:
        lo, hi = np.percentile(rgb, (2, 98))
        scaled = np.clip((rgb.astype("float32") - lo) / max(hi - lo, 1), 0, 1)
        rgb = (scaled * 255).astype(np.uint8)
    return rgb, res_m, ground_m


def arm_caption(arm: str, res_m: float, meta: dict) -> str:
    gsd = f"{res_m:.2f} m/px"
    if arm == "2020":
        return f"2020 Pennington\n1 ft leaf-off\n{gsd}"
    if arm == "2019":
        return f"2019 Pennington\nleaf-on NAIP, same window\n{gsd}"
    false_color = "false" in meta["notes"].lower()
    color = "false color" if false_color else "natural color"
    source = "Schütte" if meta["source"].startswith("Sch") else "UrbanLondon"
    return f"MillionTrees, {source}\n{color}, scene match\n{gsd}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        row["chip_id"]: row
        for row in csv.DictReader((CHIPS / "milliontrees" / "manifest.csv").open(newline=""))
    }
    ids = [f"{scene}_{i:02d}" for scene in SCENES for i in range(1, 6)]
    loaded: dict[str, dict[str, list[tuple[np.ndarray, str]]]] = {scene: {} for scene in SCENES}

    for chip_id in ids:
        scene = chip_id.split("_")[0]
        panels = []
        for arm in ARMS:
            image, res_m, ground_m = load_rgb(CHIPS / arm / f"{chip_id}.tif")
            if abs(ground_m - 100) > 1:
                raise SystemExit(f"{arm} {chip_id} ground window is {ground_m:.2f} m")
            caption = arm_caption(arm, res_m, meta[chip_id])
            panels.append((image, caption))
            loaded[scene].setdefault(chip_id, []).append((image, caption))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained")
        for ax, (image, caption) in zip(axes, panels):
            ax.imshow(image)
            ax.set_title(caption, fontsize=10)
            ax.axis("off")
        fig.suptitle(chip_id, fontsize=13)
        dest = OUT / f"{chip_id}.png"
        fig.savefig(dest, dpi=140)
        plt.close(fig)
        print(dest)

    for scene in SCENES:
        fig, axes = plt.subplots(5, 3, figsize=(12, 20), layout="constrained")
        for row, i in enumerate(range(1, 6)):
            chip_id = f"{scene}_{i:02d}"
            for col, (image, caption) in enumerate(loaded[scene][chip_id]):
                ax = axes[row, col]
                ax.imshow(image)
                title = caption if col else f"{chip_id}\n{caption}"
                ax.set_title(title, fontsize=9)
                ax.axis("off")
        fig.suptitle(f"{scene}: 100 m windows", fontsize=14)
        dest = OUT / f"sheet_{scene}.png"
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        print(dest)


if __name__ == "__main__":
    main()
