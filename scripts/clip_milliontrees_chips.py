"""Clip 15 MillionTrees chips at a 100 m ground window.

Schütte DOP20 and UrbanLondon City Centre tiles are both 20 cm.
A 100 m window is 500 x 500 pixels. One window per source tile.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely import from_wkt
from shapely.affinity import translate
from shapely.geometry import box, mapping

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "SmallTreePolygons_v0.23" / "images"
CSV_PATH = ROOT / "data" / "SmallTreePolygons_v0.23" / "within-distribution.csv"
OUT = ROOT / "data" / "chips" / "milliontrees"

GSD_M = 0.20
SIDE_PX = 500
GROUND_M = SIDE_PX * GSD_M  # 100 m
HALF_WINDOW_M = GROUND_M / 2

# (chip_id, scene, filename, x0, y0, notes)
# x0, y0 are the top-left pixel in the source PNG.
CHIPS = [
    (
        "isolated_01",
        "isolated",
        "01_Training_dop20_False_Color_22_Sch_tte_et_al_2025.png",
        0,
        200,
        "Spaced crowns on a lawn beside a building. False-color DOP20.",
    ),
    (
        "isolated_02",
        "isolated",
        "04_Training_dop20_False_Color_5_Sch_tte_et_al_2025.png",
        0,
        80,
        "Open-grown crowns on a park lawn, west of the circular clearing. False color.",
    ),
    (
        "isolated_03",
        "isolated",
        "03_Training_dop20_False_Color_0_Sch_tte_et_al_2025.png",
        500,
        450,
        "Crowns standing apart over open ground below a road. A touching group sits on the left. False color.",
    ),
    (
        "isolated_04",
        "isolated",
        "CityCentre_2017_32630_712195_5788908_160_20_32630_Zuniga_Gonzalez_et_al_2023.png",
        150,
        0,
        "Large-lot houses; yard trees stand apart over lawn. Natural color. Sports pitch left out.",
    ),
    (
        "isolated_05",
        "isolated",
        "CityCentre_2017_32630_712515_5788108_160_20_32630_Zuniga_Gonzalez_et_al_2023.png",
        0,
        0,
        "Crowns on a mown lawn. Natural color. Some crowns touch.",
    ),
    (
        "street_01",
        "street",
        "CityCentre_2017_32630_712355_5786988_160_20_32630_Zuniga_Gonzalez_et_al_2023.png",
        0,
        0,
        "Terraced houses, parked cars, and small garden trees. Natural color.",
    ),
    (
        "street_02",
        "street",
        "CityCentre_2017_32630_712995_5789548_160_20_32630_Zuniga_Gonzalez_et_al_2023.png",
        150,
        150,
        "Terraced streets, cars, and garden trees. Natural color.",
    ),
    (
        "street_03",
        "street",
        "CityCentre_2017_32630_713795_5789388_160_20_32630_Zuniga_Gonzalez_et_al_2023.png",
        200,
        200,
        "Residential streets with trees against roofs and cars. Natural color.",
    ),
    (
        "street_04",
        "street",
        "04_Training_dop20_False_Color_11_Sch_tte_et_al_2025.png",
        180,
        450,
        "Street trees against a road, cars, and a low building. False color.",
    ),
    (
        "street_05",
        "street",
        "CityCentre_2017_32630_713955_5789548_160_20_32630_Zuniga_Gonzalez_et_al_2023.png",
        0,
        0,
        "Terraced houses, a roundabout, parked cars, and garden trees. Natural color.",
    ),
    (
        "merged_01",
        "merged",
        "06_Training_dop20_False_Color_3_Sch_tte_et_al_2025.png",
        0,
        0,
        "Continuous canopy beside a road. False color.",
    ),
    (
        "merged_02",
        "merged",
        "04_Training_dop20_False_Color_10_Sch_tte_et_al_2025.png",
        0,
        280,
        "Touching park crowns. False color.",
    ),
    (
        "merged_03",
        "merged",
        "00_Training_dop20_False_Color_6_Sch_tte_et_al_2025.png",
        100,
        450,
        "Interlocking crowns along paths. False color.",
    ),
    (
        "merged_04",
        "merged",
        "CityCentre_2017_32630_712995_5787148_160_20_32630_Zuniga_Gonzalez_et_al_2023.png",
        0,
        220,
        "Touching park and garden canopy beside houses. Natural color.",
    ),
    (
        "merged_05",
        "merged",
        "06_Training_dop20_False_Color_4_Sch_tte_et_al_2025.png",
        0,
        380,
        "Continuous canopy below a playground. False color. Playground left out.",
    ),
]


def source_name(filename: str) -> str:
    if "Sch_tte" in filename or "Schütte" in filename:
        return "Schütte et al. 2025"
    if "Zuniga_Gonzalez" in filename:
        return "Zuniga-Gonzalez et al. 2023"
    raise ValueError(filename)


def source_gsd_m(filename: str, width: int, height: int) -> float:
    """GSD from the source naming, checked against the pixel grid."""
    if "dop20" in filename:
        if (width, height) != (1000, 1000):
            raise SystemExit(f"{filename} is {width}x{height}, expected 1000x1000 DOP20")
        return 0.20
    match = re.search(r"_(\d+)_(\d+)_32630_Zuniga", filename)
    if match is None:
        raise SystemExit(f"cannot read GSD token from {filename}")
    tile_m = int(match.group(1))
    gsd_cm = int(match.group(2))
    if tile_m != 160 or gsd_cm != 20:
        raise SystemExit(f"{filename} token is {tile_m} m / {gsd_cm} cm, expected 160 m / 20 cm")
    # A full tile is 160 m / 0.20 m = 800 px. 801 px is the edge pixel.
    # Shorter sides are tiles cut by the city boundary, same GSD.
    for side in (width, height):
        if not 700 <= side <= 802:
            raise SystemExit(f"{filename} side {side}px is outside the 20 cm City Centre grid")
    return gsd_cm / 100.0


def london_grid_pitch_m() -> None:
    eastings = set()
    northings = set()
    for path in IMG_DIR.glob("CityCentre_2017_*.png"):
        parts = path.name.split("_")
        # CityCentre_2017_32630_EAST_NORTH_160_20_32630_...
        eastings.add(int(parts[3]))
        northings.add(int(parts[4]))
    for name, values in (("easting", eastings), ("northing", northings)):
        ordered = sorted(values)
        gaps = [b - a for a, b in zip(ordered, ordered[1:])]
        # A missing tile leaves a 320 m gap. Every gap is still a multiple of 160 m.
        if not gaps or any(gap <= 0 or gap % 160 != 0 for gap in gaps):
            raise SystemExit(f"London {name} pitch is {gaps}, expected multiples of 160 m")


def load_polygons(filenames: set[str]) -> dict[str, list]:
    df = pd.read_csv(CSV_PATH, usecols=["filename", "polygon"])
    df = df[df["filename"].isin(filenames)]
    grouped: dict[str, list] = {name: [] for name in filenames}
    for filename, polygon in zip(df["filename"], df["polygon"]):
        grouped[filename].append(from_wkt(polygon))
    missing = [name for name, geoms in grouped.items() if not geoms]
    if missing:
        raise SystemExit(f"no polygons for {missing}")
    return grouped


def clip_polygons(geoms: list, x0: int, y0: int) -> list[dict]:
    window = box(x0, y0, x0 + SIDE_PX, y0 + SIDE_PX)
    features = []
    for geom in geoms:
        if not geom.intersects(window):
            continue
        clipped = geom.intersection(window)
        if clipped.is_empty or clipped.area < 25:
            continue
        moved = translate(clipped, xoff=-x0, yoff=-y0)
        features.append(
            {
                "type": "Feature",
                "properties": {"area_px": round(moved.area, 2)},
                "geometry": mapping(moved),
            }
        )
    return features


def main() -> None:
    if len(CHIPS) != 15:
        raise SystemExit(f"expected 15 chips, got {len(CHIPS)}")
    ids = [row[0] for row in CHIPS]
    files = [row[2] for row in CHIPS]
    if len(set(ids)) != 15 or len(set(files)) != 15:
        raise SystemExit("chip ids or source tiles are not unique")
    london_grid_pitch_m()
    polygons = load_polygons(set(files))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for chip_id, scene, filename, x0, y0, notes in CHIPS:
        src_path = IMG_DIR / filename
        image = np.asarray(Image.open(src_path).convert("RGB"))
        height, width = image.shape[:2]
        gsd = source_gsd_m(filename, width, height)
        if gsd != GSD_M:
            raise SystemExit(f"{filename} GSD {gsd} != {GSD_M}")
        if x0 < 0 or y0 < 0 or x0 + SIDE_PX > width or y0 + SIDE_PX > height:
            raise SystemExit(f"{chip_id} window leaves {filename} ({width}x{height})")
        chip = image[y0 : y0 + SIDE_PX, x0 : x0 + SIDE_PX]
        if chip.shape != (SIDE_PX, SIDE_PX, 3):
            raise SystemExit(f"{chip_id} shape {chip.shape}")
        black = float((chip.sum(axis=2) == 0).mean())
        features = clip_polygons(polygons[filename], x0, y0)
        transform = from_origin(0.0, GROUND_M, GSD_M, GSD_M)
        profile = {
            "driver": "GTiff",
            "height": SIDE_PX,
            "width": SIDE_PX,
            "count": 3,
            "dtype": "uint8",
            "transform": transform,
            "compress": "deflate",
            "photometric": "RGB",
        }
        dest = OUT / f"{chip_id}.tif"
        with rasterio.open(dest, "w", **profile) as dst:
            for band in range(3):
                dst.write(chip[:, :, band], band + 1)
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            dst.update_tags(
                source=source_name(filename),
                src_image=filename,
                x0=str(x0),
                y0=str(y0),
                gsd_m=str(GSD_M),
                scene_type=scene,
            )
        geojson = {"type": "FeatureCollection", "features": features}
        (OUT / f"{chip_id}.geojson").write_text(json.dumps(geojson))
        ground_m = SIDE_PX * gsd
        rows.append(
            {
                "chip_id": chip_id,
                "scene_type": scene,
                "source": source_name(filename),
                "src_image": filename,
                "x0": x0,
                "y0": y0,
                "side_px": SIDE_PX,
                "gsd_m": f"{gsd:.2f}",
                "half_window_m": HALF_WINDOW_M,
                "ground_m": f"{ground_m:.2f}",
                "n_polygons": len(features),
                "black_fraction": f"{black:.3f}",
                "notes": notes,
            }
        )
        print(
            f"{chip_id} {scene} {SIDE_PX}px x {gsd:.2f} m = {ground_m:.2f} m  "
            f"polygons={len(features)} black={black:.3f}"
        )

    with (OUT / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
