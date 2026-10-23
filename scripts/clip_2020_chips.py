"""Clip the 15 Pennington 2020 chips in data/chips/2020/.

Centers were hand-picked on the leaf-off COGs. Each window is 328 x 328
pixels at 1 US survey foot (328 ft = 99.97 m), centered on the chosen point.
half_window_m in the manifest is the design value, 50.
"""

from __future__ import annotations

import csv
import struct
import zlib
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
COGS = ROOT / "COGS"
OUT = ROOT / "data" / "chips" / "2020"

TILES = {
    "F10A16": COGS / "F10A16.tif",
    "F10B13": COGS / "F10B13.tif",
    "F10C4": COGS / "F10C4.tif",
    "F10D1": COGS / "F10D1.tif",
}
ORIGINS = {
    "F10A16": (405000.0, 550000.0),
    "F10B13": (410000.0, 550000.0),
    "F10C4": (405000.0, 545000.0),
    "F10D1": (410000.0, 545000.0),
}

# (chip_id, scene_type, tile, x_ft, y_ft, notes)
# x/y are EPSG:6527 US survey feet.
CHIPS = [
    (
        "isolated_01",
        "isolated",
        "F10B13",
        411562.0,
        547188.0,
        "Large-lot yards north of the borough; pool and wide lawn; crowns stand apart.",
    ),
    (
        "isolated_02",
        "isolated",
        "F10B13",
        412186.0,
        547188.0,
        "Next block east of isolated_01; wide lawns and spaced shade trees.",
    ),
    (
        "isolated_03",
        "isolated",
        "F10C4",
        407812.0,
        543438.0,
        "Open lawns southwest of downtown; one large open-grown crown and spaced yard trees.",
    ),
    (
        "isolated_04",
        "isolated",
        "F10A16",
        408436.0,
        545938.0,
        "Spaced houses and yard trees on lawn along a local road, northwest of the junction.",
    ),
    (
        "isolated_05",
        "isolated",
        "F10D1",
        412186.0,
        544800.0,
        "Large southeast lots with a circular drive and pool; crowns stand apart over lawn.",
    ),
    (
        "street_01",
        "street",
        "F10A16",
        409686.0,
        547188.0,
        "Townhouse rows and pavement; small trees in the gaps and along the drives.",
    ),
    (
        "street_02",
        "street",
        "F10B13",
        411562.0,
        545314.0,
        "Houses, sidewalks, and parked cars on a north-south street; trees in the verge and yards.",
    ),
    (
        "street_03",
        "street",
        "F10B13",
        412186.0,
        545938.0,
        "Yard trees against roofs, driveways, and parked cars.",
    ),
    (
        "street_04",
        "street",
        "F10D1",
        410936.0,
        542814.0,
        "South borough house grid; sidewalks, cars, and yard trees.",
    ),
    (
        "street_05",
        "street",
        "F10D1",
        412186.0,
        543438.0,
        "Curved residential street in the southeast subdivision; houses and cars at the lots.",
    ),
    (
        "merged_01",
        "merged",
        "F10B13",
        410936.0,
        546564.0,
        "Rear lots northeast of downtown; bare crowns touch across property lines.",
    ),
    (
        "merged_02",
        "merged",
        "F10B13",
        412186.0,
        546564.0,
        "House and drive inside a continuous canopy, east of merged_01.",
    ),
    (
        "merged_03",
        "merged",
        "F10C4",
        408436.0,
        543438.0,
        "Wooded rear lots southwest of downtown; crowns form one mass between houses.",
    ),
    (
        "merged_04",
        "merged",
        "F10D1",
        412186.0,
        542814.0,
        "Southeast lots; crowns touch around the houses.",
    ),
    (
        "merged_05",
        "merged",
        "F10D1",
        411562.0,
        543880.0,
        "Overlapping rear-lot canopy along the curved street north of merged_04.",
    ),
]

HALF_PX = 164
SIZE = 328


def _level0(path: Path) -> dict:
    with path.open("rb") as f:
        head = f.read(4096)
    off = 192
    n = struct.unpack("<H", head[off : off + 2])[0]
    tags = {}
    for i in range(n):
        tag, typ, count, val = struct.unpack("<HHII", head[off + 2 + i * 12 : off + 14 + i * 12])
        tags[tag] = (typ, count, val)
    nx = 10

    def longs(tag_id: int) -> tuple[int, ...]:
        typ, count, val = tags[tag_id]
        raw = head[val : val + 4 * count]
        return struct.unpack(f"<{count}I", raw)

    return {"offsets": longs(324), "counts": longs(325), "nx": nx}


def read_rgb(path: Path, col0: int, row0: int) -> np.ndarray:
    info = _level0(path)
    tw = th = 512
    out = np.zeros((SIZE, SIZE, 3), dtype=np.uint16)
    tx0, tx1 = col0 // tw, (col0 + SIZE - 1) // tw
    ty0, ty1 = row0 // th, (row0 + SIZE - 1) // th
    with path.open("rb") as f:
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                idx = ty * info["nx"] + tx
                f.seek(info["offsets"][idx])
                raw = zlib.decompress(f.read(info["counts"][idx]))
                tile = np.frombuffer(raw, dtype="<u2").reshape(th, tw, 4)
                x0, y0 = tx * tw, ty * th
                ix0, iy0 = max(x0, col0), max(y0, row0)
                ix1 = min(x0 + tw, col0 + SIZE)
                iy1 = min(y0 + th, row0 + SIZE)
                out[iy0 - row0 : iy1 - row0, ix0 - col0 : ix1 - col0] = tile[
                    iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0, :3
                ]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    to_wgs = Transformer.from_crs("EPSG:6527", "EPSG:4326", always_xy=True)
    rows = []
    centers = []
    for chip_id, scene, tile, x, y, notes in CHIPS:
        ox, oy = ORIGINS[tile]
        col0 = int(round((x - ox) - HALF_PX))
        row0 = int(round((oy - y) - HALF_PX))
        if not (0 <= col0 and col0 + SIZE <= 5000 and 0 <= row0 and row0 + SIZE <= 5000):
            raise SystemExit(f"{chip_id} leaves {tile}: col {col0} row {row0}")
        rgb = read_rgb(TILES[tile], col0, row0)
        if rgb.max() == 0:
            raise SystemExit(f"{chip_id} is empty")
        x_ul = ox + col0
        y_ul = oy - row0
        cx = x_ul + HALF_PX
        cy = y_ul - HALF_PX
        lon, lat = to_wgs.transform(cx, cy)
        transform = from_origin(x_ul, y_ul, 1.0, 1.0)
        profile = {
            "driver": "GTiff",
            "height": SIZE,
            "width": SIZE,
            "count": 3,
            "dtype": "uint16",
            "crs": CRS.from_epsg(6527),
            "transform": transform,
            "compress": "deflate",
            "photometric": "RGB",
        }
        dest = OUT / f"{chip_id}.tif"
        with rasterio.open(dest, "w", **profile) as dst:
            for b in range(3):
                dst.write(rgb[:, :, b], b + 1)
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        rows.append(
            {
                "chip_id": chip_id,
                "scene_type": scene,
                "lon": f"{lon:.7f}",
                "lat": f"{lat:.7f}",
                "half_window_m": 50,
                "src_tile": tile,
                "notes": notes,
            }
        )
        centers.append((chip_id, cx, cy))
        print(f"{chip_id} {tile} {lon:.7f} {lat:.7f} px {col0},{row0}")

    # 328 ft windows must not overlap.
    for i, (a, ax, ay) in enumerate(centers):
        for b, bx, by in centers[i + 1 :]:
            if abs(ax - bx) < SIZE and abs(ay - by) < SIZE:
                raise SystemExit(f"overlap {a} {b}")

    with (OUT / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chip_id", "scene_type", "lon", "lat", "half_window_m", "src_tile", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
