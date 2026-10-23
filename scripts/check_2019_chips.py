"""Check the 2019 NAIP chips against the 2020 manifest."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import rasterio

ROOT = Path(__file__).resolve().parents[1]
CHIPS_2020 = ROOT / "data" / "chips" / "2020" / "manifest.csv"
CHIPS_2019 = ROOT / "data" / "chips" / "2019"
SRC_TILE = "m_4007442_ne_18_060_20190830"


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows_2020 = load_manifest(CHIPS_2020)
    rows_2019 = {row["chip_id"]: row for row in load_manifest(CHIPS_2019 / "manifest.csv")}
    ids_2020 = [row["chip_id"] for row in rows_2020]
    if ids_2020 != list(rows_2019):
        raise SystemExit("2019 manifest chip ids do not match 2020, in order")

    failed = False
    for row in rows_2020:
        chip_id = row["chip_id"]
        rec = rows_2019[chip_id]
        path = CHIPS_2019 / f"{chip_id}.tif"
        reasons = []
        if not path.exists():
            reasons.append("missing file")
        else:
            with rasterio.open(path) as src:
                width_m = src.width * abs(src.res[0])
                height_m = src.height * abs(src.res[1])
                if src.count != 3:
                    reasons.append(f"bands {src.count}")
                if src.crs is None or src.crs.to_epsg() != 26918:
                    reasons.append(f"crs {src.crs}")
                if src.dtypes != ("uint8", "uint8", "uint8"):
                    reasons.append(f"dtype {src.dtypes}")
                if not (99 < width_m < 101 and 99 < height_m < 101):
                    reasons.append(f"size {width_m:.1f} x {height_m:.1f} m")
                size = f"{src.width} x {src.height} px, {width_m:.1f} m"
        if rec["lon"] != row["lon"] or rec["lat"] != row["lat"]:
            reasons.append("center moved")
        if rec["src_tile"] != SRC_TILE:
            reasons.append(f"src_tile {rec['src_tile']}")
        if reasons:
            failed = True
            print(f"{chip_id} BAD: {', '.join(reasons)}")
        else:
            print(f"{chip_id} ok  {size}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()