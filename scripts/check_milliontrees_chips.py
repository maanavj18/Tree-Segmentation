"""Check that each MillionTrees chip is 500 px at 0.20 m, hence 100 m."""

from __future__ import annotations

import csv
from pathlib import Path

import rasterio

ROOT = Path(__file__).resolve().parents[1]
CHIPS = ROOT / "data" / "chips" / "milliontrees"


def main() -> None:
    rows = list(csv.DictReader((CHIPS / "manifest.csv").open(newline="")))
    if len(rows) != 15:
        raise SystemExit(f"manifest has {len(rows)} rows")
    failed = False
    for rec in rows:
        path = CHIPS / f"{rec['chip_id']}.tif"
        with rasterio.open(path) as src:
            ground_x = src.width * abs(src.res[0])
            ground_y = src.height * abs(src.res[1])
            reasons = []
            if (src.width, src.height) != (500, 500):
                reasons.append(f"size {src.width}x{src.height}")
            if abs(src.res[0] - 0.20) > 1e-9 or abs(src.res[1] - 0.20) > 1e-9:
                reasons.append(f"res {src.res}")
            if abs(ground_x - 100) > 1e-6 or abs(ground_y - 100) > 1e-6:
                reasons.append(f"ground {ground_x:.4f}x{ground_y:.4f}")
            if src.count != 3 or src.dtypes != ("uint8", "uint8", "uint8"):
                reasons.append(f"bands {src.count} {src.dtypes}")
            if rec["side_px"] != "500" or rec["gsd_m"] != "0.20" or rec["ground_m"] != "100.00":
                reasons.append("manifest disagrees")
            status = "ok" if not reasons else "BAD " + ", ".join(reasons)
            if reasons:
                failed = True
            print(
                f"{rec['chip_id']:12} {src.width}x{src.height} "
                f"gsd={src.res[0]:.2f} m/px  ground={ground_x:.2f} x {ground_y:.2f} m  {status}"
            )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
