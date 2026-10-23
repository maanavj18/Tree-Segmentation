"""Write side-by-side PNGs of the matched 2020 and 2019 chips."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "chips" / "2019" / "previews"
IDS = ["isolated_01", "isolated_03", "street_01", "street_04", "merged_01"]


def show(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
    if rgb.dtype == np.uint8:
        return rgb
    lo, hi = np.percentile(rgb, (2, 98))
    scaled = np.clip((rgb.astype("float32") - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for chip_id in IDS:
        pair = [
            show(ROOT / "data" / "chips" / "2020" / f"{chip_id}.tif"),
            show(ROOT / "data" / "chips" / "2019" / f"{chip_id}.tif"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, image, year in zip(axes, pair, ("2020", "2019")):
            ax.imshow(image)
            ax.set_title(f"{chip_id} {year}")
            ax.axis("off")
        fig.tight_layout()
        dest = OUT / f"{chip_id}.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        print(dest)


if __name__ == "__main__":
    main()