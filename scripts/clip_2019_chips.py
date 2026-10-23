import csv
from pathlib import Path
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import from_bounds
from pyproj import Transformer

root = Path('.').resolve()
src_path = root / 'COGS' / 'm_4007442_ne_18_060_20190830.tif'
out = root / 'data' / 'chips' / '2019'
out.mkdir(parents=True, exist_ok=True)
to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:26918', always_xy=True)
half = 50.0
tile = 'm_4007442_ne_18_060_20190830'
rows = []
with rasterio.open(src_path) as src, (root / 'data' / 'chips' / '2020' / 'manifest.csv').open(newline='') as f:
    for rec in csv.DictReader(f):
        x, y = to_utm.transform(float(rec['lon']), float(rec['lat']))
        window = from_bounds(x - half, y - half, x + half, y + half, src.transform).round_offsets().round_lengths()
        if window.col_off < 0 or window.row_off < 0 or window.col_off + window.width > src.width or window.row_off + window.height > src.height:
            raise SystemExit(f"{rec['chip_id']} leaves the tile")
        rgb = src.read([1, 2, 3], window=window)
        if rgb.max() == 0:
            raise SystemExit(f"{rec['chip_id']} is empty")
        profile = src.profile.copy()
        profile.update(driver='GTiff', height=rgb.shape[1], width=rgb.shape[2], count=3, transform=src.window_transform(window), compress='deflate', photometric='RGB')
        dest = out / f"{rec['chip_id']}.tif"
        with rasterio.open(dest, 'w', **profile) as dst:
            dst.write(rgb)
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        rows.append({**rec, 'src_tile': tile, 'half_window_m': 50})
        print(rec['chip_id'], rgb.shape[1], rgb.shape[2])
with (out / 'manifest.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['chip_id','scene_type','lon','lat','half_window_m','src_tile','notes'])
    w.writeheader()
    w.writerows(rows)