# Tree crown segmentation — Pennington viability trial

This file is the working brief for agents in this repo. Follow the current phase before anything later in the document.

The end goal, from the lab plan with Anirudh, is to replace manual tree surveys in New Jersey suburbs with drone imagery plus a model that can **count and classify** trees. Count and species are later deliverables. The assigned work right now is a small **instance crown** viability test on public imagery of Pennington Borough. Literature review is done. No model has been trained on New Jersey.

## Current phase — do this, in this order

Leave annotation, fine-tuning, and a borough mosaic until the call in step 5 is made. The other two arms have to match the same blocks, so 2020 chips come first.

The four 2020 COGs are already in `COGS/`: `F10A16.tif`, `F10B13.tif`, `F10C4.tif`, `F10D1.tif` (about 250 MB each). They meet on Pennington Borough. Overlap previews are in `NJ_Images_Progress/`. Do not download the `2020/cog/` prefix again, and do not run a model on the full tiles.

### 1. Clip 12–15 windows from the 2020 1-ft COGs

Source, if a tile is ever missing: NJGIN Imagery Warehouse, `s3://njogis-imagery/2020/cog/`, public read, region `us-west-2`, `--no-sign-request`. Copy one named key only. The prefix is about 9,200 files and 2.2 TB, named by grid ID (`F10A16.tif`), not by town. Lookup is the 2007 orthophoto grid or the Imagery Warehouse app, not `aws s3 ls`.

Each COG is a 5,000 ft × 5,000 ft, 1-ft, **leaf-off** tile (EPSG:6527, NAD83(2011) New Jersey State Plane, US survey feet). Usually 4-band (RGB + NIR). The web basemap is RGB only and slightly softer than the COG.

Clip **12–15 windows of about 100 m on the ground**, four or five of each:

1. Isolated yard or street trees — crowns standing apart over lawn, a wide setback, or pavement.
2. Trees against houses, cars, and pavement — roofs, driveways, parked cars, sidewalks.
3. Touching backyard or park canopy — merged rear lots, school grounds, small parks.

Stay in the borough residential core (the inward corners of the four-tile block). Farmland, highway, and closed woodlot answer a different question. Spread chips across blocks. Prefer a window that sits inside one GeoTIFF.

Pick centers on the 2020 imagery (you can see crowns). Record WGS84 lon/lat (EPSG:4326), not pixel row/column and not the tile centroid. A point layer exported as CSV with `AS_XY` is enough; QGIS is optional. Browser viewing of the 2020 natural-color service is enough to choose scenes: [2020 Map Viewer](https://www.arcgis.com/apps/mapviewer/index.html?layers=a047d5d9924748019ede18ae87639996). Clip from the local COGs.

Fixed window: **100 m on a side** (the plan allows 80–150 m; do not vary it). 100 m ≈ 328.08 US survey feet, so about **328 × 328 px** at 1 ft. Reproject the lon/lat center into EPSG:6527, then clip center ± 50 m. Write RGB (bands 1–3) GeoTIFFs for the visual trial. Keep the georeferencing.

Write this before touching 2019:

```text
data/chips/2020/
  isolated_01.tif … isolated_05.tif
  street_01.tif   … street_05.tif
  merged_01.tif   … merged_05.tif
  manifest.csv
```

`manifest.csv` columns: `chip_id`, `scene_type` (`isolated` / `street` / `merged`), `lon`, `lat`, `half_window_m` (50), `src_tile`, `notes`. Check each chip still matches its scene label.

### 2. Clip 8–12 geographically matched 2019 1 m NAIP chips

Same lon/lat centers and the same 100 m ground window, from 2019 NAIP (Mercer / Pennington). That arm is only a **boundary check**: does leaf-on rescue individual crowns when GSD is ~1 m? Expect no. A 10 m tree is ~80 pixels; a 5 m ornamental is ~20.

If the first five are canopy blobs, stop. You do not need all 12–15. The ones you keep must be a subset of the 2020 manifest and should still cover the three scene types if possible. At 1 m a 100 m chip is ~100 × 100 px. Do not match 400 px across sensors: at 1 m, 400 px is 400 m and is a different experiment.

NJGIN leaf-on statewide product is NAIP at 1 m (2019, 2017, 2013, 2010, 2006). There is no recent NJGIN product that is both leaf-on and 1 ft. Newest downloadable statewide 1-ft set is still 2020, and it is leaf-off. 1977 Tidelands / 1970 Wetlands are 1 ft and leaf-on but black-and-white and unusable here. DVRPC 2025 Nearmap over Mercer is 1 ft and late winter / early spring (leaf-off again) and is not the warehouse product.

Imagery page: https://njgin.nj.gov/njgin/edata/imagery/ — rows `2020 - 1 ft leaf off` and `2019 - 1 m leaf on`.

### 3. Load MillionTrees residential chips

```python
# TreePolygons, small=True — not mini=True (~3 images per source)
# Sources: Schütte (Schuette) and UrbanLondon only
```

Hand-pick **12–15 residential chips** (yards, not closed forest), same three scene types, also ~100 m on the ground.

- **Schütte** — 20 cm urban true-ortho polygons; low/medium density; yards vs park. Closest task match.
- **UrbanLondon** — 0.25 m urban polygons. Closest GSD to 1 ft (~30 cm).

These are not Pennington. They answer: does the method work on Pennington-like yards when GSD and phenology are both in band? Match scene type, not GPS.

Skip BAMFORESTS, NEON, OAM-TCD as the segmentation control, and closed-forest UAV. OAM-TCD is a finer-GSD box check, not this control. If the MillionTrees `small` archive is thin, Schütte Zenodo tiles (ITCD Urban) are the backup.

### 4. Zero-shot one polygon model on all three arms

One of **Detectree2** (Mask R-CNN) or **CanopyRS** (DINO Swin-L + SAM). Same checkpoint and the same ~100 m ground window on every chip. DeepForest boxes are an optional overlay, the baseline to beat, not the segmentation method.

CanopyRS DINO Swin-L is the stronger published starting point (zero-shot F1 0.831 on held-out MillionTrees sources vs fine-tuned DeepForest 0.697). Detectree2 is the lower-friction polygon baseline and the model Gan actually swept through 30 cm. Pick one for this trial. Do not run a bake-off yet.

Do not use zero-shot SAM alone. Do not train from scratch. Do not fine-tune in this phase. Save overlays and GeoJSON in the image CRS.

### 5. Score by eye and make the call

Per chip: isolated trees usable, small trees visible, merged crowns, building false positives. Labels exist only on MillionTrees. Report F1 there only. Do not compute an F1 on New Jersey.

| 2020 1 ft winter | 2019 1 m summer | MillionTrees 20–25 cm | Pursue |
|---|---|---|---|
| Isolated trees OK | Fail | Pass | Stay on 2020 1 ft for a street-tree trial. Do not expect park delineation. Leaf-off was not fatal. |
| Fail | Fail | Pass | Method is fine. Do not pursue NJGIN 1 m. Next real NJ data must be leaf-on at ~20–30 cm (UAV or a later flight). Use MillionTrees as the sandbox until then. |
| Fail | Fail | Fail | Pipeline, checkpoint, or window. Fix inference before changing data. |
| Fail | Isolated trees OK | Pass | Unexpected. Phenology would be dominating GSD. Still a weak instance product; canopy-cover only unless outlines are actually tight. |
| Isolated OK | Isolated OK | Pass | 2020 is viable. 2019 is viable only if the masks are real crowns, which is unlikely. |

Do not pursue 2019 1 m for crown segmentation unless this trial shows tight instances on medium and large trees.

## What the three arms are testing

NJGIN never gives leaf-on at 1 ft. The arms are three cells of phenology × GSD, not three equal Pennington substitutes.

- **2020 1 ft leaf-off** — the assigned NJ product. Can instance crowns work on winter 1-ft Pennington?
- **2019 1 m leaf-on** — NJGIN’s actual leaf-on product. Does leaf-on rescue you at 1 m?
- **MillionTrees ~20–25 cm leaf-on urban** — the missing cell. Does the method work when GSD and phenology are both reasonable?

2020 vs 2019 alone is confounded (winter + fine vs summer + coarse). If both fail and MillionTrees was skipped, you cannot tell “1 ft is too coarse,” “leaf-off kills it,” or “the model was run wrong.”

On a data card for the 2020 tiles, record: leaf-off, flydate ([Orthophoto Flydates 2020](https://njogis-newjersey.opendata.arcgis.com/datasets/newjersey::orthophoto-flydates-2020-for-nj-3424/about)), native GSD (~0.3048 m), band count, CRS, true-ortho vs building lean, licence. Confirm those on the GeoTIFF, not the web cache. Bare branches mean the test is “can a leaf-on model see winter trees at 1 ft?”

## Full scope — only after step 5

### If 2020 isolated trees pass

Street-tree inventory on 2020 is the product to pursue. Park and merged backyard delineation is a separate, weaker target (Schütte park F1 ~69% and mIoU ~48% even at 20 cm leaf-on).

Then, in order:

1. Hand-annotate about **100–150 crowns on the NJ chips, from scratch.** Correcting a model’s own boxes was worse than drawing new labels (Burmeister). Do not bootstrap NJ labels from DeepForest or Detectree2.
2. **Spatially blocked split.** Hold out whole neighborhoods. Discard overlapping train tiles (Ball et al.). A random tile split leaks crowns.
3. Fine-tune the winner of the zero-shot pass. CanopyRS first if it was the polygon model; Mask R-CNN / Detectree2 if you need polygons and CanopyRS did not install. A few epochs, on the order of 5–10, is the published fine-tune scale — confirm against the paper you actually train from.
4. Report **standard COCO, AP50, and AP40**, plus mask F1 and mean IoU. Stratify by residential density (yard vs merged/park). Do not mix MillionTrees F1 with Gan or Schütte F1, and do not cite MillionTrees out-of-distribution scores as an estimate of NJ transfer.
5. A borough mosaic comes after that evaluation, not before.

Cheapest extra ground truth, if it exists: a municipal street-tree inventory for the same blocks. Record whether one covers Pennington before inventing labels.

### If 2020 fails and MillionTrees passes

The method is fine and the NJ product is the limiter. Do not switch the project to 2019 1 m. Use Schütte / UrbanLondon as a sandbox (fine-tune and metrics where labels exist). The next New Jersey dataset has to be **leaf-on at about 20–30 cm** (UAV or a later flight). 1 m stays canopy cover: tree vs not-tree, percent cover, large patches.

### If every arm fails

Fix the checkpoint, install, and ground-window before collecting new imagery or labels.

### After a viable instance model exists

The original survey goal is two outputs, kept separate:

- **Count** — instance crowns, the thing this trial is testing.
- **Classify** — species or type. Do not fold species into the viability call. No public RGB UAV instance dataset of temperate deciduous residential crowns was found in the review (Gan’s site, Stark’s Munich zone, and Burmeister’s DJI plots are the closest and are not public). NJ labels would be a real contribution, which is why they wait until the imagery is worth labeling.

Flight band, when a drone mission is actually planned: about **2–10 cm** is the comfortable range for forest-pretrained Detectree2 / DeepForest on closed temperate canopy (Gan: no meaningful gain from 0.7–10 cm; drop from 9 cm through 50 cm). That is one closed-canopy site, not a cliff. Schütte still got residential F1 85–91% at **20 cm** leaf-on true ortho. Flying finer than about 5 cm has not paid off after fine-tuning (Burmeister). Match **ground footprint** (~40 m for DeepForest’s released 10 cm weights, or the ~100 m window used in this trial), not a fixed pixel crop. DeepForest’s docs say widen the patch rather than resample; Burmeister downsampled to 10 cm. Both target pixels per crown.

RGB is the pipeline. LiDAR, CHM, and hyperspectral methods can be cited only with that extra requirement labeled. They are not the current method.

## Facts that constrain recommendations

- **1 foot (~30 cm) on isolated, medium-to-large residential crowns** is in Schütte’s regime, not Gan’s 2–10 cm forest-UAV regime. A ~10 m crown is ~840 px at 1 ft (just inside Hao’s “works well” band of ~800–12,800 px). A ~5 m ornamental is ~210 px: detectable, blocky. Predictions under ~4 m² are noise at this GSD (Schütte’s cutoff).
- **Closed or park canopy at 20–50 cm** is Gan’s worst bin and Schütte’s park class. Expect under-segmentation (several trees → one blob). OAM-TCD: instance recall ~0.36 for individuals vs ~0.73 if groups count.
- **1 m** is the wrong task for instance crowns. Canopy cover can still be valid.
- **Leaf-off is a bigger risk than 30 cm vs 20 cm.** Pretrained crown models are leaf-on. Schütte also used true orthophotos (no building lean) and RGN / false color. NJGIN 2020 natural color is weaker than that stack even before winter.
- **DeepForest release weights have not seen a suburb.** No NEON benchmark site is urban or residential. A Portland street-tree result in the 2020 paper was a separate fine-tune and is not in the shipped weights. Plain box precision is far below the mask-aware number the leaderboard quotes (~0.20 vs ~0.82). The “DeepForest Mask R-CNN” AP40 0.120 row is a torchvision Mask R-CNN with COCO init, not a DeepForest release. NEON sits inside MillionTrees, so DeepForest’s “out of distribution” score is not a clean OOD number.
- Off-the-shelf DeepForest on deciduous UAV is on the order of F1 ~0.22. Transfer-trained Detectree2 on Gan’s site held about 0.55–0.58 only through 8 cm. Expect roughly 0.5–0.6 on hard deciduous canopy, not 0.9.
- Going from 20 cm to 30 cm costs about 2.25× pixels. Going from DeepForest’s 10 cm training GSD to 30 cm costs about 9×. That is why a 10 cm box model is the wrong starting point for this ortho.

## Working rules

- One polygon checkpoint for the trial. Optional DeepForest boxes on the same chips.
- Match meters on the ground (~100 m). Do not match pixel size across 1 ft, 1 m, and 20 cm.
- 12–15 chips per serious arm. Thirty only if the first twelve are mixed and you cannot call it. 2019 can stop at five blobs.
- Hand-pick scene types. Random crops over-sample fields.
- Georeferenced chips plus `manifest.csv`. Screenshots alone cannot drive the 2019 clip.
- No New Jersey F1, AP, or mIoU until hand labels exist.
- No full-tile inference, no borough mosaic, no 100-crown annotation, and no fine-tune until step 5.
- Do not recurse `s3://njogis-imagery/`.
- Cite a paper for any new performance number. Keep MillionTrees F1, Gan F1, and Schütte F1 in separate columns.
- Papers and the slide deck live outside this repo (Downloads: literature review, papers tracker, `Tree Detection Literature Review` / `Tree Crown RGB Drone Review` decks, `_deck_content.py`). Do not treat those paths as project source. The decision record that belongs in-repo is this file and `.cursor/plans/three-arm_viability_trial_9be084b0.plan.md`.
