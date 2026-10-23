---
name: Three-arm viability trial
overview: A small three-arm chip test is a good idea if 2019 1 m is treated as a boundary check, not a peer of 1-ft leaf-off. MillionTrees fills the missing leaf-on ~20–25 cm cell that NJGIN does not have. About 12–15 matched chips per arm is enough to pick which option to pursue; 30 is optional extra, not required.
todos:
  - id: chips-2020
    content: Pull 1-2 Pennington 2020 1-ft COGs and clip 12-15 stratified chips (isolated / street / merged).
    status: pending
  - id: chips-2019
    content: Clip 8-12 geographically matched 2019 1 m NAIP chips of the same blocks as a boundary check.
    status: pending
  - id: chips-mt
    content: Load MillionTrees TreePolygons small=True for Schuette + UrbanLondon; pick 12-15 residential chips.
    status: pending
  - id: infer
    content: Zero-shot one polygon model on all chips with a ~100 m ground window; DeepForest boxes optional.
    status: pending
  - id: call
    content: Score isolated / small / merged / building FPs; use the 2x3 table to pick 2020 vs MT-style leaf-on, not 1 m NAIP.
    status: pending
isProject: false
---

# Three-arm Pennington viability trial

The three datasets answer **different questions**. That is useful. Do not score them as if they were three equally fair “Pennington substitutes.”

## What this design actually tests

NJGIN never gives you leaf-on at 1 ft. The three arms fill three of four cells:

- **2020 1 ft leaf-off** — the assigned NJ product. Question: can instance crowns work on winter 1-ft Pennington?
- **2019 1 m leaf-on** — NJGIN’s actual leaf-on product. Question: does leaf-on rescue you if GSD is 1 m? Expect **no** for individual crowns (a 10 m tree is ~80 pixels; a 5 m ornamental is ~20). Still worth a few chips so you can document the dead end instead of guessing.
- **MillionTrees ~20–25 cm leaf-on urban** — the missing cell. Question: does the *method* work on Pennington-like yards when GSD and phenology are both in-band?

If you skip MillionTrees, 2020 vs 2019 is confounded (winter+fine vs summer+coarse). If 2020 fails and 2019 fails, you cannot tell “1 ft is too coarse,” “leaf-off kills it,” or “we ran the model wrong.”

```mermaid
flowchart LR
  subgraph cells [Phenology vs GSD]
    A["1 ft leaf-off: NJGIN 2020"]
    B["1 m leaf-on: NAIP 2019"]
    C["20-25 cm leaf-on: Schuette / UrbanLondon"]
  end
  A --> call[Which option to pursue]
  B --> call
  C --> call
```

**Verdict:** viable and a good trial **if** arm 2 is a negative/boundary check and arm 3 is the positive control. It is a bad idea if you treat 1 m NAIP as a serious crown-segmentation candidate equal to 1 ft.

## Chip count: 12–15 per arm, not 30

For a go / no-go visual call, **12–15 chips per type is enough**. Thirty is safe but past the point of diminishing returns unless the first 12 look mixed.

Use the same three scene types in every arm (4–5 chips each):

1. Isolated large yard / street trees
2. Trees against houses, cars, pavement
3. Touching backyard or park canopy

That is **36–45 chips total**, one model, one afternoon of scoring after inference.

- **2020:** 12–15 windows (~80–150 m) from **one or two** Pennington COGs. Not 30 full tiles.
- **2019:** 8–12 chips is enough if the first 5 already show blobs instead of crowns. Keep them geographically matched to the 2020 chips if possible (same blocks at 1 m).
- **MillionTrees:** 12–15 **residential** chips, not a random mix. Prefer **Schütte** low/medium-density (20 cm, polygons, yards vs park) plus a few **UrbanLondon** (0.25 m, closest GSD to 1 ft). Skip BAMFORESTS / NEON / closed-forest UAV.

MillionTrees `mini=True` is only ~3 images per source — too few. Use `small=True` or the Schütte Zenodo tiles and hand-pick 12–15 residential chips.

## How to call the result

Score each chip by eye: isolated trees usable / small trees visible / merged under-segmented / building FPs. Labels exist only on MillionTrees; report F1 there only. Do not invent an NJ F1.

| 2020 1 ft winter | 2019 1 m summer | MT 20–25 cm | Pursue |
|---|---|---|---|
| Isolated trees OK | Fail | Pass | **Stay on 2020 1 ft** for a street-tree trial; do not expect park delineation. Leaf-off was not fatal. |
| Fail | Fail | Pass | Method is fine. **Do not pursue NJGIN 1 m.** Next real NJ data must be leaf-on at ~20–30 cm (UAV or a later flight), and use MT as the sandbox until then. |
| Fail | Fail | Fail | Pipeline/checkpoint, not the imagery. Fix inference before changing data. |
| Fail | Isolated trees OK | Pass | Unexpected. 1 m leaf-on beating 1 ft winter would mean **phenology dominates GSD**. Still a weak instance product; treat as canopy-cover only unless outlines are actually tight. |
| Isolated OK | Isolated OK | Pass | 2020 is viable; 2019 is only viable if masks are real crowns, which is unlikely. |

**Do not pursue 2019 1 m for crown segmentation** unless this trial clearly shows tight instances on medium/large trees. The literature says it will fail; the chips are there to confirm.

## Keep the trial small

- One polygon model (Detectree2 or CanopyRS DINO+SAM), plus DeepForest boxes only as a baseline overlay.
- Same checkpoint on all three arms.
- Match **ground window** (~100 m), not 400 px. At 1 m, 400 px is 400 m and is a different experiment.
- No fine-tuning, no 100-crown NJ annotation, no full-borough mosaic.

## Download (still tiny)

- 2020: 1–2 Pennington COGs from the [Imagery Warehouse](https://njgin.nj.gov/njgin/edata/imagery/) / `s3://njogis-imagery/2020/cog/`.
- 2019: Mercer NAIP via NJGIN’s NAIP stream or USDA county mosaic; clip the same Pennington extent.
- MillionTrees: `TreePolygons` with Schütte + UrbanLondon, `small=True`.

## Suggestion in one line

Run the three arms. Put most of the 12–15 chips on **2020** and **Schütte-like MT**. Use **2019 1 m** as a short boundary check. The option you pursue is whichever of 2020 vs MT-style leaf-on high-res actually produces instance crowns; 1 m is almost certainly not that option.
