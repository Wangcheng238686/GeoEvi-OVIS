"""Build the NWPU-Refer category subset (S1.3 supplement, implementation-v2).

Registered extraction rule (2026-09-03, supplement protocol — not part of the
SCORE main protocol):

- Universe: NWPU-Refer **test-split** images (3,078).
- Query items: every (image, category) pair with >= 1 instance of that
  category in the image. Images are category-centric (12,322 single-cat +
  87 two-cat), so the pair count ~= image count. The dataset ships 34,466
  "all X" category-style expressions across splits as supporting evidence.
- Query phrase: cleaned English category name. Registered cleanings:
  * drop `airplaneroad` (corrupt merged class, 66 anns) — its images/anns
    are excluded from the subset;
  * `photovolatic` -> query phrase "photovoltaic panel" (spelling fix on the
    prompt side only; GT stays bound to the original category_id);
  * `no target` excluded (0 instances; non-object query device).
- GT: ALL instances of the pair's category in the image (polygon masks from
  new_instances.json, verbatim).
- GT caveat (registered per implementation-v2 S1.3): instance masks may come
  from connected-component decomposition of semantic masks — touching
  objects can be merged into one instance; flagged in p0-report and
  gt-provenance.

Output: $SAM3_OVIS_DATA_ROOT/NWPU-Refer/val/instances_val.json (+ images
under val/images as copies-free relative paths into the source dir).

Run:  $PYTHON sam3_ovis/data/build_nwpurefer_category.py
"""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path("/data/wangcheng/dataset/NWPU-refer")
OUT_ROOT = Path("/data/wangcheng/dataset/SCORE/NWPU-Refer")

DROP_CATEGORIES = {"airplaneroad", "no target"}
QUERY_PHRASE = {"photovolatic": "photovoltaic panel"}


def clean_name(name: str) -> str:
    return QUERY_PHRASE.get(name, name)


def main() -> int:
    d = json.load(open(SRC / "new_instances.json"))
    imgs = {im["id"]: im for im in d["images"]}
    cats = {c["id"]: c["name"] for c in d["categories"]}

    test_imgs = [im for im in d["images"] if im.get("split") == "test"]
    keep_cat_ids = {cid for cid, name in cats.items()
                    if name not in DROP_CATEGORIES}

    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    out_images, out_anns = [], []
    ann_id = 1
    pairs = 0
    dropped_pairs = 0
    multi_poly_anns = 0
    for im in test_imgs:
        img_anns = [a for a in anns_by_img.get(im["id"], [])
                    if a["category_id"] in keep_cat_ids]
        if not img_anns:
            continue
        out_images.append({**im, "file_name": im["file_name"]})
        by_cat = defaultdict(list)
        for a in img_anns:
            by_cat[a["category_id"]].append(a)
        pairs += len(by_cat)
        for a in img_anns:
            seg = a["segmentation"]
            if isinstance(seg, list) and len(seg) > 1:
                multi_poly_anns += 1
            out_anns.append({
                "id": ann_id,
                "image_id": im["id"],
                "category_id": a["category_id"],
                "bbox": a["bbox"],
                "area": a["area"],
                "iscrowd": 0,
                "segmentation": seg,
            })
            ann_id += 1

    used_cat_ids = sorted({a["category_id"] for a in out_anns})
    out_cats = [{"id": cid, "name": clean_name(cats[cid])}
                for cid in used_cat_ids]

    out_dir = OUT_ROOT / "val"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    out = {"images": out_images, "annotations": out_anns,
           "categories": out_cats,
           "info": {"source": "NWPU-Refer new_instances.json (test split)",
                    "subset": "category (see build script docstring)"}}
    (out_dir / "instances_val.json").write_text(json.dumps(out))

    # link (not copy) images to keep disk usage at zero
    n_links = 0
    for im in out_images:
        dst = out_dir / "images" / im["file_name"]
        src = SRC / "image" / "image" / im["file_name"]
        if not dst.exists():
            dst.symlink_to(src.resolve())
        n_links += 1

    print(f"images: {len(out_images)} (linked {n_links})")
    print(f"(image, category) pairs: {pairs}")
    print(f"instances: {len(out_anns)}")
    print(f"categories kept: {len(out_cats)} "
          f"(dropped: {sorted(DROP_CATEGORIES)})")
    print(f"multi-polygon anns (merge-risk indicator): {multi_poly_anns}"
          f" / {len(out_anns)}")
    counts = {}
    for c in out_cats:
        counts[c["name"]] = sum(1 for a in out_anns
                                if a["category_id"] == c["id"])
    print("per-category instances:", json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
