"""Category co-occurrence priors from training sets (S3.2 §5).

Builds per-source-dataset tables P(j|i) = (#images containing both i and j)
/ (#images containing i), Laplace-smoothed, over the train COCO jsons of
iSAID and SIOR (the SCORE training sets). Vocabulary is per-dataset; at edge
time the builder matches by exact class name and falls back to
`prior_floor` with an explicit `prior=unmatched` attribute (registered
behaviour, soft scores only).

Run:  $PYTHON sam3_ovis/graph/priors.py
Out:  $SAM3_OVIS_DATA_ROOT/priors/cooccurrence_{iSAID,SIOR}.json
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

MAINLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE))

from graph.config import THRESHOLDS  # noqa: E402


def build_table(coco_dataset: dict, alpha: float) -> dict:
    """coco_dataset: dict with images/annotations/categories."""
    name_by_cat = {c["id"]: c["name"] for c in coco_dataset["categories"]}
    cats_by_img = defaultdict(set)
    for a in coco_dataset["annotations"]:
        cats_by_img[a["image_id"]].add(a["category_id"])

    counts_i = defaultdict(int)          # images containing i
    counts_ij = defaultdict(lambda: defaultdict(int))  # images with i and j
    for img_id, cat_set in cats_by_img.items():
        for i in cat_set:
            counts_i[i] += 1
            for j in cat_set:
                if j != i:
                    counts_ij[i][j] += 1

    table = {}
    for i in counts_i:
        denom = counts_i[i] + alpha * len(name_by_cat)
        row = {}
        for j in counts_ij[i]:
            p = (counts_ij[i][j] + alpha) / denom
            row[name_by_cat[j]] = round(p, 6)
        table[name_by_cat[i]] = {
            "n_images": counts_i[i],
            "cond": row,
        }
    return table


def main() -> int:
    from data.score_datasets import DATASETS

    root = Path(os.environ.get("SAM3_OVIS_DATA_ROOT", ""))
    out_dir = root / "priors"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("iSAID", "SIOR"):
        spec = DATASETS[name]
        data = json.load(open(root / spec.val_json))
        table = build_table(data, THRESHOLDS.prior_alpha)
        out = out_dir / f"cooccurrence_{name}.json"
        out.write_text(json.dumps(table, indent=1, ensure_ascii=False))
        n_rows = len(table)
        print(f"{name}: {n_rows} classes, {sum(r['n_images'] for r in table.values())} "
              f"image-class rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
