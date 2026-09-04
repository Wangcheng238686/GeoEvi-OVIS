"""Category co-occurrence priors from training sets (S3.2 §5).

Builds per-source-dataset tables P(j|i) = (#images containing both i and j)
/ (#images containing i), Laplace-smoothed with alpha, over the train COCO
jsons of iSAID and SIOR (the SCORE training sets).

Registered semantics (docs/graph-thresholds.md §5): the smoothing covers the
FULL query vocabulary — `cond` holds every j != i of the dataset vocabulary
(never-co-occurring pairs get alpha / (n_i + alpha*(K-1)), NOT the floor);
the `prior_floor` in edges.py is only a lower clamp. True "unmatched" means
the evidence class name is absent from the dataset vocabulary, decided via
the explicit `vocabulary` list written alongside the table.

Run:  $PYTHON sam3_ovis/graph/priors.py
Out:  $SAM3_OVIS_DATA_ROOT/priors/cooccurrence_{iSAID,SIOR}.json
      {"vocabulary": [names...], "classes": {name: {n_images, cond}}}
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
    """Full-vocabulary Laplace-smoothed co-occurrence table."""
    name_by_cat = {c["id"]: c["name"] for c in coco_dataset["categories"]}
    K = len(name_by_cat)
    cats_by_img = defaultdict(set)
    for a in coco_dataset["annotations"]:
        cats_by_img[a["image_id"]].add(a["category_id"])

    counts_i = defaultdict(int)                        # images containing i
    counts_ij = defaultdict(lambda: defaultdict(int))  # images with i and j
    for img_id, cat_set in cats_by_img.items():
        for i in cat_set:
            counts_i[i] += 1
            for j in cat_set:
                if j != i:
                    counts_ij[i][j] += 1

    classes = {}
    for cid, i_name in name_by_cat.items():
        if cid not in counts_i:
            continue  # no images contain this class -> no P(.|i) row
        # proper distribution over the K-1 possible j != i
        denom = counts_i[cid] + alpha * (K - 1)
        row = {}
        for jcid, j_name in name_by_cat.items():
            if jcid == cid:
                continue
            row[j_name] = round((counts_ij[cid].get(jcid, 0) + alpha) / denom, 6)
        classes[i_name] = {"n_images": counts_i[cid], "cond": row}
    return {"vocabulary": sorted(name_by_cat.values()), "classes": classes}


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
        n_rows = len(table["classes"])
        n_vocab = len(table["vocabulary"])
        n_cells = sum(len(r["cond"]) for r in table["classes"].values())
        print(f"{name}: {n_rows} class rows, vocabulary {n_vocab}, "
              f"{n_cells} full-vocab cond cells -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
