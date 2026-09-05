"""Step-2 evidence-signal probe (v3-literature-lessons.md): can a CLIP
actually discriminate candidate crops by class — and does RemoteCLIP beat
general CLIP (SCORE's Table-6 split says it may not)?

Protocol (fixed, registered before results, 2026-09-05):
- Crops: GT boxes + 10% margin (fixed candidates, same crops for both
  encoders), one crop per GT instance.
- Text: plain class name (the registered query form), plus the templated
  "a photo of a {name}" variant reported separately.
- Vocabulary: the dataset's own class list; a crop competes against all
  classes (no top-K pre-filtering, mirroring design-v3.0 §3).
- Metrics per encoder: top-1 accuracy, mean rank of the correct class,
  per-class accuracy, and the top confuser that beats the truth.

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/eval/clip_discrimination_probe.py --dataset NWPU \
          --per-class 100
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

from data.score_datasets import DATASETS  # noqa: E402


def crops_for_dataset(root, spec, cat_names, per_class, margin=0.1):
    """GT-box crops: returns [(img_id, cat_name, PIL crop), ...]."""
    gt = COCO(str(root / spec.val_json))
    out = []
    per_counter = defaultdict(int)
    for a in gt.dataset["annotations"]:
        cid = a["category_id"]
        name = cat_names[cid]
        if per_counter[name] >= per_class:
            continue
        im = gt.imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        mx, my = w * margin, h * margin
        x1 = min(max(0, int(x - mx)), im["width"] - 1)
        y1 = min(max(0, int(y - my)), im["height"] - 1)
        x2 = min(im["width"], max(x1 + 1, int(x + w + mx)))
        y2 = min(im["height"], max(y1 + 1, int(y + h + my)))
        img = Image.open(root / spec.images_dir / im["file_name"]) \
            .convert("RGB").crop((x1, y1, x2, y2))
        out.append((a["image_id"], name, img))
        per_counter[name] += 1
    return out


class Encoder:
    def __init__(self, name):
        import open_clip

        self.name = name
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if name == "RemoteCLIP":
            ckpt = "/data/wangcheng/pretrained-models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt"
        else:
            ckpt = "openai"  # general CLIP ViT-L-14
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=ckpt)
        self.tok = open_clip.get_tokenizer("ViT-L-14")
        self.model = self.model.eval().to(self.device)

    @torch.no_grad()
    def image_feats(self, crops, bs=64):
        feats = []
        for i in range(0, len(crops), bs):
            t = torch.stack([self.pre(c) for c in crops[i:i + bs]]) \
                .to(self.device)
            f = self.model.encode_image(t)
            feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
        return torch.cat(feats)

    @torch.no_grad()
    def text_feats(self, phrases):
        t = self.tok(phrases).to(self.device)
        f = self.model.encode_text(t)
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu()


def evaluate(feats_img, feats_txt, crop_names, names_list):
    """top-1 acc / mean rank over the full vocabulary."""
    gt_idx = torch.tensor([names_list.index(n) for n in crop_names])
    sim = feats_img @ feats_txt.T  # [N, K]
    top1 = sim.argmax(dim=1)
    acc = float((top1 == gt_idx).float().mean())
    order = sim.argsort(dim=1, descending=True)
    ranks = torch.tensor(
        [float((order[i] == gt_idx[i]).nonzero()[0, 0]) + 1
         for i in range(len(crop_names))])
    return {"accuracy": round(acc, 4),
            "mean_rank": round(float(ranks.mean()), 2),
            "ranks": ranks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NWPU")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    gt = COCO(str(root / spec.val_json))
    cat_names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    names_list = sorted(set(cat_names.values()))
    print(f"dataset {args.dataset}: vocabulary {len(names_list)} classes")

    items = crops_for_dataset(root, spec, cat_names, args.per_class)
    print(f"crops: {len(items)} (<= {args.per_class}/class)")
    crops = [it[2] for it in items]
    crop_names = [it[1] for it in items]

    results = {}
    for enc_name in ["RemoteCLIP", "general"]:
        enc = Encoder(enc_name)
        for mode, mk in [("plain", lambda n: n),
                         ("templated", lambda n: f"a photo of a {n}")]:
            phrases = [mk(n) for n in names_list]
            fi = enc.image_feats(crops)
            ft = enc.text_feats(phrases)
            sim = fi @ ft.T
            gt_idx = torch.tensor([names_list.index(n) for n in crop_names])
            top1 = sim.argmax(dim=1)
            acc = float((top1 == gt_idx).float().mean())
            order = sim.argsort(dim=1, descending=True)
            ranks = torch.tensor(
                [float((order[i] == gt_idx[i]).nonzero()[0, 0]) + 1
                 for i in range(len(crop_names))])
            key = f"{enc_name}_{mode}"
            results[key] = {"top1_acc": round(acc, 4),
                            "mean_rank": round(float(ranks.mean()), 2)}
            # per-class accuracy
            per_class_acc = {}
            for n in names_list:
                idxs = [i for i, n2 in enumerate(crop_names) if n2 == n]
                if idxs:
                    per_class_acc[n] = round(
                        float((top1[idxs] == gt_idx[idxs]).float().mean()), 3)
            results[key]["per_class"] = per_class_acc
            print(f"{key:18s} top1={acc:.4f} mean_rank="
                  f"{float(ranks.mean()):.2f}")
            del fi, ft, sim
            torch.cuda.empty_cache()

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n_crops": len(items), "vocabulary": names_list,
             "results": results}, indent=1, ensure_ascii=False))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
