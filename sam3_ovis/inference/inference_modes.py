"""S2.3 large-image inference-mode comparison (E2): whole / tile / window.

Registered protocol (S2.3): same 50-image subset per test set, three modes:
  whole  - processor resize to 1008 (the P0 baseline mode);
  tile   - non-overlapping 1024 grid (edge tiles padded by crop), candidates
           mapped back to global frame by offset, no dedup across tiles;
  window - 1024 window / 768 stride (25% overlap), same mapping, then
           per-class score-ordered greedy mask-IoU NMS at 0.5 across windows
           (dedup registered here; only affects window mode).

Per (dataset, mode): per-image latency, peak GPU memory, and subset mAP +
recall-proxy from the S1.4 diagnostics.

Run AFTER the S2.2 full runs (GPU contention):
  source sam3_ovis/scripts/load_environment.sh
  "$PYTHON" sam3_ovis/inference/inference_modes.py --dataset NWPU --subset 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

MAINLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

THRESHOLD = 0.05      # same registered pool as S2.2
WINDOW = 1024
STRIDE = 768
NMS_IOU = 0.5


def to_rle(m: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


class Runner:
    def __init__(self):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch = torch
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        model = build_sam3_image_model(
            checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False)
        self.processor = Sam3Processor(model, confidence_threshold=THRESHOLD)

    def run_image(self, image, cat_names):
        """image: PIL. Returns list of (cat_name, score, global RLE) — whole
        mode only; tile/window handled by caller via run_crop."""
        state = self.processor.set_image(image)
        out = []
        for name in cat_names:
            state = self.processor.set_text_prompt(state=state, prompt=name)
            scores = state["scores"]
            n = int(scores.shape[0]) if scores is not None else 0
            for j in range(n):
                m = state["masks"][j, 0].detach().cpu().numpy().astype("uint8")
                out.append((name, float(scores[j]), to_rle(m)))
        return out

    def run_crop(self, crop_pil, cat_names, ox, oy):
        """Run one tile/window; paste masks into a crop-local bool canvas and
        return (cat, score, local canvas). Caller places into global frame."""
        import numpy as np

        w, h = crop_pil.size
        state = self.processor.set_image(crop_pil)
        out = []
        for name in cat_names:
            state = self.processor.set_text_prompt(state=state, prompt=name)
            scores = state["scores"]
            n = int(scores.shape[0]) if scores is not None else 0
            if n == 0:
                continue
            masks = state["masks"]
            for j in range(n):
                m = masks[j, 0].detach().cpu().numpy()
                canvas = np.zeros((h, w), dtype=bool)
                mm = m[:h, :w]  # processor resizes masks to (crop_h, crop_w)
                canvas[: mm.shape[0], : mm.shape[1]] = mm.astype(bool)
                out.append((name, float(scores[j]), canvas, ox, oy))
        return out


def grid_windows(w, h, size, stride):
    xs = list(range(0, max(w - size, 0) + 1, stride))
    ys = list(range(0, max(h - size, 0) + 1, stride))
    if xs and xs[-1] + size < w:
        xs.append(w - size)
    if ys and ys[-1] + size < h:
        ys.append(h - size)
    if not xs:
        xs = [0]
    if not ys:
        ys = [0]
    return [(x, y) for y in ys for x in xs]


def cross_window_nms(cands):
    """cands: (cat, score, rle). Per class, score-ordered greedy mask-IoU NMS."""
    from pycocotools import mask as mask_util

    keep = []
    by_cat = {}
    for c in cands:
        by_cat.setdefault(c[0], []).append(c)
    for cat, items in by_cat.items():
        items.sort(key=lambda x: -x[1])
        kept_rles = []
        for cat_, score, rle in items:
            r = dict(rle)
            r["counts"] = r["counts"].encode("utf-8")
            suppressed = False
            for kr in kept_rles:
                iou = float(mask_util.iou([r], [kr], [0])[0][0])
                if iou >= NMS_IOU:
                    suppressed = True
                    break
            if not suppressed:
                kept_rles.append(r)
                keep.append((cat, score, rle))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset", type=int, default=50)
    ap.add_argument("--modes", nargs="+", default=["whole", "tile", "window"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    from PIL import Image
    from pycocotools.coco import COCO

    from data.score_datasets import DATASETS

    spec = DATASETS[args.dataset]
    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    gt = COCO(str(root / spec.val_json))
    cat_ids = gt.getCatIds()
    cat_names = [c["name"] for c in gt.loadCats(cat_ids)]
    id2name = {c["id"]: c["name"] for c in gt.loadCats(cat_ids)}
    img_ids = gt.getImgIds()[: args.subset]

    out_dir = Path(args.out_dir or (
        Path(os.environ["SAM3_OVIS_LOG_ROOT"]) / "inference_modes"))
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = Runner()
    torch = runner.torch

    for mode in args.modes:
        preds_path = out_dir / f"{args.dataset}_{mode}_preds.jsonl"
        mem_peak = 0
        torch.cuda.reset_peak_memory_stats()
        t_all = time.time()
        with open(preds_path, "w") as fout:
            for k, img_id in enumerate(img_ids):
                info = gt.loadImgs(img_id)[0]
                image = Image.open(
                    root / spec.images_dir / info["file_name"]).convert("RGB")
                w, h = image.size
                t0 = time.time()
                cands = []
                if mode == "whole":
                    cands = runner.run_image(image, cat_names)
                elif mode in ("tile", "window"):
                    stride = WINDOW if mode == "tile" else STRIDE
                    boxes = grid_windows(w, h, WINDOW, stride)
                    cands_local = []
                    for (x, y) in boxes:
                        crop = image.crop((x, y, min(x + WINDOW, w),
                                           min(y + WINDOW, h)))
                        cands_local.extend(runner.run_crop(crop, cat_names, x, y))
                    cands = []
                    for name, score, cmask, ox, oy in cands_local:
                        g = np.zeros((h, w), dtype=bool)
                        g[oy:oy + cmask.shape[0], ox:ox + cmask.shape[1]] = cmask
                        cands.append((name, score, g))
                    cands = [(n, s, to_rle(g)) for n, s, g in cands]
                    if mode == "window":
                        cands = cross_window_nms(cands)
                dt = time.time() - t0
                mem_peak = max(mem_peak, torch.cuda.max_memory_allocated())
                for name, score, rle in cands:
                    cid = next(c for c, n in
                               ((c, id2name[c]) for c in cat_ids)
                               if n == name)
                    fout.write(json.dumps({
                        "image_id": img_id, "category_id": int(cid),
                        "category_name": name, "score": round(score, 6),
                        "segmentation": rle}) + "\n")
                print(f"  [{mode}] {k + 1}/{len(img_ids)} {dt:.1f}s "
                      f"{len(cands)} cands")
        row = {
            "dataset": args.dataset, "mode": mode,
            "images": len(img_ids),
            "avg_latency_s": round((time.time() - t_all) / len(img_ids), 2),
            "peak_gpu_mem_gb": round(mem_peak / 2**30, 2),
        }
        (out_dir / f"{args.dataset}_{mode}_meta.json").write_text(
            json.dumps(row, indent=1))
        print(json.dumps(row))


if __name__ == "__main__":
    main()
