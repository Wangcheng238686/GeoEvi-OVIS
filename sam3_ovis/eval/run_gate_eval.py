"""P0 gate evaluation driver: predictions.jsonl -> G1/G2 numbers.

Reads the zero-shot predictions of each test set, runs the S1.4 SCORE
evaluation + diagnostics, and prints/aggregates the gate table against
notes/reports/p0-gate-criteria.md (G1 class-avg recall proxy @IoU0.5 pooled
over NWPU/SOTA/FAST; G2 per-dataset mAP + Average).

Run:
  source sam3_ovis/scripts/load_environment.sh
  "$PYTHON" sam3_ovis/eval/run_gate_eval.py --datasets NWPU SOTA FAST
"""

import argparse
import json
import sys
from pathlib import Path

MAINLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE))

from eval.score_eval import (  # noqa: E402
    compute_diagnostics, cross_dataset_average, evaluate_dataset, load_gt_coco,
)
from data.score_datasets import DATASETS  # noqa: E402

GATE_SETS = ["NWPU", "SOTA", "FAST"]  # registered in p0-gate-criteria.md


def load_predictions_jsonl(path: Path) -> list:
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def pooled_class_avg_recall(per_ds_diag: dict) -> float:
    """G1 registered aggregation (p0-gate-criteria.md §2): per class, pool
    GT instances over the gate datasets, compute coverage ratio, then average
    over the union of class names."""
    import numpy as np
    from collections import defaultdict

    pooled = defaultdict(lambda: [0, 0])  # name -> [covered, total]
    for diag in per_ds_diag.values():
        for name, st in diag["per_class"].items():
            pooled[name][0] += st["recall_hits"]
            pooled[name][1] += st["n_gt"]
    ratios = [c / t for c, t in pooled.values() if t > 0]
    return float(np.mean(ratios)) if ratios else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=GATE_SETS)
    ap.add_argument("--out", default=None, help="write gate table json here")
    args = ap.parse_args()

    import os

    root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
    log_root = Path(os.environ["SAM3_OVIS_LOG_ROOT"]) / "zeroshot"

    results = {}
    per_ds_diag = {}
    for ds in args.datasets:
        spec = DATASETS[ds]
        gt = load_gt_coco(str(root / spec.val_json))
        preds = load_predictions_jsonl(log_root / ds / "predictions.jsonl")
        summary, per_class = evaluate_dataset(gt, preds)
        diag = compute_diagnostics(gt, preds)
        per_ds_diag[ds] = diag
        results[ds] = {
            "mAP": summary["segm/mAP"],
            "mAP_50": summary["segm/mAP_50"],
            "recall_proxy_class_avg": diag["recall_proxy_class_avg"],
            "recall_proxy_micro": diag["recall_proxy_micro"],
            "precision_proxy_class_avg": diag["precision_proxy_class_avg"],
            "conditional_iou_class_avg": diag["conditional_iou_class_avg"],
            "n_predictions": len(preds),
            "per_class_ap": per_class,
        }
        print(f"[{ds}] mAP={summary['segm/mAP']:.4f} "
              f"AP50={summary['segm/mAP_50']:.4f} "
              f"G1(recall proxy)={diag['recall_proxy_class_avg']:.4f} "
              f"micro={diag['recall_proxy_micro']:.4f} "
              f"prec={diag['precision_proxy_class_avg']:.4f} "
              f"condIoU={diag['conditional_iou_class_avg']:.4f}")

    gate_sets = [d for d in GATE_SETS if d in results]
    if len(gate_sets) == len(GATE_SETS):
        g2_avg = cross_dataset_average(
            {d: {"segm/mAP": results[d]["mAP"]} for d in gate_sets})
        g1 = pooled_class_avg_recall({d: per_ds_diag[d] for d in gate_sets})
        verdict = ("GO" if g1 >= 0.40 else
                   "CONDITIONAL GO" if g1 >= 0.20 else "NO-GO")
        results["_gate"] = {
            "G1_pooled_class_avg_recall": g1,
            "G1_verdict": verdict,
            "G2_average_mAP": g2_avg,
        }
        print(f"\nG1 pooled class-avg recall @IoU0.5 = {g1:.4f} -> {verdict}")
        print(f"G2 cross-dataset average mAP      = {g2_avg:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"gate table -> {args.out}")


if __name__ == "__main__":
    main()
