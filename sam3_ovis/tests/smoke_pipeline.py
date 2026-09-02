"""M0 smoke pipeline (P-1 exit, implementation-v2 S1.5).

Chains the P-1 deliverables and fails loudly on any gap:
  1. environment: run under scripts/load_environment.sh; required vars set;
     sam3 env python imports torch with CUDA visible;
  2. data: registered datasets pass counts / category-table / image checks;
  3. eval: forged-prediction self-test of the SCORE eval wrapper passes;
  4. model: vendored sam3 importable + SAM3_CKPT present (S1.1 output).

Run:  source sam3_ovis/scripts/load_environment.sh && "$PYTHON" sam3_ovis/tests/smoke_pipeline.py
"""

import importlib
import os
import sys
from pathlib import Path

MAINLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE))

REQUIRED_ENV = [
    "PYTHON", "SAM3_REPO", "SAM3_CKPT", "SAM3_HUB_DIR",
    "SAM3_OVIS_DATA_ROOT", "SAM3_OVIS_LOG_ROOT", "SAM3_OVIS_TMP_ROOT",
    "CUDA_VISIBLE_DEVICES",
]


def check_env() -> bool:
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        print(f"[env ] FAIL missing vars: {missing} "
              "(source sam3_ovis/scripts/load_environment.sh first)")
        return False
    try:
        import torch  # noqa: PLC0415

        ok = torch.cuda.is_available()
        n = torch.cuda.device_count()
        print(f"[env ] torch {torch.__version__}, cuda={ok}, "
              f"visible GPUs={n} (CUDA_VISIBLE_DEVICES="
              f"{os.environ['CUDA_VISIBLE_DEVICES']})")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[env ] FAIL torch import: {e}")
        return False


def check_data() -> bool:
    from data.score_datasets import resolve_specs, dataset_summary

    specs = resolve_specs()
    if not specs:
        print("[data] FAIL nothing registered under $SAM3_OVIS_DATA_ROOT")
        return False
    ok_all = True
    for name, spec in specs.items():
        s = dataset_summary(spec)
        ok = (all(v for k, v in s.items() if k.endswith("_match"))
              and s["image_path_ok"])
        ok_all &= ok
        print(f"[data] {name}: imgs={s['images']} anns={s['annotations']} "
              f"cats={len(s['categories'])} -> {'PASS' if ok else 'FAIL'}")
    expected = {"NWPU", "SOTA", "FAST"}
    if not expected.issubset(specs):
        print(f"[data] FAIL core test sets missing: {expected - set(specs)}")
        ok_all = False
    return ok_all


def check_eval() -> bool:
    try:
        import numpy as np  # noqa: F401, PLC0415
        from pycocotools import mask  # noqa: F401, PLC0415
        from tests.test_score_eval import main as eval_selftest

        rc = eval_selftest()
        print(f"[eval] self-test rc={rc}")
        return rc == 0
    except ImportError as e:
        print(f"[eval] FAIL deps: {e}")
        return False


def check_model() -> bool:
    ckpt = Path(os.environ["SAM3_CKPT"])
    if not ckpt.exists():
        print(f"[model] FAIL checkpoint missing: {ckpt}")
        return False
    try:
        importlib.import_module("sam3")
        print(f"[model] sam3 importable, checkpoint {ckpt.stat().st_size} bytes")
        return True
    except ImportError as e:
        print(f"[model] FAIL sam3 import: {e}")
        return False


def main() -> int:
    results = {
        "env": check_env(),
        "data": check_data(),
        "eval": check_eval(),
        "model": check_model(),
    }
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if all(results.values()):
        print("M0 smoke pipeline: PASS")
        return 0
    print("M0 smoke pipeline: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
