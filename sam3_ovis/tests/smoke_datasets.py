"""S1.3 smoke: registered datasets match official counts / category tables.

Run:  $PYTHON sam3_ovis/tests/smoke_datasets.py
Exit 0 only if every resolved dataset passes all its registered checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.score_datasets import resolve_specs, dataset_summary  # noqa: E402


def main() -> int:
    specs = resolve_specs()
    if not specs:
        print("no registered dataset found under $SAM3_OVIS_DATA_ROOT")
        return 1
    print(f"data root: {specs[next(iter(specs))].val_json.parents[2]}")
    failed = False
    for name, spec in specs.items():
        s = dataset_summary(spec)
        ok_flags = [v for k, v in s.items() if k.endswith("_match")]
        ok = all(ok_flags) and s["image_path_ok"]
        print(f"[{name}] imgs={s['images']} anns={s['annotations']} "
              f"cats={len(s['categories'])} seg={s['segmentation_kind']} "
              f"img_ok={s['image_path_ok']} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed = True
            for k, v in s.items():
                if k.endswith("_match") and not v:
                    print(f"    mismatch on {k}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
