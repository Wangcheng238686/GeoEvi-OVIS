"""SCORE dataset registry and loaders (stdlib-only; pycocotools enters at eval).

Split authority: SCORE HF `HuangShiqi008/SCORE` (implementation-v2 S1.3, E4).
Every dataset is a COCO-instance subtree under $SAM3_OVIS_DATA_ROOT:

    <DATA_ROOT>/<DATASET>/val/instances_val.json   (+ images/)

Expectations below are cross-checked against the SCORE paper (ICCV 2025,
arXiv 2507.12857) §4.1 where it states counts (NWPU 731 imgs / 10 classes,
SOTA 874 / 18, FAST 3207 / 37); iSAID/SIOR figures are registered from the
downloaded HF jsons and re-verified at smoke time.
"""

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# (subtree, split dir, expected images, expected annotations, expected classes)
# expected=None for iSAID/SIOR until their zips land; smoke fills and pins them.
@dataclass
class DatasetSpec:
    name: str
    role: str                      # train / test / supplement
    val_json: Path
    images_dir: Path
    expected_images: Optional[int] = None
    expected_annotations: Optional[int] = None
    expected_categories: Optional[List[str]] = None
    notes: Dict[str, str] = field(default_factory=dict)


DATASETS = OrderedDict([
    ("NWPU", DatasetSpec(
        name="NWPU", role="test", val_json=Path("NWPU/val/instances_val.json"),
        images_dir=Path("NWPU/val/images"),
        expected_images=731, expected_annotations=2661,
        expected_categories=[
            "airplane", "ship", "storage_tank", "baseball_diamond",
            "tennis_court", "basketball_court", "ground_track_field",
            "harbor", "bridge", "vehicle"],
        notes={"gt": "human (manual masks; SCORE paper §4.1, ref [44])"},
    )),
    ("SOTA", DatasetSpec(
        name="SOTA", role="test", val_json=Path("SOTA/val/instances_val.json"),
        images_dir=Path("SOTA/val/images"),
        expected_images=874, expected_annotations=29247,
        expected_categories=[
            "large-vehicle", "swimming-pool", "helicopter", "bridge",
            "plane", "ship", "soccer-ball-field", "basketball-court",
            "ground-track-field", "small-vehicle", "baseball-diamond",
            "tennis-court", "roundabout", "storage-tank", "harbor",
            "container-crane", "airport", "helipad"],
        notes={"gt": "SAM pseudo-label (SAMRS from DOTA-V2.0)"},
    )),
    ("FAST", DatasetSpec(
        name="FAST", role="test", val_json=Path("FAST/val/instances_val.json"),
        images_dir=Path("FAST/val/images"),
        expected_images=3207, expected_annotations=43258,
        expected_categories=[
            "A220", "A321", "A330", "A350", "ARJ21", "Baseball-Field",
            "Basketball-Court", "Boeing737", "Boeing747", "Boeing777",
            "Boeing787", "Bridge", "Bus", "C919", "Cargo-Truck",
            "Dry-Cargo-Ship", "Dump-Truck", "Engineering-Ship", "Excavator",
            "Fishing-Boat", "Football-Field", "Intersection",
            "Liquid-Cargo-Ship", "Motorboat", "other-airplane", "other-ship",
            "other-vehicle", "Passenger-Ship", "Roundabout", "Small-Car",
            "Tennis-Court", "Tractor", "Trailer", "Truck-Tractor", "Tugboat",
            "Van", "Warship"],
        notes={"gt": "SAM pseudo-label (SAMRS from FAIR1M 2.0)"},
    )),
    ("iSAID", DatasetSpec(
        name="iSAID", role="train", val_json=Path("iSAID/train/instances_train.json"),
        images_dir=Path("iSAID/train/images"),
        expected_images=28029, expected_annotations=704684,
        expected_categories=[
            "Small_Vehicle", "Large_Vehicle", "plane", "storage_tank",
            "ship", "Swimming_pool", "Harbor", "tennis_court",
            "Ground_Track_Field", "Soccer_ball_field", "baseball_diamond",
            "Bridge", "basketball_court", "Roundabout", "Helicopter"],
        notes={
            "gt": "human (original iSAID manual instance annotation)",
            # SCORE paper §4.1 states 18,732 train images; the published HF
            # json (split authority, E4) carries 28,029. Registered from the
            # json; discrepancy flagged in p-1-report.md, not resolved here.
        },
    )),
    ("iSAID-val", DatasetSpec(
        # SCORE Table 1 uses iSAID val as the test set when trained on SIOR
        name="iSAID-val", role="test",
        val_json=Path("iSAID/val/instances_val.json"),
        images_dir=Path("iSAID/val/images"),
        expected_images=9512, expected_annotations=235531,
        expected_categories=[
            "Small_Vehicle", "Large_Vehicle", "plane", "storage_tank",
            "ship", "Swimming_pool", "Harbor", "tennis_court",
            "Ground_Track_Field", "Soccer_ball_field", "baseball_diamond",
            "Bridge", "basketball_court", "Roundabout", "Helicopter"],
        notes={"gt": "human (original iSAID manual instance annotation)"},
    )),
    ("SIOR", DatasetSpec(
        name="SIOR", role="train", val_json=Path("SIOR/train/instances_train.json"),
        images_dir=Path("SIOR/train/images"),
        expected_images=11725, expected_annotations=68062,
        expected_categories=[
            "airplane", "airport", "baseball field", "basketball court",
            "bridge", "chimney", "expressway-service-area",
            "expressway-toll-station", "dam", "golffield",
            "ground track field", "harbor", "overpass", "ship", "stadium",
            "storage tank", "tennis court", "train station", "vehicle",
            "windmill"],
        notes={"gt": "SAM pseudo-label (SAMRS from DIOR)"},
    )),
    ("SIOR-val", DatasetSpec(
        # SCORE Table 1 uses SIOR val as the test set when trained on iSAID
        name="SIOR-val", role="test",
        val_json=Path("SIOR/val/instances_val.json"),
        images_dir=Path("SIOR/val/images"),
        expected_images=11738, expected_annotations=124373,
        expected_categories=[
            "airplane", "airport", "baseball field", "basketball court",
            "bridge", "chimney", "expressway-service-area",
            "expressway-toll-station", "dam", "golffield",
            "ground track field", "harbor", "overpass", "ship", "stadium",
            "storage tank", "tennis court", "train station", "vehicle",
            "windmill"],
        notes={"gt": "SAM pseudo-label (SAMRS from DIOR)"},
    )),
    ("NWPU-Refer", DatasetSpec(
        # Supplement (implementation-v2 S1.3): category subset of the
        # NWPU-Refer test split, built by data/build_nwpurefer_category.py
        # (extraction rule registered in its docstring). 1024-2048px images:
        # the first real validation ground for the S2.3 window mode.
        name="NWPU-Refer", role="supplement",
        val_json=Path("NWPU-Refer/val/instances_val.json"),
        images_dir=Path("NWPU-Refer/val/images"),
        expected_images=2541, expected_annotations=10071,
        expected_categories=None,  # 31 cleaned names; pinned by build script
        notes={"gt": "human (referring expressions + manual instance masks)",
               "query": "cleaned English category names; airplaneroad "
                        "dropped, photovolatic -> photovoltaic panel"},
    )),
])


def resolve_specs(data_root: Optional[str] = None) -> OrderedDict:
    """Resolve registered datasets against the data root, skipping missing."""
    root = Path(data_root or os.environ.get("SAM3_OVIS_DATA_ROOT", ""))
    out = OrderedDict()
    for name, spec in DATASETS.items():
        if (root / spec.val_json).exists():
            s = DatasetSpec(**{**spec.__dict__,
                               "val_json": root / spec.val_json,
                               "images_dir": root / spec.images_dir})
            out[name] = s
    return out


def load_coco_instance_json(json_path) -> dict:
    """Read a COCO-instance json verbatim (images/annotations/categories)."""
    with open(json_path, "r") as f:
        return json.load(f)


def dataset_summary(spec: DatasetSpec) -> dict:
    """Smoke summary: counts + category table + image-path spot check."""
    d = load_coco_instance_json(spec.val_json)
    cats = [(c["id"], c["name"]) for c in d["categories"]]
    summary = {
        "name": spec.name,
        "images": len(d["images"]),
        "annotations": len(d["annotations"]),
        "categories": [c for _, c in cats],
        "segmentation_kind": type(d["annotations"][0]["segmentation"]).__name__
        if d["annotations"] else "none",
    }
    # spot-check: first image's file must exist on disk
    first = d["images"][0]
    summary["image_path_ok"] = (spec.images_dir / first["file_name"]).exists()
    if spec.expected_images is not None:
        summary["images_match"] = summary["images"] == spec.expected_images
    if spec.expected_annotations is not None:
        summary["annotations_match"] = (
            summary["annotations"] == spec.expected_annotations)
    if spec.expected_categories is not None:
        summary["categories_match"] = (
            summary["categories"] == spec.expected_categories)
    return summary
