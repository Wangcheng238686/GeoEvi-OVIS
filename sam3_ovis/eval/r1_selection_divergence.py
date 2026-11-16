r"""R1 "relation selection" vs no-graph retrieval: selection divergence rate.

Quantifies how differently R1 (graph relation-based context selection) picks
memory candidates compared with two no-graph retrieval baselines, on the T2
pre-fusion candidate cache. Pure CPU (numpy + pycocotools only, no torch).

Question (from notes/reports/2026-10-25-graph-memory-candidate-routes.md risk
column): if the graph and plain top-K retrieval select the same information,
R1's payoff is small. This script measures the selection divergence as the
existence space for R1, in the spirit of the G1 report's divergence framing.

=========================== RULE REGISTRATION ===============================
(Registered BEFORE any computation on real data; must not be adjusted after
 seeing results. Verified against real data BEFORE the metrics were computed
 were only format/integrity checks: record fields, RLE decodability,
 spatial_rivals index range, rival-pair IoU>=0.5, rival pairs all cross-class.)

Data: sam3_ovis/logs/t2_cache/candidates.jsonl (+ manifest.json in same dir).
  - Per record: img_id, cat_name, bbox (xyxy, pixels, half-open), mask_rle
    (COCO compressed RLE, size [h,w]), z (15-class CLIP cosine dict),
    spatial_rivals (keys "img{img_id}/{idx}"; idx = write order of that
    image's candidates in the file).
  - Pool floor 0.05 + class-wise NMS 0.5 were applied upstream, so same-class
    pairs in the cache have mask IoU < 0.5 (verified: 0 same-class rivals).

Per image:
  TARGETS: candidates sorted by raw_score desc (tie -> write idx asc), top
  min(8, n) are targets. POOL: the remaining candidates of the same image.
  Both U and R1 select exclusively from POOL (identical universes).
  EXCLUSION: images with pool < 4 (n < 12) are excluded entirely; the count
  and ids are reported. (Images with 0 candidates above the pool floor simply
  contribute no targets and are reported.)

  U1 (distance retrieval baseline): pool candidates ranked by Euclidean
  distance between bbox centers ((x1+x2)/2, (y1+y2)/2); K smallest.
  U2 (CLIP similarity retrieval baseline): pool candidates ranked by cosine
  similarity of the 15-dim z vectors (component order: sorted class names,
  values as stored, i.e. rounded to 6 dp); K largest.
  Ties in U1/U2 -> smaller write idx first.

  R1 (relation selection):
    O (overlap competition) = {c in pool : cat(c) != cat(t) and
                               maskIoU(t, c) >= 0.5}            [cross-class]
    A (adjacency)           = {c in pool \ O : 0.02 < bboxIoU(t, c) < 0.5}
                                                              [any class]
    Ranking: all of O first sorted by mask IoU desc, then A sorted by bbox
    IoU desc; ties -> smaller write idx first. Take the first K.
    If |O ∪ A| < K, R1 returns fewer than K (SHORTFALL; no padding); the
    shortfall rate is reported as a separate registered metric.
    mask IoU is recomputed from mask_rle via pycocotools (exact, C impl) for
    every pair with bboxIoU > 0.02 — with tight upstream bboxes this cannot
    miss a maskIoU>=0.5 pair (mask IoU >= 0.5 forces bbox IoU well above
    0.02); spatial_rivals is used only as a cross-check of this recomputation
    (O itself is defined by the recomputed IoU, never by spatial_rivals).

  K in {2, 4}.

Metrics per target t, per K, per U in {U1, U2}:
  Jaccard(R1, U) = |R1 ∩ U| / |R1 ∪ U|. R1 empty -> 0.0 (U always has K).
  R1-only share  = |R1 \ U| / |R1|; undefined (None) if R1 empty -> excluded
    from the mean, empty-R1 rate reported separately.
  U-only share   = |U \ R1| / |K| (context metric).

Stratification:
  - by target class (cat_name of t);
  - by image density quartile: n = candidate count of the image (included
    images only); cuts c1<c2<c3 = np.percentile(n_list, [25,50,75]);
    bin 1: n<=c1, bin 2: c1<n<=c2, bin 3: c2<n<=c3, bin 4: n>c3.

Summary: per (K, U): n, mean, median, p25, p75, p90 of Jaccard; mean of
R1-only (over defined values) and its n; empty-R1 rate; shortfall rate.

Outputs:
  - sam3_ovis/logs/r1_selection_divergence.json  (details + summaries)
  - console summary tables.

Self-test: synthetic small dataset, >=5 assertions, run BEFORE real data;
`--self-test` runs only the self-test. The synthetic data deliberately ships
WRONG spatial_rivals to prove R1 never reads them.
=============================================================================

Run:  python3 sam3_ovis/eval/r1_selection_divergence.py [--self-test]
"""

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_util

# ---- registered constants (mirror of the docstring) ------------------------
N_TARGETS_PER_IMG = 8
MIN_POOL = 4
KS = (2, 4)
OVERLAP_MASK_IOU = 0.5      # O: cross-class mask IoU >= 0.5
ADJ_IOU_LO = 0.02           # A: bbox IoU in open (0.02, 0.5)
ADJ_IOU_HI = 0.5
BBOX_PREFILTER = 0.02       # exact mask IoU only computed if bbox IoU > this

DEFAULT_CANDIDATES = "sam3_ovis/logs/t2_cache/candidates.jsonl"
DEFAULT_OUT = "sam3_ovis/logs/r1_selection_divergence.json"


# ============================ geometry helpers ==============================
def bbox_iou_matrix(bboxes):
    """Pairwise IoU of xyxy half-open boxes. bboxes: (n, 4) -> (n, n)."""
    b = np.asarray(bboxes, dtype=np.float64)
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area[:, None] + area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def centers_of(bboxes):
    b = np.asarray(bboxes, dtype=np.float64)
    return np.stack([(b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0],
                    axis=1)


def rle_of(rec):
    r = rec["mask_rle"]
    return {"size": list(r["size"]), "counts": r["counts"].encode("ascii")}


def mask_iou_batch(rle_t, rle_others):
    """Exact mask IoU of one RLE vs a list of RLEs (pycocotools C impl)."""
    if not rle_others:
        return np.zeros(0, dtype=np.float64)
    out = mask_util.iou([rle_t], list(rle_others), [0] * len(rle_others))
    return np.asarray(out, dtype=np.float64).ravel()


# ============================ selection rules ===============================
def select_u1(t_idx, pool_idx, centers):
    """K smallest bbox-center Euclidean distances; tie -> smaller idx."""
    c = centers[t_idx]
    order = sorted(pool_idx, key=lambda i: (float(np.hypot(*(centers[i] - c))),
                                            i))
    return order


def select_u2(t_idx, pool_idx, z_unit):
    """K largest z cosine similarities; tie -> smaller idx."""
    sims = z_unit @ z_unit[t_idx]
    order = sorted(pool_idx, key=lambda i: (-float(sims[i]), i))
    return order


def select_r1_full(t_idx, pool_idx, bbox_iou, mask_iou, cat_names):
    r"""Return (overlap_list, adjacent_list) with registered ranking.

    overlap_list: cross-class pool cands with mask IoU >= 0.5, mask IoU desc.
    adjacent_list: pool \ overlap with bbox IoU in (0.02, 0.5), bbox IoU desc.
    Ties -> smaller idx first. R1(K) = (overlap + adjacent)[:K].
    """
    ct = cat_names[t_idx]
    overlap, adjacent = [], []
    for j, i in enumerate(pool_idx):   # mask_iou is aligned with pool order
        bi = float(bbox_iou[t_idx, i])
        if bi <= BBOX_PREFILTER:
            continue
        mi = float(mask_iou[j])
        if mi >= OVERLAP_MASK_IOU and cat_names[i] != ct:
            overlap.append((-mi, i))
        elif ADJ_IOU_LO < bi < ADJ_IOU_HI:
            adjacent.append((-bi, i))
    overlap.sort()
    adjacent.sort()
    return [i for _, i in overlap], [i for _, i in adjacent]


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return None
    u = a | b
    return len(a & b) / len(u) if u else None


def r1_only_share(r1, u):
    if not r1:
        return None
    return len(set(r1) - set(u)) / len(set(r1))


def u_only_share(u, r1):
    return len(set(u) - set(r1)) / len(u)


# ============================ image processing ==============================
def load_candidates(path):
    by_img = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_img[d["img_id"]].append(d)
    return by_img


def check_rivals(recs, rles, bbox_iou):
    """Integrity cross-check of spatial_rivals vs recomputed mask IoU."""
    n = len(recs)
    rival_pairs = set()
    for i, d in enumerate(recs):
        for k in d["spatial_rivals"]:
            j = int(k.split("/")[1])
            assert 0 <= j < n and j != i, f"bad rival key {k} (n={n})"
            rival_pairs.add((min(i, j), max(i, j)))
    # recompute mask IoU for all pairs with bbox IoU > prefilter
    recomputed_ge = set()
    ii, jj = np.where(np.triu(bbox_iou, 1) > BBOX_PREFILTER)
    by_i = defaultdict(list)
    for i, j in zip(ii.tolist(), jj.tolist()):
        by_i[i].append(j)
    for i, js in by_i.items():
        vals = mask_iou_batch(rles[i], [rles[j] for j in js])
        for j, miou in zip(js, vals.tolist()):
            if miou >= OVERLAP_MASK_IOU:
                recomputed_ge.add((i, j))
    missed = recomputed_ge - rival_pairs       # our rule finds, rivals lack
    extra = rival_pairs - recomputed_ge        # rivals claim, recomputation
    return {"rival_pairs": len(rival_pairs),
            "recomputed_ge_0p5": len(recomputed_ge),
            "missed_by_rivals": len(missed),
            "unsupported_by_recompute": len(extra)}


def process_image(recs, integrity):
    """Compute per-target selections for one image. None if excluded."""
    n = len(recs)
    if n - min(N_TARGETS_PER_IMG, n) < MIN_POOL:  # pool < 4
        return None
    order = sorted(range(n), key=lambda i: (-float(recs[i]["raw_score"]), i))
    targets = order[:N_TARGETS_PER_IMG]
    pool = sorted(order[N_TARGETS_PER_IMG:])

    bboxes = np.array([r["bbox"] for r in recs], dtype=np.float64)
    cat_names = [r["cat_name"] for r in recs]
    centers = centers_of(bboxes)
    bbox_iou = bbox_iou_matrix(bboxes)

    zkeys = sorted(recs[0]["z"].keys())
    Z = np.array([[r["z"][k] for k in zkeys] for r in recs], dtype=np.float64)
    zn = np.linalg.norm(Z, axis=1, keepdims=True)
    z_unit = Z / np.maximum(zn, 1e-12)

    rles = [rle_of(r) for r in recs]

    integrity.append(check_rivals(recs, rles, bbox_iou))

    # mask IoU rows for targets, evaluated only on pool cands passing the
    # bbox prefilter (registered); others are structurally irrelevant.
    mask_iou_rows = {}
    for t in targets:
        cand = [i for i in pool if bbox_iou[t, i] > BBOX_PREFILTER]
        vals = mask_iou_batch(rles[t], [rles[i] for i in cand])
        mask_iou_rows[t] = dict(zip(cand, vals.tolist()))

    out = []
    for t in targets:
        mi_row = mask_iou_rows[t]
        # mask_iou lookup aligned with pool order (0.0 where prefiltered out)
        mi_vec = [mi_row.get(i, 0.0) for i in pool]
        u1_rank = select_u1(t, pool, centers)
        u2_rank = select_u2(t, pool, z_unit)
        ov, adj = select_r1_full(t, pool, bbox_iou, mi_vec, cat_names)
        entry = {
            "img_id": recs[t]["img_id"], "idx": t,
            "key": f"img{recs[t]['img_id']}/{t}",
            "cat_name": cat_names[t], "raw_score": recs[t]["raw_score"],
            "img_n_cands": n, "overlap_rank": ov, "adjacent_rank": adj,
            "u1_rank": u1_rank[: max(KS)], "u2_rank": u2_rank[: max(KS)],
            "per_k": {},
        }
        for k in KS:
            r1 = (ov + adj)[:k]
            rec_k = {"r1": r1, "r1_n": len(r1),
                     "r1_shortfall": len(r1) < k, "r1_empty": len(r1) == 0}
            for uname, urank in (("U1", u1_rank), ("U2", u2_rank)):
                u = urank[:k]
                rec_k[uname] = u
                rec_k[f"jac_{uname}"] = 0.0 if not r1 else jaccard(r1, u)
                rec_k[f"r1only_{uname}"] = r1_only_share(r1, u)
                rec_k[f"uonly_{uname}"] = u_only_share(u, r1)
            entry["per_k"][str(k)] = rec_k
        out.append(entry)
    return out


# ============================ aggregation ===================================
def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    a = np.array(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90))}


def summarize(target_records, n_images_included):
    ks = [str(k) for k in KS]
    summary = {"n_images_included": n_images_included,
               "n_targets": len(target_records)}
    for k in ks:
        summary[k] = {}
        for u in ("U1", "U2"):
            recs_k = [t["per_k"][k] for t in target_records]
            jac = [r[f"jac_{u}"] for r in recs_k]
            r1o = [r[f"r1only_{u}"] for r in recs_k]
            uo = [r[f"uonly_{u}"] for r in recs_k]
            n_empty = sum(1 for r in recs_k if r["r1_empty"])
            n_short = sum(1 for r in recs_k if r["r1_shortfall"])
            summary[k][u] = {
                "jaccard": _stats(jac),
                "r1_only_share": _stats(r1o),
                "u_only_share": _stats(uo),
                "r1_empty_rate": n_empty / len(recs_k) if recs_k else 0.0,
                "r1_shortfall_rate": n_short / len(recs_k) if recs_k else 0.0,
                "n_r1_empty": n_empty, "n_r1_shortfall": n_short,
            }
    return summary


def stratify(target_records, keyfn):
    groups = defaultdict(list)
    for t in target_records:
        groups[keyfn(t)].append(t)
    return {g: summarize(rs, None) for g, rs in sorted(groups.items())}


def density_bins(n_included):
    arr = np.array(sorted(n_included), dtype=np.float64)
    cuts = np.percentile(arr, [25, 50, 75]).tolist()

    def bin_of(n):
        return (1 if n <= cuts[0] else
                2 if n <= cuts[1] else
                3 if n <= cuts[2] else 4)
    return cuts, bin_of


# ============================ self-test =====================================
def _mk_rec(img_id, idx, cat, bbox, mask, z, score, rivals):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"img_id": img_id, "cat_id": 0, "cat_name": cat,
            "raw_score": score,
            "mask_rle": {"size": list(rle["size"]),
                         "counts": rle["counts"].decode("utf-8")},
            "bbox": list(bbox), "z": z, "spatial_rivals": rivals}


def self_test():
    """Synthetic data; hand-computed expectations; >=5 assertions."""
    rng = np.random.default_rng(0)

    # A1: bbox IoU hand values.
    bb = np.array([[0, 0, 10, 10], [0, 0, 10, 5], [5, 5, 15, 15],
                   [20, 20, 30, 30]], dtype=np.float64)
    m = bbox_iou_matrix(bb)
    assert abs(m[0, 1] - 50 / 100) < 1e-12          # inter 50 / union 100
    assert abs(m[0, 2] - 25 / 175) < 1e-12          # inter 25 / union 175
    assert m[0, 3] == 0.0 and np.allclose(np.diag(m), 1.0)

    # A2: mask IoU from RLE equals numpy mask IoU (round-trip through json
    # string form, as stored in the cache).
    ma = (rng.random((32, 32)) > 0.6)
    mb = (rng.random((32, 32)) > 0.6)
    rec_a = _mk_rec(1, 0, "A", [0, 0, 32, 32], ma,
                    {"a": 0.2, "b": 0.1}, 0.9, [])
    inter = int((ma & mb).sum())
    miou_np = inter / (ma.sum() + mb.sum() - inter)
    miou_rle = float(mask_iou_batch(rle_of(rec_a), [rle_of(
        _mk_rec(1, 1, "B", [0, 0, 32, 32], mb, {"a": 0.1, "b": 0.2}, 0.8, [])
    )])[0])
    assert abs(miou_np - miou_rle) < 1e-9

    # Build one synthetic image (100x100 canvas), target idx 0 by score.
    # target: box [40,40,60,60], mask fills the box (area 400)
    tgt_mask = np.zeros((100, 100), bool)
    tgt_mask[40:60, 40:60] = True
    z_t = {"a": 0.9, "b": 0.1}
    # c1: SAME-class identical box/mask (mask IoU 1.0 but same class ->
    #     EXCLUDED from O; bbox IoU 1.0 >= 0.5 so not adjacent either)
    c1_mask = np.zeros((100, 100), bool)
    c1_mask[40:60, 40:60] = True
    # c2: cross-class mask [40:60, 40:70] -> mask IoU 400/600 = 2/3 >= 0.5
    #     (O member); bbox [40,40,70,60], center (55, 50), d(target) = 5
    c2_mask = np.zeros((100, 100), bool)
    c2_mask[40:60, 40:70] = True
    # c3: adjacent, box [40,50,60,70] -> bbox IoU 200/600 = 1/3 in
    #     (0.02, 0.5); mask [50:70, 40:60] -> mask IoU 1/3 < 0.5; A member
    c3_mask = np.zeros((100, 100), bool)
    c3_mask[50:70, 40:60] = True
    # c4: far away, no bbox overlap (bbox IoU 0)
    c4_mask = np.zeros((100, 100), bool)
    c4_mask[80:90, 80:90] = True
    recs = [
        _mk_rec(7, 0, "a", [40, 40, 60, 60], tgt_mask, z_t, 0.9, []),
        _mk_rec(7, 1, "a", [40, 40, 60, 60], c1_mask,
                {"a": 0.85, "b": 0.15}, 0.8, ["img7/0"]),   # WRONG rivals
        _mk_rec(7, 2, "b", [40, 40, 70, 60], c2_mask,
                {"a": 0.2, "b": 0.8}, 0.7, []),
        _mk_rec(7, 3, "b", [40, 50, 60, 70], c3_mask,
                {"a": 0.3, "b": 0.7}, 0.6, []),
        _mk_rec(7, 4, "b", [80, 80, 90, 90], c4_mask,
                {"a": 0.1, "b": 0.9}, 0.5, []),
    ]
    bboxes = np.array([r["bbox"] for r in recs], dtype=np.float64)
    cat_names = [r["cat_name"] for r in recs]
    centers = centers_of(bboxes)
    bbox_iou = bbox_iou_matrix(bboxes)
    pool = [1, 2, 3, 4]
    rles = [rle_of(r) for r in recs]
    mi = mask_iou_batch(rles[0], [rles[i] for i in pool])

    # A3: U1 ranking by center distance to target center (50, 50):
    #     c1 (50,50) d=0, c2 (55,50) d=5, c3 (50,60) d=10, c4 d=49.5.
    u1 = select_u1(0, pool, centers)
    assert u1 == [1, 2, 3, 4]

    # A4: U2 (K=1) = highest z cosine = c1 (0.85/0.15 vs target 0.9/0.1).
    zkeys = sorted(z_t.keys())
    Z = np.array([[r["z"][k] for k in zkeys] for r in recs])
    zu = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    u2 = select_u2(0, pool, zu)
    assert u2[0] == 1

    # A5: R1 ranking: O = [c2] (cross-class mask IoU 2/3; same-class c1
    #     excluded despite mask IoU 1.0; R1 ignores the wrong spatial_rivals),
    #     A = [c3] (bbox IoU 1/3 in (0.02, 0.5)), c4 neither.
    ov, adj = select_r1_full(0, pool, bbox_iou, list(mi), cat_names)
    assert ov == [2], ov
    assert adj == [3], adj
    assert abs(mi[0] - 1.0) < 1e-9 and abs(mi[1] - 2 / 3) < 1e-9 \
        and abs(mi[2] - 1 / 3) < 1e-9 and mi[3] == 0.0

    # A6: metrics hand-computed. R1(K=2) = [2, 3]; U1(K=2) = [1, 2];
    #     Jaccard = |{2}| / |{1,2,3}| = 1/3; R1-only = |{3}|/2 = 0.5;
    #     U-only = |{1}|/2 = 0.5.
    r1_k2 = (ov + adj)[:2]
    assert r1_k2 == [2, 3]
    assert abs(jaccard(r1_k2, u1[:2]) - 1 / 3) < 1e-12
    assert abs(r1_only_share(r1_k2, u1[:2]) - 0.5) < 1e-12
    assert abs(u_only_share(u1[:2], r1_k2) - 0.5) < 1e-12

    # A7: empty R1 -> Jaccard convention 0, R1-only None (via process_image).
    #     12 records (1 target + 11 far away, zero-overlap pool) so the image
    #     passes the pool >= 4 inclusion rule; the target's R1 is empty.
    z_far = {"a": 0.1, "b": 0.9}
    solo = [_mk_rec(8, 0, "a", [0, 0, 10, 10],
                    np.pad(np.ones((10, 10), bool), ((0, 90), (0, 90))),
                    z_t, 0.9, [])]
    for i in range(1, 12):
        solo.append(_mk_rec(8, i, "b" if i % 2 else "a",
                            [50, 50 + i, 60, 60 + i],
                            np.zeros((100, 100), bool), z_far,
                            0.89 - 0.01 * i, []))
    out = process_image(solo, [])
    assert out is not None and len(out) == N_TARGETS_PER_IMG
    t0 = next(e for e in out if e["idx"] == 0)
    k2 = t0["per_k"]["2"]
    assert k2["r1_empty"] and k2["jac_U1"] == 0.0
    assert k2["r1only_U1"] is None and k2["r1_n"] == 0
    # A8: shortfall: R1 empty but K=4 (and K=2) demand more.
    k4 = t0["per_k"]["4"]
    assert k4["r1_n"] == 0 and k4["r1_shortfall"] and k2["r1_shortfall"]

    # A9: exclusion rule pool < 4 (n=5 with N_TARGETS_PER_IMG=8 -> pool 0).
    tiny = [_mk_rec(9, i, "a", [0, 0, 5, 5],
                    np.ones((5, 5), bool), z_t, 0.9 - i * 0.1, [])
            for i in range(5)]
    assert process_image(tiny, []) is None

    print("[self-test] all 9 assertion groups passed")
    return True


# ============================ main ==========================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--self-test", action="store_true",
                    help="run only the synthetic self-test")
    args = ap.parse_args()

    assert self_test(), "self-test failed"
    if args.self_test:
        return

    by_img = load_candidates(args.candidates)
    integrity = []
    target_records = []
    excluded = []
    for img_id in sorted(by_img):
        recs = by_img[img_id]
        out = process_image(recs, integrity)
        if out is None:
            excluded.append({"img_id": img_id, "n_cands": len(recs)})
        else:
            target_records.extend(out)

    included_counts = [len(by_img[t["img_id"]]) for t in
                       {t["img_id"]: t for t in target_records}.values()]
    cuts, bin_of = density_bins(included_counts)
    for t in target_records:
        t["density_bin"] = bin_of(t["img_n_cands"])

    summary = summarize(target_records, len(set(t["img_id"] for t in
                                                target_records)))
    by_class = stratify(target_records, lambda t: t["cat_name"])
    by_density = stratify(target_records, lambda t: f"Q{t['density_bin']}")

    integ_total = {
        "images_checked": len(integrity),
        "rival_pairs": sum(i["rival_pairs"] for i in integrity),
        "recomputed_ge_0p5": sum(i["recomputed_ge_0p5"] for i in integrity),
        "missed_by_rivals": sum(i["missed_by_rivals"] for i in integrity),
        "unsupported_by_recompute": sum(i["unsupported_by_recompute"]
                                        for i in integrity),
    }

    result = {
        "meta": {
            "script": "sam3_ovis/eval/r1_selection_divergence.py",
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "candidates": args.candidates,
            "rules": {
                "targets": "per-image top-8 by raw_score (tie: write idx asc)",
                "pool": "same-image non-target candidates",
                "exclusion": f"images with pool < {MIN_POOL} (n < "
                             f"{MIN_POOL + N_TARGETS_PER_IMG})",
                "U1": f"K nearest by bbox-center Euclidean distance",
                "U2": "K most cosine-similar 15-dim z (CLIP class cosines)",
                "R1_overlap": f"cross-class mask IoU >= {OVERLAP_MASK_IOU}",
                "R1_adjacent": f"bbox IoU in ({ADJ_IOU_LO}, {ADJ_IOU_HI})",
                "R1_ranking": "overlap (mask IoU desc) then adjacent "
                              "(bbox IoU desc); tie: write idx asc; no pad",
                "K": list(KS),
                "jaccard_empty_r1": 0.0,
            },
        },
        "data": {
            "n_images_manifest": None,  # filled below if manifest readable
            "n_images_with_candidates": len(by_img),
            "n_candidates": sum(len(v) for v in by_img.values()),
            "n_images_included": summary["n_images_included"],
            "n_targets": len(target_records),
            "excluded_images": excluded,
            "density_quartile_cuts": cuts,
        },
        "integrity_cross_check": integ_total,
        "overall": summary,
        "by_class": by_class,
        "by_density": by_density,
        "targets": target_records,
    }
    try:
        import pathlib
        mf = pathlib.Path(args.candidates).parent / "manifest.json"
        result["data"]["n_images_manifest"] = json.loads(
            mf.read_text())["n_images"]
    except Exception:
        pass

    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False)

    # console digest
    print(f"[data] images with candidates: {len(by_img)} "
          f"(manifest {result['data']['n_images_manifest']}), "
          f"candidates: {result['data']['n_candidates']}, "
          f"included images: {summary['n_images_included']}, "
          f"targets: {len(target_records)}, excluded images: {len(excluded)}")
    print(f"[integrity] {integ_total}")
    hdr = f"{'K':>3} {'U':>3} {'jac_med':>8} {'jac_mean':>9} {'jac_p25':>8} " \
          f"{'jac_p75':>8} {'jac_p90':>8} {'r1only_mean':>11} " \
          f"{'r1only_med':>10} {'shortfall':>9} {'empty':>6}"
    print(hdr)
    for k in (str(x) for x in KS):
        for u in ("U1", "U2"):
            s = summary[k][u]
            j, r = s["jaccard"], s["r1_only_share"]
            print(f"{k:>3} {u:>3} {j['median']:8.3f} {j['mean']:9.3f} "
                  f"{j['p25']:8.3f} {j['p75']:8.3f} {j['p90']:8.3f} "
                  f"{r['mean']:11.3f} {r['median']:10.3f} "
                  f"{s['r1_shortfall_rate']:9.3f} "
                  f"{s['r1_empty_rate']:6.3f}")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
