"""Registered v0 thresholds for the evidence graph (S3.2).

Single source of truth for edge weights/triggers. Every value mirrors
sam3_ovis/docs/graph-thresholds.md (registered 2026-09-03; do not modify —
ablations live in S3.5, design changes go through the iteration rule).

All weights are within [0, 1]; context/scene act as soft scores only
(design red line 5).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphThresholds:
    # score normalisation (§1)
    pool_floor: float = 0.05          # candidate pool floor (P0 registration)

    # text-instance (§2)
    w_text_sam3: float = 0.5
    w_text_clip: float = 0.5

    # same-class support (§3)
    theta_s: float = 0.70             # appearance arm trigger
    scale_k: float = 3.0              # scale arm: ratio < k, weight 1 - ratio/k

    # competing class (§4)
    compete_topk: int = 5
    theta_c: float = 0.50             # class-name text-cos floor
    theta_o: float = 0.30             # cross-class candidate IoU trigger

    # instance-local context (§5)
    band_ratio: float = 0.25          # ring radius = ratio * sqrt(area_px)
    prior_alpha: float = 1.0          # Laplace smoothing of P(j|i)
    prior_floor: float = 0.01
    context_conf_floor: float = 0.5   # norm_text floor for context evidence

    # unknown conflict (§7)
    theta_u_text: float = 0.20


THRESHOLDS = GraphThresholds()

# §6 scene phrase list (registered; RemoteCLIP text side)
SCENE_PHRASES = [
    "airport", "harbor", "parking lot", "residential area", "industrial area",
    "farmland", "forest", "river", "city", "port", "desert", "overpass",
]


def norm_text(score: float, pool_floor: float = THRESHOLDS.pool_floor) -> float:
    """SAM3 score -> [0,1] linear rescale from the registered pool floor."""
    return max(0.0, min(1.0, (score - pool_floor) / (1.0 - pool_floor)))


def cos01(cos: float) -> float:
    """cosine [-1,1] -> [0,1]."""
    return (cos + 1.0) / 2.0


@dataclass(frozen=True)
class CompletenessThresholds:
    """S3.3 completeness metrics (docs/graph-thresholds.md §10).

    v0 weights are hand-set equal thirds (P1 is training-free; no tuning).
    The high-confidence floor REUSES the registered S3.2
    `context_conf_floor` (0.5); the duplicate/match IoU reuses the 0.5
    convention from the evaluation/duplicate machinery.
    """

    w_unexplained_area: float = 1.0 / 3       # (1 - ECR) arm
    w_unexplained_response: float = 1.0 / 3   # URD arm
    w_conflict: float = 1.0 / 3               # CCR arm

    @property
    def tau_high(self) -> float:
        return THRESHOLDS.context_conf_floor   # 0.5, registered S3.2

    @property
    def dup_iou(self) -> float:
        return 0.5


COMPLETENESS = CompletenessThresholds()


@dataclass(frozen=True)
class MessagePassingThresholds:
    """S3.4 message passing + merge (docs/graph-thresholds.md §11)."""

    rounds: int = 2
    alpha_support: float = 0.15
    alpha_compete: float = 0.30
    alpha_context: float = 0.10
    alpha_scene: float = 0.05
    lambda_risk: float = 0.5
    theta_alarm: float = 0.5
    merge_iou: float = 0.5          # cluster pairs must overlap (dup_iou)


MP = MessagePassingThresholds()
