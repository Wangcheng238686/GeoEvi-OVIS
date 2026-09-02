"""Evidence-graph schema (design-v2.md §5.1/§5.2 instantiation, S3.1).

A graph is built per (image, query class); multi-class queries share the
underlying candidate data but get one graph each (S3.1 wording). Node kinds
and edge kinds map 1:1 to design-v2 §5.1 / §5.2; the EXEMPLAR kind is
structurally reserved for P3 and never produced by the P1 builder.

Serialization is plain JSON (to_dict / from_dict) so graphs can be dumped
alongside predictions for offline analysis. No semantic weights live here —
weights are computed by graph/builder.py (S3.2) and message passing is S3.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeKind(str, Enum):
    CATEGORY = "category"            # the open-vocabulary query phrase
    CANDIDATE = "candidate"          # SAM3 instance candidate
    LOCAL_CONTEXT = "local_context"  # ring-band region around one instance
    SCENE = "scene"                  # global scene evidence
    UNKNOWN = "unknown"              # unreliable / conflicting aggregate
    EXEMPLAR = "exemplar"            # P3 only; reserved, not built in P1


class EdgeKind(str, Enum):
    TEXT_INSTANCE = "text_instance"          # §5.2 文本—实例匹配
    SAME_CLASS_SUPPORT = "same_class_support"  # §5.2 同类实例支持
    COMPETING_CLASS = "competing_class"      # §5.2 相似类别竞争
    INSTANCE_CONTEXT = "instance_context"    # §5.2 实例—局部上下文
    INSTANCE_SCENE = "instance_scene"        # §5.2 实例—全局场景（含区域覆盖）
    UNKNOWN_CONFLICT = "unknown_conflict"    # §5.2 实例—unknown 冲突
    EXEMPLAR_SIMILARITY = "exemplar_similarity"  # P3 only; reserved


@dataclass
class Node:
    id: str
    kind: NodeKind
    payload: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # crop/region/text feature (S3.2)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind.value,
                "payload": self.payload, "embedding": self.embedding}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        return cls(id=d["id"], kind=NodeKind(d["kind"]),
                   payload=d.get("payload", {}),
                   embedding=d.get("embedding"))


@dataclass
class Edge:
    src: str
    dst: str
    kind: EdgeKind
    weight: float
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "kind": self.kind.value,
                "weight": self.weight, "attrs": self.attrs}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Edge":
        return cls(src=d["src"], dst=d["dst"], kind=EdgeKind(d["kind"]),
                   weight=float(d["weight"]), attrs=d.get("attrs", {}))


@dataclass
class EvidenceGraph:
    image_id: int
    class_name: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)  # e.g. coverage RLE

    def add_node(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Edge) -> Edge:
        for ref in (edge.src, edge.dst):
            if ref not in self.nodes:
                raise ValueError(f"edge references unknown node {ref!r}")
        if not 0.0 <= edge.weight <= 1.0:
            raise ValueError(f"edge weight must be in [0,1], got {edge.weight}")
        self.edges.append(edge)
        return edge

    def neighbors(self, node_id: str, kind: Optional[EdgeKind] = None):
        out = []
        for e in self.edges:
            if kind is not None and e.kind is not kind:
                continue
            if e.src == node_id:
                out.append((self.nodes[e.dst], e))
            elif e.dst == node_id:
                out.append((self.nodes[e.src], e))
        return out

    def by_kind(self, kind: NodeKind) -> List[Node]:
        return [n for n in self.nodes.values() if n.kind is kind]

    def to_dict(self) -> Dict[str, Any]:
        return {"image_id": self.image_id, "class_name": self.class_name,
                "attrs": self.attrs,
                "nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceGraph":
        g = cls(image_id=d["image_id"], class_name=d["class_name"],
                attrs=d.get("attrs", {}))
        for nd in d.get("nodes", []):
            g.add_node(Node.from_dict(nd))
        for ed in d.get("edges", []):
            g.add_edge(Edge.from_dict(ed))
        return g
