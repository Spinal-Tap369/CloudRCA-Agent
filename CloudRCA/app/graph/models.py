from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, order=True)
class NodeKey:
    kind: str
    name: str
    namespace: str = ""

    @property
    def id(self) -> str:
        ns = self.namespace or "_"
        return f"{self.kind}:{ns}:{self.name}"


@dataclass
class EvidenceRef:
    path: str
    category: str
    summary: str = ""


@dataclass
class GraphNode:
    key: NodeKey
    labels: dict[str, str] = field(default_factory=dict)

    affected_score: int = 0
    candidate_score: int = 0

    hypothesis_scores: dict[str, int] = field(default_factory=dict)
    evidence: list[EvidenceRef] = field(default_factory=list)
    signals: set[str] = field(default_factory=set)
    reasons: set[str] = field(default_factory=set)

    @property
    def kind(self) -> str:
        return self.key.kind

    @property
    def name(self) -> str:
        return self.key.name

    @property
    def namespace(self) -> str:
        return self.key.namespace

    @property
    def best_hypothesis(self) -> str:
        if not self.hypothesis_scores:
            return "unknown"
        return max(self.hypothesis_scores.items(), key=lambda item: item[1])[0]

    @property
    def evidence_paths(self) -> list[str]:
        return sorted({item.path for item in self.evidence})


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    evidence_path: str | None = None


@dataclass
class EvidenceFile:
    relative_path: str
    absolute_path: Path
    category: str
    text: str


@dataclass
class RootCandidate:
    name: str
    kind: str
    namespace: str
    score: int
    best_hypothesis: str
    reasons: list[str]
    evidence_paths: list[str]


@dataclass
class IncidentGraph:
    scenario_path: Path
    files_seen: int = 0

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    global_scores: dict[str, int] = field(default_factory=dict)
    signals: dict[str, object] = field(default_factory=dict)

    candidates: list[RootCandidate] = field(default_factory=list)
    affected_nodes: list[GraphNode] = field(default_factory=list)
