"""Types used by submission verifiers."""
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class Span:
    """Where in the transcript the misbehavior is.

    call:     index into rollout["calls"] (0-based)
    location: which text inside that call, one of
              "output.content", "output.reasoning_content", "output.tool_calls.<i>.arguments", "messages.<i>.content"
    start/end: optional character range inside that text (end exclusive); leave both out to point at the whole text
    text:     optional copy of the quoted text
    """
    call: int
    location: str = "output.content"
    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = None


@dataclass
class Verdict:
    """What a verifier returns for one rollout."""
    misbehavior: bool                       # True if this rollout contains the misbehavior
    explanation: str = ""                   # short description of what the model did and why it counts (or why not)
    spans: List[Span] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(misbehavior=bool(d["misbehavior"]), explanation=d.get("explanation", ""), spans=[Span(**s) for s in d.get("spans", [])])
