from dataclasses import dataclass
from typing import Any, List


@dataclass
class Violation:
    clause: str
    expected: Any
    actual: Any
    severity: str
    detail: str


@dataclass
class ViolationReport:
    passed: bool
    violations: List[Violation]
    fingerprint_id: str
    provenance_id: str
    timestamp: str
