from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TestReport:
    __test__ = False
    passed: bool
    results: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        lines.append("=========================================")
        lines.append("         IMGSHAPE TEST SUMMARY           ")
        lines.append("=========================================")
        status_str = "PASS" if self.passed else "FAIL"
        lines.append(f"Overall Result: {status_str}")
        lines.append("-----------------------------------------")

        passed_count = sum(1 for r in self.results if r["status"] == "PASS")
        warn_count = sum(1 for r in self.results if r["status"] == "WARN")
        fail_count = sum(1 for r in self.results if r["status"] == "FAIL")

        lines.append(f"Total Tests: {len(self.results)}")
        lines.append(f"Passed:      {passed_count}")
        lines.append(f"Warnings:    {warn_count}")
        lines.append(f"Failed:      {fail_count}")
        lines.append("-----------------------------------------")
        lines.append("Individual Test Results:")
        for r in self.results:
            lines.append(f"  [{r['status']}] {r['name']}: {r['message']}")
        lines.append("=========================================")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "results": self.results}
