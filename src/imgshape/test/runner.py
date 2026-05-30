from typing import Optional, Any
from imgshape.test.assertions import (
    assert_corruption,
    assert_duplicates,
    assert_resolution_min,
    assert_entropy_min,
    assert_channel_consistency,
    assert_formats,
)
from imgshape.test.report import TestReport


class TestRunner:
    @staticmethod
    def run(fingerprint: Any, only_suite: Optional[str] = None) -> TestReport:
        results = []

        run_quality = False
        run_dist = False

        if only_suite == "quality":
            run_quality = True
        elif only_suite == "distribution":
            run_dist = True
        elif only_suite is None:
            run_quality = True
            run_dist = True
        else:
            raise ValueError(f"Unknown suite: {only_suite}")

        if run_quality:
            # assert_corruption
            status, actual, expected, msg = assert_corruption(fingerprint)
            results.append(
                {
                    "name": "assert_corruption",
                    "status": status,
                    "actual": actual,
                    "expected": expected,
                    "message": msg,
                }
            )

            # assert_duplicates
            status, actual, expected, msg = assert_duplicates(fingerprint)
            results.append(
                {
                    "name": "assert_duplicates",
                    "status": status,
                    "actual": actual,
                    "expected": expected,
                    "message": msg,
                }
            )

            # assert_resolution_min
            status, actual, expected, msg = assert_resolution_min(fingerprint)
            results.append(
                {
                    "name": "assert_resolution_min",
                    "status": status,
                    "actual": actual,
                    "expected": expected,
                    "message": msg,
                }
            )

            # assert_formats
            status, actual, expected, msg = assert_formats(fingerprint)
            results.append(
                {
                    "name": "assert_formats",
                    "status": status,
                    "actual": actual,
                    "expected": expected,
                    "message": msg,
                }
            )

        if run_dist:
            # assert_entropy_min
            status, actual, expected, msg = assert_entropy_min(fingerprint)
            results.append(
                {
                    "name": "assert_entropy_min",
                    "status": status,
                    "actual": actual,
                    "expected": expected,
                    "message": msg,
                }
            )

            # assert_channel_consistency
            status, actual, expected, msg = assert_channel_consistency(fingerprint)
            results.append(
                {
                    "name": "assert_channel_consistency",
                    "status": status,
                    "actual": actual,
                    "expected": expected,
                    "message": msg,
                }
            )

        # passed is True if no status is 'FAIL'
        passed = all(r["status"] != "FAIL" for r in results)

        return TestReport(passed=passed, results=results)
