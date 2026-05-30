from imgshape.contract.violations import ViolationReport
from imgshape.test.report import TestReport
from imgshape.diff.report import DiffReport


def render_validation_report(report: ViolationReport) -> str:
    """Render a validation report to rich Markdown."""
    lines = []
    lines.append("# imgshape Contract Validation Report")
    lines.append("")

    if report.passed:
        lines.append("> [!NOTE]")
        lines.append("> **VALIDATION PASSED**")
        lines.append(
            "> All contract clauses were successfully satisfied by the dataset fingerprint."
        )
    else:
        lines.append("> [!CAUTION]")
        lines.append("> **VALIDATION FAILED**")
        num_v = len(report.violations)
        lines.append(
            f"> The dataset fingerprint failed contract validation with {num_v} violation(s)."
        )

    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| **Dataset URI** | `{report.fingerprint_id}` |")
    lines.append(f"| **Provenance ID** | `{report.provenance_id}` |")
    lines.append(f"| **Timestamp** | {report.timestamp} |")
    lines.append("")

    if not report.passed:
        lines.append("## Violations")
        lines.append("")
        lines.append("| # | Clause | Severity | Expected | Actual | Details |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for i, v in enumerate(report.violations, 1):
            expected_esc = str(v.expected).replace("|", "\\|")
            actual_esc = str(v.actual).replace("|", "\\|")
            detail_esc = str(v.detail).replace("|", "\\|")
            row = (
                f"| {i} | `{v.clause}` | **{v.severity.upper()}** | "
                f"`{expected_esc}` | `{actual_esc}` | {detail_esc} |"
            )
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def render_test_report(report: TestReport) -> str:
    """Render a test report to rich Markdown."""
    lines = []
    lines.append("# imgshape Dataset Test Report")
    lines.append("")

    passed_count = sum(1 for r in report.results if r["status"] == "PASS")
    warn_count = sum(1 for r in report.results if r["status"] == "WARN")
    fail_count = sum(1 for r in report.results if r["status"] == "FAIL")

    if report.passed:
        if warn_count > 0:
            lines.append("> [!WARNING]")
            lines.append("> **TEST RUN PASSED WITH WARNINGS**")
            lines.append(
                f"> All critical tests passed, but there were {warn_count} warning(s)."
            )
        else:
            lines.append("> [!NOTE]")
            lines.append("> **TEST RUN PASSED**")
            lines.append(
                "> All dataset quality and distribution assertions passed successfully."
            )
    else:
        lines.append("> [!CAUTION]")
        lines.append("> **TEST RUN FAILED**")
        lines.append(f"> Dataset tests failed with {fail_count} hard failure(s).")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("| --- | --- |")
    lines.append(f"| **Total Tests** | {len(report.results)} |")
    lines.append(f"| **Passed** | {passed_count} |")
    lines.append(f"| **Warnings** | {warn_count} |")
    lines.append(f"| **Failed** | {fail_count} |")
    lines.append("")

    lines.append("## Test Details")
    lines.append("")
    lines.append("| Test Case | Status | Expected | Actual | Message |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in report.results:
        status_str = f"**{r['status']}**"
        expected_esc = str(r.get("expected", "")).replace("|", "\\|")
        actual_esc = str(r.get("actual", "")).replace("|", "\\|")
        msg_esc = str(r.get("message", "")).replace("|", "\\|")
        lines.append(
            f"| `{r['name']}` | {status_str} | `{expected_esc}` | `{actual_esc}` | {msg_esc} |"
        )
    lines.append("")

    return "\n".join(lines)


def render_diff_report(report: DiffReport) -> str:
    """Render a diff report to rich Markdown."""
    lines = []
    lines.append("# imgshape Dataset Diff Report")
    lines.append("")

    if report.is_significant:
        lines.append("> [!WARNING]")
        lines.append("> **SIGNIFICANT DRIFT DETECTED**")
        lines.append(
            f"> The overall dataset drift ({report.overall_drift:.4f}) exceeded the "
            "threshold, suggesting a significant change in the dataset."
        )
    else:
        lines.append("> [!NOTE]")
        lines.append("> **DATASETS STABLE / NO SIGNIFICANT DRIFT**")
        lines.append(
            f"> The overall dataset drift ({report.overall_drift:.4f}) is within acceptable limits."
        )

    lines.append("")
    lines.append("## Datasets Compared")
    lines.append("")
    lines.append(f"- **Baseline Dataset:** `{report.baseline_uri}`")
    lines.append(f"- **Candidate Dataset:** `{report.current_uri}`")
    lines.append("")

    lines.append("## Drift Metrics Summary")
    lines.append("")
    lines.append("| Metric | Score / Value |")
    lines.append("| --- | --- |")
    lines.append(f"| **Similarity Score** | {report.similarity_score:.4f} |")
    lines.append(f"| **Overall Drift** | {report.overall_drift:.4f} |")
    lines.append(f"| **Spatial Drift** | {report.spatial_drift:.4f} |")
    lines.append(f"| **Signal Drift** | {report.signal_drift:.4f} |")
    lines.append(f"| **Semantic Drift** | {report.semantic_drift:.4f} |")
    lines.append("")

    lines.append("## Rationale")
    lines.append("")
    for item in report.rationale:
        lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines)
