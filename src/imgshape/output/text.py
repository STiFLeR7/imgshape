import click
from imgshape.contract.violations import ViolationReport
from imgshape.test.report import TestReport
from imgshape.diff.report import DiffReport


def render_validation_report(report: ViolationReport) -> str:
    """Format and colorize validation reports for console display."""
    lines = []
    lines.append(click.style("=" * 60, fg="cyan"))
    lines.append(
        click.style(
            "         IMGSHAPE CONTRACT VALIDATION REPORT", bold=True, fg="cyan"
        )
    )
    lines.append(click.style("=" * 60, fg="cyan"))

    status_str = (
        click.style("PASS", fg="green", bold=True)
        if report.passed
        else click.style("FAIL", fg="red", bold=True)
    )
    lines.append(f"Overall Verdict: {status_str}")
    lines.append(f"Dataset URI:     {report.fingerprint_id}")
    lines.append(f"Provenance ID:   {report.provenance_id}")
    lines.append(f"Timestamp:       {report.timestamp}")
    lines.append(click.style("-" * 60, fg="cyan"))

    if report.passed:
        lines.append(
            click.style("✓ All contract clauses satisfied successfully.", fg="green")
        )
    else:
        lines.append(
            click.style(
                f"Violations ({len(report.violations)} total):", fg="red", bold=True
            )
        )
        for i, violation in enumerate(report.violations, 1):
            sev_color = "red" if violation.severity.lower() == "error" else "yellow"
            sev_str = click.style(violation.severity.upper(), fg=sev_color, bold=True)
            lines.append(
                f"  {i}. [{sev_str}] Clause: "
                + click.style(violation.clause, fg="yellow", bold=True)
            )
            lines.append(f"     Expected: {violation.expected}")
            lines.append(f"     Actual:   {violation.actual}")
            lines.append(f"     Details:  {violation.detail}")
            lines.append("")

    lines.append(click.style("=" * 60, fg="cyan"))
    return "\n".join(lines)


def render_test_report(report: TestReport) -> str:
    """Format and colorize test reports for console display."""
    lines = []
    lines.append(click.style("=" * 60, fg="magenta"))
    lines.append(
        click.style(
            "             IMGSHAPE DATASET TEST REPORT", bold=True, fg="magenta"
        )
    )
    lines.append(click.style("=" * 60, fg="magenta"))

    status_str = (
        click.style("PASS", fg="green", bold=True)
        if report.passed
        else click.style("FAIL", fg="red", bold=True)
    )
    lines.append(f"Overall Verdict: {status_str}")
    lines.append(click.style("-" * 60, fg="magenta"))

    passed_count = sum(1 for r in report.results if r["status"] == "PASS")
    warn_count = sum(1 for r in report.results if r["status"] == "WARN")
    fail_count = sum(1 for r in report.results if r["status"] == "FAIL")

    lines.append(f"Total Tests: {len(report.results)}")
    lines.append(click.style(f"Passed:      {passed_count}", fg="green"))
    lines.append(click.style(f"Warnings:    {warn_count}", fg="yellow"))
    lines.append(click.style(f"Failed:      {fail_count}", fg="red"))
    lines.append(click.style("-" * 60, fg="magenta"))

    lines.append("Individual Test Results:")
    for r in report.results:
        status_val = r["status"]
        if status_val == "PASS":
            status_style = click.style("PASS", fg="green", bold=True)
        elif status_val == "WARN":
            status_style = click.style("WARN", fg="yellow", bold=True)
        else:
            status_style = click.style("FAIL", fg="red", bold=True)

        lines.append(
            f"  [{status_style}] "
            + click.style(r["name"], bold=True)
            + f": {r['message']}"
        )
        if status_val != "PASS":
            lines.append(f"         Expected: {r.get('expected')}")
            lines.append(f"         Actual:   {r.get('actual')}")

    lines.append(click.style("=" * 60, fg="magenta"))
    return "\n".join(lines)


def render_diff_report(report: DiffReport) -> str:
    """Format and colorize diff reports for console display."""
    lines = []
    lines.append(click.style("=" * 60, fg="blue"))
    lines.append(
        click.style("             IMGSHAPE DATASET DIFF REPORT", bold=True, fg="blue")
    )
    lines.append(click.style("=" * 60, fg="blue"))

    lines.append(f"Baseline Dataset:  {report.baseline_uri}")
    lines.append(f"Candidate Dataset: {report.current_uri}")
    lines.append(click.style("-" * 60, fg="blue"))

    drift_color = "red" if report.is_significant else "green"
    verdict_str = (
        click.style("SIGNIFICANT DRIFT DETECTED", fg="red", bold=True)
        if report.is_significant
        else click.style("STABLE / NO SIGNIFICANT DRIFT", fg="green", bold=True)
    )
    lines.append(f"Drift Verdict:    {verdict_str}")
    lines.append(
        f"Similarity Score: {click.style(f'{report.similarity_score:.4f}', bold=True)}"
    )
    lines.append(
        f"Overall Drift:    {click.style(f'{report.overall_drift:.4f}', fg=drift_color, bold=True)}"
    )

    lines.append(click.style("-" * 60, fg="blue"))
    lines.append("Drift Metrics Breakdown:")
    lines.append(f"  • Spatial Drift:  {report.spatial_drift:.4f}")
    lines.append(f"  • Signal Drift:   {report.signal_drift:.4f}")
    lines.append(f"  • Semantic Drift: {report.semantic_drift:.4f}")

    lines.append(click.style("-" * 60, fg="blue"))
    lines.append("Rationale:")
    for item in report.rationale:
        lines.append(f"  • {item}")

    lines.append(click.style("=" * 60, fg="blue"))
    return "\n".join(lines)
