import click
import sys
from imgshape.atlas import Atlas
from imgshape.test import TestRunner
from imgshape.output import render_test_report


@click.command(name="test")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option(
    "--only",
    type=click.Choice(["quality", "distribution"]),
    default=None,
    help="Run only the specified test suite.",
)
@click.option(
    "--output",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format.",
)
@click.option(
    "--save", type=click.Path(writable=True), help="Path to write the test report."
)
def test(dataset_path: str, only: str, output: str, save: str) -> None:
    """Run assertions/tests on a dataset."""
    # 1. Extract fingerprint
    atlas = Atlas()
    try:
        fingerprint = atlas.extract(dataset_path)
    except Exception as e:
        click.echo(
            click.style(f"Error: Fingerprint extraction failed: {e}", fg="red"),
            err=True,
        )
        sys.exit(1)

    # 2. Run TestRunner
    try:
        report = TestRunner.run(fingerprint, only_suite=only)
    except Exception as e:
        click.echo(click.style(f"Error: Test runner failed: {e}", fg="red"), err=True)
        sys.exit(1)

    # 3. Render and optionally Save output
    output_str = render_test_report(report, format=output, save_path=save)
    click.echo(output_str)

    # 4. Exits with code:
    # 0: passed
    # 1: failed (at least one FAIL status)
    # 2: warnings present (no hard failures, but at least one WARN status)
    statuses = [r["status"] for r in report.results]
    if "FAIL" in statuses:
        sys.exit(1)
    elif "WARN" in statuses:
        sys.exit(2)
    else:
        sys.exit(0)
