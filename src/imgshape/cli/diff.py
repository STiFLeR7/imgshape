import click
import sys
import json
from pathlib import Path
from typing import Any
from imgshape.atlas import Atlas
from imgshape.diff import DatasetDiff
from imgshape.output import render_diff_report


def load_fingerprint(path_str: str) -> Any:
    """
    Load a fingerprint from a path which can be:
    - A JSON file containing the fingerprint
    - A '.fingerprint_lock' file (a lock file structure)
    - A directory (where we extract the fingerprint dynamically)
    """
    path = Path(path_str)
    if path.is_file() and (
        path_str.endswith(".json") or path_str.endswith(".fingerprint_lock")
    ):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if it has a nested fingerprint key
            if "fingerprint" in data:
                return data["fingerprint"]
            # Check if it looks like a direct fingerprint structure
            elif "profiles" in data:
                return data
            # Check if it looks like a lock file structure
            elif "fingerprint_id" in data:
                return {
                    "dataset_uri": data.get("fingerprint_id", "unknown_dataset"),
                    "profiles": {},
                }
            else:
                return data
        except Exception as e:
            click.echo(
                click.style(
                    f"Error: Failed to parse fingerprint file '{path_str}': {e}",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)
    else:
        # It's a directory, extract fingerprint dynamically
        atlas = Atlas()
        try:
            return atlas.extract(path)
        except Exception as e:
            click.echo(
                click.style(
                    f"Error: Failed to extract fingerprint from '{path_str}': {e}",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)


@click.command(name="diff")
@click.argument("baseline_path", type=click.Path(exists=True))
@click.argument("candidate_path", type=click.Path(exists=True))
@click.option(
    "--output",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format.",
)
@click.option(
    "--semantic", is_flag=True, help="Enable DINOv2 semantic drift detection."
)
@click.option(
    "--threshold", type=float, default=None, help="Drift threshold limit override."
)
@click.option(
    "--save", type=click.Path(writable=True), help="Path to write the diff report."
)
def diff(
    baseline_path: str,
    candidate_path: str,
    output: str,
    semantic: bool,
    threshold: float,
    save: str,
) -> None:
    """Compare baseline and candidate datasets/fingerprints."""
    # 1. Load baseline & candidate fingerprints
    baseline = load_fingerprint(baseline_path)
    candidate = load_fingerprint(candidate_path)

    # 2. Compare using DatasetDiff
    try:
        report = DatasetDiff.compare(
            baseline=baseline, current=candidate, semantic=semantic, threshold=threshold
        )
    except Exception as e:
        click.echo(
            click.style(f"Error: Failed to compare datasets: {e}", fg="red"), err=True
        )
        sys.exit(1)

    # 3. Render and optionally Save output
    output_str = render_diff_report(report, format=output, save_path=save)
    click.echo(output_str)

    # 4. Exit with code 1 if drift is significant
    if report.is_significant:
        sys.exit(1)
    else:
        sys.exit(0)
