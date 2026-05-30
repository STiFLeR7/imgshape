import click
import sys
from pathlib import Path
from imgshape.atlas import Atlas
from imgshape.contract import Contract
from imgshape.audit import FingerprintLock
from imgshape.output import render_validation_report


@click.command(name="validate")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("contract_path", type=click.Path(exists=True))
@click.option(
    "--output",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format.",
)
@click.option(
    "--save",
    type=click.Path(writable=True),
    help="Path to write the validation report.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Treat warnings as failures (exiting with code 1 instead of 2).",
)
@click.option(
    "--lock",
    is_flag=True,
    help="Write a fingerprint lock file next to the contract on a PASS verdict.",
)
def validate(
    dataset_path: str,
    contract_path: str,
    output: str,
    save: str,
    strict: bool,
    lock: bool,
) -> None:
    """Validate a dataset fingerprint against contract clauses."""
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

    # 2. Load contract
    try:
        contract = Contract.load(contract_path)
    except Exception as e:
        click.echo(
            click.style(f"Error: Failed to load contract: {e}", fg="red"), err=True
        )
        sys.exit(1)

    # 3. Perform Validation
    report = contract.validate(fingerprint)

    # 4. Render and optionally Save output
    output_str = render_validation_report(report, format=output, save_path=save)
    click.echo(output_str)

    # 5. Lock file if PASS and --lock is requested
    if report.passed and lock:
        try:
            FingerprintLock.write_lock(
                contract_path=Path(contract_path),
                fingerprint=fingerprint,
                report_passed=True,
            )
        except Exception as e:
            click.echo(
                click.style(f"Warning: Failed to write lock file: {e}", fg="yellow"),
                err=True,
            )

    # 6. Set appropriate exit code
    # Find errors among violations
    errors = [v for v in report.violations if v.severity.lower() == "error"]

    # Non-empty violations list implies failure if passed is False
    # If passed is True, we have no violations of error type.
    if not report.passed:
        # If there are hard errors or strict mode is enabled, exit 1
        if errors or strict:
            sys.exit(1)
        else:
            sys.exit(2)
    else:
        sys.exit(0)
