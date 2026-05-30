import click
import logging
from imgshape.cli.validate import validate
from imgshape.cli.test import test
from imgshape.cli.diff import diff
from imgshape.cli.fingerprint import fingerprint


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """imgshape v5 CLI - Dataset validation, testing, and drift analysis."""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose

    # Configure logging
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Also adjust imgshape logger
    logger = logging.getLogger("imgshape")
    logger.setLevel(log_level)


main.add_command(validate)
main.add_command(test)
main.add_command(diff)
main.add_command(fingerprint)

if __name__ == "__main__":
    main()
