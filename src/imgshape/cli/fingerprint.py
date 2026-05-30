import click
import sys
import json
from pathlib import Path
from typing import Any
from imgshape.atlas import Atlas
from imgshape.audit import FingerprintLock


def render_fingerprint_text(fingerprint: Any) -> str:
    """Format the dataset fingerprint as clean, structured, human-readable text."""

    def get_val(obj, *keys):
        curr = obj
        for k in keys:
            if curr is None:
                return None
            if isinstance(curr, dict):
                curr = curr.get(k)
            else:
                curr = getattr(curr, k, None)
        return curr

    lines = []
    lines.append("=" * 60)
    lines.append("                IMGSHAPE DATASET FINGERPRINT")
    lines.append("=" * 60)

    uri = getattr(fingerprint, "dataset_uri", "")
    schema = getattr(fingerprint, "schema_version", "")
    cls_name = getattr(fingerprint, "derived_class", "")
    conf = getattr(fingerprint, "confidence", 0.0)

    lines.append(f"Dataset URI:       {uri}")
    lines.append(f"Schema Version:    {schema}")
    lines.append(f"Derived Class:     {cls_name}")
    lines.append(f"Confidence:        {conf:.4f}")
    lines.append("-" * 60)

    metadata = getattr(fingerprint, "metadata", {}) or {}
    lines.append("Metadata:")
    for k, v in metadata.items():
        lines.append(f"  • {k}: {v}")

    lines.append("-" * 60)
    lines.append("Profiles Summary:")

    profiles = getattr(fingerprint, "profiles", {}) or {}

    # Spatial
    spatial = profiles.get("spatial", {})
    if spatial:
        geo = get_val(spatial, "geometry_class")
        risk = get_val(spatial, "resize_risk")
        lines.append(f"  • Spatial:  Geometry Class: {geo}, Resize Risk: {risk}")

    # Signal
    signal = profiles.get("signal", {})
    if signal:
        ent_mean = get_val(signal, "entropy", "mean")
        noise = get_val(signal, "noise_estimate")
        lines.append(f"  • Signal:   Entropy Mean: {ent_mean}, Noise Estimate: {noise}")

    # Quality
    quality = profiles.get("quality", {})
    if quality:
        oq = get_val(quality, "overall_quality")
        corr = get_val(quality, "corruption_rate")
        lines.append(f"  • Quality:  Overall Quality: {oq}, Corruption Rate: {corr}")

    # Semantic
    semantic = profiles.get("semantic", {})
    if semantic:
        domain = get_val(semantic, "primary_domain")
        ctype = get_val(semantic, "content_type")
        lines.append(f"  • Semantic: Primary Domain: {domain}, Content Type: {ctype}")

    lines.append("=" * 60)
    return "\n".join(lines)


@click.command(name="fingerprint")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
@click.option(
    "--lock",
    is_flag=True,
    help="Create a fingerprint lock file at the default location.",
)
@click.option(
    "--save", type=click.Path(writable=True), help="Path to write the fingerprint file."
)
def fingerprint(dataset_path: str, output: str, lock: bool, save: str) -> None:
    """Extract and serialize a dataset fingerprint."""
    # 1. Extract fingerprint
    atlas = Atlas()
    try:
        fp_obj = atlas.extract(dataset_path)
    except Exception as e:
        click.echo(
            click.style(f"Error: Fingerprint extraction failed: {e}", fg="red"),
            err=True,
        )
        sys.exit(1)

    # 2. Serialize/Format
    if output == "json":
        output_str = json.dumps(fp_obj.to_dict(), indent=2)
    else:
        output_str = render_fingerprint_text(fp_obj)

    # 3. Echo output
    click.echo(output_str)

    # 4. Save output if requested
    if save:
        try:
            save_path = Path(save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(output_str)
        except Exception as e:
            click.echo(
                click.style(
                    f"Warning: Failed to save output to {save}: {e}", fg="yellow"
                ),
                err=True,
            )

    # 5. Lock if requested
    if lock:
        try:
            ds_path = Path(dataset_path)
            if ds_path.is_file():
                lock_dir = ds_path.parent
            else:
                lock_dir = ds_path

            lock_file_path = lock_dir / ".fingerprint_lock"
            contract_file_path = lock_dir / "contract.yaml"

            FingerprintLock.write_lock(
                contract_path=contract_file_path,
                fingerprint=fp_obj,
                report_passed=True,
                lock_path=lock_file_path,
            )
        except Exception as e:
            click.echo(
                click.style(f"Warning: Failed to write lock file: {e}", fg="yellow"),
                err=True,
            )
