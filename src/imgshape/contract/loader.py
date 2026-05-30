import yaml
from pathlib import Path
from typing import Union, Any
from imgshape.contract.clauses import (
    ContractClauses,
    DatasetClauses,
    QualityClauses,
    DistributionClauses,
)


class ContractSchemaError(Exception):
    """Exception raised for syntax or structural errors in contract schema."""

    pass


class ContractLoader:
    @staticmethod
    def _parse_clause(
        section: dict, key: str, expected_type: Any, clause_path: str, severities: dict
    ) -> Any:
        val = section.get(key)
        if val is None:
            return None

        severity = "error"
        if isinstance(val, dict) and "value" in val:
            severity = val.get("severity", "error")
            if severity not in ("error", "warning"):
                raise ContractSchemaError(
                    f"Severity for {clause_path} must be 'error' or 'warning', got '{severity}'"
                )
            val = val["value"]

        if val is None:
            return None

        if expected_type == int:
            if not isinstance(val, int) or isinstance(val, bool):
                raise ContractSchemaError(
                    f"{clause_path} must be an integer, got {type(val).__name__}"
                )
        elif expected_type == float:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ContractSchemaError(
                    f"{clause_path} must be a number, got {type(val).__name__}"
                )
            val = float(val)
        elif expected_type == str:
            if not isinstance(val, str):
                raise ContractSchemaError(
                    f"{clause_path} must be a string, got {type(val).__name__}"
                )
        elif expected_type == list:
            if not isinstance(val, list):
                raise ContractSchemaError(
                    f"{clause_path} must be a list, got {type(val).__name__}"
                )
        elif expected_type == "list_str":
            if not isinstance(val, list):
                raise ContractSchemaError(
                    f"{clause_path} must be a list, got {type(val).__name__}"
                )
            for item in val:
                if not isinstance(item, str):
                    raise ContractSchemaError(
                        f"{clause_path} elements must be strings, got {type(item).__name__}"
                    )
        elif expected_type == "resolution":
            if not isinstance(val, (list, tuple)):
                raise ContractSchemaError(
                    f"{clause_path} must be a list/tuple, got {type(val).__name__}"
                )
            if len(val) != 2 or not all(
                isinstance(x, int) and not isinstance(x, bool) for x in val
            ):
                raise ContractSchemaError(
                    f"{clause_path} must contain exactly 2 integers [width, height]"
                )
            val = tuple(val)

        severities[clause_path] = severity
        return val

    @staticmethod
    def load(path: Union[str, Path]) -> ContractClauses:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Contract file not found at: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ContractSchemaError(f"YAML syntax error in contract file: {e}") from e
        except Exception as e:
            raise ContractSchemaError(f"Failed to read contract file: {e}") from e

        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise ContractSchemaError(
                "Contract file must contain a top-level dictionary"
            )

        # Validate top-level keys
        allowed_top_keys = {"dataset", "quality", "distribution"}
        for k in data:
            if k not in allowed_top_keys:
                raise ContractSchemaError(
                    f"Invalid top-level key: '{k}'. Allowed keys: {list(allowed_top_keys)}"
                )

        severities = {}

        # Parse and validate 'dataset' section
        dataset_section = data.get("dataset")
        if dataset_section is None:
            dataset_section = {}
        elif not isinstance(dataset_section, dict):
            raise ContractSchemaError("Section 'dataset' must be a dictionary")

        allowed_dataset_keys = {
            "expected_channels",
            "allowed_formats",
            "resolution_min",
            "resolution_max",
        }
        for k in dataset_section:
            if k not in allowed_dataset_keys:
                allowed_list = sorted(list(allowed_dataset_keys))
                raise ContractSchemaError(
                    f"Invalid key under 'dataset': '{k}'. Allowed: {allowed_list}"
                )

        expected_channels = ContractLoader._parse_clause(
            dataset_section,
            "expected_channels",
            int,
            "dataset.expected_channels",
            severities,
        )
        allowed_formats = ContractLoader._parse_clause(
            dataset_section,
            "allowed_formats",
            "list_str",
            "dataset.allowed_formats",
            severities,
        )
        resolution_min = ContractLoader._parse_clause(
            dataset_section,
            "resolution_min",
            "resolution",
            "dataset.resolution_min",
            severities,
        )
        resolution_max = ContractLoader._parse_clause(
            dataset_section,
            "resolution_max",
            "resolution",
            "dataset.resolution_max",
            severities,
        )

        dataset_clauses = DatasetClauses(
            expected_channels=expected_channels,
            allowed_formats=allowed_formats,
            resolution_min=resolution_min,
            resolution_max=resolution_max,
        )

        # Parse and validate 'quality' section
        quality_section = data.get("quality")
        if quality_section is None:
            quality_section = {}
        elif not isinstance(quality_section, dict):
            raise ContractSchemaError("Section 'quality' must be a dictionary")

        allowed_quality_keys = {"corruption_max", "duplicate_max", "blur_threshold"}
        for k in quality_section:
            if k not in allowed_quality_keys:
                allowed_list = sorted(list(allowed_quality_keys))
                raise ContractSchemaError(
                    f"Invalid key under 'quality': '{k}'. Allowed: {allowed_list}"
                )

        corruption_max = ContractLoader._parse_clause(
            quality_section,
            "corruption_max",
            float,
            "quality.corruption_max",
            severities,
        )
        duplicate_max = ContractLoader._parse_clause(
            quality_section, "duplicate_max", float, "quality.duplicate_max", severities
        )
        blur_threshold = ContractLoader._parse_clause(
            quality_section,
            "blur_threshold",
            float,
            "quality.blur_threshold",
            severities,
        )

        quality_clauses = QualityClauses(
            corruption_max=corruption_max,
            duplicate_max=duplicate_max,
            blur_threshold=blur_threshold,
        )

        # Parse and validate 'distribution' section
        distribution_section = data.get("distribution")
        if distribution_section is None:
            distribution_section = {}
        elif not isinstance(distribution_section, dict):
            raise ContractSchemaError("Section 'distribution' must be a dictionary")

        allowed_distribution_keys = {
            "entropy_min",
            "entropy_max",
            "channel_consistency",
        }
        for k in distribution_section:
            if k not in allowed_distribution_keys:
                allowed_list = sorted(list(allowed_distribution_keys))
                raise ContractSchemaError(
                    f"Invalid key under 'distribution': '{k}'. Allowed: {allowed_list}"
                )

        entropy_min = ContractLoader._parse_clause(
            distribution_section,
            "entropy_min",
            float,
            "distribution.entropy_min",
            severities,
        )
        entropy_max = ContractLoader._parse_clause(
            distribution_section,
            "entropy_max",
            float,
            "distribution.entropy_max",
            severities,
        )
        channel_consistency = ContractLoader._parse_clause(
            distribution_section,
            "channel_consistency",
            str,
            "distribution.channel_consistency",
            severities,
        )

        distribution_clauses = DistributionClauses(
            entropy_min=entropy_min,
            entropy_max=entropy_max,
            channel_consistency=channel_consistency,
        )

        # Contradiction checks
        if resolution_min is not None and resolution_max is not None:
            if (
                resolution_min[0] > resolution_max[0]
                or resolution_min[1] > resolution_max[1]
            ):
                raise ContractSchemaError(
                    f"Contradictory resolution constraints: resolution_min {resolution_min} "
                    f"cannot be greater than resolution_max {resolution_max}"
                )

        if entropy_min is not None and entropy_max is not None:
            if entropy_min > entropy_max:
                raise ContractSchemaError(
                    f"Contradictory entropy constraints: distribution.entropy_min {entropy_min} "
                    f"cannot be greater than distribution.entropy_max {entropy_max}"
                )

        return ContractClauses(
            dataset=dataset_clauses,
            quality=quality_clauses,
            distribution=distribution_clauses,
            severities=severities,
        )
