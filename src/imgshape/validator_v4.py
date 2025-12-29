"""
imgshape.v4.validator — Schema Validation System

Validates fingerprints, decisions, and artifacts against their JSON schemas.
Ensures data integrity and version compatibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger("imgshape.validator_v4")


class SchemaType(Enum):
    """Available schema types"""
    FINGERPRINT = "fingerprint"
    DECISION = "decision"
    DECISIONS_COLLECTION = "decisions_collection"
    PIPELINE = "pipeline_v4"


class ValidationError(Exception):
    """Schema validation error"""
    pass


class SchemaValidator:
    """
    Validates data structures against JSON schemas.
    
    This validator ensures that all fingerprints, decisions, and artifacts
    conform to their versioned schemas.
    """

    def __init__(self, schemas_dir: Optional[Path] = None):
        """
        Initialize validator.
        
        Args:
            schemas_dir: Directory containing schema files (defaults to package schemas/)
        """
        if schemas_dir is None:
            # Default to package schemas directory
            schemas_dir = Path(__file__).parent.parent / "schemas"
        
        self.schemas_dir = schemas_dir
        self._schemas_cache: Dict[SchemaType, Dict] = {}
        
    def load_schema(self, schema_type: SchemaType) -> Dict[str, Any]:
        """Load a schema from disk"""
        if schema_type in self._schemas_cache:
            return self._schemas_cache[schema_type]
        
        schema_file = self.schemas_dir / f"{schema_type.value}.schema.json"
        
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        self._schemas_cache[schema_type] = schema
        return schema

    def validate_fingerprint(self, data: Dict[str, Any]) -> bool:
        """
        Validate a fingerprint against the schema.
        
        Args:
            data: Fingerprint data dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        schema = self.load_schema(SchemaType.FINGERPRINT)
        return self._validate(data, schema, "fingerprint")

    def validate_decision(self, data: Dict[str, Any]) -> bool:
        """
        Validate a decision object against the schema.
        
        Args:
            data: Decision data dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        schema = self.load_schema(SchemaType.DECISION)
        return self._validate(data, schema, "decision")

    def validate_decisions_collection(self, data: Dict[str, Any]) -> bool:
        """
        Validate a decisions collection against the schema.
        
        Args:
            data: Decisions collection data dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        schema = self.load_schema(SchemaType.DECISIONS_COLLECTION)
        return self._validate(data, schema, "decisions_collection")

    def validate_pipeline(self, data: Dict[str, Any]) -> bool:
        """
        Validate a pipeline artifact against the schema.
        
        Args:
            data: Pipeline data dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        schema = self.load_schema(SchemaType.PIPELINE)
        return self._validate(data, schema, "pipeline")

    def _validate(self, data: Dict[str, Any], schema: Dict[str, Any], name: str) -> bool:
        """
        Internal validation using simple checks.
        
        For production, this should use jsonschema library.
        For now, we do basic required field checking.
        """
        errors = []
        
        # Check schema_version if present
        if "schema_version" in schema.get("required", []):
            if "schema_version" not in data:
                errors.append("Missing required field: schema_version")
            elif data["schema_version"] != "4.0":
                errors.append(f"Invalid schema_version: {data['schema_version']} (expected 4.0)")
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check types of top-level fields
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type:
                    if not self._check_type(value, expected_type):
                        errors.append(f"Invalid type for field '{field}': expected {expected_type}")
        
        if errors:
            error_msg = f"Validation failed for {name}:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValidationError(error_msg)
        
        return True

    def _check_type(self, value: Any, expected: str) -> bool:
        """Check if value matches expected JSON schema type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list
        }
        
        expected_python_type = type_map.get(expected)
        if expected_python_type is None:
            return True  # Unknown type, skip check
        
        return isinstance(value, expected_python_type)


# Convenience functions
def validate_fingerprint(data: Dict[str, Any]) -> bool:
    """Validate fingerprint data"""
    validator = SchemaValidator()
    return validator.validate_fingerprint(data)


def validate_decision(data: Dict[str, Any]) -> bool:
    """Validate decision data"""
    validator = SchemaValidator()
    return validator.validate_decision(data)


def validate_decisions_collection(data: Dict[str, Any]) -> bool:
    """Validate decisions collection data"""
    validator = SchemaValidator()
    return validator.validate_decisions_collection(data)


def validate_pipeline(data: Dict[str, Any]) -> bool:
    """Validate pipeline data"""
    validator = SchemaValidator()
    return validator.validate_pipeline(data)
