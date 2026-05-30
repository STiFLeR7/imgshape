import json
import hashlib
import datetime
import os
from pathlib import Path
from typing import Optional, Any


class FingerprintLock:
    @staticmethod
    def _get_file_sha256(file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def _extract_dataset_id(fingerprint: Any) -> Optional[str]:
        # Try getting provenance_id or dataset_id from fields, dict, or metadata
        val = getattr(fingerprint, "provenance_id", None)
        if val is not None:
            return str(val)
        val = getattr(fingerprint, "dataset_id", None)
        if val is not None:
            return str(val)

        if isinstance(fingerprint, dict):
            val = fingerprint.get("provenance_id") or fingerprint.get("dataset_id")
            if val is not None:
                return str(val)

        # Try metadata
        metadata = getattr(fingerprint, "metadata", None)
        if metadata:
            if isinstance(metadata, dict):
                val = metadata.get("provenance_id") or metadata.get("dataset_id")
                if val is not None:
                    return str(val)
            else:
                val = getattr(metadata, "provenance_id", None) or getattr(
                    metadata, "dataset_id", None
                )
                if val is not None:
                    return str(val)

        # Check dict representation metadata
        if isinstance(fingerprint, dict):
            meta = fingerprint.get("metadata")
            if isinstance(meta, dict):
                val = meta.get("provenance_id") or meta.get("dataset_id")
                if val is not None:
                    return str(val)

        return None

    @staticmethod
    def _extract_fingerprint_id(fingerprint: Any) -> Optional[str]:
        val = getattr(fingerprint, "dataset_uri", None)
        if val is not None:
            return str(val)
        if isinstance(fingerprint, dict):
            val = fingerprint.get("dataset_uri")
            if val is not None:
                return str(val)
        return None

    @classmethod
    def write_lock(
        cls,
        contract_path: Path,
        fingerprint: Any,
        report_passed: bool,
        lock_path: Optional[Path] = None,
    ) -> dict:
        contract_path = Path(contract_path)
        if lock_path is None:
            target_lock_path = contract_path.parent / ".fingerprint_lock"
        else:
            target_lock_path = Path(lock_path)

        # Ensure parent directory exists
        target_lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate relative contract path
        try:
            contract_rel = os.path.relpath(contract_path, start=target_lock_path.parent)
        except Exception:
            contract_rel = contract_path.name
        contract_rel_posix = Path(contract_rel).as_posix()

        # Calculate contract hash
        try:
            contract_hash = cls._get_file_sha256(contract_path)
        except Exception:
            contract_hash = hashlib.sha256(
                str(contract_path).encode("utf-8")
            ).hexdigest()

        dataset_id = cls._extract_dataset_id(fingerprint)
        fingerprint_id = cls._extract_fingerprint_id(fingerprint)

        lock_data = {
            "contract": contract_rel_posix,
            "contract_hash": contract_hash,
            "dataset_id": dataset_id,
            "fingerprint_id": fingerprint_id,
            "result": "PASS" if report_passed else "FAIL",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "imgshape_version": "5.0.0",
        }

        with open(target_lock_path, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)

        return lock_data

    @staticmethod
    def read_lock(lock_path: Path) -> dict:
        with open(lock_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def verify_lock(cls, lock_path: Path, current_fingerprint: Any) -> bool:
        lock_path = Path(lock_path)
        if not lock_path.exists():
            return False
        try:
            lock_data = cls.read_lock(lock_path)
            lock_dataset_id = lock_data.get("dataset_id")
            lock_result = lock_data.get("result")

            curr_dataset_id = cls._extract_dataset_id(current_fingerprint)

            return lock_result == "PASS" and lock_dataset_id == curr_dataset_id
        except Exception:
            return False
