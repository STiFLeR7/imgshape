import datetime
import json
from pathlib import Path
from typing import Optional, Any, Dict


class AuditTrailWriter:
    """
    Simple utility to log validation verdicts to a target file (as JSON Lines).
    """

    def __init__(self, target_file: Path):
        self.target_file = Path(target_file)

    def write_verdict(
        self, dataset_uri: str, verdict: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Append a validation verdict and associated details to the audit trail log.
        """
        self.target_file.parent.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_uri": dataset_uri,
            "verdict": verdict,
            "details": details or {},
            "imgshape_version": "5.0.0",
        }

        with open(self.target_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
