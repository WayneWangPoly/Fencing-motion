"""Report writing utilities."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def write_report(output_dir: str, report_data: Dict[str, Any]) -> str:
    """Write report.json to output directory."""
    path = os.path.join(output_dir, "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    return path
