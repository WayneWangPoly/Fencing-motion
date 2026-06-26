"""Report and science-investigation output utilities."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCIENCE_FIELDS = [
    "stage",
    "trial",
    "video",
    "timing_score",
    "balance_score",
    "recovery_time_seconds",
    "notes",
    "wrist_start_frame",
    "foot_start_frame",
    "landing_frame",
    "recovery_frame",
    "hand_starts_before_foot",
    "front_knee_angle_at_landing_deg",
    "torso_lean_angle_at_landing_deg",
    "head_vertical_movement_range_px",
    "post_landing_torso_drift",
    "selected_lunge_score",
    "number_of_valid_frames",
    "number_of_input_frames",
    "fps",
    "report_path",
]


def write_report(output_dir: str, report_data: Dict[str, Any]) -> str:
    """Write report.json to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.json")
    report_data.setdefault("artifacts", {})["report_json"] = path
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    return path


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _numeric(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_key(row: Dict[str, Any]) -> tuple[str, str, str]:
    return (_clean(row.get("stage")), _clean(row.get("trial")), _clean(row.get("video")))


def append_science_trial(
    output_root: str,
    report: Dict[str, Any],
    video_path: str,
    stage: Optional[str],
    trial: Optional[int],
    timing_score: Optional[int],
    balance_score: Optional[int],
    notes: str = "",
    csv_name: str = "science_trials.csv",
) -> str:
    """Upsert one science trial row into output/science_trials.csv.

    Re-running the same stage + trial + video replaces that row instead of creating duplicates.
    """
    os.makedirs(output_root, exist_ok=True)
    csv_path = os.path.join(output_root, csv_name)
    metrics = report.get("metrics", {})
    artifacts = report.get("artifacts", {})

    row = {
        "stage": stage,
        "trial": trial,
        "video": video_path,
        "timing_score": timing_score,
        "balance_score": balance_score,
        "recovery_time_seconds": metrics.get("recovery_time_seconds"),
        "notes": notes,
        "wrist_start_frame": metrics.get("wrist_start_frame"),
        "foot_start_frame": metrics.get("foot_start_frame"),
        "landing_frame": metrics.get("landing_frame"),
        "recovery_frame": metrics.get("recovery_frame"),
        "hand_starts_before_foot": metrics.get("hand_starts_before_foot"),
        "front_knee_angle_at_landing_deg": metrics.get("front_knee_angle_at_landing_deg"),
        "torso_lean_angle_at_landing_deg": metrics.get("torso_lean_angle_at_landing_deg"),
        "head_vertical_movement_range_px": metrics.get("head_vertical_movement_range_px"),
        "post_landing_torso_drift": metrics.get("post_landing_torso_drift"),
        "selected_lunge_score": metrics.get("selected_lunge_score"),
        "number_of_valid_frames": metrics.get("number_of_valid_frames"),
        "number_of_input_frames": metrics.get("number_of_input_frames"),
        "fps": metrics.get("fps"),
        "report_path": artifacts.get("report_json"),
    }

    rows: List[Dict[str, Any]] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    new_key = _row_key(row)
    replaced = False
    for idx, existing in enumerate(rows):
        if _row_key(existing) == new_key:
            rows[idx] = {field: _clean(row.get(field)) for field in SCIENCE_FIELDS}
            replaced = True
            break
    if not replaced:
        rows.append({field: _clean(row.get(field)) for field in SCIENCE_FIELDS})

    rows.sort(key=lambda r: (r.get("stage", ""), int(r["trial"]) if r.get("trial", "").isdigit() else 9999, r.get("video", "")))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCIENCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values: Iterable[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def build_science_outputs(output_root: str, csv_path: Optional[str] = None) -> Dict[str, str]:
    """Create summary CSV and optional charts from science_trials.csv."""
    csv_path = csv_path or os.path.join(output_root, "science_trials.csv")
    rows = _read_rows(csv_path)
    if not rows:
        return {}

    by_stage: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        stage = row.get("stage") or "unknown"
        by_stage[stage].append(row)

    summary_fields = [
        "stage",
        "trial_count",
        "timing_average",
        "balance_average",
        "recovery_average_seconds",
    ]
    summary_rows: List[Dict[str, str]] = []
    for stage in sorted(by_stage):
        stage_rows = by_stage[stage]
        timing_avg = _mean([_numeric(r.get("timing_score")) for r in stage_rows])
        balance_avg = _mean([_numeric(r.get("balance_score")) for r in stage_rows])
        recovery_avg = _mean([_numeric(r.get("recovery_time_seconds")) for r in stage_rows])
        summary_rows.append(
            {
                "stage": stage,
                "trial_count": str(len(stage_rows)),
                "timing_average": "" if timing_avg is None else f"{timing_avg:.2f}",
                "balance_average": "" if balance_avg is None else f"{balance_avg:.2f}",
                "recovery_average_seconds": "" if recovery_avg is None else f"{recovery_avg:.2f}",
            }
        )

    summary_path = os.path.join(output_root, "science_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    artifacts = {"summary_csv": summary_path}
    artifacts.update(_try_write_charts(output_root, rows, summary_rows))
    return artifacts


def _try_write_charts(
    output_root: str,
    rows: List[Dict[str, str]],
    summary_rows: List[Dict[str, str]],
) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    artifacts: Dict[str, str] = {}

    stages = [r["stage"] for r in summary_rows]
    timing = [_numeric(r.get("timing_average")) for r in summary_rows]
    balance = [_numeric(r.get("balance_average")) for r in summary_rows]
    recovery = [_numeric(r.get("recovery_average_seconds")) for r in summary_rows]

    if stages and any(v is not None for v in timing + balance + recovery):
        metrics = ["Timing", "Balance", "Recovery s"]
        x = list(range(len(metrics)))
        width = 0.35
        before = next((r for r in summary_rows if r["stage"] == "before"), None)
        after = next((r for r in summary_rows if r["stage"] == "after"), None)
        if before or after:
            before_vals = [
                _numeric(before.get("timing_average")) if before else None,
                _numeric(before.get("balance_average")) if before else None,
                _numeric(before.get("recovery_average_seconds")) if before else None,
            ]
            after_vals = [
                _numeric(after.get("timing_average")) if after else None,
                _numeric(after.get("balance_average")) if after else None,
                _numeric(after.get("recovery_average_seconds")) if after else None,
            ]
            plt.figure(figsize=(8, 4.5))
            plt.bar([i - width / 2 for i in x], [v or 0 for v in before_vals], width, label="Before")
            plt.bar([i + width / 2 for i in x], [v or 0 for v in after_vals], width, label="After")
            plt.xticks(x, metrics)
            plt.ylabel("Average score / seconds")
            plt.title("Before vs After Averages")
            plt.legend()
            plt.tight_layout()
            avg_chart = os.path.join(output_root, "science_before_after_averages.png")
            plt.savefig(avg_chart, dpi=180)
            plt.close()
            artifacts["before_after_chart"] = avg_chart

    recovery_points = []
    for row in rows:
        rec = _numeric(row.get("recovery_time_seconds"))
        trial = _numeric(row.get("trial"))
        stage = row.get("stage") or "unknown"
        if rec is not None and trial is not None:
            recovery_points.append((stage, int(trial), rec))

    if recovery_points:
        plt.figure(figsize=(8, 4.5))
        for stage in sorted({p[0] for p in recovery_points}):
            pts = sorted([p for p in recovery_points if p[0] == stage], key=lambda p: p[1])
            plt.plot([p[1] for p in pts], [p[2] for p in pts], marker="o", label=stage.title())
        plt.xlabel("Trial")
        plt.ylabel("Recovery time (seconds)")
        plt.title("Recovery Time by Trial")
        plt.legend()
        plt.tight_layout()
        rec_chart = os.path.join(output_root, "science_recovery_by_trial.png")
        plt.savefig(rec_chart, dpi=180)
        plt.close()
        artifacts["recovery_chart"] = rec_chart

    return artifacts
