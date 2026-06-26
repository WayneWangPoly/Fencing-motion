from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from src.lunge_analyzer import LungeAnalyzer
from src.pose_extractor import PoseExtractor
from src.report_generator import (
    append_science_trial,
    build_science_outputs,
    write_report,
)
from src.video_io import ensure_output_dirs, read_video_frames


def _safe_stem(video_path: str) -> str:
    stem = Path(video_path).stem.strip() or "video"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return safe[:80]


def _run_output_dir(base_output: str, video_path: str, stage: Optional[str], trial: Optional[int]) -> str:
    """Keep the old output shape for normal one-off analysis, but isolate science trials."""
    if stage is None and trial is None:
        return base_output

    parts = [_safe_stem(video_path)]
    if stage:
        parts.append(stage)
    if trial is not None:
        parts.append(f"{trial:02d}")
    return os.path.join(base_output, "runs", "_".join(parts))


def _copy_evidence_frame(
    annotated_dir: str,
    evidence_dir: str,
    frame_number: Optional[int],
    label: str,
) -> Optional[str]:
    if frame_number is None or frame_number < 0:
        return None

    source = os.path.join(annotated_dir, f"frame_{frame_number:05d}.jpg")
    if not os.path.exists(source):
        return None

    os.makedirs(evidence_dir, exist_ok=True)
    target = os.path.join(evidence_dir, f"{label}_frame_{frame_number:05d}.jpg")
    shutil.copy2(source, target)
    return target


def _export_evidence_frames(run_dir: str, annotated_dir: str, report: Dict[str, Any]) -> Dict[str, Optional[str]]:
    metrics = report.get("metrics", {})
    evidence_dir = os.path.join(run_dir, "evidence")
    exported = {
        "wrist_start": _copy_evidence_frame(
            annotated_dir, evidence_dir, metrics.get("wrist_start_frame"), "wrist_start"
        ),
        "foot_start": _copy_evidence_frame(
            annotated_dir, evidence_dir, metrics.get("foot_start_frame"), "foot_start"
        ),
        "landing": _copy_evidence_frame(
            annotated_dir, evidence_dir, metrics.get("landing_frame"), "landing"
        ),
        "recovery": _copy_evidence_frame(
            annotated_dir, evidence_dir, metrics.get("recovery_frame"), "recovery"
        ),
    }
    report.setdefault("artifacts", {})["evidence_frames"] = exported
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze side-view epee lunge video.")
    parser.add_argument("--video", required=True, help="Path to input .mp4 video")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument(
        "--stage",
        choices=["before", "after"],
        default=None,
        help="Science investigation stage. Use before or after for Bridget's project.",
    )
    parser.add_argument("--trial", type=int, default=None, help="Science investigation trial number")
    parser.add_argument("--timing-score", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--balance-score", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument(
        "--recovery-frame",
        type=int,
        default=None,
        help="Manual recovered en-garde frame number. Recovery time = (recovery - landing) / fps.",
    )
    parser.add_argument("--notes", default="", help="Short student observation note for this trial")
    parser.add_argument(
        "--no-debug-video",
        action="store_true",
        help="Skip writing debug_overlay.mp4 to make repeated trial runs faster.",
    )
    args = parser.parse_args()

    frames, fps = read_video_frames(args.video)
    root_output_dir = args.output
    run_dir = _run_output_dir(root_output_dir, args.video, args.stage, args.trial)
    annotated_dir, output_dir = ensure_output_dirs(run_dir)

    extractor = PoseExtractor()
    landmarks_per_frame = []
    height, width = frames[0].shape[:2]

    writer = None
    debug_video_path = os.path.join(output_dir, "debug_overlay.mp4")
    if not args.no_debug_video:
        writer = cv2.VideoWriter(
            debug_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer = None

    for i, frame in enumerate(frames):
        timestamp_ms = int((i / fps) * 1000)
        landmarks, overlay = extractor.process_frame(frame, timestamp_ms=timestamp_ms)
        landmarks_per_frame.append(landmarks)

        cv2.putText(
            overlay,
            f"Frame: {i}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frame_path = os.path.join(annotated_dir, f"frame_{i:05d}.jpg")
        cv2.imwrite(frame_path, overlay)
        if writer is not None:
            writer.write(overlay)

    extractor.close()
    if writer is not None:
        writer.release()

    analyzer = LungeAnalyzer()
    report = analyzer.analyze(
        landmarks_per_frame,
        fps=fps,
        recovery_frame=args.recovery_frame,
    )

    report.setdefault("input", {})
    report["input"].update(
        {
            "video": args.video,
            "fps": fps,
            "frame_count": len(frames),
            "stage": args.stage,
            "trial": args.trial,
            "student_timing_score": args.timing_score,
            "student_balance_score": args.balance_score,
            "student_notes": args.notes,
        }
    )

    _export_evidence_frames(output_dir, annotated_dir, report)
    report_path = write_report(output_dir, report)

    science_csv_path = None
    if args.stage is not None or args.trial is not None:
        science_csv_path = append_science_trial(
            root_output_dir,
            report,
            video_path=args.video,
            stage=args.stage,
            trial=args.trial,
            timing_score=args.timing_score,
            balance_score=args.balance_score,
            notes=args.notes,
        )
        build_science_outputs(root_output_dir, science_csv_path)

    print("Analysis complete.")
    print(f"Report: {report_path}")
    print(f"Annotated frames: {annotated_dir}")
    if science_csv_path:
        print(f"Science trial CSV: {science_csv_path}")
        print(f"Science summary: {os.path.join(root_output_dir, 'science_summary.csv')}")
    if writer is not None:
        print(f"Debug overlay video: {debug_video_path}")
    elif not args.no_debug_video:
        print("Debug overlay video was not created (VideoWriter initialization failed).")


if __name__ == "__main__":
    main()
