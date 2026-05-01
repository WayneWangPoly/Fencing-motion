from __future__ import annotations

import argparse
import os

import cv2

from src.lunge_analyzer import LungeAnalyzer
from src.pose_extractor import PoseExtractor
from src.report_generator import write_report
from src.video_io import ensure_output_dirs, read_video_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze side-view epee lunge video.")
    parser.add_argument("--video", required=True, help="Path to input .mp4 video")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    frames, fps = read_video_frames(args.video)
    annotated_dir, output_dir = ensure_output_dirs(args.output)

    extractor = PoseExtractor()
    landmarks_per_frame = []

    height, width = frames[0].shape[:2]
    debug_video_path = os.path.join(output_dir, "debug_overlay.mp4")
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
    report = analyzer.analyze(landmarks_per_frame)
    report_path = write_report(output_dir, report)

    print(f"Analysis complete. Report: {report_path}")
    print(f"Annotated frames: {annotated_dir}")
    if writer is not None:
        print(f"Debug overlay video: {debug_video_path}")
    else:
        print("Debug overlay video was not created (VideoWriter initialization failed).")


if __name__ == "__main__":
    main()
