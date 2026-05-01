# Fencing Motion Prototype

A local Python command-line prototype for first-pass side-view epee lunge analysis using body landmarks from MediaPipe Pose.

## Scope (v1)

This prototype intentionally makes strong assumptions:
- **Side-view video only**.
- **One clearly visible fencer**.
- **Body landmarks only** (no sword tip tracking yet).
- **Single lunge-oriented analysis pass** from one video clip.

Because this is a first version, metric quality depends heavily on camera angle, lighting, and occlusion.

## Project structure

```text
fencing-motion-prototype/
  requirements.txt
  README.md
  analyze_lunge.py
  src/
    video_io.py
    pose_extractor.py
    geometry.py
    timing.py
    lunge_analyzer.py
    report_generator.py
  output/
```

## Setup

1. Use Python 3.11+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python analyze_lunge.py --video sample_videos/my_lunge.mp4 --output output
```

## Outputs

- `output/report.json`: computed metrics, issues, and correction plan.
- `output/annotated_frames/`: per-frame images with pose landmarks and frame index.
- `output/debug_overlay.mp4`: optional debug video with overlay if writer initialization succeeds.

## Notes and limitations

- Front side / weapon side inference is estimated from wrist horizontal displacement; wrong if camera is mirrored or movement is atypical.
- Landing frame is estimated from front ankle deceleration and may miss exact foot-contact moment.
- Torso lean uses shoulder-hip segment angle relative to vertical; this is a practical approximation.
- This tool is intended for beginner coaching feedback, not biomechanical-grade measurement.
