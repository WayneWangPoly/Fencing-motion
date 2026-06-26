# Bridget Science Investigation Workflow

This repository can be used as a video measurement aid for Bridget's science investigation:

**Can video feedback improve timing, balance and recovery in young fencers?**

The app should support the investigation. It should not replace the student's own scoring, observations, logbook, or conclusions.

## What the tool measures

- `wrist_start_frame`: estimated frame where the weapon-side wrist starts moving.
- `foot_start_frame`: estimated frame where the front foot/lunge segment starts.
- `landing_frame`: estimated frame where the front foot/lunge segment finishes.
- `recovery_frame`: manually marked frame where Bridget has returned to en garde.
- `recovery_time_seconds`: calculated as `(recovery_frame - landing_frame) / fps`.

The timing and balance scores are intentionally entered by the student:

- Timing: `2 = hand clearly before foot`, `1 = hand and foot almost together`, `0 = foot before hand`.
- Balance: `2 = stable landing`, `1 = slight wobble`, `0 = lost balance or needed an extra step`.

## First run a trial without recovery frame

```powershell
python analyze_lunge.py `
  --video sample_videos/before_01.mp4 `
  --output output `
  --stage before `
  --trial 1 `
  --timing-score 1 `
  --balance-score 2 `
  --notes "hand and foot close"
```

Then open the annotated frames folder and find the frame where Bridget has recovered to en garde.

Example:

```text
output/runs/before_01_before_01/annotated_frames/
```

## Re-run with recovery frame

```powershell
python analyze_lunge.py `
  --video sample_videos/before_01.mp4 `
  --output output `
  --stage before `
  --trial 1 `
  --timing-score 1 `
  --balance-score 2 `
  --recovery-frame 96 `
  --notes "hand and foot close"
```

## Outputs

- `output/science_trials.csv`: raw trial data table.
- `output/science_summary.csv`: before/after averages.
- `output/science_before_after_averages.png`: chart for the report.
- `output/science_recovery_by_trial.png`: recovery time chart.
- `output/runs/.../evidence/`: key evidence screenshots.
- `output/runs/.../report.json`: one detailed report per video.

## Important judging note

Use wording such as:

> I used video feedback and frame-by-frame checking to help measure my fencing movement before and after practice.

Avoid wording such as:

> The AI proved my fencing improved.

This keeps the project framed as a student science investigation.
