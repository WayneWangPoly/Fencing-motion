# Apply this update on Windows

Because the repository is not on your computer yet, clone it first.

```powershell
cd $env:USERPROFILE\Desktop
git clone https://github.com/WayneWangPoly/Fencing-motion.git
cd Fencing-motion
git checkout -b science-investigation
```

Unzip `fencing_motion_science_update_latest.zip` somewhere, for example to your Desktop, then copy the files into the cloned repository:

```powershell
Copy-Item ..\fencing_motion_science_update_latest\analyze_lunge.py .\analyze_lunge.py -Force
Copy-Item ..\fencing_motion_science_update_latest\requirements.txt .\requirements.txt -Force
Copy-Item ..\fencing_motion_science_update_latest\src\lunge_analyzer.py .\src\lunge_analyzer.py -Force
Copy-Item ..\fencing_motion_science_update_latest\src\report_generator.py .\src\report_generator.py -Force
Copy-Item ..\fencing_motion_science_update_latest\summarize_science_trials.py .\summarize_science_trials.py -Force
Copy-Item ..\fencing_motion_science_update_latest\SCIENCE_WORKFLOW.md .\SCIENCE_WORKFLOW.md -Force
```

Install and check:

```powershell
python -m pip install -r requirements.txt
python -m compileall .
git status
git add .
git commit -m "Add science investigation trial workflow"
git push -u origin science-investigation
```

Then GitHub will show a branch. You can open a pull request into `main`, or keep using the branch.
