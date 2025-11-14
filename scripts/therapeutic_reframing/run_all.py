#!/usr/bin/env python3
"""
Master script to run all phases of therapeutic reframing analysis sequentially.

Usage:
    .venv/bin/python scripts/therapeutic_reframing/run_all.py

Or run individual phases:
    .venv/bin/python scripts/therapeutic_reframing/01_prepare_dataset.py
    .venv/bin/python scripts/therapeutic_reframing/02_extract_all_activations.py
    ... etc
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent

PHASES = [
    ("01_prepare_dataset.py", "Phase 1.1: Dataset Preparation"),
    ("02_extract_all_activations.py", "Phase 1.2: Activation Extraction"),
    ("03_learn_pattern_paths.py", "Phase 2.1: Path Learning"),
    ("04_extract_landmarks.py", "Phase 2.2: Landmark Validation"),
    ("05_learn_universal_patterns.py", "Phase 3: Universal Aggregation"),
    ("06_compute_trajectory_properties.py", "Phase 4: Trajectory Analysis"),
    ("07_create_visualizations.py", "Phase 5: Visualizations"),
]


def run_phase(script_name: str, description: str):
    """Run a single phase script."""
    print("\n" + "=" * 80)
    print(f"🚀 Starting: {description}")
    print(f"   Script: {script_name}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    script_path = SCRIPT_DIR / script_name

    # Use .venv python
    python_path = Path(".venv/bin/python")

    try:
        result = subprocess.run(
            [str(python_path), str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )

        print(f"\n✅ Completed: {description}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed: {description}")
        print(f"   Error code: {e.returncode}")
        return False


def main():
    start_time = datetime.now()

    print("=" * 80)
    print("THERAPEUTIC REFRAMING ANALYSIS: FULL PIPELINE")
    print("=" * 80)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total phases: {len(PHASES)}")

    completed = []
    failed = []

    for script_name, description in PHASES:
        success = run_phase(script_name, description)

        if success:
            completed.append(description)
        else:
            failed.append(description)
            print(f"\n⚠️  Stopping pipeline due to failure in: {description}")
            break

    end_time = datetime.now()
    duration = end_time - start_time

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")

    print(f"\n✅ Completed phases: {len(completed)}/{len(PHASES)}")
    for phase in completed:
        print(f"   ✓ {phase}")

    if failed:
        print(f"\n❌ Failed phases: {len(failed)}")
        for phase in failed:
            print(f"   ✗ {phase}")

    if len(completed) == len(PHASES):
        print("\n🎉 All phases completed successfully!")
        return 0
    else:
        print("\n⚠️  Pipeline incomplete due to errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
