"""
Master Workflow Controller Script

This script is the central controller for the GNSS-InSAR data processing workflow.
It coordinates the execution of all processing steps in the correct order, manages
environment variables, and provides timing information for each step.

The workflow performs:
1. GNSS 3D velocity calculation
2. InSAR data filtering and parameter extraction
3. Spatial alignment between InSAR and GNSS
4. GNSS Line-of-Sight projection
5. Time series visualization and comparison
6. Grid-based spatial analysis

To customize the workflow:
- Adjust parameter values below
- Comment/uncomment scripts in the 'scripts' list
- Set DATA_DIR to the location of your input data
"""

import os               # Environment variable and path management
import subprocess       # For running Python scripts as subprocesses
import time             # For timing the workflow steps
from pathlib import Path # For platform-independent path handling
from datetime import timedelta  # For formatting execution times

# Set global data directory as an environment variable
# This directory should contain all input data and will store all outputs
data_dir = Path("C:/insar_gnss_data")
os.environ["DATA_DIR"] = str(data_dir)

# Set global parameters (these can be controlled from master.py)
# These parameters affect how the scripts operate - adjust as needed for your dataset
os.environ["MIN_TEMPORAL_COHERENCE"] = "0.7"  # Minimum temporal coherence threshold (0-1)
os.environ["INSAR_RADIUS"] = "250"            # Radius in meters for InSAR point averaging around GNSS stations
os.environ["USE_NNR_CORRECTED"] = "False"     # Standard setting for stable release
os.environ["GNSS_PROVIDER"] = "gfz"           # GNSS data provider ('gfz', 'usgs', etc.)

# Set file names for INSAR and the stations_list.
# These values will be combined with DATA_DIR in your scripts.
os.environ["INSAR_FILE"] = "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A.csv"
os.environ["STATIONS_FILE"] = "stations_list"

# List of scripts to run (uncomment any additional scripts as needed)
scripts = [
    "gnss_3d_vels.py",
    "filter_insar_save_parameters.py",
    "fit_plane_correct_insar.py",
    "gnss_los_displ.py",
    "plot_combined_time_series.py",
    "grid_amplitude_analysis.py",
]

log_file = "workflow.log"

def run_script(script):
    """Executes a script, logs output, and confirms success."""
    try:
        with open(log_file, "a") as log:
            result = subprocess.run(
                ["python", script],
                capture_output=True,
                text=True,
                env=os.environ
            )
            log.write(f"Running {script}...\n")
            log.write(result.stdout)
            log.write(result.stderr)
            log.write("\n" + "-" * 50 + "\n")
        
        if result.returncode != 0:
            print(f"Error in {script}, see {log_file}")
            return False
        
        print(f"{script} executed successfully!")
        return True
    
    except Exception as e:
        print(f"Error while executing {script}: {e}")
        return False

# Main workflow: execute each script in order and abort if any fail.
start_time = time.time()
print(f"Starting workflow at {time.strftime('%Y-%m-%d %H:%M:%S')}")

for script in scripts:
    script_start = time.time()
    success = run_script(script)
    script_duration = time.time() - script_start
    print(f"{script} took {timedelta(seconds=int(script_duration))}")
    
    if not success:
        print("Workflow aborted due to script failure.")
        break

total_duration = time.time() - start_time
print(f"Workflow completed in {timedelta(seconds=int(total_duration))}")
print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")