import os
import subprocess
from pathlib import Path
import time

# Set global data directory as an environment variable
data_dir = Path("C:/insar_gnss_data")
os.environ["DATA_DIR"] = str(data_dir)

# Set global parameters (these can be controlled from master.py)
os.environ["MIN_TEMPORAL_COHERENCE"] = "0.7"    # Minimum temporal coherence threshold
os.environ["INSAR_RADIUS"] = "500"              # Radius in m for InSAR averaging
os.environ["GRID_SIZE_KM"] = "0.5"              # Grid size in km for plot_variability.py

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
    "plot_variability.py"
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

start_time = time.time()

# Main workflow: execute each script in order and abort if any fail.
for script in scripts:
    success = run_script(script)
    if not success:
        print("Workflow aborted due to error.")
        break

end_time = time.time()
duration = end_time - start_time
if duration < 60:
    print(f"Total runtime: {duration:.2f} seconds")
else:
    print(f"Total runtime: {duration/60:.2f} minutes")