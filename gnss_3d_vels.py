"""
GNSS 3D Velocity Calculation and Visualization Script

This script processes GNSS time series data to calculate 3D velocities (North, East, Up).
It performs linear regression on station displacement data to determine velocity trends
and creates visualizations of displacement time series with fitted trend lines.

Features:
- Automated processing of multiple GNSS stations
- 3D velocity calculation using linear regression
- Time series visualization with trend lines
- Conversion between different date formats (MJD to standard dates)
- Statistical analysis of station movements
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from gnss_data_providers import load_gnss_data  # Import the modular GNSS data loader

def mjd_to_date(mjd):
    """Converts Modified Julian Date (MJD) to a readable date in dd.mm.yy format."""
    mjd_start = datetime(1858, 11, 17)  # MJD epoch start
    date = mjd_start + timedelta(days=mjd)
    return date.strftime("%d.%m.%y")

def compute_3d_velocities(input_file):
    """Computes 3D velocities (North, East, Up) in mm/year using linear regression."""
    # Use the modular GNSS data provider to load data
    gnss_data = load_gnss_data(input_file)
    
    # Check if we have enough data for regression
    if gnss_data.shape[0] < 2:
        print(f"Insufficient data for regression in {input_file}. Skipping.")
        return {"North": None, "East": None, "Up": None}

    # Perform linear regression
    north_slope, _, _, _, _ = linregress(gnss_data["MJD"], gnss_data["North"])
    east_slope, _, _, _, _ = linregress(gnss_data["MJD"], gnss_data["East"])
    up_slope, _, _, _, _ = linregress(gnss_data["MJD"], gnss_data["Up"])

    return {
        "North": north_slope * 365.25,
        "East": east_slope * 365.25,
        "Up": up_slope * 365.25
    }

def find_input_file(station_name, data_dir):
    """Finds a file matching {station_name}*NEU_TIME*.txt within DATA_DIR."""
    pattern = os.path.join(data_dir, f"{station_name}*NEU_TIME*.txt")
    matching_files = glob.glob(pattern)
    return matching_files[0] if matching_files else None

def plot_displacements(station_name, input_file, data_dir):
    """Plots GNSS displacement data (East, North, Up) with trendlines."""
    # Use the modular GNSS data provider to load data
    gnss_data = load_gnss_data(input_file)
    
    if gnss_data.shape[0] < 2:
        print(f"Insufficient data for plotting {station_name}. Skipping.")
        return {"North": None, "East": None, "Up": None}
    
    # Ensure `plots` subfolder exists
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    mjd_start = gnss_data["MJD"].iloc[0]
    gnss_data["Decimal_Years"] = (gnss_data["MJD"] - mjd_start) / 365.25
    gnss_data["Date"] = pd.to_datetime(gnss_data["MJD"].apply(mjd_to_date), format="%d.%m.%y")
    
    start_date = gnss_data["Date"].iloc[0].strftime("%Y-%m-%d")
    end_date = gnss_data["Date"].iloc[-1].strftime("%Y-%m-%d")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Date Range: {start_date} to {end_date}", fontsize=10)
    date_formatter = DateFormatter("%Y-%m-%d")

    results = {}
    
    for i, component in enumerate(["East", "North", "Up"]):
        axes[i].scatter(gnss_data["Decimal_Years"], gnss_data[component], label=f"{component} Displacement", color="blue", s=3, alpha=0.5)
        slope, intercept, _, _, _ = linregress(gnss_data["Decimal_Years"], gnss_data[component])
        axes[i].plot(gnss_data["Decimal_Years"], slope * gnss_data["Decimal_Years"] + intercept, label=f"Trend ({slope:.5f} mm/year)", color="red")
        axes[i].set_title(f"GNSS {component} Displacement and Trend for Station {station_name}")
        axes[i].set_ylabel(f"{component} Displacement (mm)")
        axes[i].grid()
        axes[i].legend()
        
        # Store velocity results
        results[component] = slope

    axes[2].set_xlabel("TIME (YYYY-MM-DD)")
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: datetime.strptime(mjd_to_date(mjd_start + x * 365.25), "%d.%m.%y").strftime("%Y-%m-%d")))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha="right")

    safe_filename = station_name.replace(" ", "_").replace("/", "-")
    plot_path = os.path.join(plots_dir, f"displacement_plot_{safe_filename}.png")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved: {plot_path}")
    
    return results

def process_stations():
    """Processes GNSS data for multiple stations, computes 3D velocities, and saves to CSV."""
    # Get data directory from environment variable or use default
    data_dir = os.environ.get("DATA_DIR", "C:\\insar_gnss_data")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return
        
    # Create directory for plots if it doesn't exist
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    station_list_file = os.path.join(data_dir, "stations_list")
    
    # Output CSV file for parameters
    parameters_file = os.path.join(data_dir, "parameters.csv")
    
    # Lists to store station data for CSV output
    station_names = []
    north_vels = []
    east_vels = []
    up_vels = []
    
    # Process each station
    with open(station_list_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # Skip empty lines and potential header lines
        if not line or line.startswith("#"):
            continue
            
        # Check if line might be a header row
        first_word = line.split()[0]
        if first_word.upper() == "STATION":
            continue
            
        # Extract station name
        station_name = first_word
        print(f"Processing GNSS data for station {station_name}...")
        
        # First, check for ETRS89 files (highest priority)
        etrs89_pattern = os.path.join(data_dir, f"{station_name}_NEU_TIME*_ETRS89.txt")
        etrs89_files = glob.glob(etrs89_pattern)
        
        if etrs89_files:
            print(f"Found ETRS89 reference frame file for {station_name}, using this file")
            input_files = etrs89_files
        else:
            # Fall back to standard files if no ETRS89 file found
            input_file = os.path.join(data_dir, f"{station_name}_NEU_TIME*.txt")
            input_files = glob.glob(input_file)
            
        if not input_files:
            print(f"No input file found for station {station_name}.")
            continue
            
        # Use the first matching file (there should typically be only one per station)
        input_file = input_files[0]
        
        # Compute and plot velocities
        velocities = plot_displacements(station_name, input_file, data_dir)
        
        # Store results for CSV output
        station_names.append(station_name)
        if velocities:
            north_vels.append(velocities["North"])
            east_vels.append(velocities["East"])
            up_vels.append(velocities["Up"])
        else:
            north_vels.append(None)
            east_vels.append(None)
            up_vels.append(None)

    # Save velocities to parameters.csv in the format expected by other scripts
    params_df = pd.DataFrame({
        "Station": station_names,
        "GNSS North Velocity (mm/year)": north_vels,
        "GNSS East Velocity (mm/year)": east_vels,
        "GNSS Up Velocity (mm/year)": up_vels
    })
    
    # Try to save the parameters file, handling the case where it might be locked
    try:
        params_df.to_csv(parameters_file, index=False)
        print(f"Parameters saved to {parameters_file}")
    except PermissionError:
        print(f"Warning: Could not write to {parameters_file} - file may be in use.")
        # Try writing to a temporary file, then rename it
        temp_file = os.path.join(data_dir, "parameters_temp.csv")
        params_df.to_csv(temp_file, index=False)
        print(f"Parameters saved to temporary file {temp_file}")

if __name__ == "__main__":
    process_stations()