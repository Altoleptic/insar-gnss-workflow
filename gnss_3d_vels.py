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

def mjd_to_date(mjd):
    """Converts Modified Julian Date (MJD) to a readable date in dd.mm.yy format."""
    mjd_start = datetime(1858, 11, 17)  # MJD epoch start
    date = mjd_start + timedelta(days=mjd)
    return date.strftime("%d.%m.%y")

def compute_3d_velocities(input_file):
    """Computes 3D velocities (North, East, Up) in mm/year using linear regression."""
    colspecs = [(0, 9), (10, 29), (30, 38), (39, 47), (48, 56)]  # Column widths
    
    gnss_data = pd.read_fwf(input_file, colspecs=colspecs, skiprows=2, names=["MJD", "TIME", "North", "East", "Up"])
    gnss_data["MJD"] = pd.to_numeric(gnss_data["MJD"], errors="coerce")
    gnss_data = gnss_data.dropna(subset=["MJD", "North", "East", "Up"])

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

def plot_displacements(station_name, gnss_data, data_dir):
    """Plots GNSS displacement data (East, North, Up) with trendlines."""
    
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

    for i, component in enumerate(["East", "North", "Up"]):
        axes[i].scatter(gnss_data["Decimal_Years"], gnss_data[component], label=f"{component} Displacement", color="blue", s=3, alpha=0.5)
        slope, intercept, _, _, _ = linregress(gnss_data["Decimal_Years"], gnss_data[component])
        axes[i].plot(gnss_data["Decimal_Years"], slope * gnss_data["Decimal_Years"] + intercept, label=f"Trend ({slope:.5f} mm/year)", color="red")
        axes[i].set_title(f"GNSS {component} Displacement and Trend for Station {station_name}")
        axes[i].set_ylabel(f"{component} Displacement (mm)")
        axes[i].grid()
        axes[i].legend()

    axes[2].set_xlabel("Date")
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: datetime.strptime(mjd_to_date(mjd_start + x * 365.25), "%d.%m.%y").strftime("%Y-%m-%d")))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

    safe_filename = station_name.replace(" ", "_").replace("/", "-")
    plot_path = os.path.join(plots_dir, f"displacement_plot_{safe_filename}.png")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved: {plot_path}")

def process_stations_to_csv(station_names, data_dir, output_file):
    """Processes GNSS data for multiple stations, computes 3D velocities, and saves to CSV."""
    results = []

    for station_name in station_names:
        input_file = find_input_file(station_name, data_dir)

        if not input_file:
            print(f"File not found for station {station_name}. Skipping.")
            continue

        print(f"Processing GNSS data for station {station_name}...")
        colspecs = [(0, 9), (10, 29), (30, 38), (39, 47), (48, 56)]
        gnss_data = pd.read_fwf(input_file, colspecs=colspecs, skiprows=2, names=["MJD", "TIME", "North", "East", "Up"])

        velocities = compute_3d_velocities(input_file)

        plot_displacements(station_name, gnss_data, data_dir)

        results.append({
            "Station": station_name,
            "GNSS North Velocity (mm/year)": velocities["North"],
            "GNSS East Velocity (mm/year)": velocities["East"],
            "GNSS Up Velocity (mm/year)": velocities["Up"]
        })

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"All station velocities saved to {output_file}")

def main():
    """Main function to handle paths and process GNSS station data."""
    # Retrieve DATA_DIR from environment variable
    data_dir = os.getenv("DATA_DIR")

    if not data_dir:
        print("Error: DATA_DIR environment variable is not set.")
        exit(1)

    data_dir = os.path.abspath(data_dir)  # Ensure correct path formatting
    station_list_file = os.path.join(data_dir, os.getenv("STATIONS_FILE", "stations_list"))
    output_file = os.path.join(data_dir, "parameters.csv")

    if not os.path.exists(station_list_file):
        print(f"Error: stations_list file not found in {data_dir}.")
        station_names = []
    else:
        station_df = pd.read_csv(station_list_file, delim_whitespace=True)
        station_names = station_df["Station"].tolist() if "Station" in station_df.columns else []

    if station_names:
        process_stations_to_csv(station_names, data_dir, output_file)
    else:
        print("No stations to process.")

if __name__ == "__main__":
    main()