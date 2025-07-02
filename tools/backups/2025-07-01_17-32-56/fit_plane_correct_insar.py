"""
Spatial Alignment Script for InSAR and GNSS Data

This script performs spatial alignment between InSAR and GNSS data by fitting a correction plane.
It calculates LOS magnitude differences between the data sources and applies a spatial correction
to align InSAR measurements with GNSS observations, reducing systematic spatial biases.

Features:
- Line-of-sight (LOS) difference calculation between GNSS and InSAR
- Plane fitting to model spatial bias patterns
- Visual representation of alignment results
- Automated correction application to InSAR data
- Diagnostic plots of before/after correction states
"""

import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import matplotlib.patheffects as path_effects
from pathlib import Path

# Set default values for environment variables if they are not set
if "DATA_DIR" not in os.environ:
    os.environ["DATA_DIR"] = str(Path("C:/insar_gnss_data"))
if "INSAR_FILE" not in os.environ:
    os.environ["INSAR_FILE"] = "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A.csv"
if "STATIONS_FILE" not in os.environ:
    os.environ["STATIONS_FILE"] = "stations_list"

# Retrieve DATA_DIR from environment variable and resolve it as an absolute Path
data_dir_value = os.getenv("DATA_DIR")
if not data_dir_value:
    raise EnvironmentError("Error: DATA_DIR environment variable is not set.")
data_dir = Path(data_dir_value).resolve()

# Build file paths using pathlib and environment variables for file names
station_list_file = data_dir / os.getenv("STATIONS_FILE", "stations_list")
parameters_file   = data_dir / "parameters.csv"
insar_file        = data_dir / os.getenv("INSAR_FILE", "insar.csv")

# Create the output filename by appending "_aligned" before the extension
output_file = insar_file.with_name(insar_file.stem + "_aligned" + insar_file.suffix)

# Directory for plots
plots_dir = data_dir / "plots"

def calculate_los_difference(parameters_file):
    """Calculates LOS magnitude difference and updates the parameters file."""
    parameters_df = pd.read_csv(parameters_file)
    parameters_df["LOS Magnitude Difference (mm/year)"] = (
        parameters_df["GNSS LOS Magnitude (mm/year)"] - parameters_df["InSAR LOS-Velocity Median (mm/year)"]
    )
    parameters_df.to_csv(parameters_file, index=False)
    print(f"LOS magnitude differences added to {parameters_file}")

def fit_plane_to_los_differences(parameters_file, station_list_file):
    """Fits a plane to LOS magnitude differences and updates the parameters file."""
    parameters_df = pd.read_csv(parameters_file)
    stations_df = pd.read_csv(station_list_file, sep=r'\s+')

    lon = stations_df["longitude"].values
    lat = stations_df["latitude"].values
    los_diff = parameters_df["LOS Magnitude Difference (mm/year)"].values

    def plane_function(params, lon, lat):
        a, b, c = params
        return a * lon + b * lat + c

    def residuals(params, lon, lat, los_diff):
        return plane_function(params, lon, lat) - los_diff

    result = least_squares(residuals, [0, 0, 0], args=(lon, lat, los_diff))
    a, b, c = result.x

    parameters_df["Plane Coefficient a"] = a
    parameters_df["Plane Coefficient b"] = b
    parameters_df["Plane Coefficient c"] = c
    parameters_df.to_csv(parameters_file, index=False)

    print(f"Plane coefficients added to {parameters_file}")
    return a, b, c

def plot_spatial_correction(insar_df, spatial_correction, stations_file):
    """Generates spatial correction plot and saves it."""
    os.makedirs(plots_dir, exist_ok=True)
    stations_df = pd.read_csv(stations_file, sep=r'\s+')

    # Color scale limits at 5th and 95th percentiles
    vmin = np.quantile(spatial_correction, 0.05)
    vmax = np.quantile(spatial_correction, 0.95)

    # Calculate correct aspect ratio for the plot based on the median latitude
    median_lat = np.median(insar_df["latitude"])
    aspect_ratio = 1.0 / np.cos(np.radians(median_lat))  # Correct the aspect ratio based on latitude
    
    # Adjust the figure size to maintain approximately correct proportions
    # Make the width greater to account for the aspect ratio
    fig_width = 10
    fig_height = fig_width / aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot the scatter points with the spatial correction values
    scatter = ax.scatter(insar_df["longitude"], insar_df["latitude"], 
                        c=spatial_correction, cmap="viridis", 
                        s=10, alpha=0.8, vmin=vmin, vmax=vmax)

    # Add GNSS station markers and labels
    for _, station in stations_df.iterrows():
        ax.scatter(station["longitude"], station["latitude"], color="yellow", marker="^", s=72, zorder=5, edgecolor='black')
        ax.text(station["longitude"] + 0.01, station["latitude"] - 0.005, station["Station"],
                 fontsize=12, ha="left", va="top", color="black", weight="bold",
                 path_effects=[path_effects.withStroke(linewidth=3, foreground="white")])

    # Set labels and title
    ax.set_xlabel("Longitude (decimal degrees)")
    ax.set_ylabel("Latitude (decimal degrees)")
    ax.set_title("Spatial Distribution of Correction Values")
    
    # Create a legend with a yellow triangle matching the station markers
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='^', color='w', markerfacecolor='yellow', 
                             markeredgecolor='black', markersize=10, linestyle='None',
                             label='GNSS Stations')]
    ax.legend(handles=legend_elements, loc="upper left")
    
    # Set the correct aspect ratio based on the median latitude
    ax.set_aspect(aspect_ratio)
    
    # Add grid for better spatial reference
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add colorbar below the plot
    cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15)
    cbar.set_label("Spatial Correction Value")

    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plot_path = os.path.join(plots_dir, "spatial_correction.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Spatial correction plot saved: {plot_path}")
    plt.close()

def align_insar_to_gnss(insar_file, a, b, c, output_file, station_list_file):
    """Aligns InSAR LOS values to GNSS using plane coefficients."""
    insar_df = pd.read_csv(insar_file)
    time_columns = insar_df.columns[insar_df.columns.str.match(r"^\d{8}$")]

    start_date = pd.to_datetime(time_columns[0], format="%Y%m%d")
    t = [(pd.to_datetime(date, format="%Y%m%d") - start_date).days / 365.25 for date in time_columns]

    spatial_correction = a * insar_df["longitude"] + b * insar_df["latitude"] + c
    for i, col in enumerate(time_columns):
        insar_df[col] += t[i] * spatial_correction

    insar_df.to_csv(output_file, index=False)
    print(f"Aligned InSAR data saved to {output_file}")

    plot_spatial_correction(insar_df, spatial_correction, station_list_file)

def main():
    """Main function to process InSAR and GNSS alignment."""
    if not os.path.exists(station_list_file):
        print(f"Error: stations_list file not found in {data_dir}.")
        exit(1)

    print(f"Processing GNSS and InSAR alignment in {data_dir}...")

    calculate_los_difference(parameters_file)
    a, b, c = fit_plane_to_los_differences(parameters_file, station_list_file)
    align_insar_to_gnss(insar_file, a, b, c, output_file, station_list_file)

    print("InSAR alignment completed successfully!")

if __name__ == "__main__":
    main()