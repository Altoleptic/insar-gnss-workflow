"""
Combined Time Series Visualization Script

This script creates comparative visualizations of InSAR and GNSS time series data.
It generates plots showing both data sources together, allowing direct comparison
of displacement patterns, trends, and s    # The station processing has been moved to the plot_station_map function
    # which is called in parallel by plot_station_velocity_mapal variations between measurement techniques.

Features:
- Side-by-side visualization of InSAR and GNSS time series
- Automatic data alignment to common time references
- Statistical comparison between measurement techniques
- Support for seasonal pattern analysis
- Multiple visualization formats (raw, detrended, normalized)
"""

# Import necessary libraries for data processing, mathematical operations, and visualization
import os                       # File and directory operations
import glob                     # File pattern matching
import pandas as pd             # Data manipulation and analysis
import matplotlib.pyplot as plt # Visualization
import numpy as np              # Numerical operations
from scipy.stats import linregress  # Linear regression for trend analysis
from datetime import datetime   # Date and time operations
from geopy.distance import geodesic  # Geodetic distance calculations
import matplotlib.patheffects as path_effects  # Text visual effects for better readability
from pathlib import Path        # Modern, object-oriented file system paths
from gnss_data_providers import load_gnss_data  # GNSS data loading utilities
import multiprocessing          # For parallel processing
from functools import partial   # For creating partial functions with fixed arguments

# Read DATA_DIR from the environment variable and set up global paths
# This is the root directory containing all input and output data
data_dir_value = os.getenv("DATA_DIR", "C:\\insar_gnss_data")  # Set default DATA_DIR to C:\insar_gnss_data if not provided.
if not data_dir_value:
    print("Error: DATA_DIR environment variable is not set and no default value is provided.")
    exit(1)
data_dir = Path(data_dir_value).resolve()

# Global parameters that can be configured from master.py (e.g., via environment variables)
# Temporal coherence threshold for filtering InSAR points
MIN_TEMPORAL_COHERENCE = float(os.getenv("MIN_TEMPORAL_COHERENCE", "0.7"))
# Radius around GNSS stations to consider InSAR points (in meters)
INSAR_RADIUS = int(os.getenv("INSAR_RADIUS", "250"))
# Whether to use NNR-corrected files
USE_NNR_CORRECTED = os.getenv("USE_NNR_CORRECTED", "True").lower() == "true"
print(f"Using NNR-corrected files: {USE_NNR_CORRECTED}")

# Global file paths (all files are assumed to be in DATA_DIR)
stations_file = data_dir / os.getenv("STATIONS_FILE", "stations_list")
parameters_file = data_dir / "parameters.csv"
insar_before = data_dir / os.getenv("INSAR_FILE", "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A.csv")
insar_after = insar_before.with_name(insar_before.stem + "_aligned" + insar_before.suffix)

# All plots will be saved in a single folder named "plots"
plots_dir = data_dir / "plots"

def find_stations_file():
    """
    Returns the path to the stations_list file located in DATA_DIR.
    """
    stations_path = os.path.join(data_dir, "stations_list")
    return stations_path if os.path.exists(stations_path) else None

def decimal_year(date_str, start_date):
    """
    Converts a date string (YYYYMMDD) to a decimal year relative to start_date.
    
    Args:
        date_str: Date string in YYYYMMDD format
        start_date: Reference date to calculate the decimal year from
        
    Returns:
        Float representing years elapsed since start_date
    """
    date = datetime.strptime(date_str, "%Y%m%d")
    delta_days = (date - start_date).days
    return delta_days / 365.25


def load_gnss_data(filepath):
    """
    Loads GNSS data from a file and handles inconsistent formatting.
    Can load both CSV files from gnss_los_displ.py and traditional space-delimited GNSS files.
    Converts MJD to datetime and computes a decimal year based on the first date.
    
    Args:
        filepath: Path to the GNSS data file
        
    Returns:
        DataFrame containing processed GNSS data with converted dates
        
    Raises:
        ValueError: If the file contains no valid MJD data
    """
    # First try to load as a CSV file (the format produced by gnss_los_displ.py)
    try:
        df = pd.read_csv(filepath)
        if "MJD" in df.columns and "LOS" in df.columns:
            # File is in the expected CSV format
            if "North" not in df.columns:
                # Add placeholder North, East, Up columns if they don't exist
                df["North"] = 0
                df["East"] = 0 
                df["Up"] = 0
            
            # Add DATE column if it doesn't exist
            if "DATE" not in df.columns:
                # Convert Modified Julian Date to datetime (origin date is 1858-11-17)
                df["DATE"] = pd.to_datetime(df["MJD"], origin="1858-11-17", unit="D")
            
            # Add decimal_year if it doesn't exist
            if "decimal_year" not in df.columns:
                # Calculate decimal years since the first date for trend analysis
                start_date = df["DATE"].iloc[0]
                df["decimal_year"] = ((df["DATE"] - start_date).dt.total_seconds() /
                                    (365.25 * 24 * 3600))
            
            return df
    except Exception as e:
        print(f"Failed to read {filepath} as CSV, trying fixed-width format: {e}")
    
    # Fall back to the original fixed-width format parsing
    data = []
    # Read file line by line to handle inconsistent formatting
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines and empty lines
            if not line or line.startswith('MJD') or line.startswith('---') or "in mm" in line:
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    # Parse each column from the line
                    mjd = float(parts[0])
                    time_str = parts[1] + " " + parts[2]
                    north = float(parts[3])
                    east = float(parts[4])
                    up = float(parts[5])
                    los = float(parts[6])
                    data.append([mjd, time_str, north, east, up, los])
                except ValueError:
                    print(f"Skipping invalid line in GNSS file: {line}")
                    continue
    
    # Convert to DataFrame and add date information
    df = pd.DataFrame(data, columns=["MJD", "TIME", "North", "East", "Up", "LOS"])
    if df.empty or df["MJD"].isnull().all():
        raise ValueError(f"GNSS file {filepath} contains no valid MJD data.")
    
    # Convert Modified Julian Date to datetime (origin date is 1858-11-17)
    df["DATE"] = pd.to_datetime(df["MJD"], origin="1858-11-17", unit="D")
    
    # Calculate decimal years since the first date for trend analysis
    start_date = df["DATE"].iloc[0]
    df["decimal_year"] = ((df["DATE"] - start_date).dt.total_seconds() /
                          (365.25 * 24 * 3600))
    return df


def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate the geodetic distance between two points using the Haversine formula.
    Vectorized implementation for efficient computation with NumPy arrays.
    
    Args:
        lat1, lon1: Latitude and longitude of the first point (single value)
        lat2, lon2: Latitude and longitude of the second points (arrays)
        
    Returns:
        Array of distances in meters
    """
    R = 6371000  # Earth's radius in meters
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def find_insar_average_within_radius(insar_df, station_lat, station_lon, radius=INSAR_RADIUS):
    """
    Computes the median displacement of all InSAR points within a given radius (in meters)
    around the GNSS station. The median is used instead of the mean because it is more robust 
    to outliers in InSAR data, which can result from phase unwrapping errors, atmospheric 
    effects, or localized deformation anomalies.
    
    Args:
        insar_df: DataFrame containing InSAR data
        station_lat, station_lon: GNSS station coordinates
        radius: Search radius in meters (default from INSAR_RADIUS global variable)
        
    Returns:
        Series with the median displacement for each time column
    """
    # Calculate distances from the station to all InSAR points
    distances = haversine_distance_vectorized(
        station_lat, station_lon,
        insar_df["latitude"].values, insar_df["longitude"].values
    )
    # Filter points within specified radius
    within_radius = insar_df[distances <= radius]
    # Extract time columns (identified by numeric names)
    time_columns = [col for col in within_radius.columns if col.isdigit()]
    # Return median displacement for each time column (more robust to outliers than mean)
    return within_radius[time_columns].median()


def convert_dates_to_decimal_years(time_dates):
    """
    Converts datetime objects to decimal years from the first date.
    This enables proper time-based trend analysis accounting for irregular sampling.
    
    Args:
        time_dates: List of datetime objects representing acquisition dates
        
    Returns:
        Array of decimal years relative to the first date
    """
    start_date = time_dates[0]
    decimal_years = [(date - start_date).total_seconds() / (365.25 * 24 * 3600) for date in time_dates]
    return np.array(decimal_years)


def plot_combined_time_series():
    """
    Generates combined time series plots for each GNSS station with two subplots:
      - InSAR time series before alignment,
      - Combined InSAR time series after alignment and GNSS LOS displacement.
    The plots are saved in the "plots" folder.
    """
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load InSAR datasets (before and after alignment)
    before_df = pd.read_csv(insar_before)
    after_df = pd.read_csv(insar_after)
    
    # Load station information
    stations_df = pd.read_csv(stations_file, sep=r'\s+')
    stations_df.columns = stations_df.columns.str.strip()
    
    # Filter InSAR data by temporal coherence to ensure quality
    before_df = before_df[before_df["temporal_coherence"] >= MIN_TEMPORAL_COHERENCE]
    after_df = after_df[after_df["temporal_coherence"] >= MIN_TEMPORAL_COHERENCE]

    # Precompute coordinates as numpy arrays for fast access (performance optimization)
    before_coords = before_df[["latitude", "longitude"]].values
    after_coords = after_df[["latitude", "longitude"]].values
    before_time_columns = [col for col in before_df.columns if col.isdigit()]
    after_time_columns = [col for col in after_df.columns if col.isdigit()]

    # Process each station individually
    for idx, station in stations_df.iterrows():
        station_name = station["Station"]
        station_lat = station["latitude"]
        station_lon = station["longitude"]

        # Find GNSS file with LOS values for this station
        if USE_NNR_CORRECTED:
            # First try to find LOS files that came from NNR-corrected data
            # These files might have been generated from NNR files but won't have "NNR" in their name
            # The LOS calculation happens after NNR correction in the workflow
            gnss_pattern = os.path.join(data_dir, f"{station_name}_NEU_TIME*_LOS.txt")
            gnss_files = glob.glob(gnss_pattern)
            
            # If there are multiple LOS files, we'd prefer ones with more recent timestamps
            gnss_files.sort(reverse=True)  # Sort by filename, reverse to get newest first
            
            if not gnss_files:
                print(f"GNSS LOS file not found for pattern: {gnss_pattern}. Skipping station {station_name}.")
                continue
                
        else:
            # If we're not using NNR-corrected files, find normal LOS files
            gnss_pattern = os.path.join(data_dir, f"{station_name}_NEU_TIME*_LOS.txt")
            gnss_files = glob.glob(gnss_pattern)
            
            if not gnss_files:
                print(f"GNSS file not found for pattern: {gnss_pattern}. Skipping station {station_name}.")
                continue
                
        gnss_file = gnss_files[0]
        print(f"Using GNSS file: {gnss_file} for station {station_name}")

        # Use pre-computed distances if available, otherwise calculate them
        # This significantly reduces computation time by avoiding redundant calculations
        if not hasattr(plot_combined_time_series, 'distance_cache'):
            plot_combined_time_series.distance_cache = {}
            
        # Create a cache key based on station coordinates and radius
        cache_key = f"{station_lat}_{station_lon}"
        
        if cache_key in plot_combined_time_series.distance_cache:
            before_mask, after_mask = plot_combined_time_series.distance_cache[cache_key]
        else:
            # Define efficient Haversine function for performance optimization
            def fast_haversine(lat1, lon1, coords):
                """Optimized Haversine formula for array operations."""
                R = 6371000
                lat1 = np.radians(lat1)
                lon1 = np.radians(lon1)
                lat2 = np.radians(coords[:, 0])
                lon2 = np.radians(coords[:, 1])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                return R * c

            # Calculate distances and filter points within radius for both datasets
            before_dist = fast_haversine(station_lat, station_lon, before_coords)
            after_dist = fast_haversine(station_lat, station_lon, after_coords)
            before_mask = before_dist <= INSAR_RADIUS
            after_mask = after_dist <= INSAR_RADIUS
            
            # Cache the results to avoid recalculation
            plot_combined_time_series.distance_cache[cache_key] = (before_mask, after_mask)
        
        # Calculate median displacements for points within radius (more robust to outliers)
        before_displacement = before_df.loc[before_mask, before_time_columns].median()
        after_displacement = after_df.loc[after_mask, after_time_columns].median()

        # Load GNSS data for comparison
        gnss_data = load_gnss_data(gnss_file)
        
        # Convert time columns to datetime for plotting
        time_dates = [pd.to_datetime(col, format="%Y%m%d") for col in before_time_columns]
        
        # Convert dates to decimal years for proper time-based trend analysis
        decimal_years = convert_dates_to_decimal_years(time_dates)

        # Create figure with two vertically stacked subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Subplot 1: InSAR before alignment
        axes[0].plot(time_dates, before_displacement, 'r.', label="Before Alignment (Displacement)")
        
        # Calculate and plot linear trend for before alignment using actual time intervals
        slope_before, intercept_before, r_value_before, p_value_before, std_err_before = linregress(
            decimal_years, before_displacement)
        trend_before = slope_before * decimal_years + intercept_before
        axes[0].plot(time_dates, trend_before, 'r-', 
                    label=f"Trend (Slope: {slope_before:.5f} mm/year)", 
                    linewidth=2.5)
        
        # Add labels and formatting
        axes[0].set_title(f"InSAR Time Series Before Alignment - Station {station_name}")
        axes[0].set_ylabel("Displacement (mm)")
        axes[0].legend()
        axes[0].grid()
        
        # Subplot 2: Combined time series (after alignment and GNSS)
        axes[1].plot(time_dates, after_displacement, 'b.', label="After Alignment (Displacement)")
        
        # Calculate and plot linear trend for after alignment using actual time intervals
        slope_after, intercept_after, r_value_after, p_value_after, std_err_after = linregress(
            decimal_years, after_displacement)
        trend_after = slope_after * decimal_years + intercept_after
        axes[1].plot(time_dates, trend_after, 'b-', 
                    label=f"InSAR Trend (Slope: {slope_after:.5f} mm/year)", 
                    linewidth=2.5)
        
        # Add GNSS data and trend (GNSS already uses proper decimal years)
        axes[1].plot(gnss_data["DATE"], gnss_data["LOS"], 'g.', label="GNSS LOS Displacement")
        slope_gnss, intercept_gnss, r_value_gnss, p_value_gnss, std_err_gnss = linregress(
            gnss_data["decimal_year"], gnss_data["LOS"])
        axes[1].plot(gnss_data["DATE"], slope_gnss * gnss_data["decimal_year"] + intercept_gnss,
                     'g-', label=f"GNSS Trend (Slope: {slope_gnss:.5f} mm/year)", 
                     linewidth=2.5)
        
        # Add labels and formatting
        axes[1].set_title(f"Combined InSAR After Alignment and GNSS LOS - Station {station_name}")
        axes[1].set_ylabel("Displacement (mm)")
        axes[1].legend()
        axes[1].grid()
        axes[1].set_xlabel("TIME (YYYY-MM-DD)")
        axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Add overall title with date range information
        date_start = time_dates[0].strftime("%Y-%m-%d")
        date_end = time_dates[-1].strftime("%Y-%m-%d")
        fig.suptitle(f"Date Range: {date_start} to {date_end}", fontsize=10)
        
        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = os.path.join(plots_dir, f"{station_name}_combined_plot.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Combined time series plot saved for station {station_name}: {output_path}")


def plot_global_velocity_map(before_df, after_df, stations_df, parameters_file, output_dir, title="Regional Velocity Map", suffix=""):
    """
    Generates a scatter plot of velocities in longitude-latitude space, consisting of three subplots:
      - Before alignment,
      - After alignment, and
      - The velocity correction plane.
      
    Args:
        before_df: DataFrame containing InSAR data before alignment
        after_df: DataFrame containing InSAR data after alignment
        stations_df: DataFrame containing GNSS station information
        parameters_file: Path to file containing plane coefficients
        output_dir: Directory to save output plots
        title: Title for the plot (default: "Regional Velocity Map")
        suffix: String suffix for output filename (default: "")
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read plane coefficients from parameters file
    parameters_df = pd.read_csv(parameters_file)
    a = parameters_df["Plane Coefficient a"].iloc[0]
    b = parameters_df["Plane Coefficient b"].iloc[0]
    c = parameters_df["Plane Coefficient c"].iloc[0]

    def filter_normal_points(df):
        """Filter outliers using IQR method and return normal range points/velocities."""
        # Identify time columns and calculate median velocity (more robust to outliers than mean)
        time_columns = [col for col in df.columns if col.isdigit()]
        velocities = df[time_columns].median(axis=1)
        
        # Calculate IQR and filter outliers
        q1 = velocities.quantile(0.25)
        q3 = velocities.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 2.0 * iqr
        upper_bound = q3 + 2.0 * iqr
        normal_points = df[(velocities >= lower_bound) & (velocities <= upper_bound)]
        normal_velocities = velocities[(velocities >= lower_bound) & (velocities <= upper_bound)]
        return normal_points, normal_velocities

    # Filter outliers from both datasets
    before_points, before_velocities = filter_normal_points(before_df)
    after_points, after_velocities = filter_normal_points(after_df)
    
    # Calculate statistics for color scaling using median for robustness to outliers
    before_median = before_velocities.median()
    before_std = before_velocities.std()  # Still using std for spread measure
    after_median = after_velocities.median()
    after_std = after_velocities.std()
    global_median = np.median([before_median, after_median])
    global_std = np.median([before_std, after_std])  # Using median for consistency
    color_min = global_median - 3 * global_std
    color_max = global_median + 3 * global_std

    # Create figure with three vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(18, 40), sharex=True)
    
    # Subplot 1: Before alignment
    scatter1 = axes[0].scatter(before_points["longitude"], before_points["latitude"],
                               c=before_velocities, cmap="seismic", s=1, alpha=0.7, marker=".")
    axes[0].set_title("Before Alignment", fontsize=16)
    axes[0].set_ylabel("Latitude (decimal degrees)", fontsize=14)
    axes[0].grid(alpha=0.5)
    axes[0].set_aspect('equal')
    fig.colorbar(scatter1, ax=axes[0], pad=0.02).set_label("Velocity (mm/year)")
    
    # Subplot 2: After alignment
    scatter2 = axes[1].scatter(after_points["longitude"], after_points["latitude"],
                               c=after_velocities, cmap="seismic", s=1, alpha=0.7, marker=".")
    axes[1].set_title("After Alignment", fontsize=16)
    axes[1].set_ylabel("Latitude (decimal degrees)", fontsize=14)
    axes[1].grid(alpha=0.5)
    axes[1].set_aspect('equal')
    fig.colorbar(scatter2, ax=axes[1], pad=0.02).set_label("Velocity (mm/year)")
    
    # Subplot 3: Correction plane
    lons = before_points["longitude"]
    lats = before_points["latitude"]
    # Calculate correction plane values at each point
    correction_plane = a * lons + b * lats + c
    scatter3 = axes[2].scatter(lons, lats, c=correction_plane, cmap="plasma", s=1, alpha=0.7, marker=".")
    axes[2].set_title("Velocity Correction Plane", fontsize=16)
    axes[2].set_xlabel("Longitude (decimal degrees)", fontsize=14)
    axes[2].set_ylabel("Latitude (decimal degrees)", fontsize=14)
    axes[2].grid(alpha=0.5)
    axes[2].set_aspect('equal')
    fig.colorbar(scatter3, ax=axes[2], pad=0.02).set_label("Correction Value (mm/year)")
    
    # Add GNSS station markers and labels to all subplots
    for _, station in stations_df.iterrows():
        station_name = station["Station"]
        station_lat = station["latitude"]
        station_lon = station["longitude"]
        for ax in axes:
            ax.scatter(station_lon, station_lat, color="black", edgecolor="white", s=50, marker="^", zorder=5)
            ax.text(station_lon, station_lat, station_name,
                    color="black", fontsize=10, ha="left", va="bottom",
                    path_effects=[path_effects.withStroke(linewidth=1, foreground="white")])
    
    # Add title and save figure
    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, f"{suffix}_velocity_map_with_correction.png")
    plt.savefig(output_path, dpi=200)  # Lower DPI for faster saving
    plt.close('all')  # Explicitly close all figures to free memory
    
    # Explicitly clear some large variables to help with memory management
    del before_points, after_points, before_velocities, after_velocities
    
    print(f"Velocity map with correction saved: {output_path}")


def plot_station_map(station, before_df, after_df, output_dir, radius=INSAR_RADIUS):
    """
    Generates a velocity map for a single station.
    This function is designed to be called in parallel for multiple stations.
    
    Args:
        station: Series containing station information
        before_df: DataFrame containing InSAR data before alignment
        after_df: DataFrame containing InSAR data after alignment
        output_dir: Directory to save output plots
        radius: Radius around stations to include InSAR points
    """
    station_name = station["Station"]
    station_lat = station["latitude"]
    station_lon = station["longitude"]
    
    # Function implementation continues below...
    # (The rest of the plotting code for a single station)
    def filter_within_radius(df):
        """Filter points within radius of station and remove outliers."""
        # Use vectorized Haversine instead of applying geodesic to each row
        # This is much faster than using apply with geodesic
        lat_rad = np.radians(station_lat)
        lon_rad = np.radians(station_lon)
        df_lat_rad = np.radians(df["latitude"])
        df_lon_rad = np.radians(df["longitude"])
        
        # Haversine formula
        dlon = df_lon_rad - lon_rad
        dlat = df_lat_rad - lat_rad
        a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(df_lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distances = 6371000 * c  # Earth radius in meters
        
        # Keep only points within specified radius
        within_radius = df[distances <= radius]
        
        # Calculate median velocities and filter outliers using IQR method
        time_columns = [col for col in within_radius.columns if col.isdigit()]
        velocities = within_radius[time_columns].median(axis=1)
        q1 = velocities.quantile(0.25)
        q3 = velocities.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        normal_points = within_radius[(velocities >= lower_bound) & (velocities <= upper_bound)]
        normal_velocities = velocities[(velocities >= lower_bound) & (velocities <= upper_bound)]
        return normal_points, normal_velocities

    # Filter points within radius for both datasets
    before_points, before_velocities = filter_within_radius(before_df)
    after_points, after_velocities = filter_within_radius(after_df)

    # Create figure with two side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # Subplot 1: Before alignment
    scatter1 = axes[0].scatter(before_points["longitude"], before_points["latitude"],
                               c=before_velocities, cmap="plasma", s=15, alpha=0.7)
    axes[0].set_title("Before Alignment", fontsize=14)
    axes[0].set_xlabel("Longitude", fontsize=12)
    axes[0].set_ylabel("Latitude", fontsize=12)
    axes[0].grid(alpha=0.5)
    fig.colorbar(scatter1, ax=axes[0], label="Velocity (mm/year)")
    
    # Subplot 2: After alignment
    scatter2 = axes[1].scatter(after_points["longitude"], after_points["latitude"],
                               c=after_velocities, cmap="plasma", s=15, alpha=0.7)
    axes[1].set_title("After Alignment", fontsize=14)
    axes[1].set_xlabel("Longitude", fontsize=12)
    axes[1].grid(alpha=0.5)
    fig.colorbar(scatter2, ax=axes[1], label="Velocity (mm/year)")
    
    # Add station marker and label to both subplots
    for ax in axes:
        ax.scatter(station_lon, station_lat, color="black", edgecolor="white", s=50, marker="^", zorder=5)
        ax.text(station_lon, station_lat, station_name,
                color="black", fontsize=10, ha="left", va="bottom",
                path_effects=[path_effects.withStroke(linewidth=1, foreground="white")])
    
    # Add title and save figure
    fig.suptitle(f"Velocity Map for Station {station_name}", fontsize=16)
    output_path = os.path.join(output_dir, f"{station_name}_velocity_map.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)  # Lower DPI for faster saving
    plt.close('all')  # Explicitly close all figures to free memory
    return f"Velocity map saved for station {station_name}: {output_path}"


def plot_station_velocity_map(before_df, after_df, stations_df, output_dir, radius=INSAR_RADIUS):
    """
    Generates a scatter plot for each station showing velocities in longitude-latitude space,
    with two subplots: before and after alignment.
    
    Args:
        before_df: DataFrame containing InSAR data before alignment
        after_df: DataFrame containing InSAR data after alignment
        stations_df: DataFrame containing GNSS station information
        output_dir: Directory to save output plots
        radius: Radius around stations to include InSAR points (default from INSAR_RADIUS)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use multiprocessing to create station plots in parallel
    # Determine number of CPU cores to use (leave one core free for system)
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_cores} CPU cores for parallel processing of station maps")
    
    # Create a partial function with fixed arguments
    plot_func = partial(plot_station_map, before_df=before_df, after_df=after_df, 
                       output_dir=output_dir, radius=radius)
    
    # Convert stations_df to list of Series for multiprocessing
    stations_list = [station for _, station in stations_df.iterrows()]
    
    # Process stations in parallel
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(plot_func, stations_list)
        
    # Print results (could be collected and returned if needed)
    for result in results:
        print(result)
        station_name = station["Station"]
        station_lat = station["latitude"]
        station_lon = station["longitude"]

        def filter_within_radius(df):
            """Filter points within radius of station and remove outliers."""
            # Use vectorized Haversine instead of applying geodesic to each row
            # This is much faster than using apply with geodesic
            lat_rad = np.radians(station_lat)
            lon_rad = np.radians(station_lon)
            df_lat_rad = np.radians(df["latitude"])
            df_lon_rad = np.radians(df["longitude"])
            
            # Haversine formula
            dlon = df_lon_rad - lon_rad
            dlat = df_lat_rad - lat_rad
            a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(df_lat_rad) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distances = 6371000 * c  # Earth radius in meters
            
            # Keep only points within specified radius
            within_radius = df[distances <= radius]
            
            # Calculate median velocities and filter outliers using IQR method
            time_columns = [col for col in within_radius.columns if col.isdigit()]
            velocities = within_radius[time_columns].median(axis=1)
            q1 = velocities.quantile(0.25)
            q3 = velocities.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            normal_points = within_radius[(velocities >= lower_bound) & (velocities <= upper_bound)]
            normal_velocities = velocities[(velocities >= lower_bound) & (velocities <= upper_bound)]
            return normal_points, normal_velocities

        # Filter points within radius for both datasets
        before_points, before_velocities = filter_within_radius(before_df)
        after_points, after_velocities = filter_within_radius(after_df)

        # Create figure with two side-by-side subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        
        # Subplot 1: Before alignment
        scatter1 = axes[0].scatter(before_points["longitude"], before_points["latitude"],
                                   c=before_velocities, cmap="plasma", s=15, alpha=0.7)
        axes[0].set_title("Before Alignment", fontsize=14)
        axes[0].set_xlabel("Longitude", fontsize=12)
        axes[0].set_ylabel("Latitude", fontsize=12)
        axes[0].grid(alpha=0.5)
        fig.colorbar(scatter1, ax=axes[0], label="Velocity (mm/year)")
        
        # Subplot 2: After alignment
        scatter2 = axes[1].scatter(after_points["longitude"], after_points["latitude"],
                                   c=after_velocities, cmap="plasma", s=15, alpha=0.7)
        axes[1].set_title("After Alignment", fontsize=14)
        axes[1].set_xlabel("Longitude", fontsize=12)
        axes[1].grid(alpha=0.5)
        fig.colorbar(scatter2, ax=axes[1], label="Velocity (mm/year)")
        
        # Add station marker and label to both subplots
        for ax in axes:
            ax.scatter(station_lon, station_lat, color="black", edgecolor="white", s=50, marker="^", zorder=5)
            ax.text(station_lon, station_lat, station_name,
                    color="black", fontsize=10, ha="left", va="bottom",
                    path_effects=[path_effects.withStroke(linewidth=1, foreground="white")])
        
        # Add title and save figure
        fig.suptitle(f"Velocity Map for Station {station_name}", fontsize=16)
        output_path = os.path.join(output_dir, f"{station_name}_velocity_map.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=150)  # Lower DPI for faster saving
        plt.close('all')  # Explicitly close all figures to free memory
        print(f"Velocity map saved for station {station_name}: {output_path}")


if __name__ == "__main__":
    # First, generate the combined time series plots
    # These show the time series of displacements for each station
    # Use a single data load for all plotting functions
    print("Loading data for all plots...")
    before_df = pd.read_csv(insar_before)
    after_df = pd.read_csv(insar_after)
    stations_df = pd.read_csv(stations_file, sep=r'\s+')
    stations_df.columns = stations_df.columns.str.strip()

    # Start with the most frequently used data: time series plots
    print("Generating time series plots...")
    plot_combined_time_series()

    # Create global velocity map plot (including velocity correction plane)
    # This shows the regional view of velocities and the correction plane
    print("Generating global velocity map...")
    plot_global_velocity_map(before_df, after_df, stations_df, parameters_file, plots_dir,
                             title="Regional Velocity Map", suffix="combined")

    # Create velocity maps for each station
    # These show detailed local views around each GNSS station
    print("Generating station velocity maps...")
    plot_station_velocity_map(before_df, after_df, stations_df, plots_dir, radius=INSAR_RADIUS)

    print("All plots were successfully saved in the folder 'plots'.")