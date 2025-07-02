"""
Combined Time Series Visualization Script

This script creates comparative visualizations of InSAR and GNSS time series data.
It generates plots showing both data sources together, allowing direct comparison
of displacement patterns, trends, and spatial variations between measurement techniques.
The script provides insights into how well the InSAR and GNSS measurements align after
spatial correction.

Features:
- Side-by-side visualization of InSAR and GNSS time series
- Automatic data alignment to common time references
- Statistical comparison between measurement techniques 
- Linear trend analysis and visualization
- Multiple visualization formats (time series, velocity maps)
- Parallel processing with batch handling for improved performance and memory efficiency
- Comprehensive station-by-station comparison plots
- Global velocity maps with correction plane visualization
- Memory-optimized processing for large datasets

Usage:
- Run directly to process all stations: `python plot_combined_time_series.py`
- Run via master.py to include it in the full workflow
- Outputs are saved to the 'plots' directory in DATA_DIR

Note: This is a computationally intensive script that benefits from multiprocessing.
The batch processing approach balances memory usage with processing efficiency.
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
import gc                       # Garbage collection
import time                     # For timing operations
import math                     # For mathematical operations

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

# Configure matplotlib to use a non-interactive backend for better performance in multiprocessing
plt.switch_backend('agg')

# Global file paths (all files are assumed to be in DATA_DIR)
stations_file = data_dir / os.getenv("STATIONS_FILE", "stations_list")
parameters_file = data_dir / "parameters.csv"
insar_before = data_dir / os.getenv("INSAR_FILE", "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A.csv")
insar_after = insar_before.with_name(insar_before.stem + "_aligned" + insar_before.suffix)

# All plots will be saved in a single folder named "plots"
plots_dir = data_dir / "plots"

# Create plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

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


def create_station_time_series_plot(station_data):
    """
    Creates a time series plot for a single station.
    This function is designed to be called in parallel for multiple stations.
    
    Args:
        station_data: Tuple containing (
            station: Series containing station information,
            before_df: DataFrame containing InSAR data before alignment,
            after_df: DataFrame containing InSAR data after alignment,
            before_time_columns: List of time columns in before_df,
            after_time_columns: List of time columns in after_df,
            before_coords: Array of coordinates from before_df,
            after_coords: Array of coordinates from after_df
        )
        
    Returns:
        Status message indicating success or failure
    """
    # Unpack the input tuple
    station, before_df, after_df, before_time_columns, after_time_columns, before_coords, after_coords = station_data
    
    station_name = station["Station"]
    station_lat = station["latitude"]
    station_lon = station["longitude"]
    
    # Find GNSS file with LOS values for this station
    if USE_NNR_CORRECTED:
        # First try to find LOS files that came from NNR-corrected data
        gnss_pattern = os.path.join(data_dir, f"{station_name}_NEU_TIME*_LOS.txt")
        gnss_files = glob.glob(gnss_pattern)
        
        # If there are multiple LOS files, we'd prefer ones with more recent timestamps
        gnss_files.sort(reverse=True)  # Sort by filename, reverse to get newest first
        
        if not gnss_files:
            return f"GNSS LOS file not found for pattern: {gnss_pattern}. Skipping station {station_name}."
            
    else:
        # If we're not using NNR-corrected files, find normal LOS files
        gnss_pattern = os.path.join(data_dir, f"{station_name}_NEU_TIME*_LOS.txt")
        gnss_files = glob.glob(gnss_pattern)
        
        if not gnss_files:
            return f"GNSS file not found for pattern: {gnss_pattern}. Skipping station {station_name}."
            
    gnss_file = gnss_files[0]
    
    try:
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
        
        # Calculate median displacements for points within radius (more robust to outliers)
        before_displacement = before_df.loc[before_mask, before_time_columns].median()
        after_displacement = after_df.loc[after_mask, after_time_columns].median()

        # Load GNSS data for comparison
        gnss_data = load_gnss_data(gnss_file)
        
        # Convert time columns to datetime for plotting
        time_dates = [pd.to_datetime(col, format="%Y%m%d") for col in before_time_columns]
        
        # Convert dates to decimal years for proper time-based trend analysis
        decimal_years = convert_dates_to_decimal_years(time_dates)

        # Create figure with two vertically stacked subplots - use a smaller figure for faster rendering
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=100)
        
        # Subplot 1: InSAR before alignment
        axes[0].plot(time_dates, before_displacement, 'r.', label="Before Alignment")
        
        # Calculate and plot linear trend for before alignment using actual time intervals
        slope_before, intercept_before, _, _, _ = linregress(
            decimal_years, before_displacement)
        trend_before = slope_before * decimal_years + intercept_before
        axes[0].plot(time_dates, trend_before, 'r-', 
                    label=f"Trend ({slope_before:.5f} mm/year)", 
                    linewidth=2)
        
        # Add labels and formatting
        axes[0].set_title(f"InSAR Before Alignment - {station_name}")
        axes[0].set_ylabel("Displacement (mm)")
        axes[0].legend()
        axes[0].grid(alpha=0.5)
        
        # Subplot 2: Combined time series (after alignment and GNSS)
        axes[1].plot(time_dates, after_displacement, 'b.', label="After Alignment")
        
        # Calculate and plot linear trend for after alignment using actual time intervals
        slope_after, intercept_after, _, _, _ = linregress(
            decimal_years, after_displacement)
        trend_after = slope_after * decimal_years + intercept_after
        axes[1].plot(time_dates, trend_after, 'b-', 
                    label=f"InSAR ({slope_after:.5f} mm/year)", 
                    linewidth=2)
        
        # Add GNSS data and trend (GNSS already uses proper decimal years)
        axes[1].plot(gnss_data["DATE"], gnss_data["LOS"], 'g.', label="GNSS LOS")
        slope_gnss, intercept_gnss, _, _, _ = linregress(
            gnss_data["decimal_year"], gnss_data["LOS"])
        axes[1].plot(gnss_data["DATE"], slope_gnss * gnss_data["decimal_year"] + intercept_gnss,
                     'g-', label=f"GNSS ({slope_gnss:.5f} mm/year)", 
                     linewidth=2)
        
        # Add labels and formatting
        axes[1].set_title(f"Combined After Alignment and GNSS - {station_name}")
        axes[1].set_ylabel("Displacement (mm)")
        axes[1].legend()
        axes[1].grid(alpha=0.5)
        axes[1].set_xlabel("TIME (YYYY-MM-DD)")
        axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Adjust layout and save figure
        plt.tight_layout()
        output_path = os.path.join(plots_dir, f"{station_name}_combined_plot.png")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
        # Explicitly clean up to reduce memory usage
        del before_displacement, after_displacement, gnss_data
        gc.collect()
        
        return f"Combined time series plot saved for station {station_name}: {output_path}"
    
    except Exception as e:
        return f"Error creating plot for station {station_name}: {str(e)}"


def process_stations_in_batches(stations_list, plot_func, batch_size=5):
    """
    Process stations in batches to limit memory usage.
    
    Args:
        stations_list: List of stations to process
        plot_func: Function to apply to each station
        batch_size: Number of stations to process in each batch
        
    Returns:
        List of results from all batches
    """
    all_results = []
    num_batches = math.ceil(len(stations_list) / batch_size)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(stations_list))
        batch = stations_list[start_idx:end_idx]
        
        # Process this batch
        print(f"Processing batch {i+1}/{num_batches} (stations {start_idx+1}-{end_idx})")
        
        # Use multiprocessing for this batch
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(processes=num_cores) as pool:
            batch_results = pool.map(plot_func, batch)
            
        all_results.extend(batch_results)
        
        # Force garbage collection
        gc.collect()
        
    return all_results


def plot_combined_time_series():
    """
    Generates combined time series plots for each GNSS station with two subplots:
      - InSAR time series before alignment,
      - Combined InSAR time series after alignment and GNSS LOS displacement.
    The plots are saved in the "plots" folder.
    
    Uses parallel processing with batching to optimize performance and memory usage.
    """
    start_time = time.time()
    
    # Load InSAR datasets (before and after alignment)
    print("Loading InSAR data for time series plots...")
    before_df = pd.read_csv(insar_before)
    after_df = pd.read_csv(insar_after)
    
    # Load station information
    stations_df = pd.read_csv(stations_file, sep=r'\s+')
    stations_df.columns = stations_df.columns.str.strip()
    
    # Filter InSAR data by temporal coherence to ensure quality
    print("Filtering InSAR data by temporal coherence threshold...")
    before_df = before_df[before_df["temporal_coherence"] >= MIN_TEMPORAL_COHERENCE]
    after_df = after_df[after_df["temporal_coherence"] >= MIN_TEMPORAL_COHERENCE]

    # Precompute coordinates as numpy arrays for fast access (performance optimization)
    before_coords = before_df[["latitude", "longitude"]].values
    after_coords = after_df[["latitude", "longitude"]].values
    before_time_columns = [col for col in before_df.columns if col.isdigit()]
    after_time_columns = [col for col in after_df.columns if col.isdigit()]
    
    # Create a list of data tuples for each station
    stations_data = [
        (station, before_df, after_df, before_time_columns, after_time_columns, before_coords, after_coords)
        for _, station in stations_df.iterrows()
    ]
    
    # Process stations in batches to limit memory usage
    print(f"Creating time series plots for {len(stations_data)} stations...")
    results = process_stations_in_batches(stations_data, create_station_time_series_plot, batch_size=5)
    
    # Print results
    for result in results:
        print(result)
    
    # Free memory
    del before_df, after_df, before_coords, after_coords
    gc.collect()
    
    elapsed_time = time.time() - start_time
    print(f"Time series plots completed in {elapsed_time:.2f} seconds")


def plot_station_map(station_data):
    """
    Generates a velocity map for a single station.
    This function is designed to be called in parallel for multiple stations.
    
    Args:
        station_data: Tuple containing (
            station: Series containing station information,
            before_df: DataFrame containing InSAR data before alignment,
            after_df: DataFrame containing InSAR data after alignment
        )
        
    Returns:
        Status message indicating success or failure
    """
    station, before_df, after_df = station_data
    station_name = station["Station"]
    station_lat = station["latitude"]
    station_lon = station["longitude"]
    
    try:
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
            within_radius = df[distances <= INSAR_RADIUS]
            
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
        output_path = os.path.join(plots_dir, f"{station_name}_velocity_map.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=150)  # Lower DPI for faster saving
        plt.close(fig)
        
        # Clean up memory
        del before_points, after_points, before_velocities, after_velocities
        gc.collect()
        
        return f"Velocity map saved for station {station_name}: {output_path}"
    except Exception as e:
        return f"Error creating velocity map for station {station_name}: {str(e)}"


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
    plt.savefig(output_path, dpi=150)  # Lower DPI for faster saving
    plt.close(fig)  # Explicitly close all figures to free memory
    
    # Explicitly clear some large variables to help with memory management
    del before_points, after_points, before_velocities, after_velocities
    gc.collect()
    
    print(f"Velocity map with correction saved: {output_path}")


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
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a list of data tuples for each station
    stations_data = [
        (station, before_df, after_df)
        for _, station in stations_df.iterrows()
    ]
    
    # Process stations in batches to limit memory usage
    print(f"Creating velocity maps for {len(stations_data)} stations...")
    results = process_stations_in_batches(stations_data, plot_station_map, batch_size=5)
    
    # Print results
    for result in results:
        print(result)
    
    elapsed_time = time.time() - start_time
    print(f"Station velocity maps completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    script_start_time = time.time()
    
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data once for all visualizations
    print("Loading data for all plots...")
    before_df = pd.read_csv(insar_before)
    after_df = pd.read_csv(insar_after)
    stations_df = pd.read_csv(stations_file, sep=r'\s+')
    stations_df.columns = stations_df.columns.str.strip()
    
    # Filter InSAR data by temporal coherence to ensure quality
    print("Filtering InSAR data by temporal coherence threshold...")
    before_df = before_df[before_df["temporal_coherence"] >= MIN_TEMPORAL_COHERENCE]
    after_df = after_df[after_df["temporal_coherence"] >= MIN_TEMPORAL_COHERENCE]

    # Start with the most frequently used data: time series plots
    print("\n=== Generating time series plots ===")
    start_time = time.time()
    plot_combined_time_series()
    print(f"Time series plots completed in {time.time() - start_time:.2f} seconds")

    # Create global velocity map plot (including velocity correction plane)
    # This shows the regional view of velocities and the correction plane
    print("\n=== Generating global velocity map ===")
    start_time = time.time()
    plot_global_velocity_map(before_df, after_df, stations_df, parameters_file, plots_dir,
                         title="Regional Velocity Map", suffix="combined")
    print(f"Global velocity map completed in {time.time() - start_time:.2f} seconds")

    # Create velocity maps for each station
    # These show detailed local views around each GNSS station
    print("\n=== Generating station velocity maps ===")
    start_time = time.time()
    plot_station_velocity_map(before_df, after_df, stations_df, plots_dir, radius=INSAR_RADIUS)
    print(f"Station velocity maps completed in {time.time() - start_time:.2f} seconds")

    # Report total run time and completion
    total_time = time.time() - script_start_time
    print(f"\nAll plots were successfully saved in '{plots_dir}'")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")