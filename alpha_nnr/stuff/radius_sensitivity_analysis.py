"""
Radius Sensitivity Analysis for InSAR-GNSS Data

This script analyzes the influence of different radius values on the creation of
InSAR time series for GNSS stations. It provides the following functions:
1. Sensitivity analysis with different radius values for individual stations
2. Spatial analysis of InSAR point distribution around the stations
3. Comparison of nearby stations with different radius values

The results are saved as graphs and CSV files.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
import seaborn as sns
from pathlib import Path
from geopy.distance import geodesic

# Make sure the script works as an import
if __name__ == "__main__" and not hasattr(sys, "ps1"):
    # Get environment variables
    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        data_dir = Path("C:/insar_gnss_data")
    else:
        data_dir = Path(data_dir)

    # File paths
    stations_file = data_dir / os.getenv("STATIONS_FILE", "stations_list")
    insar_after = data_dir / "insar_after_correction.csv"
    insar_before = data_dir / "insar_filtered.csv"
    gnss_los_dir = data_dir / "gnss_los"
    output_dir = data_dir / "radius_analysis"
    os.makedirs(output_dir, exist_ok=True)


def haversine_distance_vectorized(lat1, lon1, lat2_array, lon2_array):
    """
    Calculates the Haversine distance between one point and an array of points.
    
    Args:
        lat1 (float): Latitude of the first point (in degrees)
        lon1 (float): Longitude of the first point (in degrees)
        lat2_array (array): Array of latitudes of the second points (in degrees)
        lon2_array (array): Array of longitudes of the second points (in degrees)
        
    Returns:
        array: Array of distances in kilometers
    """
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_array)
    lon2_rad = np.radians(lon2_array)
      # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in kilometers
    
    return c * r


def find_gnss_los_file(station_name, los_dir=None):
    """Finds the GNSS-LOS file for a specific station."""
    potential_files = []
    
    # Search in gnss_los directory if specified
    if los_dir is None:
        los_dir = os.path.join(data_dir, "gnss_los")
    
    if os.path.exists(los_dir):
        potential_files.extend([
            os.path.join(los_dir, f"{station_name.lower()}_los_displacement.txt"),
            os.path.join(los_dir, f"{station_name.upper()}_los_displacement.txt")
        ])
    
    # Also search in the main directory with the actual filename pattern
    potential_files.extend([
        os.path.join(data_dir, f"{station_name.lower()}_NEU_TIME_no_tide_LOS.txt"),
        os.path.join(data_dir, f"{station_name.upper()}_NEU_TIME_no_tide_LOS.txt")
    ])
    
    for file_path in potential_files:
        if os.path.exists(file_path):
            print(f"GNSS-LOS file found: {file_path}")
            return file_path
    
    print(f"No GNSS-LOS file found for station {station_name}.")
    return None


def load_gnss_data(gnss_file):
    """Loads GNSS data from a file and handles the formatting."""
    try:
        # Check if the file is *_LOS.txt (specific format)
        if gnss_file.endswith('_LOS.txt'):
            # Specific format for LOS files: Check first few lines
            data = []
            with open(gnss_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('MJD') or line.startswith('---') or "in mm" in line:
                        continue
                    parts = line.split()
                    if len(parts) >= 7:  # Full format with North, East, Up, LOS
                        try:
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
                    elif len(parts) >= 3:  # Simple format with MJD, TIME, LOS
                        try:
                            mjd = float(parts[0])
                            # Time might be a single column or split
                            if len(parts) >= 4:  # Time is split into date and time
                                time_str = parts[1] + " " + parts[2]
                                los = float(parts[3])
                            else:  # Time is a single column
                                time_str = parts[1]
                                los = float(parts[2])
                            data.append([mjd, time_str, los])
                        except ValueError:
                            print(f"Skipping invalid line in GNSS file: {line}")
                            continue
            
            # Create DataFrame based on the number of columns
            if not data:
                print(f"No valid data found in {gnss_file}")
                return None
                
            if len(data[0]) == 6:  # Full format
                df = pd.DataFrame(data, columns=["MJD", "TIME", "North", "East", "Up", "LOS"])
            else:  # Simple format
                df = pd.DataFrame(data, columns=["MJD", "TIME", "LOS"])
            
            # Convert MJD to datetime format
            df["DATE"] = pd.to_datetime(df["MJD"], origin="1858-11-17", unit="D")
            
            return df
        else:
            # Standard CSV format for other files
            gnss_data = pd.read_csv(gnss_file, sep='\s+')
            return gnss_data
    except Exception as e:
        print(f"Error loading GNSS data: {e}")
        return None


def perform_radius_sensitivity_analysis(station_name, station_lat, station_lon, radii=None, output_dir=None, insar_df=None):
    """
    Creates a comparative visualization of InSAR time series for different radius values
    for a specific GNSS station.
    
    Args:
        station_name (str): Name of the GNSS station
        station_lat (float): Latitude of the station
        station_lon (float): Longitude of the station
        radii (list): List of radius values in meters
        output_dir (str): Output directory for plots
        insar_df (DataFrame): DataFrame with InSAR data (optional)
    
    Returns:
        DataFrame: DataFrame with InSAR points within the maximum radius
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, "radius_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    if radii is None:
        radii = [100, 200, 300, 500, 750, 1000]
      # Load InSAR data (after alignment) with fallback options
    if insar_df is None:
        # Option 1: Standard file "insar_after_correction.csv"
        insar_after = os.path.join(data_dir, "insar_after_correction.csv")
        if os.path.exists(insar_after):
            print(f"Loading InSAR data from {insar_after}...")
            insar_df = pd.read_csv(insar_after)
        
        # Option 2: Aligned EGMS file
        if insar_df is None:
            aligned_file = os.path.join(data_dir, "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A_aligned.csv")
            if os.path.exists(aligned_file):
                print(f"Loading InSAR data from {aligned_file}...")
                insar_df = pd.read_csv(aligned_file)
        
        # Option 3: Try to load a station-specific file
        if insar_df is None:
            station_file = os.path.join(data_dir, f"INSAR_{station_name}_filtered.csv")
            if os.path.exists(station_file):
                print(f"Loading InSAR data from {station_file}...")
                insar_df = pd.read_csv(station_file)
    
    # Extract timestamps from column names (format: YYYYMMDD)
    time_columns = [col for col in insar_df.columns if col.isdigit() and len(col) == 8]
    time_dates = [pd.to_datetime(col, format="%Y%m%d") for col in time_columns]
    
    fig, ax = plt.subplots(figsize=(12, 8))
      # Colors for different radii
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Calculate Haversine distances to all InSAR points (once)
    distances = haversine_distance_vectorized(
        station_lat, station_lon,
        insar_df["latitude"].values, insar_df["longitude"].values
    ) * 1000  # Convert to meters
    
    # DataFrame to store statistical data
    stats_data = []
    
    for i, radius in enumerate(radii):
        # Filter points within the current radius
        within_radius = insar_df[distances <= radius]
        
        # Calculate the median displacement for each time point (more robust to outliers)
        median_displacement = within_radius[time_columns].median() if len(within_radius) > 0 else pd.Series(np.nan, index=time_columns)
        
        # Number of points for this radius
        num_points = len(within_radius)
          # Calculate regression line
        if num_points > 0 and not median_displacement.isna().all():
            # Convert data for regression
            dates_numeric = mdates.date2num(time_dates)
            valid_dates = [d for i, d in enumerate(dates_numeric) if not np.isnan(median_displacement.iloc[i])]
            valid_displacements = [d for d in median_displacement if not np.isnan(d)]
            
            if len(valid_dates) > 1:
                # Linear fit
                z = np.polyfit(valid_dates, valid_displacements, 1)
                p = np.poly1d(z)
                
                # Calculate slope in mm/year
                days_in_year = 365.25
                slope_mm_per_day = z[0]
                slope_mm_per_year = slope_mm_per_day * days_in_year
                
                # Plot the regression line
                ax.plot(time_dates, p(dates_numeric), '--', color=colors[i % len(colors)], linewidth=1)
            else:
                slope_mm_per_year = np.nan
        else:
            slope_mm_per_year = np.nan
          # Plot the time series for this radius
        label = f"Radius: {radius}m ({num_points} points)"
        if not np.isnan(slope_mm_per_year):
            label += f", {slope_mm_per_year:.2f} mm/year"
            
        ax.plot(time_dates, median_displacement, 'o-', color=colors[i % len(colors)], 
                label=label, linewidth=1.5, markersize=4, alpha=0.7)
        
        # Save statistics
        stats_data.append({
            'Radius (m)': radius,
            'Number of points': num_points,
            'Slope (mm/year)': slope_mm_per_year,
            'Mean coherence': within_radius['temporal_coherence'].mean() if 'temporal_coherence' in within_radius.columns and num_points > 0 else np.nan
        })
      # Add GNSS time series (if available)
    gnss_file = find_gnss_los_file(station_name)
    if gnss_file:
        gnss_data = load_gnss_data(gnss_file)
        if gnss_data is not None and 'DATE' in gnss_data.columns:
            # Use converted datetime objects directly
            gnss_dates = gnss_data['DATE']
            ax.plot(gnss_dates, gnss_data['LOS'], 's-', color='black', 
                    label="GNSS LOS", linewidth=1.5, markersize=5)
            
            # Calculate GNSS trend
            gnss_dates_numeric = mdates.date2num(gnss_dates)
            gnss_z = np.polyfit(gnss_dates_numeric, gnss_data['LOS'], 1)
            gnss_p = np.poly1d(gnss_z)
            
            # Calculate slope in mm/year
            days_in_year = 365.25
            gnss_slope_mm_per_year = gnss_z[0] * days_in_year
            
            # Plot GNSS regression line
            ax.plot(gnss_dates, gnss_p(gnss_dates_numeric), '--', color='black', linewidth=1.5,
                   label=f"GNSS Trend: {gnss_slope_mm_per_year:.2f} mm/year")
    
    ax.set_title(f"Radius Sensitivity Analysis - Station {station_name}", fontsize=16)
    ax.set_ylabel("Displacement (mm)", fontsize=14)
    ax.set_xlabel("Date", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # Format x-axis date labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Save the graph
    plot_path = os.path.join(output_dir, f"radius_sensitivity_{station_name}.png")    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Sensitivity analysis for station {station_name} saved to: {plot_path}")
    
    # Save statistics
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_path = os.path.join(output_dir, f"radius_statistics_{station_name}.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Statistics for different radii saved to: {stats_path}")
    
    # Return DataFrame for further analysis
    max_radius = max(radii)
    return insar_df[distances <= max_radius].copy()


def analyze_insar_spatial_distribution(station_name, station_lat, station_lon, max_radius=1000, output_dir=None, insar_df=None):
    """
    Creates a map showing the spatial distribution of InSAR points around a GNSS station,
    as well as statistics on point density in different radius ranges.
    
    Args:
        station_name (str): Name of the GNSS station
        station_lat (float): Latitude of the station        station_lon (float): Longitude of the station
        max_radius (int): Maximum radius in meters for the analysis
        output_dir (str): Output directory for plots
        insar_df (DataFrame): DataFrame with InSAR data (optional)
        
    Returns:
        list: List with statistical data for different radii
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, "radius_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load InSAR data (after alignment) with fallback options
    if insar_df is None:
        # Option 1: Standard file "insar_after_correction.csv"
        insar_after = os.path.join(data_dir, "insar_after_correction.csv")
        if os.path.exists(insar_after):
            print(f"Loading InSAR data from {insar_after}...")
            insar_df = pd.read_csv(insar_after)
        
        # Option 2: Aligned EGMS file
        if insar_df is None:
            aligned_file = os.path.join(data_dir, "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A_aligned.csv")
            if os.path.exists(aligned_file):
                print(f"Loading InSAR data from {aligned_file}...")
                insar_df = pd.read_csv(aligned_file)
          # Option 3: Try to load a station-specific file
        if insar_df is None:
            station_file = os.path.join(data_dir, f"INSAR_{station_name}_filtered.csv")
            if os.path.exists(station_file):
                print(f"Loading InSAR data from {station_file}...")
                insar_df = pd.read_csv(station_file)
      # Calculate Haversine distances to all InSAR points
    distances = haversine_distance_vectorized(
        station_lat, station_lon,
        insar_df["latitude"].values, insar_df["longitude"].values
    ) * 1000  # Convert to meters
    
    insar_df = insar_df.copy()
    insar_df['distance_to_station'] = distances
    
    # Filter points up to the maximum radius
    insar_filtered = insar_df[insar_df['distance_to_station'] <= max_radius].copy()
    
    # If not enough points are found, increase the radius adaptively
    if len(insar_filtered) < 10:
        print(f"Too few points within {max_radius}m found. Increasing the radius...")
        # Find the nearest 100 points
        insar_df_sorted = insar_df.sort_values('distance_to_station')
        insar_filtered = insar_df_sorted.head(100).copy()
        max_radius = insar_filtered['distance_to_station'].max()
        print(f"New radius: {max_radius:.1f}m")
    
    # Create a map of InSAR points around the station
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Spatial distribution
    scatter = ax[0].scatter(
        insar_filtered['longitude'], insar_filtered['latitude'],
        c=insar_filtered['distance_to_station'], cmap='viridis', 
        alpha=0.7, s=15, vmin=0, vmax=max_radius
    )
    
    # Mark the GNSS station
    ax[0].scatter([station_lon], [station_lat], color='red', marker='^', s=150, 
                 label=f'GNSS Station {station_name}', zorder=5, edgecolors='black')
    
    # Draw circles for different radii
    radii_to_plot = [100, 300, 500, 1000]
    for radius in radii_to_plot:
        if radius <= max_radius:            # Convert meters to degrees
            lat_km_per_deg = 111.32
            lon_km_per_deg = 111.32 * np.cos(np.deg2rad(station_lat))
            
            radius_deg_lat = radius / (lat_km_per_deg * 1000)
            radius_deg_lon = radius / (lon_km_per_deg * 1000)
            
            # Draw ellipse (appears as circle with correct aspect ratio)
            circle = Circle((station_lon, station_lat), radius_deg_lat, 
                           fill=False, edgecolor='red', linestyle='--', alpha=0.7)
            ax[0].add_patch(circle)
            
            # Add text for the radius
            angle = np.deg2rad(45)  # Position of text at 45 degrees
            x_text = station_lon + radius_deg_lon * np.cos(angle)
            y_text = station_lat + radius_deg_lat * np.sin(angle)
            ax[0].text(x_text, y_text, f'{radius}m', fontsize=9, color='red',
                      horizontalalignment='left', verticalalignment='bottom')
    ax[0].set_title(f'InSAR Point Distribution around Station {station_name}', fontsize=14)
    ax[0].set_xlabel('Longitude (°)', fontsize=12)
    ax[0].set_ylabel('Latitude (°)', fontsize=12)
    ax[0].grid(True, alpha=0.3)
      # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax[0])
    cbar.set_label('Distance to Station (m)', fontsize=12)
    
    # 2. Point density histogram for different radii
    radii_bins = np.linspace(0, max_radius, 20)
    hist_data = []
    for i in range(len(radii_bins)-1):
        r_min, r_max = radii_bins[i], radii_bins[i+1]
        points_in_ring = insar_filtered[(insar_filtered['distance_to_station'] >= r_min) & 
                                     (insar_filtered['distance_to_station'] < r_max)]
        hist_data.append(len(points_in_ring))
    ax[1].bar(radii_bins[:-1], hist_data, width=radii_bins[1]-radii_bins[0], 
             alpha=0.7, color='skyblue', edgecolor='navy')
    
    # Mark the currently used radius
    current_radius = float(os.getenv("INSAR_RADIUS", "500"))
    if current_radius <= max_radius:
        ax[1].axvline(current_radius, color='red', linestyle='--', 
                     label=f'Current Radius: {current_radius}m')
    
    ax[1].set_title(f'InSAR Point Density by Distance', fontsize=14)
    ax[1].set_xlabel('Distance to Station (m)', fontsize=12)
    ax[1].set_ylabel('Number of InSAR Points', fontsize=12)
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"spatial_distribution_{station_name}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Spatial analysis for station {station_name} saved to: {plot_path}")
    
    # Table with statistical information for different radii
    stats_data = []
    standard_radii = [100, 200, 300, 500, 750, 1000, 1500]
    time_columns = [col for col in insar_df.columns if col.isdigit() and len(col) == 8]
    
    for radius in standard_radii:
        if radius <= max(max_radius, current_radius * 1.5):
            points = insar_df[insar_df['distance_to_station'] <= radius]
            num_points = len(points)
              # Calculate statistical metrics of the time series
            if num_points > 0 and time_columns:
                # Calculate mean temporal coherence and velocity if available
                mean_coh = points['temporal_coherence'].mean() if 'temporal_coherence' in points.columns else np.nan
                mean_vel = points['mean_velocity'].mean() if 'mean_velocity' in points.columns else np.nan
                std_vel = points['mean_velocity'].std() if 'mean_velocity' in points.columns else np.nan
                
                # Calculate mean velocity also from time series
                mean_displ = points[time_columns].mean()
                dates = [pd.to_datetime(col, format="%Y%m%d") for col in time_columns]
                dates_numeric = mdates.date2num(dates)
                
                # Remove NaN values
                valid_indices = ~np.isnan(mean_displ)
                valid_dates = [dates_numeric[i] for i, v in enumerate(valid_indices) if v]
                valid_displ = [mean_displ[i] for i, v in enumerate(valid_indices) if v]
                if len(valid_dates) > 1:
                    # Linear fit
                    z = np.polyfit(valid_dates, valid_displ, 1)
                    # Calculate slope in mm/year
                    days_in_year = 365.25
                    slope_mm_per_year = z[0] * days_in_year
                else:
                    slope_mm_per_year = np.nan
                
                stats_data.append({
                    'Radius (m)': radius,
                    'Number of points': num_points,
                    'Point density (per km²)': num_points / (np.pi * (radius/1000)**2),
                    'Mean velocity (mm/year)': mean_vel,
                    'Calculated velocity (mm/year)': slope_mm_per_year,
                    'Std. dev. velocity (mm/year)': std_vel,
                    'Mean coherence': mean_coh
                })
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_path = os.path.join(output_dir, f"spatial_statistics_{station_name}.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Statistics for different radii saved to: {stats_path}")
        
    return stats_data


def compare_nearby_stations(station_names, radii=None, output_dir=None, insar_df=None):
    """
    Compares InSAR time series for nearby stations at different radius values.
    
    Args:
        station_names (list): List of station names
        radii (list): List of radius values in meters
        output_dir (str): Output directory for plots
        insar_df (DataFrame): DataFrame with InSAR data (optional)
        
    Returns:
        tuple: (DataFrame with summarized statistics, distances between stations)
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, "radius_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    if radii is None:
        radii = [100, 200, 300, 500, 750, 1000]
    
    # Load station data
    stations_file_path = os.path.join(data_dir, os.getenv("STATIONS_FILE", "stations_list"))
    stations_df = pd.read_csv(stations_file_path, sep='\s+')
    stations_df = stations_df[stations_df['Station'].isin(station_names)]
    
    if len(stations_df) < 2:
        print(f"Not enough stations found. Available: {stations_df['Station'].tolist()}")
        return None, None
      # Load InSAR data (after alignment) with fallback options
    if insar_df is None:
        # Option 1: Standard file "insar_after_correction.csv"
        insar_after = os.path.join(data_dir, "insar_after_correction.csv")
        if os.path.exists(insar_after):
            print(f"Loading InSAR data from {insar_after}...")
            insar_df = pd.read_csv(insar_after)
        
        # Option 2: Aligned EGMS file
        if insar_df is None:
            aligned_file = os.path.join(data_dir, "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A_aligned.csv")
            if os.path.exists(aligned_file):
                print(f"Loading InSAR data from {aligned_file}...")
                insar_df = pd.read_csv(aligned_file)
          # Option 3: Combine individual station files
        if insar_df is None:
            station_dfs = []
            for station_name in station_names:
                station_file = os.path.join(data_dir, f"INSAR_{station_name}_filtered.csv")
                if os.path.exists(station_file):
                    print(f"Loading InSAR data from {station_file}...")
                    df = pd.read_csv(station_file)
                    station_dfs.append(df)
            
            if station_dfs:
                insar_df = pd.concat(station_dfs, ignore_index=True)
    
    time_columns = [col for col in insar_df.columns if col.isdigit() and len(col) == 8]
    time_dates = [pd.to_datetime(col, format="%Y%m%d") for col in time_columns]
    
    # Calculate the distances between stations
    station_distances = {}
    for i in range(len(stations_df)):
        for j in range(i+1, len(stations_df)):
            station1 = stations_df.iloc[i]
            station2 = stations_df.iloc[j]
            station_distance = geodesic(
                (station1['latitude'], station1['longitude']), 
                (station2['latitude'], station2['longitude'])
            ).meters
            station_pair = f"{station1['Station']}_{station2['Station']}"
            station_distances[station_pair] = station_distance
            print(f"Distance between {station1['Station']} and {station2['Station']}: {station_distance:.1f} meters")
      # Create a combined time series plot for all stations and radii
    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))
      # Colors for stations (case-insensitive with duplicate entries)
    station_colors = {
        'BUHZ': 'blue', 'buhz': 'blue',
        'GLEE': 'red', 'glee': 'red'
    }  # Specific colors for known stations
    default_colors = ['green', 'purple', 'orange', 'brown', 'pink', 'gray']
    color_index = 0
    
    # Style for different radii
    line_styles = ['-', '--', ':', '-.']
    marker_styles = ['o', 's', '^', 'v', 'D', '*']
    
    # Statistics for all stations and radii
    all_stats = []
    
    for i, station in stations_df.iterrows():
        station_name = station['Station']
        station_lat = station['latitude']
        station_lon = station['longitude']
        
        # Choose color for this station
        if station_name in station_colors:
            station_color = station_colors[station_name]
        else:
            station_color = default_colors[color_index % len(default_colors)]
            color_index += 1
        
        # We calculate the distances only once per station
        distances = haversine_distance_vectorized(
            station_lat, station_lon,
            insar_df["latitude"].values, insar_df["longitude"].values
        ) * 1000  # Convert to meters
        
        # Load GNSS data for this station
        gnss_file = find_gnss_los_file(station_name)
        if gnss_file:
            gnss_data = load_gnss_data(gnss_file)
            if gnss_data is not None:
                gnss_dates = [pd.to_datetime(d) for d in gnss_data['TIME']]
                
                # Plot GNSS data in the combined plot
                gnss_label = f"GNSS {station_name}"
                ax_combined.plot(gnss_dates, gnss_data['LOS'], 'o-', color=station_color, 
                               alpha=0.7, linewidth=2, label=gnss_label, markersize=5)
                
                # Calculate GNSS trend
                gnss_dates_numeric = mdates.date2num(gnss_dates)
                if len(gnss_dates) > 1:  # At least 2 points for linear regression
                    gnss_z = np.polyfit(gnss_dates_numeric, gnss_data['LOS'], 1)
                    gnss_p = np.poly1d(gnss_z)
                    
                    # Calculate slope in mm/year
                    days_in_year = 365.25
                    gnss_slope_mm_per_year = gnss_z[0] * days_in_year
                    
                    # Plot the GNSS regression line
                    ax_combined.plot(gnss_dates, gnss_p(gnss_dates_numeric), '--', color=station_color, linewidth=1,
                                  alpha=0.7)
                      # Add GNSS trend to the legend (with dashed line style)
                    gnss_line = plt.Line2D([0], [0], color=station_color, linestyle='--', linewidth=1)
                    
                    # Remove current legend if present
                    if ax_combined.get_legend() is not None:
                        ax_combined.get_legend().remove()
                    
                    handles, labels = ax_combined.get_legend_handles_labels()
                    handles.append(gnss_line)
                    labels.append(f"{gnss_label} Trend: {gnss_slope_mm_per_year:.2f} mm/year")
                    ax_combined.legend(handles, labels, fontsize=9)
        
        for j, radius in enumerate(radii):
            # Filter points within the current radius
            within_radius = insar_df[distances <= radius]
            num_points = len(within_radius)
            
            # Calculate median displacement for each time point (more robust to outliers)
            median_displacement = within_radius[time_columns].median() if num_points > 0 else pd.Series(np.nan, index=time_columns)
            
            # Calculate trend
            if num_points > 0 and not median_displacement.isna().all():
                # Convert data for regression
                dates_numeric = mdates.date2num(time_dates)
                valid_dates = [d for i, d in enumerate(dates_numeric) if i < len(median_displacement) and not np.isnan(median_displacement.iloc[i])]
                valid_displacements = [d for d in median_displacement if not np.isnan(d)]
                
                if len(valid_dates) > 1:
                    # Linear fit
                    z = np.polyfit(valid_dates, valid_displacements, 1)
                    p = np.poly1d(z)
                    
                    # Calculate slope in mm/year
                    days_in_year = 365.25
                    slope_mm_per_year = z[0] * days_in_year
                    
                    # Plot the regression line in the combined plot
                    ax_combined.plot(time_dates, p(dates_numeric), '--', color=station_color, alpha=0.5, linewidth=1)
                else:
                    slope_mm_per_year = np.nan
            else:
                slope_mm_per_year = np.nan
            
            # Choose line style and marker
            line_style = line_styles[j % len(line_styles)]
            marker = marker_styles[j % len(marker_styles)]
            
            # Plot in the combined plot
            label = f"InSAR {station_name}, R={radius}m ({num_points} points)"
            if not np.isnan(slope_mm_per_year):
                label += f", {slope_mm_per_year:.2f} mm/year"
            
            ax_combined.plot(time_dates, median_displacement, marker=marker, linestyle=line_style, color=station_color, 
                           alpha=0.6, linewidth=1.5, label=label, markersize=4)
            
            # Add statistics
            if num_points > 0:
                all_stats.append({
                    'Station': station_name,
                    'Radius (m)': radius,
                    'Number of points': num_points,
                    'Point density (per km²)': num_points / (np.pi * (radius/1000)**2),
                    'InSAR Trend (mm/year)': slope_mm_per_year,
                    'Mean coherence': within_radius['temporal_coherence'].mean() if 'temporal_coherence' in within_radius.columns else np.nan
                })
      # Finalize combined plot
    station_names_str = '_'.join(station_names)
    distance_str = ', '.join([f"{pair.split('_')[0]}-{pair.split('_')[1]}: {dist:.0f}m" for pair, dist in station_distances.items()])
    ax_combined.set_title(f"Comparison of Stations ({distance_str})", fontsize=14)
    ax_combined.set_xlabel("Date", fontsize=12)
    ax_combined.set_ylabel("Displacement (mm)", fontsize=12)
    ax_combined.grid(True, alpha=0.3)
    
    # Optimize the legend
    handles, labels = ax_combined.get_legend_handles_labels()
    
    # Group the legend entries by stations
    by_station = {}
    for h, l in zip(handles, labels):
        station = l.split(',')[0]  # Extract station names
        if station not in by_station:
            by_station[station] = []
        by_station[station].append((h, l))
    
    # Create a new, grouped legend
    all_handles, all_labels = [], []
    for station, items in by_station.items():
        for h, l in items:
            all_handles.append(h)
            all_labels.append(l)
        # Add separator line except after the last group
        if station != list(by_station.keys())[-1]:
            all_handles.append(plt.Line2D([0], [0], color='none'))
            all_labels.append('')
    
    ax_combined.legend(all_handles, all_labels, fontsize=9, loc='best')
      # Format date axis
    ax_combined.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax_combined.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Save the combined plot
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, f"station_radius_comparison_{station_names_str}.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.close()
    
    print(f"Station comparison saved to: {combined_plot_path}")
    
    # Create a heatmap plot of the statistical data
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        
        # Create a pivot table for the heatmap
        pivot_stats = {}
        for metric in ['Number of points', 'InSAR Trend (mm/year)', 'Mean coherence']:
            if metric in stats_df.columns:
                pivot = stats_df.pivot(index='Station', columns='Radius (m)', values=metric)
                  # Create heatmap
                plt.figure(figsize=(10, 6))
                sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.2f' if metric == 'InSAR Trend (mm/year)' else '.1f')
                plt.title(f"{metric} by station and radius", fontsize=14)
                plt.tight_layout()
                
                # Save heatmap                # Replace invalid characters in filenames
                metric_filename = metric.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
                heat_path = os.path.join(output_dir, f"heatmap_{metric_filename}_{station_names_str}.png")
                plt.savefig(heat_path, dpi=300)
                plt.close()
                
                print(f"Heatmap for {metric} saved to: {heat_path}")
                
                pivot_stats[metric] = pivot
          # Save the statistics
        stats_path = os.path.join(output_dir, f"comparison_statistics_{station_names_str}.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Comparison statistics saved to: {stats_path}")
        
        return stats_df, station_distances
    
    return None, station_distances


def run_full_analysis(station_names=None, radii=None, output_dir=None, force_rerun=False):
    """
    Performs a complete analysis for specific stations, including
    radius sensitivity, spatial distribution, and comparison of nearby stations.
    
    Args:
        station_names (list): List of stations to analyze, or None for all
        radii (list): List of radius values to test in meters
        output_dir (str): Output directory
        force_rerun (bool): Whether to overwrite existing analyses
        
    Returns:
        dict: Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, "radius_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    if radii is None:
        radii = [100, 200, 300, 500, 750, 1000]
    
    # Load station data
    stations_file_path = os.path.join(data_dir, os.getenv("STATIONS_FILE", "stations_list"))
    stations_df = pd.read_csv(stations_file_path, sep='\s+')
      # Filter by specific stations if specified
    if station_names:
        # Case-insensitive search: Convert station names to lowercase
        stations_df['Station_lower'] = stations_df['Station'].str.lower()
        station_names_lower = [sname.lower() for sname in station_names]
        stations_df = stations_df[stations_df['Station_lower'].isin(station_names_lower)]
        stations_df = stations_df.drop('Station_lower', axis=1)  # Remove temporary column
        if len(stations_df) == 0:        print(f"None of the specified stations found: {station_names}")
        return None
      # Load InSAR data (only once) with fallback options
    insar_df = None
    
    # Option 1: Standard file "insar_after_correction.csv"
    insar_after = os.path.join(data_dir, "insar_after_correction.csv")
    if os.path.exists(insar_after):
        print(f"Loading InSAR data from {insar_after}...")
        insar_df = pd.read_csv(insar_after)
    
    # Option 2: Aligned EGMS file
    if insar_df is None:
        aligned_file = os.path.join(data_dir, "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A_aligned.csv")
        if os.path.exists(aligned_file):
            print(f"Loading InSAR data from {aligned_file}...")
            insar_df = pd.read_csv(aligned_file)
    
    # Option 3: Load individual filtered station files and combine them
    if insar_df is None:
        print("Load filtered station files and combine them...")
        station_dfs = []
        for station_name in stations_df['Station']:
            station_file = os.path.join(data_dir, f"INSAR_{station_name}_filtered.csv")
            if os.path.exists(station_file):
                print(f"  - Loading {station_file}")
                df = pd.read_csv(station_file)
                station_dfs.append(df)
        
        if station_dfs:
            insar_df = pd.concat(station_dfs, ignore_index=True)
    
    if insar_df is None:
        print("Error: No InSAR data found. Please ensure that the files exist.")
        return None
    
    print(f"Analyzing {len(stations_df)} stations with {len(radii)} different radius values and {len(insar_df)} InSAR points...")
    
    results = {}
      # 1. Run individual analyses for each station
    for i, station in stations_df.iterrows():
        station_name = station['Station']
        print(f"\nAnalyzing station {station_name}...")
          # Perform radius sensitivity analysis
        perform_radius_sensitivity_analysis(
            station_name, station['latitude'], station['longitude'],
            radii=radii, output_dir=output_dir, insar_df=insar_df
        )
        
        # Perform spatial analysis
        analyze_insar_spatial_distribution(
            station_name, station['latitude'], station['longitude'],
            max_radius=max(radii), output_dir=output_dir, insar_df=insar_df
        )
      # 2. Perform comparison analyses for nearby stations
    if len(stations_df) >= 2:
        print("\nPerforming comparison analyses for station groups...")
          # BUHZ and GLEE as special case, if available (case-insensitive)
        if any(s.lower() == 'buhz' for s in stations_df['Station'].str.lower()) and any(s.lower() == 'glee' for s in stations_df['Station'].str.lower()):            # Find the correct spellings of the station names as they appear in the file
            buhz_name = stations_df[stations_df['Station'].str.lower() == 'buhz']['Station'].iloc[0]
            glee_name = stations_df[stations_df['Station'].str.lower() == 'glee']['Station'].iloc[0]
            print(f"\nComparing the stations {buhz_name} and {glee_name}...")
            stats_df, distances = compare_nearby_stations(
                [buhz_name, glee_name], radii=radii, output_dir=output_dir, insar_df=insar_df
            )
            if stats_df is not None:
                results['BUHZ_GLEE'] = {
                    'statistics': stats_df,
                    'distances': distances
                }
          # Compare all stations (if more than 2)
        if len(stations_df) > 2:
            print("\nComparing all specified stations...")
            all_stats_df, all_distances = compare_nearby_stations(
                stations_df['Station'].tolist(), radii=radii, output_dir=output_dir, insar_df=insar_df
            )
            if all_stats_df is not None:
                results['all_stations'] = {
                    'statistics': all_stats_df,
                    'distances': all_distances
                }
    
    print(f"\nAnalysis completed. All results were saved in the directory {output_dir}.")
    
    return results


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Radius Sensitivity Analysis for InSAR-GNSS Data')
    
    parser.add_argument('--stations', type=str, nargs='+', default=None,
                        help='List of stations to analyze (default: all)')
    parser.add_argument('--radii', type=int, nargs='+', default=[100, 200, 300, 500, 750, 1000],
                        help='List of radius values to test in meters')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for analysis results')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing analyses')
                        
    args = parser.parse_args()
    print("Starting radius sensitivity analysis...")
    print(f"Stations: {args.stations or 'all'}")
    print(f"Radius values: {args.radii} meters")
    
    results = run_full_analysis(
        station_names=args.stations,
        radii=args.radii,
        output_dir=args.output,
        force_rerun=args.force
    )
    
    print("Analysis completed.")
