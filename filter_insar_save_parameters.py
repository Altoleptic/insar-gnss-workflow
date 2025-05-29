"""
InSAR Filtering and Parameter Extraction Script

This script filters InSAR data points based on temporal coherence and proximity to GNSS stations.
It processes spatial relationships between GNSS stations and InSAR measurements, extracting
relevant parameters and saving filtered data for further analysis.

Features:
- Coherence-based filtering of InSAR points
- Radius-based selection around GNSS stations
- Vectorized Haversine distance calculations for efficiency
- Line-of-sight (LOS) component normalization
- Parameter extraction for GNSS-InSAR integration
"""

import pandas as pd
import numpy as np
import os

# Get DATA_DIR from environment variable
data_dir = os.getenv("DATA_DIR")

if not data_dir:
    print("Error: DATA_DIR environment variable is not set.")
    exit(1)

data_dir = os.path.abspath(data_dir)
station_list_file = os.path.join(data_dir, os.getenv("STATIONS_FILE", "stations_list"))
output_file = os.path.join(data_dir, "parameters.csv")

# Define the Haversine formula using NumPy for vectorized calculations
def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Calculate the geodetic distance between two points."""
    R = 6371.0

    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Filtering function optimized for vectorized calculations
def filter_insar_points_by_station(gnss_stations, insar_data, radius_km, coherence_threshold):
    """Filters InSAR points within a given radius around GNSS stations."""
    filtered_points_dict = {}

    insar_data = insar_data[insar_data['temporal_coherence'] >= coherence_threshold]  # Pre-filter by coherence

    for station in gnss_stations:
        station_name = station['Station']
        station_lat, station_lon = station['latitude'], station['longitude']

        distances = haversine_distance_vectorized(station_lat, station_lon, insar_data['latitude'].values, insar_data['longitude'].values)
        filtered_points_dict[station_name] = insar_data[distances <= radius_km]

    return filtered_points_dict

def normalize_los_components(points):
    """Normalizes the LOS components (east, north, up)."""
    magnitude = np.sqrt(points["los_east"]**2 + points["los_north"]**2 + points["los_up"]**2)
    magnitude = magnitude.replace(0, np.nan)  # Avoid division by zero

    return points.assign(
        los_east=points["los_east"] / magnitude,
        los_north=points["los_north"] / magnitude,
        los_up=points["los_up"] / magnitude
    )

def normalize_vector(east, north, up):
    """Normalizes a vector given its components."""
    magnitude = np.sqrt(east**2 + north**2 + up**2)
    return (east / magnitude, north / magnitude, up / magnitude) if magnitude != 0 else (0, 0, 0)

def calculate_gnss_los_magnitude(results_df):
    """Computes GNSS LOS Magnitude using the dot product of LOS vectors and GNSS velocities."""
    results_df["GNSS LOS Magnitude (mm/year)"] = (
        results_df["Los Unit Vector East norm"] * results_df["GNSS East Velocity (mm/year)"] +
        results_df["Los Unit Vector North norm"] * results_df["GNSS North Velocity (mm/year)"] +
        results_df["Los Unit Vector Up norm"] * results_df["GNSS Up Velocity (mm/year)"]
    )
    return results_df

def save_parameters_to_csv(filtered_data, output_file):
    """Stores velocity and LOS normalization data for GNSS stations."""
    results_df = pd.read_csv(output_file) if os.path.exists(output_file) else pd.DataFrame()

    parameters_data = []
    for station_name, points in filtered_data.items():
        points = normalize_los_components(points)

        median_velocity = points["mean_velocity"].median()
        median_los_east = points["los_east"].median()
        median_los_north = points["los_north"].median()
        median_los_up = points["los_up"].median()

        norm_los_east, norm_los_north, norm_los_up = normalize_vector(median_los_east, median_los_north, median_los_up)

        parameters_data.append({
            "Station": station_name,
            "InSAR LOS-Velocity Median (mm/year)": median_velocity,
            "Los Unit Vector East norm": norm_los_east,
            "Los Unit Vector North norm": norm_los_north,
            "Los Unit Vector Up norm": norm_los_up
        })

    parameters_df = pd.DataFrame(parameters_data)

    if not results_df.empty:
        for col in ["InSAR LOS-Velocity Median (mm/year)", "Los Unit Vector East norm", "Los Unit Vector North norm", "Los Unit Vector Up norm"]:
            if col in results_df.columns:
                results_df = results_df.drop(columns=[col])

        results_df = results_df.merge(parameters_df, on="Station", how="left")
    else:
        results_df = parameters_df

    results_df = calculate_gnss_los_magnitude(results_df)
    results_df.to_csv(output_file, index=False)

def main():
    """Main function to load GNSS and InSAR data, process filtering, and save results."""
    if not os.path.exists(station_list_file):
        print(f"Error: stations_list file not found in {data_dir}.")
        exit(1)

    gnss_stations_df = pd.read_csv(station_list_file, delim_whitespace=True)
    gnss_stations = gnss_stations_df.to_dict(orient="records")

    insar_file = os.path.join(data_dir, os.getenv("INSAR_FILE", "default_insar.csv"))
    insar_data = pd.read_csv(insar_file)

    radius_km = int(os.getenv("INSAR_RADIUS", "500")) / 1000
    coherence_threshold = float(os.getenv("MIN_TEMPORAL_COHERENCE", "0.7"))

    filtered_data = filter_insar_points_by_station(gnss_stations, insar_data, radius_km, coherence_threshold)

    for station_name, points in filtered_data.items():
        filename = os.path.join(data_dir, f"INSAR_{station_name}_filtered.csv")
        points.to_csv(filename, index=False)
        print(f"Saved {len(points)} points for {station_name} to {filename}")

    save_parameters_to_csv(filtered_data, output_file)

if __name__ == "__main__":
    main()