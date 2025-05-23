import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load InSAR file and data directory from environment variables or set default values
DATA_DIR = os.getenv("DATA_DIR", r"C:\insar_gnss_data")
INSAR_FILE = os.getenv("INSAR_FILE", "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A.csv")
# Modify the filename to use the aligned version
insar_name, insar_ext = os.path.splitext(INSAR_FILE)
INSAR_FILE_ALIGNED = f"{insar_name}_aligned{insar_ext}"
INSAR_PATH = os.path.join(DATA_DIR, INSAR_FILE_ALIGNED)

# Parameters for the grid
GRID_SIZE_KM = float(os.getenv("GRID_SIZE_KM", "0.5"))

# Load InSAR data
insar = pd.read_csv(INSAR_PATH)

# Determine the extent of the study area
grid_min_lon = insar['longitude'].min()
grid_max_lon = insar['longitude'].max()
grid_min_lat = insar['latitude'].min()
grid_max_lat = insar['latitude'].max()

# Conversion km -> degree
# 1 degree of latitude = approx. 111.32 km
lat_km_per_deg = 111.32
lon_km_per_deg = 111.32 * np.cos(np.deg2rad((insar['latitude'].min() + insar['latitude'].max()) / 2))

lat_grid_size_deg = GRID_SIZE_KM / lat_km_per_deg
lon_grid_size_deg = GRID_SIZE_KM / lon_km_per_deg

# Create the grid
lon_bins = np.arange(grid_min_lon, grid_max_lon + lon_grid_size_deg, lon_grid_size_deg)
lat_bins = np.arange(grid_min_lat, grid_max_lat + lat_grid_size_deg, lat_grid_size_deg)

# Assign each InSAR measurement to a grid cell
insar['lon_bin'] = np.digitize(insar['longitude'], lon_bins) - 1
insar['lat_bin'] = np.digitize(insar['latitude'], lat_bins) - 1

# Identify time columns (format: YYYYMMDD)
time_cols = [col for col in insar.columns if col.isdigit() and len(col) == 8]

def detrend_timeseries(df, time_cols):
    """
    Remove linear trend from each row's time series.
    Returns a DataFrame with the same shape as df[time_cols].
    """
    from scipy.stats import linregress
    times = np.arange(len(time_cols))
    detrended = np.empty_like(df[time_cols].values, dtype=float)
    for i, row in enumerate(df[time_cols].values):
        slope, intercept, _, _, _ = linregress(times, row)
        detrended[i, :] = row - (slope * times + intercept)
    return pd.DataFrame(detrended, columns=time_cols, index=df.index)

def calculate_amplitude(df, time_cols, use_detrended=True):
    """
    Calculate amplitude (max-min) per point, optionally using detrended time series.
    """
    if use_detrended:
        df_detrended = detrend_timeseries(df, time_cols)
        amplitude = df_detrended.max(axis=1) - df_detrended.min(axis=1)
    else:
        amplitude = df[time_cols].max(axis=1) - df[time_cols].min(axis=1)
    return amplitude

def plot_grid_amplitude(grid_amplitude, lon_bins, lat_bins, vmin, vmax, DATA_DIR, insar, title, grid_size_km):
    """
    Plot the grid amplitude map and save to file.
    """
    import matplotlib.pyplot as plt
    plots_dir = os.path.join(DATA_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    # Include grid size in the filename (in meters)
    grid_size_m = int(grid_size_km * 1000)
    plot_path = os.path.join(plots_dir, f"grid_amplitude_map_{grid_size_m}.png")
    fig, ax = plt.subplots(figsize=(16, 12))
    cax = ax.imshow(grid_amplitude, origin='lower',
                    extent=[lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]],
                    aspect='equal', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', pad=0.15)
    cbar.set_label('Median Amplitude (mm)')
    
    # Add GNSS stations to the map
    stations_file = os.path.join(DATA_DIR, os.getenv("STATIONS_FILE", "stations_list"))
    if os.path.exists(stations_file):
        stations_df = pd.read_csv(stations_file, sep=r'\s+')
        for _, station in stations_df.iterrows():
            ax.scatter(station["longitude"], station["latitude"], color="black", marker="^", s=72, zorder=5)
            ax.text(station["longitude"] + 0.01, station["latitude"] - 0.005, str(station["Station"]),
                    fontsize=12, ha="left", va="top", color="black", weight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_path}")

# --- Main workflow ---
def main():
    # Calculate amplitude (set use_detrended=True/False as needed)
    insar['amplitude'] = calculate_amplitude(insar, time_cols, use_detrended=True)

    # Calculate median amplitude per grid cell
    grid_amplitude = insar.groupby(['lat_bin', 'lon_bin'])['amplitude'].median().unstack()

    # Limit color scale to 5th and 95th percentile
    vmin = insar['amplitude'].quantile(0.05)
    vmax = insar['amplitude'].quantile(0.95)    # Dynamically set grid size and detrending info in title
    detrend_str = " (detrended)" if True else ""
    grid_size_str = f"{GRID_SIZE_KM:g} x {GRID_SIZE_KM:g} km (central latitude, varies with latitude)"
    title = f"Seasonal Amplitudes (max-min) per Grid Cell{detrend_str} - Aligned Data\nGrid size ≈ {grid_size_str}"    # Plot
    plot_grid_amplitude(
        grid_amplitude, lon_bins, lat_bins, vmin, vmax, DATA_DIR, insar,
        title=title, grid_size_km=GRID_SIZE_KM
    )

if __name__ == "__main__":
    main()
