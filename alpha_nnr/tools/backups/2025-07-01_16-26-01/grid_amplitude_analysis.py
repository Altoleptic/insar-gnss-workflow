"""
Grid Amplitude Analysis Script

This script analyzes seasonal displacement amplitudes in InSAR data.
It creates spatial visualizations of amplitude values at different grid resolutions, 
allowing multi-resolution comparison to find the optimal scale for analysis.
The script can display raw or detrended time series data and supports different
amplitude calculation methods (half-amplitude or peak-to-peak range).

Features:
- Multiple grid resolution support (e.g., 0.25km to 5km)
- Visualization with GNSS station overlay
- Statistical comparison across resolutions
- Optional detrending of time series
"""

# Import necessary libraries for data processing, numerical computation, and visualization
import os                          # File and directory operations
import pandas as pd                # Data manipulation and analysis
import numpy as np                 # Numerical operations
import matplotlib.pyplot as plt    # Visualization
from matplotlib.gridspec import GridSpec  # Advanced subplot grid layouts
import matplotlib.patheffects as path_effects  # Text visual effects for better readability

# Load InSAR file and data directory from environment variables or set default values
# DATA_DIR is the root directory containing all input and output data
DATA_DIR = os.getenv("DATA_DIR", r"C:/insar_gnss_data")
# Name of the unaligned InSAR file (will be modified to use aligned version)
INSAR_FILE = os.getenv("INSAR_FILE", "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A.csv")
# Modify the filename to use the aligned version (_aligned suffix)
insar_name, insar_ext = os.path.splitext(INSAR_FILE)
INSAR_FILE_ALIGNED = f"{insar_name}_aligned{insar_ext}"
INSAR_PATH = os.path.join(DATA_DIR, INSAR_FILE_ALIGNED)

# Parameters for the grid - default is 0.5 km but can be overridden
# This defines the resolution of the analysis grid
GRID_SIZE_KM = float(os.getenv("GRID_SIZE_KM", "0.5"))

# Load InSAR data from the aligned file
insar = pd.read_csv(INSAR_PATH)

# Determine the extent of the study area from the data bounds
# These values define the outer boundaries of the grid
grid_min_lon = insar['longitude'].min()
grid_max_lon = insar['longitude'].max()
grid_min_lat = insar['latitude'].min()
grid_max_lat = insar['latitude'].max()

# Conversion from kilometers to degrees for grid generation
# This conversion accounts for latitude-dependent distance scaling
# 1 degree of latitude = approx. 111.32 km globally
lat_km_per_deg = 111.32
# Longitude distance varies with latitude (cos rule)
lon_km_per_deg = 111.32 * np.cos(np.deg2rad((insar['latitude'].min() + insar['latitude'].max()) / 2))

# Convert grid size from km to degrees for both latitude and longitude
lat_grid_size_deg = GRID_SIZE_KM / lat_km_per_deg
lon_grid_size_deg = GRID_SIZE_KM / lon_km_per_deg

# Create the grid by generating bin edges for longitude and latitude
lon_bins = np.arange(grid_min_lon, grid_max_lon + lon_grid_size_deg, lon_grid_size_deg)
lat_bins = np.arange(grid_min_lat, grid_max_lat + lat_grid_size_deg, lat_grid_size_deg)

# Assign each InSAR measurement to a grid cell using numpy's digitize function
# This creates bin indices for each data point (0-based indexing)
insar['lon_bin'] = np.digitize(insar['longitude'], lon_bins) - 1
insar['lat_bin'] = np.digitize(insar['latitude'], lat_bins) - 1

# Identify time columns (format: YYYYMMDD) to extract the time series data
time_cols = [col for col in insar.columns if col.isdigit() and len(col) == 8]

def detrend_timeseries(df, time_cols):
    """
    Remove linear trend from each row's time series.
    
    This function fits a linear trend to each point's time series and
    subtracts it, returning the detrended residuals. This isolates
    seasonal patterns by removing long-term linear displacement.
    
    Args:
        df: DataFrame containing InSAR time series data
        time_cols: List of column names containing time series measurements
        
    Returns:
        DataFrame with the same shape as df[time_cols] containing detrended values
    """
    from scipy.stats import linregress  # Import here for statistical regression calculations
    
    # Use integer indices for time (assumes equal spacing between measurements)
    times = np.arange(len(time_cols))
    
    # Create empty array with same shape as input data to store detrended values
    detrended = np.empty_like(df[time_cols].values, dtype=float)
    
    # Process each row (each InSAR point) individually to remove its unique trend
    for i, row in enumerate(df[time_cols].values):
        # Calculate linear regression parameters (slope, intercept) for this point's time series
        slope, intercept, _, _, _ = linregress(times, row)
        
        # Subtract the linear trend from the original data to isolate seasonal components
        detrended[i, :] = row - (slope * times + intercept)
    
    return pd.DataFrame(detrended, columns=time_cols, index=df.index)

def calculate_amplitude(df, time_cols, use_detrended=True, half_amplitude=True):
    """
    Calculate amplitude per point, optionally using detrended time series.
    
    Args:
        df: DataFrame containing InSAR time series data
        time_cols: List of column names containing time series measurements
        use_detrended: If True, detrend the time series before calculating amplitude
        half_amplitude: If True, calculate scientific amplitude (max-min)/2
                        If False, calculate peak-to-peak range (max-min)
    
    Returns:
        Series with amplitude values for each point
    """
    if use_detrended:
        # Detrend the time series to isolate seasonal patterns from long-term trends
        # This helps identify true seasonal amplitude independent of secular motion
        df_detrended = detrend_timeseries(df, time_cols)
        peak_to_peak = df_detrended.max(axis=1) - df_detrended.min(axis=1)
    else:
        # Calculate raw amplitude without detrending (includes both seasonal and long-term variations)
        peak_to_peak = df[time_cols].max(axis=1) - df[time_cols].min(axis=1)
    
    if half_amplitude:
        # Convert peak-to-peak range to scientific amplitude definition (half the range)
        # This represents the average deviation from the mean position
        return peak_to_peak / 2
    else:
        # Return full range (peak-to-peak), representing the total displacement range
        return peak_to_peak

def plot_grid_amplitude(grid_amplitude, lon_bins, lat_bins, vmin, vmax, DATA_DIR, insar, title, grid_size_km):
    """
    Plot the grid amplitude map and save to file.
    
    Args:
        grid_amplitude: DataFrame containing amplitude values for each grid cell
        lon_bins, lat_bins: Arrays defining the grid boundaries
        vmin, vmax: Minimum and maximum values for the color scale
        DATA_DIR: Directory for saving output
        insar: Original InSAR DataFrame (for reference)
        title: Plot title
        grid_size_km: Grid cell size in kilometers (for filename)
    """
    # Create plots directory if it doesn't exist
    # This ensures the output directory is available before saving files
    plots_dir = os.path.join(DATA_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calculate the proper figure dimensions based on the data extent
    # This ensures the map has the correct aspect ratio for the geographic region
    lon_extent = lon_bins[-1] - lon_bins[0]
    lat_extent = lat_bins[-1] - lat_bins[0]
    
    # Convert geographic extents to same units (km) for proper aspect ratio
    # Account for latitude-dependent distance scaling
    lon_km_per_deg = 111.32 * np.cos(np.deg2rad((lat_bins[0] + lat_bins[-1]) / 2))
    lat_km_per_deg = 111.32
    
    width_km = lon_extent * lon_km_per_deg
    height_km = lat_extent * lat_km_per_deg
    aspect_ratio = height_km / width_km
    
    # Set figure size based on actual geographic proportions
    # Use a fixed width and adjust height based on aspect ratio
    fig_width = 10
    fig_height = fig_width * aspect_ratio
    
    # Include grid size in the filename (in meters)
    # This helps identify different resolution outputs
    grid_size_m = int(grid_size_km * 1000)
    plot_path = os.path.join(plots_dir, f"grid_amplitude_analysis_{grid_size_m}.png")
    
    # Create figure with proper dimensions
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create geographic plot that respects Earth's coordinate system
    # This ensures proper representation of the geographic area
    ax = plt.axes()
    
    # Adjust aspect ratio to account for longitude/latitude distortion at current latitude
    # This ensures proper geographic representation of the area
    ax.set_aspect(1/np.cos(np.deg2rad((lat_bins[0] + lat_bins[-1]) / 2)))
    
    # Create a proper visualization of the grid data
    # Mask invalid values (NaN) to avoid display issues
    masked_grid = np.ma.masked_invalid(grid_amplitude.values)
    
    # Get the actual dimensions of the grid data
    n_rows, n_cols = masked_grid.shape
    
    # Create adjusted coordinate arrays that match the grid data
    # This ensures we're using the correct number of bins for our grid data
    adjusted_lon_bins = lon_bins[:n_cols+1]  # Grid is one less than coordinates for pcolor
    adjusted_lat_bins = lat_bins[:n_rows+1]
    
    # If adjusted arrays are too short (which might happen with data gaps),
    # extend them using the same grid spacing to prevent plotting errors
    if len(adjusted_lon_bins) < n_cols + 1:
        spacing = lon_bins[1] - lon_bins[0]
        while len(adjusted_lon_bins) < n_cols + 1:
            adjusted_lon_bins = np.append(adjusted_lon_bins, adjusted_lon_bins[-1] + spacing)
    
    if len(adjusted_lat_bins) < n_rows + 1:
        spacing = lat_bins[1] - lat_bins[0]
        while len(adjusted_lat_bins) < n_rows + 1:
            adjusted_lat_bins = np.append(adjusted_lat_bins, adjusted_lat_bins[-1] + spacing)
    
    # Create mesh grid with adjusted coordinates for plotting
    # This defines the boundaries of each grid cell
    X, Y = np.meshgrid(adjusted_lon_bins, adjusted_lat_bins)
    
    # Use pcolormesh with consistent coordinates and grid data dimensions
    # Add interpolation for smoother appearance especially for larger grid sizes
    cax = ax.pcolormesh(X, Y, masked_grid, 
                     cmap='viridis', vmin=vmin, vmax=vmax, 
                     edgecolors='face', linewidth=0.1)
    
    # Add axis labels with clear descriptions
    ax.set_xlabel('Longitude (decimal degrees)')
    ax.set_ylabel('Latitude (decimal degrees)')
    ax.set_title(title)
    
    # Increase pad parameter to create more space between colorbar and x-axis
    # This improves readability of the colorbar labels
    cbar = plt.colorbar(cax, orientation='horizontal', pad=0.15, shrink=0.75)
    cbar.set_label('Median Amplitude (mm)')
    
    # Add GNSS stations to the map if stations file exists
    # This provides reference points for comparing InSAR and GNSS data
    stations_file = os.path.join(DATA_DIR, os.getenv("STATIONS_FILE", "stations_list"))
    if os.path.exists(stations_file):
        stations_df = pd.read_csv(stations_file, sep=r'\s+')
        for _, station in stations_df.iterrows():
            # Use smaller triangle markers for better clarity
            ax.scatter(station["longitude"], station["latitude"], color="black", marker="^", s=50, zorder=5)
            
            # Optimize text size and position relative to marker
            # Slight offset ensures station labels are clearly visible
            ax.text(station["longitude"] + 0.008, station["latitude"] - 0.004, str(station["Station"]),
                    fontsize=10, ha="left", va="top", color="black", weight="bold")
    
    # Add grid to the plot for better readability of coordinate values
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add a frame around the plot for better visual definition
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_visible(True)
    
    # Improve tick formatting for better readability
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0, length=5)
    ax.tick_params(axis='both', which='minor', width=0.5, length=3)
    
    # Format longitude ticks to 1 decimal place and latitude to 2 decimal places
    # This provides sufficient precision without cluttering the axes
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Optimize layout and save the plot at high resolution
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")

# --- Main workflow ---
def process_grid_size(grid_size_km, insar_df, time_columns):
    """
    Process data for a specific grid size and return the result.
    
    Args:
        grid_size_km: Grid cell size in kilometers
        insar_df: DataFrame containing InSAR data
        time_columns: List of column names containing time series measurements
        
    Returns:
        Dictionary containing grid data, boundaries, and visualization parameters
    """
    # Calculate grid size in degrees based on the specific grid_size_km
    # This conversion accounts for latitude-dependent distance scaling, ensuring consistent spatial analysis
    lat_km_per_deg = 111.32  # Approximate km per degree of latitude (global average)
    
    # Longitude distance varies with latitude following the cosine rule
    # Points at higher latitudes are closer together in terms of longitude
    lon_km_per_deg = 111.32 * np.cos(np.deg2rad((insar_df['latitude'].min() + insar_df['latitude'].max()) / 2))
    
    # Convert grid dimensions from km to degrees for latitude and longitude separately
    # This accounts for the Earth's ellipsoidal shape and ensures proper geographic binning
    lat_grid_size_deg = grid_size_km / lat_km_per_deg
    lon_grid_size_deg = grid_size_km / lon_km_per_deg
    
    # Create the grid for this specific grid size by determining area bounds
    grid_min_lon = insar_df['longitude'].min()
    grid_max_lon = insar_df['longitude'].max()
    grid_min_lat = insar_df['latitude'].min()
    grid_max_lat = insar_df['latitude'].max()
    
    # Generate bin edges for the grid using arange to create evenly spaced coordinates
    # These bin edges will be used to assign data points to specific grid cells
    lon_bins = np.arange(grid_min_lon, grid_max_lon + lon_grid_size_deg, lon_grid_size_deg)
    lat_bins = np.arange(grid_min_lat, grid_max_lat + lat_grid_size_deg, lat_grid_size_deg)
    
    # Assign each InSAR measurement to a grid cell for this specific grid size
    # Each point gets assigned to exactly one cell based on its coordinates
    insar_copy = insar_df.copy()  # Create a copy to avoid modifying the original data
    insar_copy['lon_bin'] = np.digitize(insar_copy['longitude'], lon_bins) - 1
    insar_copy['lat_bin'] = np.digitize(insar_copy['latitude'], lat_bins) - 1
    
    # Get amplitude calculation settings from environment variables
    # These control how seasonal patterns are extracted and quantified
    use_detrended = os.getenv("USE_DETRENDED", "True").lower() == "true"
    half_amplitude = os.getenv("HALF_AMPLITUDE", "True").lower() == "true"
    
    # Calculate amplitude based on environment settings
    # This quantifies the seasonal displacement pattern for each InSAR point
    insar_copy['amplitude'] = calculate_amplitude(insar_copy, time_columns, 
                                                use_detrended=use_detrended, 
                                                half_amplitude=half_amplitude)
    
    # Calculate median amplitude per grid cell to reduce noise and outlier effects
    # Using median instead of mean makes the result more robust to outliers
    grid_amplitude = insar_copy.groupby(['lat_bin', 'lon_bin'])['amplitude'].median().unstack()
    
    # Limit color scale to 5th and 95th percentile to exclude extreme outliers
    # This provides better visualization contrast for the majority of data points
    vmin = insar_copy['amplitude'].quantile(0.05)
    vmax = insar_copy['amplitude'].quantile(0.95)
    
    # Return all necessary information for visualization and comparison
    return {
        'grid_amplitude': grid_amplitude,
        'lon_bins': lon_bins,
        'lat_bins': lat_bins,
        'vmin': vmin,
        'vmax': vmax,
        'grid_size_km': grid_size_km
    }

def create_multi_resolution_comparison(results, data_dir, use_detrended, half_amplitude):
    """
    Create a comparison plot showing how amplitude varies with grid resolution.
    
    Args:
        results: List of dictionaries with results for each grid size
        data_dir: Directory to save the plot
        use_detrended: Whether detrending was used
        half_amplitude: Whether half amplitude (scientific definition) was used
    """    
    # Create plots directory if it doesn't exist
    # This ensures the output directory is available before saving files
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load GNSS station data for plotting if available
    # Stations will be overlaid on the map for reference
    stations_file = os.path.join(data_dir, os.getenv("STATIONS_FILE", "stations_list"))
    if os.path.exists(stations_file):
        stations_df = pd.read_csv(stations_file, sep=r'\s+')
    else:
        stations_df = None
    
    # Sort results by grid size for consistent ordering in the comparison plot
    # This ensures a logical progression from finest to coarsest resolution
    results = sorted(results, key=lambda x: x['grid_size_km'])
    
    # Create a compact figure with 2×3 grid layout for the resolution comparison
    # This layout accommodates up to 6 different grid resolutions
    fig = plt.figure(figsize=(21, 12))
    
    # Configure grid layout with optimized spacing between subplots
    # This ensures good use of space while keeping plots readable
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    # Calculate the median extent and proportions for all plots
    # This ensures consistent display across all resolutions
    avg_lon_extent = np.median([r['lon_bins'][-1] - r['lon_bins'][0] for r in results])
    avg_lat_extent = np.median([r['lat_bins'][-1] - r['lat_bins'][0] for r in results])
    
    # Convert geographic coordinates to same units (km) for proper aspect ratio
    # This accounts for latitude-dependent distance scaling
    avg_lat = np.median([np.median([r['lat_bins'][0], r['lat_bins'][-1]]) for r in results])
    lon_km_per_deg = 111.32 * np.cos(np.deg2rad(avg_lat))
    lat_km_per_deg = 111.32
    
    width_km = avg_lon_extent * lon_km_per_deg
    height_km = avg_lat_extent * lat_km_per_deg
    aspect_ratio = height_km / width_km
    
    # Find global min and max values for consistent colormaps across all plots
    # This ensures fair visual comparison between different resolutions
    all_values = []
    for result in results:
        flat_values = result['grid_amplitude'].stack().dropna().values
        all_values.extend(flat_values)
    
    # Use percentiles to avoid outliers affecting the color scale
    # 5th and 95th percentiles provide good contrast while excluding extremes
    global_vmin = np.percentile(all_values, 5)
    global_vmax = np.percentile(all_values, 95)
      
    # Create subplot for each resolution
    # Limit to 6 plots (2x3 grid) for readability
    for i, result in enumerate(results):
        if i >= 6:  # Limit to 6 plots (2x3 grid)
            break
            
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        grid_size = result['grid_size_km']
        
        # Ensure all subplots have the exact same geographical extent for consistency
        # This allows fair comparison between different resolutions
        lon_min = min([r['lon_bins'][0] for r in results])
        lon_max = max([r['lon_bins'][-1] for r in results])
        lat_min = min([r['lat_bins'][0] for r in results])
        lat_max = max([r['lat_bins'][-1] for r in results])
        
        # Set same limits for all subplots to ensure consistent scale
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        
        # Set equal aspect ratio for proper geographic representation
        # This accounts for Earth's curvature at the current latitude
        ax.set_aspect(1/np.cos(np.deg2rad((result['lat_bins'][0] + result['lat_bins'][-1]) / 2)))
        
        # Use pcolormesh with masked data to handle NaN values safely
        # This prevents display issues when some grid cells have no data
        masked_grid = np.ma.masked_invalid(result['grid_amplitude'].values)
        
        # Get the actual dimensions of the grid data
        n_rows, n_cols = masked_grid.shape
        
        # Create adjusted coordinate arrays that match the grid data
        # This ensures proper alignment between grid cells and coordinates
        adjusted_lon_bins = result['lon_bins'][:n_cols+1]
        adjusted_lat_bins = result['lat_bins'][:n_rows+1]
        
        # If adjusted arrays are too short (which might happen with data gaps),
        # extend them using the same grid spacing to prevent plotting errors
        if len(adjusted_lon_bins) < n_cols + 1:
            spacing = result['lon_bins'][1] - result['lon_bins'][0]
            while len(adjusted_lon_bins) < n_cols + 1:
                adjusted_lon_bins = np.append(adjusted_lon_bins, adjusted_lon_bins[-1] + spacing)
        if len(adjusted_lat_bins) < n_rows + 1:
            spacing = result['lat_bins'][1] - result['lat_bins'][0]
            while len(adjusted_lat_bins) < n_rows + 1:
                adjusted_lat_bins = np.append(adjusted_lat_bins, adjusted_lat_bins[-1] + spacing)
                
        # Create mesh grid with adjusted coordinates for plotting
        # This defines the boundaries of each grid cell
        X, Y = np.meshgrid(adjusted_lon_bins, adjusted_lat_bins)
        
        # Configure visualization settings for consistent appearance across all grid sizes
        edgecolor = 'face'
        linewidth = 0.1
        alpha = 1.0
            
        # Create the visualization with consistent color scale across all subplots
        # Using viridis colormap for better perceptual uniformity
        cax = ax.pcolormesh(X, Y, masked_grid, 
                         cmap='viridis', vmin=global_vmin, vmax=global_vmax,
                         edgecolors=edgecolor, linewidth=linewidth, alpha=alpha)
        
        # Add GNSS stations to each subplot with enhanced visibility
        # This provides reference points that are consistent across all resolutions
        if stations_df is not None:
            for _, station in stations_df.iterrows():
                # Mark station with triangle marker with white edge for better visibility
                ax.scatter(station["longitude"], station["latitude"], 
                          color="black", marker="^", s=50, zorder=5, 
                          edgecolors='white', linewidths=0.8)
                
                # Add station label with white outline for visibility against any background
                ax.text(station["longitude"] + 0.008, station["latitude"] - 0.004, 
                        str(station["Station"]), fontsize=9, ha="left", va="top", color="black", weight="bold",
                        path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")])
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Improve formatting of subplot borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_visible(True)
        
        # Improve tick formatting with consistent number of ticks across all subplots
        ax.tick_params(axis='both', which='major', labelsize=9, width=1.0, length=4)        
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        # Set the same number of ticks for all plots for consistency
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        
        # Set concise title showing only grid size information
        ax.set_title(f"Grid size: {grid_size} km", fontsize=11, fontweight='bold', pad=5)
        
        # Add x-axis labels to all plots for improved readability
        ax.set_xlabel('Longitude (decimal degrees)', fontsize=10)
        
        # Add y-axis labels only to plots in the leftmost column
        # This reduces clutter while maintaining clarity
        if col == 0:
            ax.set_ylabel('Latitude (decimal degrees)', fontsize=10)
        else:
            ax.set_ylabel('')
        
        # Add a horizontal colorbar below each plot with padding to avoid overlap
        # Adjust the colorbar size to be more compact
        cbar = plt.colorbar(cax, ax=ax, orientation='horizontal', pad=0.15, shrink=0.9)
        cbar.set_label('Median Amplitude (mm)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)  # Smaller tick labels for compactness
    
    # Overall title - more compact positioning
    # Include information about the data processing method used
    detrend_str = "detrended" if use_detrended else "raw"
    amplitude_method = "half amplitude" if half_amplitude else "peak-to-peak"
    fig.suptitle(f'Spatial Resolution Comparison - {amplitude_method} ({detrend_str} data)', 
                fontsize=16, y=0.98)
    
    # Adjust subplot spacing manually instead of using tight_layout to avoid warnings
    # This provides better control over the final layout
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.90, wspace=0.3, hspace=0.35)
    
    # Save figure with higher DPI for better quality
    # Use descriptive filename for the multi-resolution comparison plot
    plot_path = os.path.join(plots_dir, f"multi_resolution_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-resolution comparison plot saved to: {plot_path}")
    
    # Now create a separate statistical comparison plot
    # This provides quantitative metrics to complement the visual comparison
    create_statistical_comparison(results, plots_dir, use_detrended, half_amplitude)

def create_statistical_comparison(results, plots_dir, use_detrended, half_amplitude):
    """
    Create a separate statistical comparison plot for amplitude statistics vs. grid resolution.
    
    This function generates a plot showing how amplitude statistics (mean, std, min, max)
    vary with grid resolution, which helps identify the optimal scale for analysis.
    
    Args:
        results: List of dictionaries with results for each grid size
        plots_dir: Directory to save the plot
        use_detrended: Whether detrending was used
        half_amplitude: Whether half amplitude (scientific definition) was used
    """
    # Extract statistics for comparison
    # Collect grid sizes and initialize empty lists for storing statistics
    grid_sizes = [r['grid_size_km'] for r in results]
    avg_amplitudes = []
    std_amplitudes = []
    max_amplitudes = []
    min_amplitudes = []
    
    # Calculate statistics for each grid size
    # This provides quantitative metrics for comparing different spatial resolutions
    for result in results:
        # Flatten and remove NaNs to get a clean list of all amplitude values in this grid
        flat_amplitudes = result['grid_amplitude'].stack().dropna().values
        
        # Calculate statistical metrics that describe the amplitude distribution
        avg_amplitudes.append(np.median(flat_amplitudes))  # Median amplitude (more robust to outliers than mean)
        std_amplitudes.append(np.std(flat_amplitudes))     # Standard deviation (variability)
        max_amplitudes.append(np.max(flat_amplitudes))     # Maximum amplitude
        min_amplitudes.append(np.min(flat_amplitudes))     # Minimum amplitude
    
    # Create a dedicated figure for statistics with better proportions
    # This provides a clear view of how statistics change with resolution
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Plot statistics with improved styling
    # Median with error bars showing standard deviation
    ax.errorbar(grid_sizes, avg_amplitudes, yerr=std_amplitudes, fmt='o-', capsize=5, 
              label='Median ± Std Dev', color='#1f77b4', linewidth=2, markersize=10)
    
    # Maximum values across resolutions
    ax.plot(grid_sizes, max_amplitudes, 's--', label='Maximum', color='#ff7f0e', 
          linewidth=2, markersize=10)
    
    # Minimum values across resolutions
    ax.plot(grid_sizes, min_amplitudes, 'v--', label='Minimum', color='#2ca02c', 
          linewidth=2, markersize=10)
    
    # Improve the statistics chart appearance with clear title and labels
    ax.set_title('InSAR Seasonal Displacement Amplitude vs. Grid Resolution', fontsize=16)
    ax.set_xlabel('Grid Cell Size (km)', fontsize=14)
    
    # Use appropriate y-axis label based on the amplitude calculation method
    amplitude_type = "Amplitude (mm)" if half_amplitude else "Peak-to-Peak Range (mm)"
    ax.set_ylabel(amplitude_type, fontsize=14)
    
    # Use logarithmic x-axis for better visualization of multiple scales
    # This ensures both small and large grid sizes are clearly visible
    ax.set_xscale('log')
    
    # Better x-axis tick formatting for log scale
    # Display actual grid size values rather than logarithmic notation
    grid_sizes_array = np.array(grid_sizes)
    ax.set_xticks(grid_sizes_array)
    ax.set_xticklabels([f"{size:.1f}" for size in grid_sizes_array], fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add minor gridlines and additional styling for readability
    # This helps track values across the plot
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=14, loc='best', framealpha=0.9)
    
    # Add a box around the plot to make it stand out
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Add descriptive subtitle with information about the data processing method
    detrend_str = "Detrended" if use_detrended else "Raw"
    amplitude_method = "Half-Amplitude" if half_amplitude else "Peak-to-Peak Range"
    
    # Add more space between x-axis and figure caption
    # This ensures the caption is clearly separated from the plot
    plt.figtext(0.5, 0.03, f"{detrend_str} data, {amplitude_method} method", 
               ha='center', fontsize=12, style='italic')
    
    # Save figure with higher DPI for better quality
    # Add more space at the bottom margin with adjusted tight_layout parameters
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Use descriptive filename for the statistical comparison plot
    plot_path = os.path.join(plots_dir, f"amplitude_statistics_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Statistical comparison plot saved to: {plot_path}")

def main():
    """
    Main function to orchestrate the amplitude analysis workflow.
    
    This function:
    1. Determines whether to use single or multi-resolution analysis
    2. Processes data for each selected grid size
    3. Creates visualizations for the results
    """
    # Get amplitude calculation settings from environment variables
    # These control how amplitudes are calculated and displayed
    use_detrended = os.getenv("USE_DETRENDED", "True").lower() == "true"  # Default to detrended analysis
    half_amplitude = os.getenv("HALF_AMPLITUDE", "True").lower() == "true"  # Default to scientific amplitude definition
    
    # Check if we should process multiple grid sizes for resolution comparison
    # Multi-resolution analysis helps identify optimal spatial scale for analysis
    multi_resolution = os.getenv("MULTI_RESOLUTION", "True").lower() == "true"
    
    if multi_resolution:
        # Process multiple grid sizes to compare how spatial resolution affects results
        # This helps identify the best scale to capture displacement patterns
        custom_grid_sizes = os.getenv("GRID_SIZES")
        if custom_grid_sizes:
            try:
                # Parse comma-separated list of custom grid sizes from environment variable
                grid_sizes = [float(x.strip()) for x in custom_grid_sizes.split(',')]
                print(f"Using custom grid sizes: {grid_sizes}")
            except ValueError:
                print("Warning: Could not parse GRID_SIZES. Using default values.")
                grid_sizes = [0.25, 0.5, 1.0, 1.5, 2.5, 5.0]  # Enhanced grid sizes in km
        else:
            # Default set of grid resolutions covering a wide range of scales
            grid_sizes = [0.25, 0.5, 1.0, 1.5, 2.5, 5.0]  # Default grid sizes in km
        
        results = []
        
        # Process each grid size separately and collect results
        for grid_size in grid_sizes:
            print(f"Processing grid size: {grid_size} km")
            result = process_grid_size(grid_size, insar, time_cols)
            results.append(result)
            
            # Print debug info about grid dimensions for verification
            print(f"Grid size {grid_size}km - dimensions: {result['grid_amplitude'].shape}")
            print(f"Grid size {grid_size}km - Lat bins: {len(result['lat_bins'])}, Lon bins: {len(result['lon_bins'])}")
            
            # Create individual plot for this specific grid size
            detrend_str = " (detrended)" if use_detrended else ""
            amplitude_type = "Amplitude" if half_amplitude else "Peak-to-Peak Range"
            grid_size_str = f"{grid_size:g} x {grid_size:g} km (central latitude, varies with latitude)"
            title = f"InSAR Seasonal Displacement {amplitude_type}{detrend_str} Map\nGrid size ≈ {grid_size_str}"
            
            plot_grid_amplitude(
                result['grid_amplitude'], result['lon_bins'], result['lat_bins'], 
                result['vmin'], result['vmax'], DATA_DIR, insar,
                title=title, grid_size_km=grid_size
            )
            
        # Create comparison plots to visualize how grid resolution affects results
        create_multi_resolution_comparison(results, DATA_DIR, use_detrended, half_amplitude)
    else:
        # Process single grid size (original behavior)
        # Calculate amplitude for each InSAR point using the configured settings
        insar['amplitude'] = calculate_amplitude(insar, time_cols, use_detrended=use_detrended, half_amplitude=half_amplitude)
        
        # Calculate median amplitude per grid cell to reduce noise and outlier effects
        grid_amplitude = insar.groupby(['lat_bin', 'lon_bin'])['amplitude'].median().unstack()
        
        # Print debug info for grid dimensions
        print(f"Grid dimensions: {grid_amplitude.shape}")
        print(f"Lat bins: {len(lat_bins)}, Lon bins: {len(lon_bins)}")
        
        # Set color scale limits to 5th and 95th percentile to exclude extreme outliers
        vmin = insar['amplitude'].quantile(0.05)
        vmax = insar['amplitude'].quantile(0.95)
        
        # Create plot title based on current settings
        detrend_str = " (detrended)" if use_detrended else ""
        amplitude_type = "Amplitude" if half_amplitude else "Peak-to-Peak Range"
        grid_size_str = f"{GRID_SIZE_KM:g} x {GRID_SIZE_KM:g} km (central latitude, varies with latitude)"
        title = f"InSAR Seasonal Displacement {amplitude_type}{detrend_str} Map\nGrid size ≈ {grid_size_str}"
        
        # Create and save the plot for the single grid size
        plot_grid_amplitude(
            grid_amplitude, lon_bins, lat_bins, vmin, vmax, DATA_DIR, insar,
            title=title, grid_size_km=GRID_SIZE_KM
        )

# Entry point of the script when run directly (not imported)
# This allows the script to be both imported as a module and run as a standalone program
if __name__ == "__main__":
    main()
