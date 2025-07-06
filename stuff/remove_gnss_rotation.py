"""
Remove GNSS Net Rotation Script

This script estimates and removes the net rotation from GNSS velocity data using geodetic coordinates.
It defines a local, internally consistent reference frame by applying rotation corrections.
The script also corrects GNSS displacement time series for net rotation effects and generates
visualization plots showing before and after correction comparisons.

Features:
- Converts geodetic coordinates to ECEF for global calculations
- Estimates rotation vector (omega) using least-squares optimization
- Applies rotation corrections to GNSS velocities and displacement time series
- Visualizes original and corrected velocity vectors on a map
- Creates time series plots showing before/after corrections and difference plots
- Saves corrected GNSS displacement files with _NNR suffix
- Saves the estimated rotation vector to a file for reference
"""


## corrected - original
## remove the avgs from the original data


import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import linregress
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import sys
import glob
from gnss_data_providers import load_gnss_data  # Import the modular GNSS data loader

# Function to convert geodetic coordinates to ECEF (Earth-Centered, Earth-Fixed)
def geodetic_to_ecef(lat, lon, h):
    """Convert geodetic coordinates (degrees, degrees, meters) to ECEF (X, Y, Z) in meters.
    
    This function transforms geodetic coordinates (latitude, longitude, height) into
    Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates using the WGS84 reference
    ellipsoid. ECEF is a Cartesian coordinate system with:
    - Origin at the Earth's center of mass
    - X-axis passing through the intersection of the equator and prime meridian (0°, 0°)
    - Z-axis passing through the North Pole
    - Y-axis completing the right-handed system, 90° East of the X-axis
    
    The transformation accounts for the ellipsoidal shape of the Earth by using the
    WGS84 parameters and applies the appropriate mathematical formulas for conversion.
    
    Mathematical formulas used:
    X = (N + h) * cos(φ) * cos(λ)
    Y = (N + h) * cos(φ) * sin(λ)
    Z = (N * (1 - e²) + h) * sin(φ)
    
    Where:
    - φ is latitude in radians
    - λ is longitude in radians
    - h is height above the ellipsoid in meters
    - N is the radius of curvature in the prime vertical: a / √(1 - e² * sin²(φ))
    - a is the semi-major axis of the WGS84 ellipsoid
    - e² is the squared eccentricity of the WGS84 ellipsoid
    
    Args:
        lat (float or array): Latitude in degrees
        lon (float or array): Longitude in degrees
        h (float or array): Height above ellipsoid in meters
        
    Returns:
        tuple: X, Y, Z coordinates in ECEF reference frame (meters)
    """
    # WGS84 reference ellipsoid parameters
    a = 6378137.0             # Semi-major axis (equatorial radius) in meters
    e2 = 6.69437999014e-3     # Eccentricity squared (e² = 2f - f²), where f is flattening
                              # WGS84 flattening f = 1/298.257223563

    # Convert latitude and longitude from degrees to radians for trigonometric functions
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Calculate the radius of curvature in the prime vertical (N)
    # This represents the distance from the surface to the Z-axis along the ellipsoid normal
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    # Compute ECEF Cartesian coordinates
    # X: distance from Earth center to the point projected onto the equatorial plane, along prime meridian
    X = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    # Y: distance from Earth center to the point projected onto the equatorial plane, 90° East of prime meridian
    Y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    # Z: distance from Earth center to the point projected onto the polar axis
    # The (1 - e²) factor accounts for the ellipsoidal flattening at the poles
    Z = (N * (1 - e2) + h) * np.sin(lat_rad)
    
    return X, Y, Z

# Function to create a skew-symmetric matrix for cross-product calculations
def cross_matrix(vec):
    """Return skew-symmetric cross-product matrix of vector.
    
    This matrix is used to model rotational effects in vector calculations.
    For a vector ω = [ωx, ωy, ωz], creates a matrix that when multiplied 
    by another vector v, gives the same result as the cross product ω × v.
    
    Args:
        vec (array): 3D vector [x, y, z]
        
    Returns:
        ndarray: 3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

# Function to estimate the rotation vector using least-squares optimization
def estimate_rotation(positions, velocities):
    """Estimate the rotation vector (omega) using least-squares optimization.
    
    The rotation vector represents the angular velocity causing the net rotation
    of the GNSS network. This function finds the optimal rotation vector that,
    when applied to station positions, best explains the observed velocities.
    
    Args:
        positions (ndarray): Station positions in ECEF coordinates, shape (n, 3)
        velocities (ndarray): Observed velocities, shape (n, 3)
        
    Returns:
        ndarray: Estimated rotation vector omega [Ωx, Ωy, Ωz]
    """
    def residuals(omega):
        # Construct the skew-symmetric matrix for omega
        Omega = cross_matrix(omega)
        # Model the velocities based on the rotation
        modeled = (Omega @ positions.T).T
        # Calculate residuals between observed and modeled velocities
        return (velocities - modeled).ravel()

    # Initial guess for omega (no rotation)
    omega0 = np.zeros(3)

    # Perform least-squares optimization to estimate omega
    result = least_squares(residuals, omega0)

    return result.x  # [Ωx, Ωy, Ωz]

# Function to apply rotation correction to GNSS velocities
def apply_rotation_correction(positions, velocities, omega):
    """Apply rotation correction to GNSS velocities to remove net rotation effect.
    
    For each station, computes the corrected velocity by subtracting the rotational
    component (omega × position) from the observed velocity.
    
    Args:
        positions (ndarray): Station positions in ECEF coordinates, shape (n, 3)
        velocities (ndarray): Observed velocities, shape (n, 3)
        omega (ndarray): Rotation vector [Ωx, Ωy, Ωz]
        
    Returns:
        ndarray: Corrected velocities with net rotation removed, shape (n, 3)
    """
    return np.array([v - np.cross(omega, x) for x, v in zip(positions, velocities)])

def plot_velocity_vectors(merged, corrected_velocities, plots_dir):
    """Plot original and corrected GNSS velocity vectors on a map.
    
    Creates a map visualization showing both original and corrected velocity vectors
    for each station, with a reference arrow for scale.
    
    Args:
        merged (DataFrame): DataFrame containing station coordinates and original velocities
        corrected_velocities (ndarray): Array of corrected velocities (East, North)
        plots_dir (str): Directory to save the plot
    """
    # Extract latitude and longitude for station positions
    lats = merged["latitude"]
    lons = merged["longitude"]

    # Extract original and corrected velocity components (East, North)
    v_orig = merged[ [
        "GNSS East Velocity (mm/year)",
        "GNSS North Velocity (mm/year)"
    ]].values
    v_corr = corrected_velocities[:, :2]  # East, North

    # Ensure valid and non-empty data for plotting
    valid_indices = (
        ~np.isnan(lats) & ~np.isnan(lons) &
        ~np.isnan(v_orig).any(axis=1) & ~np.isnan(v_corr).any(axis=1)
    )

    # Filter valid data
    lats = lats[valid_indices]
    lons = lons[valid_indices]
    v_orig = v_orig[valid_indices]
    v_corr = v_corr[valid_indices]

    # Check if filtered data is non-empty
    if lats.size == 0 or lons.size == 0 or v_orig.shape[0] == 0 or v_corr.shape[0] == 0:
        print("Warning: No valid data available for plotting velocity vectors.")
        return

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.title("GNSS Velocity Vectors: Before and After Net Rotation Correction", loc='center')
    plt.xlabel("Longitude (decimal degrees)")
    plt.ylabel("Latitude (decimal degrees)")

    # Plot original velocities (in gray)
    plt.quiver(lons, lats, v_orig[:, 0], v_orig[:, 1],
               color="lightgray", scale=20, label="Original GNSS Velocity")

    # Plot corrected velocities (in blue)
    plt.quiver(lons, lats, v_corr[:, 0], v_corr[:, 1],
               color="dodgerblue", scale=20, label="Corrected Velocity (NNR)")

    # Add a reference arrow for scale under the legend
    ref_arrow = plt.quiver([lons.min() - 0.05], [lats.min() - 0.05], [3], [0],
                           color="black", scale=20)
    plt.quiverkey(ref_arrow, X=0.85, Y=0.1, U=3,
                  label="Reference: 3 mm/year", labelpos='S',
                  fontproperties={'size': 10})

    # Add legend and grid
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)

    # Adjust axis limits to zoom out slightly (30% of the previous amount)
    plt.xlim(lons.min() - 0.03, lons.max() + 0.03)
    plt.ylim(lats.min() - 0.03, lats.max() + 0.03)

    # Add station names to the arrows
    for i, station in enumerate(merged['Station']):
        plt.text(lons.iloc[i], lats.iloc[i], station, fontsize=12, ha='center', va='center')

    # Ensure the plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Save the plot
    plot_path = os.path.join(plots_dir, "net_rotation_correction_map.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    print(f"Velocity vector map saved to {plot_path}")

# Function to plot time series before and after correction for each GNSS station
def plot_time_series_before_after_correction(station_name, gnss_data, corrected_displacements, gnss_folder):
    """Plot time series before and after correction for a GNSS station.
    
    Creates a figure with three subplots showing:
    - Original and corrected time series for the North component
    - Original and corrected time series for the East component
    - Original and corrected time series for the Up component
    
    Each subplot includes trend lines calculated with linear regression
    to show the velocity changes due to the NNR correction.
    
    Args:
        station_name (str): Name of the GNSS station
        gnss_data (DataFrame): Original GNSS displacement data
        corrected_displacements (ndarray): Corrected displacement values
        gnss_folder (str): Directory to save the plots
    """
    # Ensure the plots directory exists
    plots_dir = os.path.join(gnss_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Extract time (MJD) and displacement components
    time = gnss_data["MJD"]
    original_north = gnss_data["North"].values  # Convert to numpy array
    original_east = gnss_data["East"].values
    original_up = gnss_data["Up"].values

    corrected_north = corrected_displacements[:, 0]
    corrected_east = corrected_displacements[:, 1]
    corrected_up = corrected_displacements[:, 2]

    # Calculate differences
    diff_north = np.array(corrected_north - original_north, dtype=np.float64)
    diff_east = np.array(corrected_east - original_east, dtype=np.float64)
    diff_up = np.array(corrected_up - original_up, dtype=np.float64)

    # Create time in decimal years for trend calculation
    mjd_start = time.iloc[0]
    decimal_years = (time - mjd_start) / 365.25

    # Convert MJD to datetime for better readability
    date_start = datetime(1858, 11, 17) + timedelta(days=float(mjd_start))
    date_end = datetime(1858, 11, 17) + timedelta(days=float(time.iloc[-1]))
    start_date = date_start.strftime("%Y-%m-%d")
    end_date = date_end.strftime("%Y-%m-%d")

    # Create subplots for North, East, and Up components - same style as gnss_3d_vels.py
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.3)  # Add more space between subplots
    fig.suptitle(f"Date Range: {start_date} to {end_date}", fontsize=10)
    
    components = ["North", "East", "Up"]
    original_data = [original_north, original_east, original_up]
    corrected_data = [corrected_north, corrected_east, corrected_up]
    
    for i, component in enumerate(components):
        # Original values (red)
        axes[i].scatter(decimal_years, original_data[i], label=f"Original", 
                        color="red", s=3, alpha=0.5)
        
        # Calculate and plot original trend
        orig_slope, orig_intercept, _, _, _ = linregress(decimal_years, original_data[i])
        axes[i].plot(decimal_years, orig_intercept + orig_slope * decimal_years, 
                     label=f"Orig {orig_slope:.2f} mm/yr", color="red", linestyle="-")
        
        # Corrected values (blue)
        axes[i].scatter(decimal_years, corrected_data[i], label=f"NNR-Corrected", 
                        color="blue", s=3, alpha=0.5)
        
        # Calculate and plot corrected trend
        corr_slope, corr_intercept, _, _, _ = linregress(decimal_years, corrected_data[i])
        axes[i].plot(decimal_years, corr_intercept + corr_slope * decimal_years, 
                     label=f"NNR {corr_slope:.2f} mm/yr", color="blue", linestyle="-")
        
        # Labels and formatting
        axes[i].set_title(f"GNSS {component} Displacement and Trend for Station {station_name}")
        # Set fixed position for y-axis labels to ensure perfect alignment
        axes[i].set_ylabel(f"{component} Displacement (mm)", labelpad=15)
        # Format y-axis to have the same width regardless of minus signs
        axes[i].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:8.1f}"))
        axes[i].grid(True)
        # More compact legend with smaller font, better positioned
        axes[i].legend(loc='upper right', fontsize=8, framealpha=0.7, handlelength=1.5, 
                      ncol=2, columnspacing=1, handletextpad=0.5)

    # Add x-axis label and format
    axes[2].set_xlabel("TIME (YYYY-MM-DD)")
    
    # Apply the same date formatting as in gnss_3d_vels.py
    axes[2].xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: (date_start + timedelta(days=x*365.25)).strftime("%Y-%m-%d")))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Clean up filename for saving
    safe_filename = station_name.replace(" ", "_").replace("/", "-")
    
    # Use consistent naming with gnss_3d_vels.py but indicate NNR correction
    plot_path = os.path.join(plots_dir, f"displacement_plot_{safe_filename}_NNR_comparison.png")
    
    # Adjust layout - give extra space for titles and rotated x-axis labels
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(plot_path, dpi=150)  # Higher DPI for better quality
    plt.close()

    print(f"Time series plot saved to {plot_path}")

# Function to correct GNSS displacement time series for net rotation
# This function processes all GNSS station files and saves corrected data.
def correct_displacement_time_series(stations_list_file, gnss_folder, omega):
    """Correct GNSS displacement time series for net rotation.
    
    For each station:
    1. Loads displacement time series data
    2. Applies time-dependent rotation correction to each displacement vector
    3. Creates visualization of before/after correction
    4. Saves corrected data to new files with _NNR suffix
    
    The correction formula is:
    corrected_displacement = original_displacement - (omega × position) × time_elapsed
    
    where:
    - omega is the rotation vector (angular velocity)
    - position is the station's position in ECEF coordinates
    - time_elapsed is the time since the first measurement, in years
    
    This makes the displacement correction consistent with the velocity correction:
    corrected_velocity = original_velocity - (omega × position)
    
    Args:
        stations_list_file (str): Path to the file containing station names and coordinates
        gnss_folder (str): Directory containing GNSS displacement files
        omega (ndarray): Rotation vector (omega) for correction
    """
    # Load station names and coordinates from the stations_list file
    stations_df = pd.read_csv(stations_list_file, sep=r'\s+')
    
    # Create a dictionary to store station coordinates for faster lookup
    station_coords = {}
    for _, row in stations_df.iterrows():
        station_name = row['Station']
        lat = row['latitude']
        lon = row['longitude'] 
        height = row['height']
        X, Y, Z = geodetic_to_ecef(lat, lon, height)
        station_coords[station_name] = np.array([X, Y, Z])

    # Process each station
    for _, row in stations_df.iterrows():
        station_name = row["Station"]
        
        # Skip if the station name is "Station" (header row)
        if station_name.upper() == "STATION":
            continue
            
        # Find GNSS files dynamically using glob pattern matching
        gnss_pattern = os.path.normpath(os.path.join(gnss_folder, f"{station_name}_NEU_TIME*.txt"))
        gnss_files = glob.glob(gnss_pattern)
        
        # Exclude NNR and LOS files from the pattern match
        gnss_files = [f for f in gnss_files if "_NNR" not in f and "_LOS" not in f]

        # Skip stations with missing GNSS files
        if not gnss_files:
            print(f"No GNSS file found matching pattern: {gnss_pattern}. Skipping station {station_name}.")
            continue

        # Use the first matching file (usually there's only one)
        gnss_file = gnss_files[0]

        # Use the modular GNSS data provider to load data
        gnss_data = load_gnss_data(gnss_file)

        # Get station position in ECEF
        if station_name not in station_coords:
            print(f"Warning: No coordinates found for station {station_name}, skipping.")
            continue
            
        position = station_coords[station_name]
        
        # Calculate time since reference epoch (first measurement)
        reference_time = gnss_data["MJD"].min()
        time_elapsed = gnss_data["MJD"] - reference_time  # in days
        
        # Convert days to years for applying the correction
        years_elapsed = time_elapsed / 365.25
        
        # Apply the rotation correction to the displacement time series
        # The correction is omega × position × time_elapsed_years
        # This matches the velocity correction formula: v_corrected = v_original - (omega × position)
        corrected_displacements = np.array([
            [row["North"], row["East"], row["Up"]] - np.cross(omega, position) * years_elapsed[i]
            for i, (_, row) in enumerate(gnss_data.iterrows())
        ])

        # Ensure corrected values for plots are derived directly from corrected_displacements
        corrected_df = pd.DataFrame(corrected_displacements, columns=["North", "East", "Up"])
        corrected_df["MJD"] = gnss_data["MJD"].values

        # Plot time series before and after correction
        plot_time_series_before_after_correction(station_name, gnss_data, corrected_displacements, gnss_folder)

        # Create output filename with _NNR suffix
        base_name = os.path.basename(gnss_file)
        output_file = os.path.join(gnss_folder, base_name.replace(".txt", "_NNR.txt"))

        # Write corrected data to the output file
        with open(output_file, "w") as f:
            # Keep the same header format as the original file
            f.write("                                   --------in mm----------\n")
            f.write("MJD      TIME                      North    East      UP\n")
            for idx, row in gnss_data.iterrows():
                time_str = str(row["TIME"]) if pd.notnull(row["TIME"]) else "N/A"
                # Use the corrected displacement values from corrected_displacements
                corr_north = corrected_displacements[idx, 0]
                corr_east = corrected_displacements[idx, 1]
                corr_up = corrected_displacements[idx, 2]
                f.write(f"{row['MJD']:.2f} {time_str}    {corr_north:>8.2f} {corr_east:>8.2f} {corr_up:>8.2f}\n")

        print(f"Corrected GNSS file saved to {output_file}")

# Main function
def main():
    """Main function to handle paths and process GNSS data for net rotation removal.
    
    This function:
    1. Loads GNSS velocity data and station coordinates
    2. Converts coordinates to ECEF reference frame
    3. Estimates the rotation vector causing the net rotation
    4. Applies rotation correction to GNSS velocities
    5. Saves corrected velocities to parameters.csv
    6. Corrects displacement time series for all stations
    7. Generates plots showing before and after correction
    """
    # Define paths for input and output files
    data_dir = os.getenv("DATA_DIR", "C:/insar_gnss_data")
    params_path = os.path.join(data_dir, "parameters.csv")
    stations_path = os.path.join(data_dir, "stations_list")

    # Load GNSS parameters (e.g., velocities) and station coordinates
    df = pd.read_csv(params_path, sep=",")  # GNSS parameters file (comma-separated)
    stations_df = pd.read_csv(stations_path, sep=r'\s+')  # Station coordinates file (whitespace-separated)

    # Merge station coordinates into the GNSS parameters dataframe
    merged = pd.merge(df, stations_df, on="Station", how="left")

    # Ensure required columns exist after merging
    required_columns = ["latitude", "longitude", "height"]
    for col in required_columns:
        if col not in merged.columns:
            print(f"Error: Missing column '{col}' in merged DataFrame.")
            print("Available columns in merged DataFrame:")
            print(merged.columns)
            exit(1)

    # Convert geodetic coordinates to ECEF for global calculations
    X, Y, Z = geodetic_to_ecef(merged["latitude"], merged["longitude"], merged["height"])
    positions = np.column_stack((X, Y, Z))  # Shape: (n, 3)

    # Extract GNSS velocities (East, North, Up components)
    velocities = merged[ [
        "GNSS East Velocity (mm/year)",
        "GNSS North Velocity (mm/year)",
        "GNSS Up Velocity (mm/year)"
    ]].values  # Shape: (n, 3)

    # Estimate the rotation vector (omega) causing the net rotation
    omega = estimate_rotation(positions, velocities)

    # Save the estimated rotation vector to a file
    omega_path = os.path.join(data_dir, "rotation_vector_omega.txt")
    with open(omega_path, "w") as f:
        f.write("Estimated rotation vector (omega):\n")
        f.write(f"{omega[0]:.6f}, {omega[1]:.6f}, {omega[2]:.6f}\n")
    
    print(f"Rotation vector saved to {omega_path}")

    # Apply rotation correction to remove the net rotation
    corrected = apply_rotation_correction(positions, velocities, omega)

    # Add corrected velocities as new columns in the dataframe
    merged["GNSS North Velocity NNR (mm/year)"] = corrected[:, 1]
    merged["GNSS East Velocity NNR (mm/year)"] = corrected[:, 0]
    merged["GNSS Up Velocity NNR (mm/year)"] = corrected[:, 2]

    # Save only relevant columns to the updated parameters.csv file
    output_columns = [
        "Station",
        "GNSS North Velocity (mm/year)",
        "GNSS East Velocity (mm/year)",
        "GNSS Up Velocity (mm/year)",
        "GNSS North Velocity NNR (mm/year)",
        "GNSS East Velocity NNR (mm/year)",
        "GNSS Up Velocity NNR (mm/year)"
    ]
    merged[output_columns].to_csv(params_path, index=False)

    # Define the plots directory
    plots_dir = os.path.join(data_dir, "plots")

    # Plot velocity vectors
    plot_velocity_vectors(merged, corrected, plots_dir)

    # Correct GNSS displacement time series for net rotation
    correct_displacement_time_series(stations_path, data_dir, omega)

    print(f"Net rotation removed and updated parameters saved to {params_path}")
    print("GNSS displacement time series corrected for net rotation.")

if __name__ == "__main__":
    main()