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

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import os
import matplotlib.pyplot as plt
import sys
import glob

# Function to convert geodetic coordinates to ECEF (Earth-Centered, Earth-Fixed)
def geodetic_to_ecef(lat, lon, h):
    """Convert geodetic coordinates (degrees, degrees, meters) to ECEF (X, Y, Z) in meters.
    
    Args:
        lat (float or array): Latitude in degrees
        lon (float or array): Longitude in degrees
        h (float or array): Height above ellipsoid in meters
        
    Returns:
        tuple: X, Y, Z coordinates in ECEF reference frame (meters)
    """
    a = 6378137.0             # WGS84 semi-major axis (meters)
    e2 = 6.69437999014e-3     # WGS84 eccentricity squared

    # Convert latitude and longitude to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Calculate the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    # Compute ECEF coordinates
    X = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
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

    # Debug: Print initial guess for omega
    print(f"Initial guess for omega: {omega0}")

    # Debug: Print residuals for initial guess
    initial_residuals = residuals(omega0)
    print(f"Residuals for initial guess: {initial_residuals[:5]} (first 5 values)")

    # Perform least-squares optimization to estimate omega
    result = least_squares(residuals, omega0)

    # Debug: Print optimization result
    print(f"Optimization result: {result}")

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
    
    Creates a multi-panel figure showing:
    1. Original vs corrected time series for North, East, and Up components
    2. Difference plots (corrected - original) for each component
    
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

    # Calculate differences - ensure we're working with numpy arrays and maintain precision
    diff_north = np.array(corrected_north - original_north, dtype=np.float64)
    diff_east = np.array(corrected_east - original_east, dtype=np.float64)
    diff_up = np.array(corrected_up - original_up, dtype=np.float64)

    # Debug: Print statistical summary of the differences
    print("Differences statistics:")
    print(f"North - Mean: {np.mean(diff_north):.8f}, Min: {np.min(diff_north):.8f}, Max: {np.max(diff_north):.8f}")
    print(f"East - Mean: {np.mean(diff_east):.8f}, Min: {np.min(diff_east):.8f}, Max: {np.max(diff_east):.8f}")
    print(f"Up - Mean: {np.mean(diff_up):.8f}, Min: {np.min(diff_up):.8f}, Max: {np.max(diff_up):.8f}")

    # Debug: Print the first 5 values for the difference plots
    print("First 5 values for difference plots:")
    print("Difference North:", diff_north[:5])
    print("Difference East:", diff_east[:5])
    print("Difference Up:", diff_up[:5])

    # Additional debug output: Check data types and detailed info for differences
    print(f"Differences data types: North: {type(diff_north)}, East: {type(diff_east)}, Up: {type(diff_up)}")
    
    # Check if we have any zero values
    zero_north = np.sum(np.abs(diff_north) < 1e-10)
    zero_east = np.sum(np.abs(diff_east) < 1e-10)
    zero_up = np.sum(np.abs(diff_up) < 1e-10)
    print(f"Number of values very close to zero (< 1e-10): North: {zero_north}, East: {zero_east}, Up: {zero_up} out of {len(diff_north)} points")
    
    # Check original vs corrected directly for a few rows
    print("\nDetailed comparison for first 5 rows:")
    for i in range(min(5, len(original_north))):
        print(f"Row {i}: Original North: {original_north[i]:.10f}, Corrected North: {corrected_north[i]:.10f}, Diff: {diff_north[i]:.10f}")
        print(f"Row {i}: Original East: {original_east[i]:.10f}, Corrected East: {corrected_east[i]:.10f}, Diff: {diff_east[i]:.10f}")
        print(f"Row {i}: Original Up: {original_up[i]:.10f}, Corrected Up: {corrected_up[i]:.10f}, Diff: {diff_up[i]:.10f}")

    # Create subplots for North, East, and Up components
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    components = ["North", "East", "Up"]
    original_data = [original_north, original_east, original_up]
    corrected_data = [corrected_north, corrected_east, corrected_up]
    differences = [diff_north, diff_east, diff_up]

    for i, component in enumerate(components):
        # Original and Corrected values
        axes[i, 0].scatter(time, original_data[i], label=f"Original {component}", color="red", s=10, alpha=0.7)
        axes[i, 0].scatter(time, corrected_data[i], label=f"Corrected {component}", color="blue", s=10, alpha=0.7)
        axes[i, 0].set_title(f"{component} Displacement for Station {station_name}")
        axes[i, 0].set_ylabel(f"{component} (mm)")
        axes[i, 0].legend()
        axes[i, 0].grid(True)

        # Difference plot - use scatter points only
        axes[i, 1].scatter(time, differences[i], label=f"Difference {component}", color="green", s=20, alpha=0.7)
        axes[i, 1].set_title(f"Difference in {component} Displacement for Station {station_name}")
        axes[i, 1].set_ylabel(f"Difference (mm)")
        
        # Add a more visible zero line
        axes[i, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add a text annotation showing the mean difference
        mean_diff = np.mean(differences[i])
        axes[i, 1].annotate(f"Mean: {mean_diff:.6f} mm", 
                          xy=(0.05, 0.95), xycoords='axes fraction',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                          ha='left', va='top')
        
        axes[i, 1].legend()
        axes[i, 1].grid(True)

        # Completely dynamic y-axis range based on the data
        diff_data = differences[i]
        diff_mean = np.mean(diff_data)
        diff_std = np.std(diff_data)
        
        # Use ±3 standard deviations from the mean, ensuring at least ±0.0005 mm range
        # This ensures small differences are still visible
        min_range = 0.0005  # 0.5 microns minimum half-range
        range_half = max(3 * diff_std, min_range)
        
        # Set initial y-limits centered on the mean
        y_min = diff_mean - range_half
        y_max = diff_mean + range_half
        
        # Ensure zero is always visible in the plot if it's close to the data range
        if 0 > y_max:  # If zero is above our range
            y_max = 0
        elif 0 < y_min:  # If zero is below our range
            y_min = 0
        
        # Make sure all data points are visible by checking actual min/max
        data_min = np.min(diff_data)
        data_max = np.max(diff_data)
        
        # Extend the range if any points would be cut off
        if data_min < y_min:
            y_min = data_min - 0.1 * range_half  # Add some padding
        if data_max > y_max:
            y_max = data_max + 0.1 * range_half  # Add some padding
            
        # Apply the y-axis limits
        axes[i, 1].set_ylim(y_min, y_max)

    # Set common x-axis label
    axes[-1, 0].set_xlabel("MJD")
    axes[-1, 1].set_xlabel("MJD")

    # Adjust layout and save the plot
    fig.tight_layout()
    plot_path = os.path.join(plots_dir, f"{station_name}_displacement_time_series_with_difference.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"Time series plot with differences saved to {plot_path}")

# Function to correct GNSS displacement time series for net rotation
# This function processes all GNSS station files and saves corrected data.
def correct_displacement_time_series(stations_list_file, gnss_folder, omega):
    """Correct GNSS displacement time series for net rotation.
    
    For each station:
    1. Loads displacement time series data
    2. Applies rotation correction to each displacement vector
    3. Creates visualization of before/after correction
    4. Saves corrected data to new files with _NNR suffix
    
    Args:
        stations_list_file (str): Path to the file containing station names
        gnss_folder (str): Directory containing GNSS displacement files
        omega (ndarray): Rotation vector (omega) for correction
    """
    # Load station names from the stations_list file
    stations_df = pd.read_csv(stations_list_file, sep=r'\s+')

    # Process each station
    for station_name in stations_df["Station"]:
        # Find GNSS files dynamically using glob pattern matching
        gnss_pattern = os.path.normpath(os.path.join(gnss_folder, f"{station_name}_NEU_TIME*.txt"))
        gnss_files = glob.glob(gnss_pattern)

        # Skip stations with missing GNSS files
        if not gnss_files:
            print(f"No GNSS file found matching pattern: {gnss_pattern}. Skipping station {station_name}.")
            continue

        # Use the first matching file (usually there's only one)
        gnss_file = gnss_files[0]

        # Read GNSS data from fixed-width format file
        colspecs = [(0, 8), (9, 28), (32, 37), (41, 46), (49, 59)]
        gnss_data = pd.read_fwf(gnss_file, colspecs=colspecs, skiprows=2, names=["MJD", "TIME", "North", "East", "Up"])
        gnss_data[["North", "East", "Up"]] = gnss_data[["North", "East", "Up"]].apply(pd.to_numeric, errors='coerce')

        # Correct displacement data using the rotation vector
        corrected_displacements = np.array([
            [row["North"], row["East"], row["Up"]] - np.cross(omega, [row["North"], row["East"], row["Up"]])
            for _, row in gnss_data.iterrows()
        ])

        # Ensure corrected values for plots are derived directly from corrected_displacements
        corrected_df = pd.DataFrame(corrected_displacements, columns=["North", "East", "Up"])
        corrected_df["MJD"] = gnss_data["MJD"].values

        # Debug: Compare original and corrected values for the first row
        first_row = gnss_data.iloc[0]
        original_vec = np.array([first_row["North"], first_row["East"], first_row["Up"]], dtype=float)
        corrected_vec = original_vec - np.cross(omega, original_vec)
        print(f"  Original   : {original_vec}")
        print(f"  Corrected : {corrected_vec}")

        # Debug: Print the first 5 values for the plotted original and corrected values
        print("Original North:", gnss_data["North"].head(5).values)
        print("Corrected North:", corrected_displacements[:5, 0])
        print("Original East:", gnss_data["East"].head(5).values)
        print("Corrected East:", corrected_displacements[:5, 1])
        print("Original Up:", gnss_data["Up"].head(5).values)
        print("Corrected Up:", corrected_displacements[:5, 2])

        # Plot time series before and after correction
        plot_time_series_before_after_correction(station_name, gnss_data, corrected_displacements, gnss_folder)

        # Create output filename with _NNR suffix
        base_name = os.path.basename(gnss_file)
        output_file = os.path.join(gnss_folder, base_name.replace(".txt", "_NNR.txt"))

        # Debug: Verify first row correction again before writing
        if not gnss_data.empty:
            first_row = gnss_data.iloc[0]
            original_vec = np.array([first_row["North"], first_row["East"], first_row["Up"]], dtype=float)
            corrected_vec = original_vec - np.cross(omega, original_vec)
            print(f"  Original   : {original_vec}")
            print(f"  Corrected : {corrected_vec}")

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