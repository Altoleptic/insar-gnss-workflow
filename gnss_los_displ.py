"""
GNSS Line-of-Sight Displacement Calculation Script

This script projects GNSS displacement measurements into the satellite's line-of-sight (LOS) direction.
It processes 3D GNSS data (North, East, Up) and calculates equivalent LOS values using
the InSAR geometry components, enabling direct comparison with InSAR measurements.

Features:
- Automatic processing of multiple GNSS stations
- Projection of 3D displacements into LOS direction
- Compatible output format with InSAR time series
- Handling of various time formats
- Support for batch processing of stations
- Support for using NNR-corrected files when specified
"""

# Import necessary libraries for data handling and file operations
import pandas as pd  # For data manipulation and analysis
import os            # For file and directory path management
import glob          # For file pattern matching
from gnss_data_providers import load_gnss_data  # Import the modular GNSS data loader

# Retrieve DATA_DIR from environment variable
# This directory contains all input and output data for the workflow
data_dir = os.getenv("DATA_DIR", "C:\\insar_gnss_data")

# Exit if the environment variable is not set
if not data_dir:
    print("Error: DATA_DIR environment variable is not set.")
    exit(1)

# ETRS89 files are automatically prioritized if available, otherwise standard files are used

# Ensure proper path formatting by converting to absolute path
data_dir = os.path.abspath(data_dir)
# Define file paths for station list and parameters
station_list_file = os.path.join(data_dir, "stations_list")
parameters_file = os.path.join(data_dir, "parameters.csv")
# GNSS data is expected to be in the main data directory
gnss_folder = data_dir

def calc_los_values(input_file, output_file, los_east, los_north, los_up):
    """Processes GNSS data to calculate LOS values and saves the output.
    
    Args:
        input_file: Path to the GNSS displacement file (North, East, Up)
        output_file: Path to save the processed file with LOS values
        los_east: LOS unit vector component in east direction
        los_north: LOS unit vector component in north direction
        los_up: LOS unit vector component in up direction
    """
    # Use the modular GNSS data provider to load data
    gnss_data = load_gnss_data(input_file)
    
    # Calculate LOS values as dot product of LOS unit vector and GNSS displacements
    # This projects the 3D displacement vector onto the satellite's line-of-sight direction
    try:
        # Ensure LOS components are float values
        los_north_float = float(los_north)
        los_east_float = float(los_east)
        los_up_float = float(los_up)
        
        # Calculate LOS with verified numeric values
        gnss_data["LOS"] = (los_north_float * gnss_data["North"] + 
                            los_east_float * gnss_data["East"] +
                            los_up_float * gnss_data["Up"])
        
        # Save the processed data to the output file in space-delimited format
        # to match the original GNSS file format, keeping North, East, Up and adding LOS
        with open(output_file, "w") as f:
            # Keep the same header format as the original file but add LOS column
            f.write("                                   --------in mm----------\n")
            f.write("MJD      TIME                      North    East      UP      LOS\n")
            
            # Write each row in space-delimited format with proper alignment
            for idx, row in gnss_data.iterrows():
                time_str = str(row["TIME"]) if pd.notnull(row["TIME"]) else "N/A"
                north_value = row["North"]
                east_value = row["East"]
                up_value = row["Up"]
                los_value = row["LOS"]
                f.write(f"{row['MJD']:.2f} {time_str}    {north_value:>8.2f} {east_value:>8.2f} {up_value:>8.2f} {los_value:>8.2f}\n")
                
        print(f"LOS values calculated and saved to {output_file}")
    except Exception as e:
        print(f"Error calculating LOS values: {e}")

def process_stations():
    """Processes GNSS data for multiple stations by calculating LOS values.
    
    This function reads InSAR parameters to get the LOS components,
    processes each station in the station list, and handles file naming
    based on whether NNR-corrected files are used.
    """
    # Verify required files exist
    if not os.path.exists(station_list_file):
        print(f"Error: Station list file not found: {station_list_file}")
        return
    
    if not os.path.exists(parameters_file):
        print(f"Error: Parameters file not found: {parameters_file}")
        return
    
    # Read parameters to get LOS components
    parameters_df = pd.read_csv(parameters_file)
    
    # Map between expected column names and what might be in the file
    los_column_map = {
        "los_east": ["los_east", "Los Unit Vector East norm"],
        "los_north": ["los_north", "Los Unit Vector North norm"],
        "los_up": ["los_up", "Los Unit Vector Up norm"]
    }
    
    # Check for any of the possible column names
    missing_components = []
    for component, possible_names in los_column_map.items():
        if not any(col in parameters_df.columns for col in possible_names):
            missing_components.append(component)
    
    if missing_components:
        print(f"Error: Parameters file does not contain required LOS components: {', '.join(missing_components)}")
        print(f"Available columns: {', '.join(parameters_df.columns)}")
        return
    
    # Extract LOS components (use the median values for robustness)
    # Try the alternative column names if the primary ones aren't available
    if "los_east" in parameters_df.columns:
        los_east = parameters_df["los_east"].median()
    else:
        los_east = parameters_df["Los Unit Vector East norm"].median()
        
    if "los_north" in parameters_df.columns:
        los_north = parameters_df["los_north"].median()
    else:
        los_north = parameters_df["Los Unit Vector North norm"].median()
        
    if "los_up" in parameters_df.columns:
        los_up = parameters_df["los_up"].median()
    else:
        los_up = parameters_df["Los Unit Vector Up norm"].median()
    
    print(f"Using LOS components - East: {los_east}, North: {los_north}, Up: {los_up}")
    
    # Create directory for plots if it doesn't exist
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Process each station
    with open(station_list_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # Skip empty lines and potential header lines
        if not line or line.startswith("#"):
            continue
            
        # Check if line might be a header row
        first_word = line.split()[0]
        if first_word.upper() == "STATION":
            continue
            
        # Extract station name
        station_name = first_word
        print(f"Processing station {station_name}...")
        
        # First, check for ETRS89 files (highest priority)
        etrs89_pattern = os.path.join(gnss_folder, f"{station_name}_NEU_TIME*_ETRS89.txt")
        etrs89_files = glob.glob(etrs89_pattern)
        
        if etrs89_files:
            print(f"Found ETRS89 reference frame file for {station_name}, using this file")
            input_file = etrs89_files[0]
        else:
            # Fall back to standard files
            gnss_pattern = os.path.join(gnss_folder, f"{station_name}_NEU_TIME*.txt")
            gnss_files = glob.glob(gnss_pattern)
            
            # Skip if no files found
            if not gnss_files:
                print(f"No GNSS data file found for station {station_name}, skipping")
                continue
            
            # Use the first matching file
            input_file = gnss_files[0]
        
        # Determine the output file name
        # Preserve any existing suffix in the filename
        if "_ETRS89" in input_file:
            output_file = input_file.replace("_ETRS89.txt", "_ETRS89_LOS.txt")
        else:
            output_file = input_file.replace(".txt", "_LOS.txt")
        
        # Calculate and save the LOS values
        calc_los_values(input_file, output_file, los_east, los_north, los_up)
            

    
    print("All stations processed successfully.")

if __name__ == "__main__":
    process_stations()