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
"""

import pandas as pd
import os
import glob

# Retrieve DATA_DIR from environment variable
data_dir = os.getenv("DATA_DIR")

if not data_dir:
    print("Error: DATA_DIR environment variable is not set.")
    exit(1)

data_dir = os.path.abspath(data_dir)  # Ensure proper path formatting
station_list_file = os.path.join(data_dir, "stations_list")
parameters_file = os.path.join(data_dir, "parameters.csv")
gnss_folder = data_dir

def calc_los_values(input_file, output_file, los_east, los_north, los_up):
    """Processes GNSS data to calculate LOS values and saves the output."""
    colspecs = [(0, 9), (10, 29), (30, 38), (39, 47), (48, 56)]  # Column widths
    
    gnss_data = pd.read_fwf(input_file, colspecs=colspecs, skiprows=2, names=["MJD", "TIME", "North", "East", "Up"])

    # Correct TIME column format
    gnss_data["TIME"] = gnss_data["TIME"].apply(
        lambda x: x if x[:4].isdigit() and len(x[:4]) == 4 else "20" + x[1:]
    )

    # Calculate LOS values
    gnss_data["LOS"] = los_north * gnss_data["North"] + los_east * gnss_data["East"] + los_up * gnss_data["Up"]

    # Write output file
    with open(output_file, "w") as f:
        f.write("                                    --------in mm----------\n")
        f.write("MJD      TIME                       North     East       UP       LOS\n")
        for _, row in gnss_data.iterrows():
            f.write(f"{row['MJD']:.2f} {row['TIME']}    {row['North']:>8.2f} {row['East']:>8.2f} {row['Up']:>8.2f} {row['LOS']:>8.2f}\n")

    print(f"Processed GNSS file saved to {output_file}")

def process_stations(station_list_file, parameters_file, gnss_folder):
    """Processes GNSS data for multiple stations by calculating LOS values."""
    parameters_df = pd.read_csv(parameters_file)

    for _, station in parameters_df.iterrows():
        station_name = station["Station"]

        # Find GNSS file dynamically
        gnss_pattern = os.path.join(gnss_folder, f"{station_name}_NEU_TIME*.txt")
        gnss_files = glob.glob(gnss_pattern)

        if not gnss_files:
            print(f"No GNSS file found matching pattern: {gnss_pattern}. Skipping station {station_name}.")
            continue

        gnss_file = gnss_files[0]

        # Extract time suffix from file name
        base_name = os.path.basename(gnss_file)
        time_suffix = base_name.split("TIME")[1].replace(".txt", "")

        output_file = os.path.join(gnss_folder, f"{station_name}_NEU_TIME{time_suffix}_LOS.txt")

        print(f"Processing LOS data for GNSS station {station_name} using file {gnss_file}...")
        calc_los_values(
            gnss_file,
            output_file,
            station["Los Unit Vector East norm"],
            station["Los Unit Vector North norm"],
            station["Los Unit Vector Up norm"]
        )

def main():
    """Main function to process GNSS stations."""
    if not os.path.exists(station_list_file):
        print(f"Error: stations_list file not found in {data_dir}.")
        exit(1)

    print(f"Processing GNSS LOS values in {data_dir}...")
    process_stations(station_list_file, parameters_file, gnss_folder)
    print("GNSS LOS value calculation completed successfully.")

if __name__ == "__main__":
    main()