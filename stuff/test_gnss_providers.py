"""
Test script for the GNSS data provider module

This script tests the modular GNSS data provider system by
loading a sample GNSS file and displaying its contents.

Usage:
    python test_gnss_providers.py

It will:
1. Try to find a real GNSS data file in the configured data directory
2. If no valid file is found, create a mock GNSS data file for testing
3. Test loading with the default provider (from GNSS_PROVIDER env var)
4. Test loading with explicit GFZ provider
5. Test loading with USGS provider (which should fail gracefully)

This is useful for:
- Testing if the GNSS data provider module is working correctly
- Verifying that environment variable selection works
- Testing newly added data providers
- Debugging issues with GNSS data loading
"""

import os
import sys
import glob
import pandas as pd
from gnss_data_providers import load_gnss_data, load_gnss_data_gfz

def create_mock_gnss_file(filepath):
    """Creates a mock GNSS file for testing purposes."""
    print(f"Creating mock GNSS file at {filepath}")
    
    # Create a directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create mock content that matches GFZ format expected in load_gnss_data_gfz
    # Note: Needs 7 columns (MJD, TIME1, TIME2, North, East, Up, LOS)
    content = """GNSS Displacement Time Series
MJD mm/dd/yyyy TIME North East Up LOS
59000.0 01/01/2020 2020.400 2.34 1.45 -0.78 1.23
59010.0 01/11/2020 2020.426 2.45 1.56 -0.82 1.24
59020.0 01/21/2020 2020.453 2.67 1.76 -0.92 1.25
59030.0 01/31/2020 2020.479 2.89 1.89 -1.05 1.26
59040.0 02/10/2020 2020.505 3.12 2.01 -1.23 1.27
59050.0 02/20/2020 2020.532 3.34 2.12 -1.45 1.28
"""
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(content)
    
    return filepath

def test_provider_selection():
    """Test provider selection using environment variables."""
    # Get data directory from environment variable or use default
    data_dir = os.environ.get("DATA_DIR", "C:\\insar_gnss_data")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return
        
    # Find a GNSS file to test with
    station_list_file = os.path.join(data_dir, "stations_list")
    if not os.path.exists(station_list_file):
        print(f"Error: Station list file '{station_list_file}' not found.")
        return
    
    # Read the first valid station from the list (skip header line if present)
    with open(station_list_file, 'r') as f:
        lines = f.readlines()
        
    # Skip empty lines and potential header lines
    station_found = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Check if first word looks like a header (all uppercase or contains "STATION")
        first_word = line.split()[0]
        if first_word.isupper() or "STATION" in first_word.upper():
            continue
            
        # Assume this is a valid station name
        first_station = first_word
        station_found = True
        break
    
    if not station_found:
        print("No valid station found in station list file.")
        # Use a default station name pattern for testing
        first_station = "GNSS"
        
    print(f"Testing with station: {first_station}")
    
    # Find any GNSS file for testing
    gnss_pattern = os.path.join(data_dir, "*_NEU_TIME*.txt")
    gnss_files = glob.glob(gnss_pattern)
    
    if not gnss_files:
        print(f"Error: No GNSS files found matching pattern {gnss_pattern}.")
        # Try a more generic pattern
        gnss_pattern = os.path.join(data_dir, "*.txt")
        gnss_files = glob.glob(gnss_pattern)
        
        if not gnss_files:
            print(f"Error: No text files found in {data_dir}.")
            return
            
        # Filter for likely GNSS files
        gnss_files = [f for f in gnss_files if "NEU" in f or "gnss" in f.lower()]
        if not gnss_files:
            print(f"No likely GNSS files found in {data_dir}.")
            print(f"Creating a mock GNSS file for testing purposes...")
            
            # Create a mock GNSS file for testing
            mock_file = os.path.join(data_dir, f"{first_station}_NEU_TIME.txt")
            test_file = create_mock_gnss_file(mock_file)
            gnss_files = [test_file]
        
    # Use the first file found
    test_file = gnss_files[0]
    print(f"Using file: {test_file}")
    
    # Check if the file exists and has content
    if not os.path.exists(test_file) or os.path.getsize(test_file) == 0:
        print(f"File {test_file} does not exist or is empty.")
        # Create a mock file for testing
        test_file = os.path.join(data_dir, "mock_gnss_data.txt")
        test_file = create_mock_gnss_file(test_file)
    
    # Test with default provider (from environment)
    print("\nTesting with default provider (from environment):")
    try:
        # First make sure environment variable is set
        provider = os.environ.get("GNSS_PROVIDER", "gfz")
        print(f"GNSS_PROVIDER environment variable: {provider}")
        
        # Load data using default provider
        data = load_gnss_data(test_file)
        print(f"Successfully loaded data with shape: {data.shape}")
        print(f"Columns: {', '.join(data.columns)}")
        print(f"First 5 rows:\n{data.head(5)}")
    except Exception as e:
        print(f"Error loading with default provider: {e}")
        print("Creating and using a mock GNSS file instead...")
        mock_file = os.path.join(data_dir, "mock_gnss_data.txt")
        test_file = create_mock_gnss_file(mock_file)
        try:
            data = load_gnss_data(test_file)
            print(f"Successfully loaded mock data with shape: {data.shape}")
        except Exception as e2:
            print(f"Error loading mock data: {e2}")
    
    # Test with explicit GFZ provider
    print("\nTesting with explicit GFZ provider:")
    try:
        # Load data using GFZ provider directly
        data_gfz = load_gnss_data(test_file, provider="GFZ")
        print(f"Successfully loaded GFZ data with shape: {data_gfz.shape}")
    except Exception as e:
        print(f"Error loading with GFZ provider: {e}")
        print("Using mock file instead...")
        mock_file = os.path.join(data_dir, "mock_gnss_data.txt")
        if not os.path.exists(mock_file):
            mock_file = create_mock_gnss_file(mock_file)
        try:
            data_gfz = load_gnss_data(mock_file, provider="GFZ")
            print(f"Successfully loaded mock data with GFZ provider with shape: {data_gfz.shape}")
        except Exception as e2:
            print(f"Error loading mock data with GFZ provider: {e2}")
    
    # Test with USGS provider (should raise NotImplementedError)
    print("\nTesting with USGS provider (should fail gracefully):")
    try:
        # Load data using USGS provider
        data_usgs = load_gnss_data(test_file, provider="USGS")
        print(f"Successfully loaded USGS data with shape: {data_usgs.shape}")
    except NotImplementedError as e:
        print(f"Expected error: {e}")
        print("This is expected behavior since USGS provider is not yet implemented.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
    print("\nTests completed.")

if __name__ == "__main__":
    # Allow command-line override of provider
    if len(sys.argv) > 1:
        provider = sys.argv[1]
        print(f"Setting GNSS_PROVIDER environment variable to: {provider}")
        os.environ["GNSS_PROVIDER"] = provider
    
    # Create a mock GNSS file for testing if no real files are found
    data_dir = os.environ.get("DATA_DIR", "C:\\insar_gnss_data")
    gnss_pattern = os.path.join(data_dir, "*_NEU_TIME*.txt")
    gnss_files = glob.glob(gnss_pattern)
    
    if not gnss_files:
        # No real GNSS files found, create a mock file
        mock_file = create_mock_gnss_file(os.path.join(data_dir, "mock_gnss_file.txt"))
        print(f"Using mock GNSS file for testing: {mock_file}")
    
    test_provider_selection()
