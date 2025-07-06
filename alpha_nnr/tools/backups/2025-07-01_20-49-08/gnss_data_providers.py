"""
GNSS Data Provider Module

This module contains functions for loading GNSS data from various data providers.
Each provider has its own specific file format and requirements, and this module
abstracts away those details with a common interface.

Supported providers:
- GFZ (Helmholtz Centre Potsdam - German Research Centre for Geosciences)
- [Future providers can be added here]

Usage:
    from gnss_data_providers import load_gnss_data
    
    # Automatically select the right parser based on the provider
    df = load_gnss_data(filepath, provider="GFZ")
    
    # Or use a specific provider's parser directly
    df = load_gnss_data_gfz(filepath)
"""

import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
import os           # For environment variables and file paths
from datetime import datetime

def load_gnss_data_gfz(filepath):
    """
    Loads GNSS data from a GFZ format file and handles inconsistent formatting.
    Handles various formats, including:
    - Format 1: MJD, TIME (YYYY-MM-DD HH:MM:SS), North, East, Up
    - Format 2: MJD, TIME1, TIME2, North, East, Up, LOS
    
    Args:
        filepath: Path to the GNSS data file
        
    Returns:
        DataFrame containing processed GNSS data with converted dates
        
    Raises:
        ValueError: If the file contains no valid MJD data
    """
    data = []
    los_available = False
    # Read file line by line to handle inconsistent formatting
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines and empty lines
            if not line or line.startswith('MJD') or line.startswith('---') or "in mm" in line:
                continue
            parts = line.split()
            
            # Need at least 5 columns for MJD, TIME, North, East, Up
            if len(parts) < 5:
                continue
                
            try:
                # Parse each column from the line
                mjd = float(parts[0])
                
                # Handle different time formats
                if len(parts) >= 6 and '-' in parts[1]:
                    # Format: MJD, TIME (YYYY-MM-DD HH:MM:SS), North, East, Up
                    time_str = parts[1] + " " + parts[2]
                    north = float(parts[3])
                    east = float(parts[4])
                    up = float(parts[5])
                    # LOS might not be available
                    los = float(parts[6]) if len(parts) >= 7 else np.nan
                elif len(parts) >= 7:
                    # Format: MJD, TIME1, TIME2, North, East, Up, LOS
                    time_str = parts[1] + " " + parts[2]
                    north = float(parts[3])
                    east = float(parts[4])
                    up = float(parts[5])
                    los = float(parts[6])
                    los_available = True
                else:
                    # Simplest format: MJD, TIME, North, East, Up
                    time_str = parts[1]
                    north = float(parts[2])
                    east = float(parts[3])
                    up = float(parts[4])
                    los = np.nan
                
                data.append([mjd, time_str, north, east, up, los])
            except ValueError:
                print(f"Skipping invalid line in GNSS file: {line}")
                continue
    
    # Convert to DataFrame and add date information
    df = pd.DataFrame(data, columns=["MJD", "TIME", "North", "East", "Up", "LOS"])
    if df.empty or df["MJD"].isnull().all():
        raise ValueError(f"GNSS file {filepath} contains no valid MJD data.")
        
    # If LOS column is all NaN, drop it to avoid downstream issues
    if df["LOS"].isnull().all():
        df = df.drop(columns=["LOS"])
    
    # Convert Modified Julian Date to datetime (origin date is 1858-11-17)
    df["DATE"] = pd.to_datetime(df["MJD"], origin="1858-11-17", unit="D")
    
    # Calculate decimal years since the first date for trend analysis
    start_date = df["DATE"].iloc[0]
    df["decimal_year"] = ((df["DATE"] - start_date).dt.total_seconds() /
                         (365.25 * 24 * 3600))
    return df

# This is a placeholder for future USGS implementation
def load_gnss_data_usgs(filepath):
    """
    Loads GNSS data from a USGS format file.
    [Implementation to be added in the future]
    
    Args:
        filepath: Path to the GNSS data file
        
    Returns:
        DataFrame containing processed GNSS data with converted dates
        
    Raises:
        NotImplementedError: This parser is not yet implemented
    """
    raise NotImplementedError("USGS data format parser is not yet implemented")

# Main function to select the appropriate parser based on provider
def load_gnss_data(filepath, provider=None):
    """
    Loads GNSS data using the appropriate parser for the specified provider.
    
    Args:
        filepath: Path to the GNSS data file
        provider: Data provider name (default: None, will use environment variable)
        
    Returns:
        DataFrame containing processed GNSS data with converted dates
        
    Raises:
        ValueError: If the provider is not supported
    """
    # If provider is not specified, get it from environment variable
    if provider is None:
        provider = os.environ.get("GNSS_PROVIDER", "gfz")
    
    if provider.upper() == "GFZ":
        return load_gnss_data_gfz(filepath)
    elif provider.upper() == "USGS":
        return load_gnss_data_usgs(filepath)
    else:
        raise ValueError(f"Unsupported GNSS data provider: {provider}")
