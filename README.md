# InSAR and GNSS Data Processing Workflow

This repository contains scripts for processing, analyzing, and visualizing InSAR and GNSS data, with a focus on spatial correction and time series analysis.

## Overview

The workflow provides tools for:
- Processing GNSS 3D velocities
- Filtering and aligning InSAR data
- Spatial correction using plane fitting
- Calculating and visualizing seasonal amplitude patterns
- Plotting combined time series of InSAR and GNSS data

## Scripts

### `master.py`
The main script that orchestrates the entire workflow. Controls parameters via environment variables and runs all other scripts in the correct order.

### Data Processing Scripts

- **`gnss_3d_vels.py`**: Processes GNSS 3D velocities.
- **`filter_insar_save_parameters.py`**: Filters InSAR data based on temporal coherence and saves parameters.
- **`fit_plane_correct_insar.py`**: Fits a plane to the InSAR data for spatial correction, aligning InSAR with GNSS observations.
- **`gnss_los_displ.py`**: Calculates GNSS displacements in the Line-of-Sight direction.

### Visualization Scripts

- **`plot_combined_time_series.py`**: Plots time series of both InSAR and GNSS data for comparison.
- **`grid_amplitude_analysis.py`**: Analyzes and visualizes seasonal amplitudes using a grid-based approach, with optional detrending and enhanced visualization features. Supports multi-resolution analysis to compare spatial patterns across different grid sizes.

## Configuration

All parameters are set in `master.py` as environment variables:

- **`DATA_DIR`**: Path to the data directory (default: "C:/insar_gnss_data")
- **`MIN_TEMPORAL_COHERENCE`**: Minimum temporal coherence threshold (default: 0.7)
- **`INSAR_RADIUS`**: Radius in meters for InSAR averaging around GNSS stations (default: 500m)
- **`GRID_SIZE_KM`**: Grid size in km for amplitude calculations (default: 0.5km)
- **`USE_DETRENDED`**: Whether to detrend time series before amplitude calculation (default: True)
- **`HALF_AMPLITUDE`**: Whether to use scientific amplitude definition (max-min)/2 (default: True)
- **`MULTI_RESOLUTION`**: Whether to create plots at multiple resolutions (default: False)
- **`GRID_SIZES`**: Comma-separated list of grid sizes in km to use when MULTI_RESOLUTION=True (default: "0.25, 0.5, 1.0, 2.0, 5.0")
- **`INSAR_FILE`**: Name of the InSAR data file
- **`STATIONS_FILE`**: Name of the GNSS stations list file

## Input Data Requirements

1. **InSAR Data**: CSV file containing InSAR time series with:
   - `longitude`, `latitude` columns for coordinates
   - Date columns in YYYYMMDD format containing displacement values
   - Temporal coherence (`temporal_coherence`) column

2. **GNSS Data**: A stations list file with:
   - Station name, longitude, latitude
   - Corresponding GNSS time series files

## Usage

1. Set your parameters in `master.py` (data paths, filtering thresholds, grid size).
2. Run the master script to execute the full workflow:
   ```powershell
   python master.py
   ```
3. For individual scripts (with environment variables already set):
   ```powershell
   python script_name.py
   ```

## Output

The workflow generates:
- Filtered and spatially corrected InSAR data files
- Various plots in the `plots` directory:
  - Combined time series plots
  - Spatial correction (plane fitting) visualizations
  - Grid-based amplitude maps showing seasonal patterns

## Grid-based Amplitude Analysis

The amplitude analysis in `grid_amplitude_analysis.py`:
- Creates a grid with configurable cell size via `GRID_SIZE_KM`
- Calculates the median amplitude (max-min)/2 or peak-to-peak of time series per cell
- Optionally detrends time series to focus on seasonal patterns (controlled by `USE_DETRENDED`)
- Visualizes the result with GNSS stations overlaid
- Can analyze and compare multiple spatial resolutions (set `MULTI_RESOLUTION=True`)
- Creates comparison plots showing amplitude statistics vs. grid resolution
- Provides consistent visualization across different grid sizes with improved station markers
- Features optimized subplot layouts to prevent element overlap and reduce white space

Note about grid cells: Due to the nature of longitude/latitude coordinates, grid cells that are square in degrees appear rectangular in kilometers, especially away from the central latitude.

## Dependencies

- Python 3.7+
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0 (with GridSpec and patheffects support)
- scipy >= 1.7.0

## Example Workflow

1. GNSS velocities are processed
2. InSAR data is filtered based on coherence
3. A correction plane is fitted to align InSAR with GNSS
4. Time series are plotted
5. Spatial amplitude patterns are visualized

For any issues, check the `workflow.log` file for detailed error messages.
