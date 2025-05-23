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
- **`plot_variability.py`**: Analyzes and visualizes seasonal amplitudes using a grid-based approach, with optional detrending.

## Configuration

All parameters are set in `master.py` as environment variables:

- **`DATA_DIR`**: Path to the data directory (default: "C:/insar_gnss_data")
- **`MIN_TEMPORAL_COHERENCE`**: Minimum temporal coherence threshold (default: 0.7)
- **`INSAR_RADIUS`**: Radius in meters for InSAR averaging around GNSS stations (default: 500m)
- **`GRID_SIZE_KM`**: Grid size in km for amplitude calculations (default: 0.5km)
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

The amplitude analysis in `plot_variability.py`:
- Creates a grid with cells approximately 1x1 km (configurable via `GRID_SIZE_KM`)
- Calculates the median amplitude (max-min of time series) per cell
- Optionally detrends time series to focus on seasonal patterns
- Visualizes the result with GNSS stations overlaid

Note about grid cells: Due to the nature of longitude/latitude coordinates, grid cells that are square in degrees appear rectangular in kilometers, especially away from the central latitude.

## Dependencies

- Python 3.7+
- numpy
- pandas
- matplotlib
- scipy

## Example Workflow

1. GNSS velocities are processed
2. InSAR data is filtered based on coherence
3. A correction plane is fitted to align InSAR with GNSS
4. Time series are plotted
5. Spatial amplitude patterns are visualized

For any issues, check the `workflow.log` file for detailed error messages.
