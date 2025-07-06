# InSAR and GNSS Data Integration Workflow

This repository contains a comprehensive suite of Python scripts for processing, aligning, analyzing, and visualizing InSAR and GNSS displacement data. The workflow focuses on creating a spatially and temporally consistent dataset by removing systemic biases and providing robust visualization tools.

## Repository Structure

This repository is organized as follows:

- **Main directory**: The production-ready, standard version
- **`alpha_nnr` directory**: Contains an experimental version with additional functionality

## Overview

The workflow addresses several key challenges in GNSS-InSAR integration:

1. **Spatial Alignment**: Corrects spatial biases between InSAR and GNSS measurements using plane fitting techniques.
2. **Temporal Analysis**: Provides tools for comparing time series and analyzing seasonal patterns.
3. **Visualization**: Offers comprehensive visualization capabilities for comparing data sources and understanding spatial patterns.
4. **Extensibility**: Uses a modular approach for data loading to support different data providers.

### Key Features

- **GNSS Processing**:
  - 3D velocity calculation from time series
  - Line-of-Sight projection for InSAR compatibility

- **InSAR Processing**:
  - Quality filtering based on temporal coherence
  - Spatial alignment with GNSS data using plane correction
  - Batch processing for large datasets

- **Analysis Tools**:
  - Combined time series visualization
  - Multi-resolution grid-based amplitude analysis
  - Statistical comparison between measurement techniques
  - Trend analysis and displacement pattern detection
  
- **Performance**:
  - Parallel processing for computationally intensive tasks
  - Memory-optimized operations for large datasets
  - Batch processing capabilities

## Scripts

### `master.py`
The main script that orchestrates the entire workflow. Controls parameters via environment variables and runs all other scripts in the correct order.

### Core Modules

- **`gnss_data_providers.py`**: A modular system for loading GNSS data from various providers (currently supports GFZ, with placeholder for USGS). This abstracts away the differences in file formats and allows easy extension for additional data providers.

### Data Processing Scripts

- **`gnss_3d_vels.py`**: Processes GNSS 3D velocities.
- **`filter_insar_save_parameters.py`**: Filters InSAR data based on temporal coherence and saves parameters.
- **`fit_plane_correct_insar.py`**: Fits a plane to the InSAR data for spatial correction, aligning InSAR with GNSS observations.
- **`gnss_los_displ.py`**: Calculates GNSS displacements in the Line-of-Sight direction.

### Visualization Scripts

- **`plot_combined_time_series.py`**: Plots time series of both InSAR and GNSS data for comparison, with statistical analysis and trend visualization.
- **`grid_amplitude_analysis.py`**: Analyzes and visualizes seasonal amplitudes using a grid-based approach, with optional detrending and enhanced visualization features. Supports multi-resolution analysis to compare spatial patterns across different grid sizes.

## Installation and Setup

### Prerequisites

- Python 3.8 or newer
- Required Python packages (install using `pip install -r requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - scipy
  - pathlib
  - geopy

### Directory Structure

```
C:/insar_gnss_data/         # Default data directory
├── stations_list           # List of GNSS stations with coordinates
├── INSAR_FILE.csv          # InSAR data file
├── parameters.csv          # Generated parameters file
├── [Station]_NEU_TIME.txt  # GNSS time series files
└── plots/                  # Output directory for plots
```

### Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare your data directory with input files
4. Update parameters in `master.py` as needed
5. Run the workflow using one of these options:
   - Full workflow: `python master.py`
   - Interactive menu: `launcher.bat` (Windows only)
   - Individual scripts: `python script_name.py`

## Configuration

All parameters are set in `master.py` as environment variables:

- **`DATA_DIR`**: Path to the data directory (default: "C:/insar_gnss_data")
- **`MIN_TEMPORAL_COHERENCE`**: Minimum temporal coherence threshold (default: 0.7)
- **`INSAR_RADIUS`**: Radius in meters for InSAR averaging around GNSS stations (default: 250m)
- **`USE_NNR_CORRECTED`**: Advanced setting for GNSS file processing (default: False)
- **`GNSS_PROVIDER`**: GNSS data provider to use ('gfz', 'usgs', etc.) (default: 'gfz')
- **`INSAR_FILE`**: Name of the InSAR CSV file in the data directory
- **`STATIONS_FILE`**: Name of the stations list file in the data directory
- **`GRID_SIZE_KM`**: Grid size in km for amplitude calculations (default: 0.5km)

### Input File Requirements

#### stations_list
```
Station  latitude  longitude
GNSS1    50.123    6.456
GNSS2    50.234    6.567
```

#### GNSS Time Series Files
The workflow supports different GNSS data providers. The default format is from GFZ, but you can extend support by modifying `gnss_data_providers.py`.

## Versioning and Backup System

This workflow includes a robust versioning and backup system located in the `tools/` directory:

### Backup Tools

- **`create_backup.bat`**: Creates a timestamped backup of all Python scripts
- **`restore_backup.bat`**: Restores scripts from a selected backup
- **`setup_git.bat`**: Sets up Git version control for more advanced versioning

### Documentation

- **`BACKUP_GUIDE.md`**: Instructions for using the backup system
- **`VERSION_CONTROL.md`**: Information about version control with Git

### Usage

1. Run `launcher.bat` and select the backup option to create a backup
2. Backups are stored in `tools/backups/` with timestamp-based folder names
3. To restore a previous version, run `launcher.bat` and select the restore option

For more information on using the backup system, see `tools/BACKUP_GUIDE.md`.

## Performance Considerations

Some scripts (particularly `plot_combined_time_series.py` and `grid_amplitude_analysis.py`) perform intensive calculations that may take significant time with large datasets. The following optimizations are implemented:

- **Multiprocessing**: Parallel execution for CPU-intensive tasks
- **Batch Processing**: Processing data in manageable chunks to limit memory usage
- **Vectorization**: Using NumPy's vectorized operations for faster calculations

For very large datasets, you may need to adjust batch sizes or processing parameters.
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
   - Corresponding GNSS time series files in NEU (North-East-Up) format

## Usage

1. Set your parameters in `master.py` (data paths, filtering thresholds, radius size).
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
- LOS-projected GNSS data (with _LOS suffix)
- Filtered and spatially corrected InSAR data files
- Various plots in the `plots` directory:
  - Combined time series plots showing GNSS and InSAR data
  - Spatial correction (plane fitting) visualizations
  - Grid-based amplitude maps showing seasonal patterns

# Advanced Features

See the experimental version in the `alpha_nnr` directory for additional features that may be incorporated in future releases.

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

## Statistical Robustness

Throughout the workflow, median-based statistics are used instead of mean-based approaches where appropriate:
- InSAR point averaging uses median values to reduce the impact of outliers
- Velocity calculations use median values for robustness
- Statistical analysis applies IQR (Inter-Quartile Range) filtering for improved data quality

## Dependencies

- Python 3.7+
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0 (with GridSpec and patheffects support)
- scipy >= 1.7.0
- geopy >= 2.0.0 (for geodesic distance calculations)
- seaborn >= 0.11.0 (for enhanced visualizations)

## Extending the Workflow

### Adding a New GNSS Data Provider

To add support for a new GNSS data provider:

1. Open `gnss_data_providers.py`
2. Create a new function `load_gnss_data_[provider_name]()` that follows the same interface pattern as the existing providers
3. Update the `load_gnss_data()` function to handle your new provider
4. Modify `master.py` to include your new provider as an option in the `GNSS_PROVIDER` environment variable

Example:
```python
def load_gnss_data_mynewprovider(filepath):
    """Loads GNSS data from a MyNewProvider format file."""
    # Your code here for loading and processing the file
    # ...
    return processed_data_df
```

## Example Workflow

1. Set up your data directory with GNSS and InSAR files
2. Adjust parameters in `master.py` as needed
3. Run the full workflow: `python master.py`
4. Check the `plots` folder for visualizations
5. Examine the corrected data files for further analysis

## Data Flow

1. GNSS 3D velocity calculation (`gnss_3d_vels.py`)
2. Net rotation removal from GNSS data (`remove_gnss_rotation.py`)
3. InSAR filtering and parameter extraction (`filter_insar_save_parameters.py`)
4. Plane fitting and InSAR correction (`fit_plane_correct_insar.py`)
5. GNSS line-of-sight displacement calculation (`gnss_los_displ.py`)
6. Combined time series plotting (`plot_combined_time_series.py`)
7. Optional: Grid-based amplitude analysis (`grid_amplitude_analysis.py`)

## License

See the LICENSE file for details.

1. GNSS velocities are processed
2. InSAR data is filtered based on coherence
3. A correction plane is fitted to align InSAR with GNSS
4. Time series are plotted
5. Spatial amplitude patterns are visualized

For any issues, check the `workflow.log` file for detailed error messages.
