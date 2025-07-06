# GNSS-InSAR Integration Workflow Flowchart

## Overview
This flowchart shows the data flow and dependencies between scripts and data files in the GNSS-InSAR integration workflow.

## Simplified Workflow Overview

```mermaid
---
config:
  flowchart:
    curve: linear
  theme: forest
  themeVariables: {}
---
flowchart TD
    %% Top level inputs
    INPUT1["GNSS Time Series"]
    INPUT2["InSAR Data CSV"]
    
    %% GNSS Processing Chain (Left)
    INPUT1 --> B1["1\. GNSS Velocities - gnss_3d_vels.py"]
    B1 --> PARAMS1["parameters.csv"]
    PARAMS1 --> B2["2\. Remove Rotation - remove_gnss_rotation.py"]
    B2 --> PARAMS2["parameters.csv + NNR corrections"]
    B2 --> NNR_FILES["GNSS NNR Files"]
    
    %% InSAR Processing Chain (Right)
    INPUT2 --> B3["3\. Filter InSAR - filter_insar_save_parameters.py"]
    B3 --> PARAMS3["parameters.csv + LOS components"]
    INPUT2 --> B4["4\. Spatial Alignment - fit_plane_correct_insar.py"]
    PARAMS3 --> B4
    B4 --> ALIGNED_INSAR["InSAR Aligned CSV"]
    
    %% Integration Steps (Bottom)
    NNR_FILES --> B5["5\. LOS Projection - gnss_los_displ.py"]
    PARAMS3 --> B5
    B5 --> LOS_FILES["GNSS LOS Files"]
    
    ALIGNED_INSAR --> B6["6\. Visualization - plot_combined_time_series.py"]
    LOS_FILES --> B6
    B6 --> PLOTS["Combined Plots"]
    
    %% Optional Analysis
    NNR_FILES --> B7["7\. Grid Analysis - grid_amplitude_analysis.py"]
    B7 --> GRID_PLOTS["Amplitude Maps"]
    
    %% Auxiliary input (side)
    INPUT3["stations file"]
    INPUT3 -.-> B1
    INPUT3 -.-> B2
    INPUT3 -.-> B4
    INPUT3 -.-> B5
    INPUT3 -.-> B6
    
    %% Node styling
    INPUT1:::input
    INPUT2:::input
    INPUT3:::input
    B1:::script
    B2:::script
    B3:::script
    B4:::script
    B5:::script
    B6:::script
    B7:::script
    PARAMS1:::data
    PARAMS2:::data
    PARAMS3:::data
    NNR_FILES:::data
    ALIGNED_INSAR:::data
    LOS_FILES:::data
    PLOTS:::output
    GRID_PLOTS:::output
    
    classDef script fill:#F3E5F5,stroke:#7B1FA2,stroke-width:3px,color:#000
    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:3px,color:#000
    classDef data fill:#E8F5E8,stroke:#388E3C,stroke-width:3px,color:#000
    classDef output fill:#FFF3E0,stroke:#F57C00,stroke-width:3px,color:#000
```

---

---

## Detailed Workflow Dependencies and Data Flow

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}, 'theme': 'forest', 'themeVariables': {'primaryColor': '#ffffff', 'lineColor': '#333333', 'primaryTextColor': '#000000', 'primaryBorderColor': '#333333', 'secondaryColor': '#f0f0f0', 'tertiaryColor': '#ffffff', 'lineWidth': '4px'}}}%%
flowchart TD
    %% Input Data
    A[Input Data Directory] --> A1[stations file]
    A -->    A2[GNSS Time Series data]
    A --> A3[InSAR Data CSV]
    
    %% Step 1: GNSS 3D Velocities
    A1 --> B1[Step 1: gnss 3d vels script]
    A2 --> B1
    B1 --> B1_OUT[parameters.csv with GNSS velocities]
    B1 --> B1_PLOT[Displacement plots]
    
    %% Step 2: Net Rotation Removal
    B1_OUT --> B2[Step 2: remove gnss rotation script]
    A1 --> B2
    A2 --> B2
    B2 --> B2_OUT1[parameters.csv updated with NNR]
    B2 --> B2_OUT2[GNSS NNR data files]
    B2 --> B2_OUT3[rotation_vector_omega.txt]
    B2 --> B2_PLOT[NNR comparison plots]
    
    %% Step 3: InSAR Filtering
    A3 --> B3[Step 3: filter insar save parameters script]
    B3 --> B3_OUT[parameters.csv updated with LOS]
    
    %% Step 4: Spatial Alignment
    A3 --> B4[Step 4: fit plane correct insar script]
    A1 --> B4
    B3_OUT --> B4
    B4 --> B4_OUT[InSAR Aligned CSV]
    B4 --> B4_PLOT[Spatial correction plot]
    
    %% Step 5: GNSS LOS Projection
    B2_OUT2 --> B5[Step 5: gnss los displ script]
    A1 --> B5
    B3_OUT --> B5
    B5 --> B5_OUT[GNSS LOS data files]
    
    %% Step 6: Time Series Visualization
    B4_OUT --> B6[Step 6: plot combined time series script]
    B5_OUT --> B6
    A1 --> B6
    B6 --> B6_OUT1[Global velocity maps]
    B6 --> B6_OUT2[Station velocity maps]
    B6 --> B6_OUT3[Time series comparisons]
    
    %% Step 7: Grid Analysis (Optional)
    B2_OUT2 --> B7[Step 7: grid amplitude analysis script]
    B7 --> B7_OUT[Seasonal amplitude maps]
    
    %% Master Controller
    MASTER[master.py Workflow Controller] -.-> B1
    MASTER -.-> B2
    MASTER -.-> B3
    MASTER -.-> B4
    MASTER -.-> B5
    MASTER -.-> B6
    MASTER -.-> B7
    
    %% Output Directory
    PLOTS[plots directory]
    B1_PLOT --> PLOTS
    B2_PLOT --> PLOTS
    B4_PLOT --> PLOTS
    B6_OUT1 --> PLOTS
    B6_OUT2 --> PLOTS
    B6_OUT3 --> PLOTS
    B7_OUT --> PLOTS
    
    %% Styling with better contrast
    classDef inputData fill:#E3F2FD,stroke:#1976D2,stroke-width:3px,color:#000
    classDef script fill:#F3E5F5,stroke:#7B1FA2,stroke-width:3px,color:#000
    classDef dataFile fill:#E8F5E8,stroke:#388E3C,stroke-width:3px,color:#000
    classDef plot fill:#FFF3E0,stroke:#F57C00,stroke-width:3px,color:#000
    classDef controller fill:#FCE4EC,stroke:#C2185B,stroke-width:3px,color:#000
    
    class A,A1,A2,A3 inputData
    class B1,B2,B3,B4,B5,B6,B7 script
    class B1_OUT,B2_OUT1,B2_OUT2,B2_OUT3,B3_OUT,B4_OUT,B5_OUT dataFile
    class B1_PLOT,B2_PLOT,B4_PLOT,B6_OUT1,B6_OUT2,B6_OUT3,B7_OUT,PLOTS plot
    class MASTER controller
```

## Script Execution Order

The workflow follows this strict sequential order:

1. **`gnss_3d_vels.py`** - Calculate 3D velocities from GNSS time series
2. **`remove_gnss_rotation.py`** - Remove net rotation from GNSS data (NNR correction)
3. **`filter_insar_save_parameters.py`** - Filter InSAR data and extract LOS parameters
4. **`fit_plane_correct_insar.py`** - Align InSAR data spatially with GNSS using plane correction
5. **`gnss_los_displ.py`** - Project GNSS displacements to Line-of-Sight direction
6. **`plot_combined_time_series.py`** - Create comprehensive time series visualizations
7. **`grid_amplitude_analysis.py`** - Optional grid-based spatial analysis

## Key Data Dependencies

### Critical Files Created and Used:
- **`parameters.csv`**: Created by `gnss_3d_vels.py`, updated by `remove_gnss_rotation.py` and `filter_insar_save_parameters.py`
- **`_NNR.txt` files**: Created by `remove_gnss_rotation.py`, used by `gnss_los_displ.py`
- **`_aligned.csv` file**: Created by `fit_plane_correct_insar.py`, used by `plot_combined_time_series.py`
- **`_NNR_LOS.txt` files**: Created by `gnss_los_displ.py`, used by `plot_combined_time_series.py`

### Environment Variables Configuration:
All scripts are controlled via environment variables set in `master.py`:
- `DATA_DIR`: Base directory for all data
- `INSAR_RADIUS`: Radius for InSAR point averaging
- `MIN_TEMPORAL_COHERENCE`: Quality threshold for InSAR filtering
- `USE_NNR_CORRECTED`: Whether to use NNR-corrected GNSS files
- `GNSS_PROVIDER`: Data provider for GNSS format handling

## Input Requirements

### Essential Input Files:
1. **`stations_list`**: Text file with station coordinates
2. **GNSS time series files**: `[Station]_NEU_TIME*.txt` format
3. **InSAR data file**: CSV with temporal coherence and displacement columns

### Directory Structure:
```
C:/insar_gnss_data/
├── stations_list
├── [Station]_NEU_TIME.txt files
├── EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A.csv
└── plots/ (created automatically)
```

## Output Products

### Generated Data Files:
- Net-rotation-corrected GNSS time series (`_NNR.txt`)
- Line-of-sight projected GNSS data (`_NNR_LOS.txt`)
- Spatially aligned InSAR data (`_aligned.csv`)
- Combined parameters file (`parameters.csv`)
- Rotation vector estimates (`rotation_vector_omega.txt`)

### Visualization Outputs:
- Individual GNSS displacement plots
- NNR correction comparison plots
- Spatial correction visualization
- Combined velocity maps
- Station-specific velocity maps
- Time series comparison plots
- Grid-based amplitude analysis maps

## Performance Notes

- **Multiprocessing**: `plot_combined_time_series.py` uses parallel processing for speed
- **Memory optimization**: Large datasets are processed in batches
- **Runtime**: Complete workflow typically takes 8-10 minutes for moderate datasets
- **Critical dependencies**: Each step must complete successfully before the next begins
