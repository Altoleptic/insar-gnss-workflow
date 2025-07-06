#!/usr/bin/env python3
"""
Script to fix remaining German text in the radius_sensitivity_analysis.py file.
"""
import re
import os
import sys

def replace_in_file(file_path, replacements):
    """Replace multiple string patterns in a file."""
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False

    # Apply replacements
    print(f"Processing {file_path}...")
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            print(f"  Replaced: {old} -> {new}")

    # Write the file back
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")
        return False

def main():
    # Define the file to process
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(script_dir, "radius_sensitivity_analysis.py")
    
    # Define German to English replacements
    replacements = {
        "# Option 1: Standard Datei": "# Option 1: Standard file",
        "# Option 2: Aligned EGMS Datei": "# Option 2: Aligned EGMS file",
        "# Option 3: Versuche eine stationsspezifische Datei zu laden": "# Option 3: Try to load a station-specific file",
        "# Option 3: Lade individuelle gefilterte Stationsdateien und kombiniere sie": "# Option 3: Load individual filtered station files and combine them",
        "print(\"Lade gefilterte Stationsdateien und kombiniere sie...": "print(\"Loading filtered station files and combining them...",
        "print(f\"  - Lade {station_file}": "print(f\"  - Loading {station_file}",
        "print(\"Fehler: Keine InSAR-Daten gefunden. Bitte stellen Sie sicher, dass die Dateien existieren.": "print(\"Error: No InSAR data found. Please make sure the files exist.",
        "# Formatiere Datumsachse": "# Format date axis",
        "# Erstelle einen gemeinsamen Zeitreihen-Plot für alle Stationen und Radii": "# Create a combined time series plot for all stations and radii",
        "# Erstelle einen Heatmap-Plot der statistischen Daten": "# Create a heatmap plot of statistical data",
        "# Erstelle eine Pivot-Tabelle für die Heatmap": "# Create a pivot table for the heatmap",
        "# Erstelle Heatmap": "# Create heatmap",
        "# Ersetze ungültige Zeichen in Dateinamen": "# Replace invalid characters in filenames",
        "# Finde die korrekten Schreibweisen der Stationsnamen wie sie in der Datei vorkommen": "# Find the correct spellings of station names as they appear in the file",
        "print(f\"Distance between {station1['Station']} und {station2['Station']}": "print(f\"Distance between {station1['Station']} and {station2['Station']}",
        "# Add GNSS trend to the legend (mit gestricheltem Linienstil)": "# Add GNSS trend to the legend (with dashed line style)",
        "# Plot the regression line im kombinierten Plot": "# Plot the regression line in the combined plot",
        "# Colors for stations (case-insensitive durch Doppeleinträge)": "# Colors for stations (case-insensitive with duplicate entries)",
        "label = f\"InSAR {station_name}, R={radius}m ({num_points} Pkte)": "label = f\"InSAR {station_name}, R={radius}m ({num_points} points)",
        "slope_mm_per_year:.2f} mm/J": "slope_mm_per_year:.2f} mm/year",
        "'Anzahl Punkte'": "'Number of points'",
        "'Punktdichte (pro km²)'": "'Point density (per km²)'",
        "'Mittlere Kohärenz'": "'Mean coherence'",
        "print(f\"None of the specified stations found: {station_names}": "print(f\"None of the specified stations found: {station_names}",
    }
    
    # Process the file
    if replace_in_file(target_file, replacements):
        print("Processing completed successfully.")
    else:
        print("Processing failed.")

if __name__ == "__main__":
    main()
