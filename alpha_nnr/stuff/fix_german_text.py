#!/usr/bin/env python3
"""
Script to translate German comments and string literals to English in the radius_sensitivity_analysis.py file.
"""
import re
import sys

# Define German to English translation mappings
translations = {
    # Comments and docstrings
    "# Dateipfade": "# File paths",
    "# Regressionslinie berechnen": "# Calculate regression line",
    "# Erstelle einen gemeinsamen Zeitreihen-Plot für alle Stationen und Radien": "# Create a combined time series plot for all stations and radii",
    "# Erstelle eine neue, gruppierte Legende": "# Create a new, grouped legend",
    "# Erstelle einen Heatmap-Plot der statistischen Daten": "# Create a heatmap plot of statistical data",
    "# Erstelle eine Pivot-Tabelle für die Heatmap": "# Create a pivot table for the heatmap",
    "# Erstelle Heatmap": "# Create heatmap",
    "# Formatiere Datumsachse": "# Format date axis",
    "# Ersetze ungültige Zeichen in Dateinamen": "# Replace invalid characters in filenames",
    "# Finde die korrekten Schreibweisen der Stationsnamen wie sie in der Datei vorkommen": "# Find the correct spellings of station names as they appear in the file",
    
    # Code comments for file loading options
    "# Option 1: Standard Datei \"insar_after_correction.csv\"": "# Option 1: Standard file \"insar_after_correction.csv\"",
    "# Option 2: Aligned EGMS Datei": "# Option 2: Aligned EGMS file",
    "# Option 3: Versuche eine stationsspezifische Datei zu laden": "# Option 3: Try to load a station-specific file",
    "# Option 3: Kombiniere einzelne Stationsdateien": "# Option 3: Combine individual station files",
    "# Option 3: Lade individuelle gefilterte Stationsdateien und kombiniere sie": "# Option 3: Load individual filtered station files and combine them",
    
    # Print statements
    "Lade gefilterte Stationsdateien und kombiniere sie...": "Loading filtered station files and combining them...",
    "Fehler: Keine InSAR-Daten gefunden. Bitte stellen Sie sicher, dass die Dateien existieren.": "Error: No InSAR data found. Please make sure the files exist.",
    "Keine der angegebenen Stationen gefunden:": "None of the specified stations found:",
    
    # Plot labels
    "f\"InSAR {station_name}, R={radius}m ({num_points} Pkte)\"": "f\"InSAR {station_name}, R={radius}m ({num_points} pts)\"",
    
    # Variable names in dictionaries
    "'Anzahl Punkte'": "'Number of points'",
    "'Punktdichte (pro km²)'": "'Point density (per km²)'",
    "'Mittlere Kohärenz'": "'Mean coherence'",
    
    # Other German text fragments that need translation
    "Distance between {station1['Station']} und {station2['Station']}": "Distance between {station1['Station']} and {station2['Station']}",
    "mm/J": "mm/yr"
}

# Load the file
file_path = "radius_sensitivity_analysis.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Apply translations
for german, english in translations.items():
    content = content.replace(german, english)

# Save the translated file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Translation completed for {file_path}")
