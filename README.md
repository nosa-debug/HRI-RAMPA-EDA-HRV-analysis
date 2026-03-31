# HRI-RAMPA-EDA-HRV-analysis
Custom scripts for EDA and HRV analysis 
# Data Analysis Pipeline for ACM THRI Paper

This repository contains the custom Python scripts used for the data processing, normalization, and statistical analysis presented in our paper.

## Workflow Overview
The analysis follows a sequential pipeline:
1. [cite_start]**Preprocessing:** Raw HRV data is converted from microseconds to milliseconds, followed by robust intra-subject Z-score normalization.
2. [cite_start]**Feature Extraction:** HRV (SDNN, RMSSD, CV) and EDA (SCL, SCR) metrics are calculated[cite: 11, 24].
3. [cite_start]**Statistical Analysis:** Includes ANOVA with Post Hoc testing and correlation matrices between physiological signals and user preferences[cite: 26].

## Requirements
- Python 3.x
- pandas
- numpy
- scipy
- statsmodels (for ANOVA/Post Hoc)
- matplotlib/seaborn (for visualization)

## How to use
The scripts are numbered in the order they should be executed. [cite_start]Ensure your input data matches the column headers defined in the `02_preprocessing_normalization.py` script (e.g., Subject, Modo, Tarea)[cite: 11].

## License
This code is provided under the MIT License.
