# K-Means Clustering of NOAA Weather Data

A comprehensive Python application that implements custom K-Means clustering to analyze NOAA Global Surface Summary of the Day (GSOD) weather data. This project demonstrates data cleaning, preprocessing, algorithm implementation, and evaluation metrics for weather pattern identification.

## Features

- **Data Cleaning**: Handles missing value codes (9999.9, 99.99) properly
- **Custom K-Means Implementation**: Complete from-scratch implementation
- **Multiple Distance Metrics**: Support for Euclidean and Manhattan distances
- **Comprehensive Evaluation**: WCSS, Silhouette Score, and comparison with scikit-learn
- **Visualization**: Elbow method, silhouette plots, cluster analysis, and convergence tracking
- **Automatic k-selection**: Determines optimal cluster count using silhouette scores

## Dataset

The project uses the NOAA GSOD dataset from the Registry of Open Data on AWS. The dataset contains daily weather measurements from global stations.

Key features used:
- Temperature (TEMP)
- Dew point temperature (DEWP) 
- Sea-level pressure (SLP)
- Wind speed (WDSP)
- Precipitation (PRCP)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Favourchiwendu/kmeans
cd kmeans

2. Run the file
```bash
python main.py
