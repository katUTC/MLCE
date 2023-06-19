# MLCE
**Machine Learning Classification for Excel-Based Datasets (A User-Friendly Interface)

This repository has been designed to provide a user-friendly graphic user interface for the exploration of machine learning classification strategies.
Notably, the analysis of data pertaining to muscle characterization is included to encourage the use of machine learning strategies for prediction in the biological and medical sciences.
Often, clinical research publishes statistical data, and the prediction power of machine learning is not always explored. To this effect, the interface has been developed to facilitate the exploration of machine learning for biological data.

# Installation
For those with little coding experience, a full tutorial is available on how to download and implement the code:
[MLCE_installation_conda.zip](UTCBioML/MLCE_installation_conda.zip)

# Data examples
Example data is provided in the [UTCBioML/ML_muscle.xlsx](UTCBioML/ML_muscle.xlsx) file, and example results are provided in the [ExamplesCharles19/](ExamplesCharles19/) directory.

# Code considerations
## Solutions for missing data
The software allows users to select which information they would like to analyze from an Excel-based dataset, and identify the data labels, and primary feature of interest for which they seek to use in their predictions.
The issue of missing data can be easily addressed, with users being able to select to either delete the row of data, or replace missing or NaN values identified in Excel with a group mean or median value, based on the label. This inference allows for data to be maintained when limited.
Users can explore the effect of interence on their results.

## Report production
Reports are created to track results. This includes reports on the number of data points imported, the number of missing data points, the values used for inference, and the results of the machine learning analysis.

## Hyper-parameter tuning
Hyper-parameters are values or implementation considerations for the algorithms used in machine learning that must be carefully tuned. Tuning is when these values are adapted to the dataset. 
For example, KNN or K nearest neighbours creates centroids in the data and groups the nearest points to each centroid to create a classification group. The number of centroids used is often tested, perhaps commencing with 2 and increasing to a value that is logical for the number of values in the dataset and the researcher's understanding of the data.
The interface presented here, allows users to test different hyper-parameters and see their effect on the results. Once again, a report is produced which can easily be saved with the results from machine learning.
