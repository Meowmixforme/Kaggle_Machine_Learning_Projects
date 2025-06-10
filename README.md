# Kaggle Machine Learning Projects

This repository contains various machine learning projects based on Kaggle datasets, implemented in different programming languages and frameworks.

## Project 1: Aviation Crash Analysis for the Past 100 Years

### Overview

This project analyses aviation crash data spanning more than a century (from 1908 onwards), using R for data processing, exploratory analysis, and machine learning classification. The primary objective is to predict crash causes based on various flight parameters, temporal data, and text analysis of crash descriptions.

### Dataset

The dataset contains detailed records of airplane crashes including:
- Date and time of crashes
- Location information
- Aircraft type and operator details
- Casualty figures (aboard, fatalities, ground casualties)
- Summary descriptions of crash incidents

### Implementation Details

The analysis follows a comprehensive machine learning workflow:

1. **Data Cleansing**
   - Treatment of missing values
   - Date and time format standardisation
   - Text data preparation

2. **Feature Engineering**
   - Creation of derived features like total fatalities and survival rates
   - Text mining from crash summaries
   - Categorical time of day classification
   - Decade grouping for temporal analysis

3. **Crash Cause Classification**
   - Natural language processing to extract crash causes from summaries
   - Categories include:
     - Bad Weather / Natural Disasters
     - Human Error
     - Acts of War / Terrorism
     - Mechanical Failure
     - Other causes

4. **Machine Learning Models**
   - Random Forest classification
   - XGBoost implementation
   - Support Vector Machines
   - Cross-validation and hyperparameter tuning

5. **Model Evaluation**
   - Confusion matrix analysis
   - Feature importance assessment
   - Classification metrics by crash cause category

### Key Findings

- Human error and mechanical failures constitute the majority of crash causes
- Weather-related crashes show seasonal patterns
- Crash causes have evolved significantly over different decades
- Text-based features provide strong predictive signal for crash classification
- Random Forest model achieves the highest accuracy for cause prediction

### Tools and Libraries

- **R Language**: Primary implementation
- **Data Manipulation**: dplyr, tidyr, zoo
- **Text Analysis**: tidytext, tm
- **Machine Learning**: caret, randomForest, xgboost, e1071
- **Visualisation**: ggplot2

### Future Work

- Integration of geographical analysis
- Advanced deep learning text classification
- Time series forecasting for crash prediction
- Anomaly detection for unusual crash patterns
- Interactive dashboard for findings exploration

### Usage

The project is implemented in R Markdown notebook format. To run the analysis:

1. Clone this repository
2. Ensure R and required packages are installed
3. Open the R Markdown notebook in RStudio
4. Run all chunks sequentially
