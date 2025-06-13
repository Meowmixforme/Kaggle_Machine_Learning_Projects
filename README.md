# Kaggle Machine Learning Projects

This repository contains various machine learning projects based on Kaggle datasets, implemented in different programming languages and frameworks. The first two are projects that I was originally going to use for my university module but were abandoned at the time due to their large scope.

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

**Data Cleansing**
- Treatment of missing values
- Date and time format standardisation
- Text data preparation

**Feature Engineering**
- Creation of derived features like total fatalities and survival rates
- Text mining from crash summaries
- Categorical time of day classification
- Decade grouping for temporal analysis

**Crash Cause Classification**
- Natural language processing to extract crash causes from summaries
- Categories include:
  - Bad Weather / Natural Disasters
  - Human Error
  - Acts of War / Terrorism
  - Mechanical Failure
  - Other causes

**Machine Learning Models**
- Random Forest classification
- XGBoost implementation
- Naive Bayes classification
- Cross-validation and hyperparameter tuning

**Model Evaluation**
- Confusion matrix analysis
- Feature importance assessment
- Classification metrics by crash cause category

### Key Findings
- Human error and mechanical failures constitute the majority of crash causes
- Weather-related crashes show distinct patterns across flight phases
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

## Project 2: Medical Appointment No-Shows in BrazilAdd commentMore actions

### Overview
This project analyses a dataset of medical appointments in Brazil to predict patient no-shows using advanced machine learning techniques. The implementation uses R for data processing, feature engineering, and ensemble model development. The primary objective is to identify factors that contribute to appointment no-shows and develop a predictive model that healthcare facilities can use to reduce missed appointments.

### Dataset
The dataset contains over 100,000 medical appointments from the Brazilian public healthcare system including:
- Patient demographic information (age, gender)
- Appointment details (scheduling date, appointment date)
- Medical conditions (hypertension, diabetes, alcoholism)
- Social determinants (neighbourhood, welfare benefits)
- Communication factors (SMS reminders)

### Implementation Details
The analysis follows a comprehensive machine learning workflow:

**Data Cleansing**
- Treatment of missing values
- Date and time format standardisation
- Removal of extreme outliers (ages > 100)
- Correction of inconsistent categorical values

**Feature Engineering**
- Creation of waiting days variable between scheduling and appointment
- Detailed waiting time categories (Same day to Over month)
- Time-of-day appointment categories
- NHS age group classification
- Neighbourhood risk categorisation based on historical no-show rates
- Health condition combinations and interactions
- Seasonal and day-of-week patterns

**Advanced Class Balancing**
- Random upsampling of minority class (no-shows)
- Equal representation of show/no-show appointments in training data
- Preservation of original distribution in testing data

**Machine Learning Models**
- XGBoost implementation
- Random Forest classification
- Logistic Regression
- Ensemble model combining all three approaches
- Cross-validation and hyperparameter tuning

**Threshold Optimisation**
- F1-score based threshold determination
- Adjustment for class imbalance in predictions
- Precision-recall trade-off analysis

**Model Evaluation**
- ROC curve analysis with AUC metrics
- Confusion matrix assessment
- Feature importance ranking
- Classification performance across patient demographics

### Key Findings
- Waiting time is the strongest predictor of appointment no-shows
- SMS reminders significantly reduce no-show probability
- Young patients with long waiting times have higher no-show rates
- Neighbourhood characteristics influence attendance patterns
- Health conditions impact attendance in complex ways
- The ensemble model achieves superior performance compared to individual models

### Tools and Libraries
- **R Language**: Primary implementation
- **Data Manipulation**: dplyr, tidyr, lubridate
- **Machine Learning**: caret, xgboost, randomForest
- **Model Evaluation**: pROC, confusionMatrix
- **Visualisation**: ggplot2

### Future Work
- Integration of distance/travel time to medical facility
- Inclusion of weather data on appointment days
- Text message content analysis for effectiveness
- Development of an early warning system for high-risk no-shows
- Cost-benefit analysis of intervention strategies

### Usage
The project is implemented in R script format. To run the analysis:
1. Clone this repository
2. Ensure R and required packages are installed
3. Run the improved_medical_noshow_prediction.R script
4. Model outputs will be saved in the project directory
Add comment


## Project 3: Diabetes Risk Prediction

### Overview
This project predicts diabetes risk using a range of classical machine learning models on a dataset of patient symptoms and demographic attributes.

### Dataset
- Features: Age, Gender, and 14 symptom indicators (e.g., Polyuria, Polydipsia, sudden weight loss, etc.)
- Target: Diabetes class (Positive/Negative)
- Size: 520 entries

### Implementation Details

#### Data Preprocessing
- Encode categorical variables and binary symptoms as 0/1
- Encode target variable
- Stratified train-test split (70/30)
- Feature standardisation

#### Model Training
Trained and evaluated the following classifiers:
- Logistic Regression
- K-Nearest Neighbours
- Decision Tree
- Support Vector Machine (Linear and RBF kernels)
- Neural Network (MLP)
- Random Forest
- Gradient Boosting

#### Evaluation
- Accuracy, precision, recall, F1-score, confusion matrix
- Feature importance analysis for tree-based models
- ROC curves and AUC for key models
- 5-fold cross-validation and hyperparameter tuning via grid search

### Results (Example)
- All models achieve high accuracy (94–98%)
- Decision Tree and Random Forest achieve 98% accuracy
- Most important features include Polyuria, Polydipsia, sudden weight loss, Age

### Usage
The project is implemented in Python (Jupyter/Colab notebook or script).
1. Ensure Python and required packages are installed (`numpy`, `pandas`, `scikit-learn`, `matplotlib`)
2. Place `diabetes_data_upload.csv` in the working directory
3. Run the notebook or script


## Project 4: Cats vs Dogs Image Classification (Transfer Learning with MobileNetV2)

### Overview
This project classifies images of cats and dogs using transfer learning with MobileNetV2, leveraging a Kaggle dataset. The workflow demonstrates the use of pretrained convolutional neural networks, data augmentation, and fine-tuning to achieve high accuracy on a small dataset.

### Dataset
- [Kaggle Cats and Dogs Image Classification Dataset](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification)
- Contains separate folders for training and testing, with images labeled as "cats" or "dogs".

### Implementation Details

#### Data Preparation
- Automatic download and extraction of the dataset from Kaggle.
- Construction of pandas DataFrames for training, validation, and test splits.
- Data augmentation (rotation, shifts, flips) applied to the training set.
- Images resized to 224x224 and normalized.

#### Model Architecture
- **Base:** Pretrained MobileNetV2 (ImageNet weights, base frozen for initial training)
- **Top:** GlobalAveragePooling2D + Dense layer with sigmoid activation

#### Training Workflow
1. **Initial Training:**  
   - Only the custom classification head is trained, MobileNetV2 base is frozen.
   - Early stopping and learning rate reduction callbacks.
2. **Fine-tuning:**  
   - The last 20 layers of MobileNetV2 are unfrozen and trained at a lower learning rate.
   - Further early stopping to prevent overfitting.

#### Evaluation
- Model evaluated on a held-out test set.
- Performance metrics: accuracy, precision, recall, F1-score, confusion matrix (per class).
- Achieved ~94% accuracy on the test set.

#### Key Results (Example)
- **Test accuracy:** 94%
- **Cat:** precision 1.00, recall 0.87, F1 0.93
- **Dog:** precision 0.89, recall 1.00, F1 0.94

#### Tools and Libraries
- **Python** (Jupyter/Colab)
- **TensorFlow/Keras** for deep learning and transfer learning
- **pandas**, **numpy**, **matplotlib**, **seaborn** for data handling and visualisation
- **scikit-learn** for evaluation metrics

### Usage
1. Download or clone this repository.
2. Obtain the [Kaggle dataset](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification) and your Kaggle API key (`kaggle.json`).
3. Run the provided Colab notebook or Python script.  
   - The code will handle dataset download, extraction, preprocessing, training, and evaluation.
4. The trained model is saved in Keras format (`catdog_classifier_mobilenetv2.keras`).

### Future Work
- Experiment with alternative pretrained models (e.g., EfficientNet, ResNet).
- Deploy the model as a web or mobile application.
- Test on user-supplied images for real-world validation.
- Explore interpretability methods (Grad-CAM, SHAP) for model explanations.
