# Kaggle Machine Learning Projects

This repository contains various machine learning projects based on Kaggle (and other) datasets, implemented in different programming languages and frameworks. The first two are projects that I was originally going to use for my university module but were abandoned at the time due to their large scope.

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



## Project 2: Medical Appointment No-Shows in Brazil

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


## Project 5: Image Super-Resolution with SRCNN (Keras)

This project demonstrates image super-resolution using a Super-Resolution Convolutional Neural Network (SRCNN) implemented in Keras with TensorFlow backend. The workflow includes image degradation, quality assessment (PSNR, MSE, SSIM), SRCNN model building, and evaluation.

- **Dataset:** Example images (e.g., baboon, lenna, etc.)
- **Image Preparation:** Images are artificially degraded by downsampling and upsampling.
- **Quality Metrics:** PSNR, MSE, and SSIM are computed before and after super-resolution.
- **SRCNN Model:** Custom SRCNN with three convolutional layers, trained weights are loaded (`3051crop_weight_200.h5`).
- **Prediction:** The model reconstructs high-resolution images from degraded versions; results are evaluated and visualized.
- **Libraries:** Python, Keras, TensorFlow, NumPy, OpenCV, scikit-image, matplotlib.

#### Usage

1. Place original images in `source/` folder.
2. Run the script to generate degraded images in `images/`.
3. Use the `predict` function to upscale and evaluate any image.
4. Example code and full pipeline in `srcnn_super_resolution.py`.

#### Key Files

- `srcnn_super_resolution.py` — Main pipeline and model code.
- `3051crop_weight_200.h5` — Pretrained SRCNN weights.
- `source/` — Folder for original high-res images.
- `images/` — Folder for degraded images.

#### Results

Sample metrics for `flowers.bmp`:
- Degraded Image: PSNR 27.25, MSE 367.56, SSIM 0.87
- Reconstructed Image: PSNR 29.66, MSE 210.95, SSIM 0.89

#### References

- [SRCNN Paper and Project Page](https://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
- [Keras SRCNN Implementation](https://github.com/MarkPrecursor/SRCNN-keras)


## Project 6: Iris Dataset Classification with Classical Machine Learning

### Overview
This project demonstrates classical supervised machine learning techniques using the renowned Iris dataset. The objective is to classify iris flowers into three species (Iris-setosa, Iris-versicolor, Iris-virginica) based on their sepal and petal measurements. The workflow includes exploratory data analysis, visualisation, model training, and evaluation using several algorithms.

### Dataset
- **Source:** [UCI Machine Learning Repository - Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)
- **Features:**
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target:** Iris species (Setosa, Versicolor, Virginica)
- **Size:** 150 samples (50 per class)

### Implementation Details

#### Data Exploration and Visualisation
- Summary statistics and distribution plots for all features
- Class distribution analysis
- Histograms and scatter plot matrix to visualise relationships between variables

#### Model Training and Evaluation
- Data split: 80% training, 20% validation (stratified)
- Models trained:
  - Logistic Regression
  - K-Nearest Neighbours (KNN)
  - Support Vector Machine (SVM, linear kernel)
- 10-fold cross-validation for model selection and performance estimation
- Final evaluation on the validation set

#### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score (per class)
- Confusion matrix
- Model comparison based on average cross-validation accuracy

### Example Results
- All models achieve high accuracy (87–90%) on the validation set
- Setosa class is always perfectly classified; most errors are between Versicolor and Virginica
- KNN achieves the highest validation accuracy in the sample run

### Tools and Libraries
- **Python**: Primary implementation (Jupyter/Colab notebook or script)
- **Data Manipulation**: pandas, numpy
- **Visualisation**: matplotlib, pandas.plotting
- **Machine Learning**: scikit-learn

### Usage

1. Ensure Python and required packages are installed:
   - `pandas`, `numpy`, `matplotlib`, `scikit-learn`
2. Download or clone this repository.
3. Run the `iris_classification.py` script or use the code in a Jupyter notebook.
   - The script will automatically download the Iris dataset from the UCI repository.
4. Review printed model evaluation reports and plots.


## Project 7: SMS Spam Detection with NLTK and scikit-learn

### Overview
This project demonstrates the use of natural language processing (NLP) and classical machine learning algorithms to detect spam in SMS messages. The workflow includes data cleansing, feature engineering, model training using several classifiers, and ensemble methods, all implemented in Python with scikit-learn and NLTK.

### Dataset
- **Source:** [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Features:**
  - Message label: "ham" (legitimate) or "spam"
  - Text: raw SMS message content
- **Size:** 5,572 messages

### Implementation Details

#### Data Exploration and Preprocessing
- Inspection of dataset shape, class distribution, and sample messages
- Conversion of class labels to binary (0 = ham, 1 = spam)
- Text cleansing using regular expressions:
  - Removal of email addresses, URLs, phone numbers, money symbols, and numbers
  - Stripping of punctuation and extra whitespace
  - Lowercasing of all text
- Removal of English stop words
- Stemming using the Porter stemmer

#### Feature Engineering
- Bag-of-words model created using tokenisation and frequency distribution
- Selection of the 1,500 most common words as features
- Feature extraction for each message using presence/absence of common words

#### Dataset Preparation
- Conversion of messages into feature-label pairs
- Random shuffling and splitting: 75% training, 25% testing

#### Model Training and Evaluation
- Training of multiple classifiers using NLTK's scikit-learn wrapper:
  - K-Nearest Neighbours
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - Stochastic Gradient Descent (SGD) Classifier
  - Multinomial Naive Bayes
  - Support Vector Machine (linear kernel)
- Evaluation of each classifier on the test set (accuracy reported)
- Ensemble learning using a hard-voting classifier for improved performance

#### Results
- All models achieve high accuracy, with SVM linear and ensemble classifier exceeding 98%
- Example confusion matrix for ensemble classifier:

  |              | predicted ham | predicted spam |
  |--------------|:------------:|:--------------:|
  | actual ham   |     1197     |       2        |
  | actual spam  |      31      |      163       |


  
## Project 8: Unsupervised Clustering of Handwritten Digits (MNIST) with K-Means

### Overview
This project applies unsupervised machine learning to the MNIST handwritten digits dataset, using the MiniBatchKMeans algorithm to discover natural groupings in the images. The objective is to explore how well clustering can recover digit classes without using labels, and to visualise the cluster centroids as representative digit images.

### Dataset
- **Source:** [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- **Features:** 28 × 28 greyscale images of handwritten digits (0–9)
- **Target:** Digit class (used only for evaluation)
- **Size:** 60,000 training images, 10,000 test images

### Implementation Details

**Data Preparation**
- Load the MNIST dataset via Keras.
- Flatten images from 28 × 28 to 784-dimensional vectors.
- Normalise pixel values to the range [0, 1].

**Clustering**
- Use MiniBatchKMeans with various numbers of clusters (10–256).
- Fit the model to the training data.
- For each configuration:
  - Evaluate inertia, homogeneity, and clustering accuracy (by inferring cluster labels).
  - Test on the held-out test set for final accuracy.

**Cluster Label Inference**
- Assign the most probable digit label to each cluster using majority voting within each cluster.
- Map cluster assignments to inferred digit labels for accuracy calculation.

**Visualisation of Cluster Centroids**
- Reshape cluster centroids back to 28 × 28 images.
- Display the centroids using matplotlib to observe typical digit patterns discovered by the clustering.

**Key Functions**
- `infer_cluster_labels`: Associates each cluster with a digit label based on the majority of ground-truth labels within the cluster.
- `infer_data_labels`: Maps cluster assignments to inferred labels for evaluation.

### Results
- With 256 clusters, K-Means achieves approximately 90% accuracy on the MNIST test set (without using labels for training).
- Visualised centroids display clear digit-like patterns, showing that K-Means can discover meaningful features in the data.

### Example Results

| Number of Clusters | Homogeneity | Accuracy (Train) | Accuracy (Test) |
|--------------------|-------------|------------------|-----------------|
| 10                 | 0.49        | 0.62             | –               |
| 36                 | 0.67        | 0.75             | –               |
| 256                | 0.84        | 0.90             | 0.90            |

### Tools and Libraries
- **Python**: Jupyter/Colab notebook or script
- **Machine Learning**: scikit-learn
- **Data Processing**: numpy
- **Visualisation**: matplotlib
- **Dataset Loading**: keras

### Usage

1. Ensure Python and required packages are installed:
   - `numpy`, `matplotlib`, `scikit-learn`, `keras`
2. Run the Python script or Jupyter notebook.
   - The script will download MNIST if needed, perform clustering, and display visualisations and metrics.
3. Review the printed evaluation metrics and the subplot grid of cluster centroids.



## Project 9: Image Classification on CIFAR-10 with the All-CNN

### Overview
This project demonstrates image classification using the CIFAR-10 dataset and the All Convolutional Neural Network (All-CNN) architecture, implemented in Keras. The All-CNN is a deep convolutional model that replaces traditional fully connected layers with 1×1 and global average pooling layers, enabling high accuracy with fewer parameters.

### Dataset
- **Source:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Features:** 32 × 32 colour images, 3 channels (RGB)
- **Target:** 10 object classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Size:** 50,000 training images, 10,000 test images

### Implementation Details

**Data Preparation**
- Load CIFAR-10 dataset via Keras.
- Visualise sample images as a 3×3 grid.
- Normalise pixel values to the range [0, 1].
- Convert class labels to one-hot vectors.

**Model: All-CNN Architecture**
- Stacks of 3×3 and 1×1 convolutional layers with ReLU activation and Dropout for regularisation.
- No fully connected layers; final classification via global average pooling and softmax.
- ~1.37 million trainable parameters.

**Training**
- Optimiser: SGD with momentum and Nesterov acceleration.
- Loss function: Categorical cross-entropy.
- Trained for up to 350 epochs (early stopping recommended in practice).
- Optionally, load pretrained weights for rapid evaluation.

**Evaluation**
- Model accuracy assessed on the CIFAR-10 test set.
- Typical accuracy with pretrained weights: ~91%.
- Class label mapping provided for easy interpretation of predictions.

**Prediction and Visualisation**
- Generate predictions on a batch of test images.
- Display a 3×3 grid of images with predicted and actual labels for qualitative assessment.

### Example Results

| Metric     | Value      |
|------------|------------|
| Test Accuracy | 0.91   |
| Classes      | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |

Example prediction (for batch of 9 images):  
Each subplot shows model's predicted class and actual class for visual inspection.

### Tools and Libraries
- **Python**: Jupyter/Colab notebook or script
- **Deep Learning**: Keras, TensorFlow backend
- **Data Handling**: numpy, PIL
- **Visualisation**: matplotlib

### Usage

1. Ensure Python and required packages are installed:
   - `keras`, `tensorflow`, `numpy`, `matplotlib`, `PIL`
2. Download or clone this repository.
3. (Optional) Download pretrained weights file (`all_cnn_weights_0.9088_0.4994.hdf5`) for rapid evaluation.
4. Run the script or notebook.
   - The code will load data, train or load the model, and report results.
5. Review printed accuracy and the image grid of predictions.


## Project 10: Natural Language Processing Fundamentals with NLTK

### Overview
This project explores fundamental natural language processing (NLP) techniques using the NLTK library in Python. The workflow covers tokenisation, stop word removal, stemming, part-of-speech tagging, chunking, named entity recognition, and sentiment analysis using the movie reviews corpus. The project also demonstrates how to integrate NLTK tokenisation and feature extraction with scikit-learn classifiers.

### Dataset
- **Sample Texts:** Provided within the code and NLTK corpora.
- **Corpora Used:**
  - [state_union](https://www.nltk.org/nltk_data/) (presidential speeches)
  - [movie_reviews](https://www.nltk.org/nltk_data/) (labelled positive/negative reviews)
  - Universal Declaration of Human Rights (UDHR) for multilingual examples

### Implementation Details

**Tokenisation & Stop Word Removal**
- Sentence and word tokenisation with `sent_tokenize` and `word_tokenize`.
- Removal of English stop words using `nltk.corpus.stopwords`.

**Stemming**
- Application of the Porter Stemmer to words and whole sentences for root word extraction.

**Part-of-Speech Tagging**
- Tokenisation and tagging of words in sentences using `nltk.pos_tag`.

**Chunking & Chinking**
- Use of regular expressions to extract and visualise chunks (noun phrases) and chinks from tagged sentences with `nltk.RegexpParser`.

**Named Entity Recognition**
- Extraction and visualisation of named entities using `nltk.ne_chunk`.

**Text Corpus Exploration**
- Use of the state_union and UDHR corpora for practical NLP examples.

**Sentiment Analysis with Movie Reviews**
- Creation of a document list from the NLTK movie_reviews corpus, labelled as positive or negative.
- Construction of a frequency distribution and selection of the 4,000 most common words as features.
- Feature extraction for each review (bag-of-words, presence/absence of features).

**Model Training and Evaluation**
- Split of feature sets into training and testing datasets using scikit-learn's `train_test_split`.
- Use of scikit-learn's SVM (linear kernel) via NLTK's `SklearnClassifier`.
- Training on the training set and evaluation on the test set.
- Example accuracy: SVC achieves ~81% accuracy on the test split.

### Example Results

| Step                        | Output Example                                      |
|-----------------------------|-----------------------------------------------------|
| Sentence tokenisation       | ['Hello students, how are you doing today?', ...]   |
| Word tokenisation           | ['Hello', 'students', ',', 'how', 'are', ...]      |
| Stop word removal           | ['This', 'sample', 'text', ..., 'filtration', '.'] |
| Stemming                    | ['ride', 'ride', 'rider', 'ride']                  |
| POS tagging                 | [('Hello', 'NNP'), ('students', 'NNS'), ...]       |
| Chunking                    | (Chunk PRESIDENT/NNP GEORGE/NNP W./NNP BUSH/NNP)   |
| SVC sentiment accuracy      | 0.814                                               |

### Tools and Libraries
- **Python**: Jupyter/Colab notebook or script
- **Natural Language Processing**: NLTK
- **Machine Learning**: scikit-learn

### Usage

1. Ensure Python and required packages are installed:
   - `nltk`, `scikit-learn`
2. Download required NLTK data packages (stopwords, punkt, averaged_perceptron_tagger, maxent_ne_chunker, words, movie_reviews, etc.) using `nltk.download()`.
3. Run the script or notebook.
   - The code will demonstrate tokenisation, stemming, POS tagging, chunking, named entity recognition, and sentiment analysis.
4. Review outputs and example accuracy on the test set.

### Future Work
- Extend sentiment analysis to more advanced classifiers (e.g., Logistic Regression, Random Forest).
- Integrate word embeddings (Word2Vec, GloVe) for richer features.
- Expand to named entity recognition in other domains.
- Build a pipeline for document classification on new datasets.



## Project 11: Predicting Board Game Ratings with Machine Learning

### Overview
This project applies exploratory data analysis and machine learning techniques to a dataset of board games to predict their average ratings. The workflow includes data cleansing, feature engineering, visualisation, and the use of linear regression and random forest models for regression tasks.

### Dataset
- **Source:** [Kaggle - Board Games](https://www.kaggle.com/datasets/andrewmvd/board-games)
- **Features:**  
  - Game metadata (id, type, name, year published, player counts, playtime, min age)
  - Community statistics (users rated, average rating, Bayes average rating, number of owners, traders, wanters, wishers, comments, weight ratings)
  - Target variable: `average_rating` (average user rating for the game)
- **Size:** 81,312 rows × 20 columns (before cleaning)

### Implementation Details

**Data Exploration and Visualisation**
- Initial exploration of dataset shape and columns.
- Histogram visualisation of all board game average ratings.
- Inspection of games with zero and nonzero ratings.

**Data Cleansing**
- Removal of games without user reviews (`users_rated` == 0).
- Removal of rows with missing values.
- Re-visualisation of the cleaned ratings distribution.

**Feature Engineering**
- Construction of a correlation heatmap for numerical features.
- Selection of predictive features, with removal of identifiers and target columns.

**Train/Test Split**
- 80% of the data for training, 20% for testing (random sampling).

**Model Training and Evaluation**
- **Linear Regression**
  - Training on selected features.
  - Evaluation with mean squared error (MSE) on the test set.
- **Random Forest Regressor**
  - Training with 100 trees, minimum leaf size of 10, fixed random seed.
  - Evaluation with MSE on the test set (lower MSE than linear regression).

**Example Predictions**
- Both models make predictions for a single test game, compared to its actual rating.

### Example Results

| Model                   | Mean Squared Error |
|-------------------------|-------------------|
| Linear Regression       | 2.08              |
| Random Forest Regressor | 1.45              |

Sample prediction for a test game:
- Linear Regression: 8.12
- Random Forest: 7.91
- Actual: 8.08

### Visualisation
- Histograms of average ratings before and after cleaning.
- Correlation heatmap of numeric features.

### Tools and Libraries
- **Python**: Jupyter/Colab notebook or script
- **Data Manipulation**: pandas
- **Visualisation**: matplotlib, seaborn
- **Machine Learning**: scikit-learn

### Usage

1. Ensure Python and required packages are installed:
   - `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
2. Download the dataset (`games.csv`) from [Kaggle - Board Games](https://www.kaggle.com/datasets/andrewmvd/board-games).
3. Run the script or notebook.
   - The code will load, clean, visualise, and model the data.
4. Review printed mean squared errors, sample predictions, and displayed plots.

### Future Work
- Experiment with additional regression algorithms (XGBoost, Gradient Boosting).
- Incorporate categorical encoding for features such as year or type.
- Perform hyperparameter tuning for the random forest.
- Build a web app for interactive board game rating prediction.



## Project 12: # Credit Card Default Prediction with SMOTE

## Overview
This project demonstrates how to predict credit card defaults using supervised machine learning. The workflow handles class imbalance using Synthetic Minority Over-sampling Technique (SMOTE) and includes data preprocessing, feature engineering, and model comparison across several algorithms.

## Dataset
- **Source:** UCI Credit Card Default Dataset
- **Features:**
  - Demographic information (SEX, EDUCATION, MARRIAGE, AGE)
  - Credit history (LIMIT_BAL)
  - Payment history (PAY_0 to PAY_6)
  - Bill statements (BILL_AMT1 to BILL_AMT6) 
  - Previous payments (PAY_AMT1 to PAY_AMT6)
  - DEFAULT: Target variable (1 = default, 0 = no default)
- **Size:** 30,000 clients

## Implementation Details

### Data Preprocessing
- Statistical analysis and visualisation of features
- Handling of categorical variables through one-hot encoding
- Replacement of special values in EDUCATION and MARRIAGE columns
- Feature scaling with StandardScaler

### Feature Engineering
- Utilisation ratio (bill amount to credit limit)
- Payment to bill ratio
- Delay history (count of months with payment delays)
- Maximum delay indicator
- Average bill and payment amounts
- Bill trend analysis
- Payment consistency metrics

### Class Imbalance Handling
- The positive class (default) represents only 22.12% of the data
- SMOTE applied to balance the training data
- Test data kept in original distribution to maintain realistic evaluation

### Machine Learning Models
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Linear SVM**
- **CatBoost**

### Evaluation Metrics
- Accuracy, precision, recall, and F1-score
- ROC curves and AUC scores
- Cross-validation results for model stability
- Threshold optimisation for precision-recall tradeoff

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 78.16% | 50.56% | 56.65% | 53.43% |
| CatBoost | 77.99% | 50.25% | 51.38% | 50.81% |
| XGBoost | 78.12% | 50.56% | 49.82% | 50.19% |
| Linear SVM | 72.67% | 41.38% | 56.55% | 47.79% |
| Logistic Regression | 71.49% | 39.94% | 57.36% | 47.09% |

- **Best Model:** Random Forest with an F1 score of 53.43%
- **Optimal Threshold:** 0.4867 for improved precision-recall balance

## Tools and Libraries
- **Python**: Jupyter/Colab notebook
- **Data Manipulation**: numpy, pandas
- **Visualisation**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, XGBoost, CatBoost
- **Imbalance Handling**: imbalanced-learn (SMOTE)

## Usage
1. Ensure Python and required packages are installed:
2. Download the UCI Credit Card dataset
3. Run the notebook to:
- Preprocess data and engineer features
- Apply SMOTE to balance training data
- Train and evaluate multiple models
- Visualise results and optimise thresholds

## Future Work
- Further hyperparameter tuning for improved performance
- Cost-sensitive learning approaches
- Additional feature engineering based on domain knowledge
- Ensemble methods combining multiple model predictions
- Deploy as a real-time default prediction service



