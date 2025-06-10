# Aviation Crash Cause Classification
# A comprehensive analysis of aviation crashes from 1908 onwards
# James Fothergill

# LIBRARIES
library(dplyr)       # For data manipulation
library(readr)       # For reading data files efficiently
library(zoo)         # For time series functions and NA handling
library(lubridate)   # For date and time handling
library(tidytext)    # For text mining and analysis
library(ggplot2)     # For creating visualisations
library(caret)       # For machine learning functions
library(randomForest) # For Random Forest algorithm
library(xgboost)     # For gradient boosting algorithm
library(e1071)       # For Naive Bayes algorithm
library(glmnet)      # For regularised regression
library(tidyverse)   # Collection of data science packages
library(missForest)  # For imputation of missing values
library(reshape2)    # For reshaping data

# DATA IMPORT
# Import the dataset of airplane crashes since 1908
aero <- read.csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")

# Examine the data structure to understand available variables
str(aero)

# 1. DATA CLEANING
# Calculate proportion of missing values in each column
misscol <- colSums(is.na(aero)) / nrow(aero)
print(round(misscol, 3))

# Replace NA values with column means for numerical variables
# and round to whole numbers as fractional people aren't sensible
aero$Fatalities <- na.aggregate(aero$Fatalities)
aero$Fatalities <- round(aero$Fatalities)

aero$Ground <- na.aggregate(aero$Ground)
aero$Ground <- round(aero$Ground)

aero$Aboard <- na.aggregate(aero$Aboard)
aero$Aboard <- round(aero$Aboard)

# Create new features to enhance the dataset
# Total fatalities combines both on-board and ground casualties
aero$Total_Fatalities <- aero$Fatalities + aero$Ground

# Create a categorical severity measure based on total fatalities
aero$Crash_Severity <- ifelse(aero$Total_Fatalities == 0, "Low",
                              ifelse(aero$Total_Fatalities < 50, "Medium", "High"))

# Convert Crash_Severity to factor for categorical analysis
aero$Crash_Severity <- as.factor(aero$Crash_Severity)

# Clean text data by replacing empty values with NA for proper handling
aero$Location <- ifelse(aero$Location == "", NA, aero$Location)
aero$Operator <- ifelse(aero$Operator == "", NA, aero$Operator)
aero$Flight.. <- ifelse(aero$Flight.. == "", NA, aero$Flight..)
aero$Flight.. <- ifelse(aero$Flight.. == "-", NA, aero$Flight..)
aero$Route <- ifelse(aero$Route == "", NA, aero$Route)
aero$Type <- ifelse(aero$Type == "", NA, aero$Type)
aero$Registration <- ifelse(aero$Registration == "", NA, aero$Registration)
aero$cn.In <- ifelse(aero$cn.In == "", NA, aero$cn.In)
aero$Summary <- ifelse(aero$Summary == "", NA, aero$Summary)

# Remove rows with NA in Summary as it's essential for our crash cause classification
aero <- aero[!is.na(aero$Summary), ]

# Convert Date to proper date format for temporal analysis
aero$Date <- as.Date(aero$Date, format = "%m/%d/%Y")

# Extract Year for chronological analysis
aero$Year <- as.numeric(format(aero$Date, "%Y"))

# Replace NA in categorical columns with "Unavailable" for analytical completeness
aero <- aero %>%
  mutate(Location = replace(Location, is.na(Location), "Unavailable")) %>%
  mutate(Type = replace(Type, is.na(Type), "Unavailable")) %>%
  mutate(Operator = replace(Operator, is.na(Operator), "Unavailable"))

# Calculate survivors by subtracting fatalities from total aboard
aero$Survived <- aero$Aboard - aero$Fatalities

# Clean Time data which has various inconsistencies
aero$Time[aero$Time == ""] <- NA 
aero$Time <- ifelse(aero$Time == "-", NA, aero$Time)

# Fix Time format issues with regular expressions
aero$Time <- gsub("^c", "", aero$Time)  # Remove "c:" prefix (circa time indicator)
aero$Time <- gsub("'", ":", aero$Time)  # Replace apostrophe with colon
aero$Time <- gsub("^:", "", aero$Time)  # Remove leading colon
aero$Time <- trimws(aero$Time)          # Remove leading/trailing whitespace
aero$Time <- sprintf("%02s", aero$Time) # Add leading zero for consistency

# Fix specific Time entries with known formatting issues
time_corrections <- list(
  "2:00" = "02:00",
  "1:00" = "01:00",
  "1:30" = "01:30",
  "18.40" = "18:40", 
  "114:20" = "14:20",
  "0943" = "09:43",
  "2:40" = "02:40",
  "9:40" = "09:40",
  "8:02" = "08:02",
  "9:30" = "09:30"
)

# Apply time corrections to standardise format
for(wrong_time in names(time_corrections)) {
  aero$Time[aero$Time == wrong_time] <- time_corrections[[wrong_time]]
}

# Convert time to proper time format and extract hour
aero$Time <- as.POSIXct(aero$Time, format = "%H:%M")
aero$Hour <- hour(aero$Time)

# Create TimeOfDay categories for temporal pattern analysis
aero$TimeOfDay <- cut(
  aero$Hour,
  breaks = c(0, 6, 12, 18, 24),
  labels = c("Night", "Morning", "Afternoon", "Evening"),
  include.lowest = TRUE
)

# Convert TimeOfDay to factor and create numeric encoding
aero$TimeOfDay <- as.factor(aero$TimeOfDay)
aero$Time_Of_Day <- as.integer(aero$TimeOfDay)

# Drop intermediate Time and Hour columns as they're no longer needed
aero <- aero %>%
  select(-Time, -Hour)

# 2. FEATURE ENGINEERING WITH TEXT DATA
# Create function to determine cause of crash from summary text
# Using keyword identification approach
get_cause_of_crash <- function(tokens) {
  tokens <- gsub("[\\.,]", "", tokens)
  tokens <- tolower(tokens)
  if ("fog" %in% tokens || "storm" %in% tokens || "weather" %in% tokens ||
      "poor" %in% tokens || "rain" %in% tokens || "icing" %in% tokens ||
      "thunderstorm" %in% tokens || "ice" %in% tokens || "thunderstorms" %in% tokens ||
      "storm" %in% tokens || "lightning" %in% tokens || "snow" %in% tokens ||
      "rainstorm" %in% tokens || "gust" %in% tokens || "overcast" %in% tokens ||
      "winds" %in% tokens || "snowstorm" %in% tokens || "winds" %in% tokens ||
      "volcano" %in% tokens || " hurricane" %in% tokens || "tornado" %in% tokens ||
      "meteorological" %in% tokens){
    return("Bad_Weather_Natural_Disaster")
  } else if ("pilot" %in% tokens || "crew" %in% tokens || "pilot's" %in% tokens ||
             "captain" %in% tokens || "pilots" %in% tokens || "decision" %in% tokens ||
             "fatigue" %in% tokens || "procedure" %in% tokens || "training" %in% tokens ||
             "operation" %in% tokens || "captain's" %in% tokens || "misjudged" %in% tokens ||
             "planning" %in% tokens || "officer" %in% tokens || "judgement" %in% tokens ||
             "experience" %in% tokens || "instructions" %in% tokens || "judgment" %in% tokens ||
             "management" %in% tokens || "crews" %in% tokens || "awareness" %in% tokens ||
             "flightcrew" %in% tokens || "flightcrew's" %in% tokens || "preparation" %in% tokens ||
             "engineer" %in% tokens || "exhaustion" %in% tokens || "operator" %in% tokens ||
             "alcohol" %in% tokens || "incapacitation" %in% tokens || "officer's" %in% tokens) {
    return("Human_Error")
  } else if ("war" %in% tokens || "terrorism" %in% tokens || "political" %in% tokens || 
             "shot" %in% tokens || "hijackers" %in% tokens || "suicide" %in% tokens || 
             "hijacker" %in% tokens || "enemy" %in% tokens || "fighters" %in% tokens || 
             "hijacked" %in% tokens || "bomb"  %in% tokens || "bombs" %in% tokens ||
             "mission" %in% tokens || "explosive" %in% tokens || "detonation" %in% tokens || 
             "missile" %in% tokens || "fighter" %in% tokens || "military" %in% tokens || 
             "shelling" %in% tokens) {
    return("Act_of_War_Terrorism")
  } else if ("mechanical" %in% tokens || "failure" %in% tokens || "engine" %in% tokens || 
             "wing" %in% tokens || "power" %in% tokens || "engines" %in% tokens || 
             "instrument" %in% tokens || "nose" %in% tokens || "gear" %in% tokens || 
             "system" %in% tokens || "propeller"  %in% tokens || "burst" %in% tokens ||
             "aircraft's" %in% tokens || "cockpit"  %in% tokens || "radar" %in% tokens || 
             "fuselage" %in% tokens || "flaps" %in% tokens || "navigational" %in% tokens || 
             "rudder"  %in% tokens || "jet" %in% tokens || "separation" %in% tokens || 
             "controls" %in% tokens || "instruments" %in% tokens || "structural" %in% tokens ||
             "wings" %in% tokens || "tank" %in% tokens || "design" %in% tokens || 
             "door" %in% tokens || "electrical" %in% tokens || "smoke" %in% tokens || 
             "rear" %in% tokens || "malfunction" %in% tokens || "stalled" %in% tokens || 
             "stall" %in% tokens || "pressure" %in% tokens || "stabilizer" %in% tokens|| 
             "equipment" %in% tokens || "uncontrollable" %in% tokens || 
             "altimeter" %in% tokens || "blade" %in% tokens || "feathered" %in% tokens || 
             "uncontrolled" %in% tokens || "faulty" %in% tokens || "deteriorating" %in% tokens || 
             "device" %in% tokens || "starboard" %in% tokens || "mechan" %in% tokens || 
             "autopilot" %in% tokens || "malfunctioning" %in% tokens || "airplane's" %in% tokens ||
             "plane's" %in% tokens || "errors" %in% tokens || "contamination" %in% tokens || 
             "tanks"%in% tokens || "developed" %in% tokens || "fracture" %in% tokens || 
             "inoperative" %in% tokens || "valve" %in% tokens || "cylinder"  %in% tokens || 
             "feather" %in% tokens || "breaking" %in% tokens || "damaged" %in% tokens ||
             "rotation"  %in% tokens || "wheel" %in% tokens || "drifted" %in% tokens || 
             "ignition" %in% tokens || "turbine" %in% tokens || "crack" %in% tokens || 
             "propellers" %in% tokens || "blades" %in% tokens || "bolts" %in% tokens || 
             "corrosion" %in% tokens || "cracks" %in% tokens || "decompression" %in% tokens || 
             "metal" %in% tokens || "rod" %in% tokens || "ruptured" %in% tokens ||
             "wires" %in% tokens || "broken" %in% tokens || "leak" %in% tokens || 
             "throttle" %in% tokens || "bolt" %in% tokens || "compressor" %in% tokens || 
             "gas" %in% tokens || "generator" %in% tokens || "loose" %in% tokens || 
             "pump" %in% tokens || "switch" %in% tokens || "brake" %in% tokens || 
             "brakes" %in% tokens || "cables" %in% tokens || "handle" %in% tokens)  {
    return("Mechanical_Failure")
  } else {
    return("Other")
  }
}

# Apply function to classify crash causes based on summary text
aero$Cause_Of_Crash <- sapply(aero$Summary, function(summary) {
  tokens <- strsplit(summary, "\\s+")[[1]]  # Split summary into tokens
  get_cause_of_crash(tokens)
})

# Create flight phase classification from summary text
get_Phase_of_Flight <- function(tokens) {
  tokens <- gsub("[\\.,]", "", tokens)
  tokens <- tolower(tokens)
  if("takeoff" %in% tokens || "taking" %in% tokens){
    return("Takeoff")
  } else if("landing" %in% tokens || "land" %in% tokens || 
            "landed" %in% tokens || "descending" %in% tokens) {
    return("Landing")
  } else {
    return("In_Flight")
  }
}

# Apply flight phase classification
aero$Phase_of_Flight <- sapply(aero$Summary, function(summary) {
  tokens <- strsplit(summary, "\\s+")[[1]]  # Split summary into tokens
  get_Phase_of_Flight(tokens)
})

# Convert target variable to factor for classification modelling
aero$Cause_Of_Crash <- as.factor(aero$Cause_Of_Crash)

# 3. FEATURE SELECTION USING TEXT MINING
# Extract text features using TF-IDF (Term Frequency-Inverse Document Frequency)

# Create a corpus from summary text - CORRECTED VERSION
summary_words <- aero %>%
  # Add row number as ID column for proper joining later
  mutate(id = row_number()) %>%
  select(id, Summary) %>%
  unnest_tokens(word, Summary) %>%
  anti_join(stop_words, by = "word") %>%  # Remove common stop words
  filter(nchar(word) > 2) %>%             # Filter out very short words
  count(id, word) %>%                     # Count word occurrences
  bind_tf_idf(word, id, n) %>%            # Calculate TF-IDF
  arrange(desc(tf_idf))                   # Sort by highest TF-IDF score

# Get top words by TF-IDF for feature selection
# Use fewer words (50 instead of 100) to reduce dimensionality
top_tfidf_words <- summary_words %>%
  group_by(word) %>%
  summarize(mean_tf_idf = mean(tf_idf)) %>%
  top_n(50, mean_tf_idf) %>%              # Select top 50 most important words
  pull(word)

# Create document-term matrix for top words to use as features
dtm <- summary_words %>%
  filter(word %in% top_tfidf_words) %>%
  select(id, word, tf_idf) %>%
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = list(tf_idf = 0))

# Sort document-term matrix by ID for proper alignment
dtm <- dtm[order(dtm$id), ]

# Join with original data frame using the ID column we created
# First add the ID to the aero dataframe for joining
aero$id <- row_number(aero)
aero <- left_join(aero, dtm, by = "id")

# Replace NA values in TF-IDF columns with 0
tfidf_cols <- setdiff(names(dtm), "id")  # All columns except ID
aero[, tfidf_cols] <- lapply(aero[, tfidf_cols], function(x) ifelse(is.na(x), 0, x))

# Create decade features for historical pattern analysis
aero$Decade <- floor(aero$Year / 10) * 10
aero$Decade <- as.factor(aero$Decade)

# Create operator category (major airlines vs. others)
# This reduces dimensionality by grouping less frequent operators
top_operators <- aero %>%
  count(Operator) %>%
  top_n(20, n) %>%
  pull(Operator)

aero$Operator_Category <- ifelse(aero$Operator %in% top_operators, 
                                 aero$Operator, "Other")
aero$Operator_Category <- as.factor(aero$Operator_Category)

# Create Aircraft Type Category
# Grouping less common aircraft types to reduce dimensionality
top_types <- aero %>%
  count(Type) %>%
  top_n(20, n) %>%
  pull(Type)

aero$Type_Category <- ifelse(aero$Type %in% top_types, aero$Type, "Other")
aero$Type_Category <- as.factor(aero$Type_Category)

# Convert Phase_of_Flight to factor for use in modelling
aero$Phase_of_Flight <- as.factor(aero$Phase_of_Flight)

# 4. EXPLORATORY DATA ANALYSIS FOR ML
# Analyse relationships between features and target
# This visualisation shows how crash causes vary by flight phase
ggplot(aero, aes(x = Cause_Of_Crash, fill = Phase_of_Flight)) +
  geom_bar(position = "dodge") +
  labs(title = "Crash Causes by Phase of Flight",
       x = "Cause of Crash", 
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Time trend of crash causes to identify historical patterns
aero %>%
  group_by(Decade, Cause_Of_Crash) %>%
  summarize(Count = n(), .groups = "drop") %>%
  ggplot(aes(x = Decade, y = Count, fill = Cause_Of_Crash)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Crash Causes by Decade",
       x = "Decade", 
       y = "Number of Crashes") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Severity by crash cause to explore impact patterns
ggplot(aero, aes(x = Cause_Of_Crash, fill = Crash_Severity)) +
  geom_bar(position = "fill") +
  labs(title = "Crash Severity by Cause",
       x = "Cause of Crash", 
       y = "Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 5. FEATURE SELECTION AND PREPARATION FOR ML
# Select features for modelling based on relevance and analysis
model_features <- aero %>%
  select(
    # Numeric features
    Year, Aboard, Fatalities, Ground, Total_Fatalities, Survived, Time_Of_Day,
    
    # Factor features 
    Crash_Severity, Phase_of_Flight, Decade,
    
    # Text-derived features
    all_of(tfidf_cols),
    
    # Target variable
    Cause_Of_Crash
  )

# Handle missing values in selected features for clean modelling
model_features <- na.omit(model_features)

# 6. MACHINE LEARNING MODEL TRAINING
# Convert factors to appropriate format for ML algorithms
model_features$Crash_Severity <- as.factor(model_features$Crash_Severity)
model_features$Phase_of_Flight <- as.factor(model_features$Phase_of_Flight)
model_features$Decade <- as.factor(model_features$Decade)

# Split data into training and testing sets (80/20 split)
set.seed(42)  # For reproducibility
trainIndex <- createDataPartition(model_features$Cause_Of_Crash, p = 0.8, list = FALSE)
train_data <- model_features[trainIndex, ]
test_data <- model_features[-trainIndex, ]

# Define predictors and target variables
predictors <- setdiff(names(train_data), "Cause_Of_Crash")
target <- "Cause_Of_Crash"

# Define cross-validation strategy for robust model evaluation
ctrl <- trainControl(
  method = "cv",               # Cross-validation method
  number = 5,                  # Number of folds
  verboseIter = TRUE,          # Print progress
  classProbs = TRUE,           # Calculate class probabilities
  savePredictions = "final"    # Save predictions for analysis
)

# MODEL 1: RANDOM FOREST
# Random Forests are excellent for classification with mixed data types
# Define parameter grid for tuning
rf_grid <- expand.grid(mtry = c(5, 10, 15))  # Number of variables at each split

# Train Random Forest model with cross-validation
set.seed(42)
rf_model <- train(
  x = train_data[, predictors],
  y = train_data[[target]],
  method = "rf",
  tuneGrid = rf_grid,
  trControl = ctrl,
  importance = TRUE  # Calculate variable importance
)

# Display model results
print(rf_model)
plot(rf_model)

# Examine feature importance
varImp(rf_model)

# Evaluate Random Forest on test set
rf_pred <- predict(rf_model, test_data[, predictors])
rf_cm <- confusionMatrix(rf_pred, test_data[[target]])
print(rf_cm)

# MODEL 2: XGBoost
# XGBoost often provides excellent performance through gradient boosting
# Define parameter grid for tuning
xgb_grid <- expand.grid(
  nrounds = c(50, 100),           # Number of boosting rounds
  max_depth = c(3, 6),            # Maximum tree depth
  eta = c(0.1, 0.3),              # Learning rate
  gamma = 0,                      # Minimum loss reduction
  colsample_bytree = c(0.75, 1),  # Subsample ratio of columns
  min_child_weight = 1,           # Minimum sum of instance weight
  subsample = 1                   # Subsample ratio of training instances
)

# Convert factor variables to dummy variables for XGBoost
# XGBoost requires numerical input
dummy_model <- dummyVars(" ~ .", data = train_data[, c("Crash_Severity", "Phase_of_Flight", "Decade")])
train_dummies <- predict(dummy_model, train_data)
test_dummies <- predict(dummy_model, test_data)

# Replace original factor columns with dummy variables
train_xgb <- train_data
train_xgb <- train_xgb[, !names(train_xgb) %in% c("Crash_Severity", "Phase_of_Flight", "Decade")]
train_xgb <- cbind(train_xgb, train_dummies)

test_xgb <- test_data
test_xgb <- test_xgb[, !names(test_xgb) %in% c("Crash_Severity", "Phase_of_Flight", "Decade")]
test_xgb <- cbind(test_xgb, test_dummies)

# Update predictors list for XGBoost model
xgb_predictors <- setdiff(names(train_xgb), "Cause_Of_Crash")

# Train XGBoost model with cross-validation
set.seed(42)
xgb_model <- train(
  x = train_xgb[, xgb_predictors],
  y = train_xgb[[target]],
  method = "xgbTree",
  tuneGrid = xgb_grid,
  trControl = ctrl,
  verbose = FALSE
)

# Display model results
print(xgb_model)
plot(xgb_model)

# Evaluate XGBoost on test set
xgb_pred <- predict(xgb_model, test_xgb[, xgb_predictors])
xgb_cm <- confusionMatrix(xgb_pred, test_xgb[[target]])
print(xgb_cm)

# MODEL 3: NAIVE BAYES
# Naive Bayes works well with text classification problems

# Remove zero variance predictors as they provide no discriminatory power
nzv <- nearZeroVar(train_data[, predictors])
if(length(nzv) > 0) {
  predictors_nb <- predictors[-nzv]
} else {
  predictors_nb <- predictors
}

# Train a Naive Bayes model
set.seed(42)
nb_model <- naiveBayes(
  x = train_data[, predictors_nb],
  y = train_data[[target]]
)

# Evaluate on test set
nb_pred <- predict(nb_model, test_data[, predictors_nb])
nb_cm <- confusionMatrix(nb_pred, test_data[[target]])
print(nb_cm)

# MODEL COMPARISON
# Compare performance across all models
model_summary <- data.frame(
  Model = c("Random Forest", "XGBoost", "Naive Bayes"),
  Accuracy = c(
    rf_cm$overall["Accuracy"],
    xgb_cm$overall["Accuracy"],
    nb_cm$overall["Accuracy"]
  ),
  Kappa = c(
    rf_cm$overall["Kappa"],
    xgb_cm$overall["Kappa"],
    nb_cm$overall["Kappa"]
  )
)

print(model_summary)

# Visualize model comparison
ggplot(model_summary, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  ylim(0, 1) +
  labs(title = "Model Accuracy Comparison",
       x = "Model", 
       y = "Accuracy") +
  theme_minimal()

# Select the best model based on performance
best_model_name <- model_summary$Model[which.max(model_summary$Accuracy)]
cat("Best model based on accuracy:", best_model_name, "\n")

# Assign best model
if (best_model_name == "Random Forest") {
  best_model <- rf_model
  best_pred <- rf_pred
  best_cm <- rf_cm
} else if (best_model_name == "XGBoost") {
  best_model <- xgb_model
  best_pred <- xgb_pred
  best_cm <- xgb_cm
} else {
  best_model <- nb_model
  best_pred <- nb_pred
  best_cm <- nb_cm
}

# 7. MODEL INTERPRETATION
# Plot feature importance for the Random Forest model
# This helps identify which variables are most predictive
if (best_model_name == "Random Forest") {
  # Extract feature importance - handling different possible structures
  imp_data <- varImp(rf_model)
  
  # Check the structure and extract appropriately
  if ("importance" %in% names(imp_data)) {
    # Standard structure with 'importance' attribute
    imp_df <- as.data.frame(imp_data$importance)
    imp_df$Feature <- rownames(imp_df)
  } else {
    # Alternative structure
    imp_df <- as.data.frame(imp_data) 
    if ("Overall" %in% colnames(imp_df)) {
      # Already has Overall column
      imp_df$Feature <- rownames(imp_df)
    } else if (ncol(imp_df) > 1) {
      # Multi-class importance (one column per class)
      # Create an overall importance by averaging across classes
      imp_df$Overall <- rowMeans(imp_df, na.rm = TRUE)
      imp_df$Feature <- rownames(imp_df)
    } else {
      # Single column but not named "Overall"
      colnames(imp_df)[1] <- "Overall"
      imp_df$Feature <- rownames(imp_df)
    }
  }
  
  # Now we can proceed with sorting and plotting
  if ("Overall" %in% colnames(imp_df)) {
    top_features <- imp_df %>%
      dplyr::select(Feature, Overall) %>%
      arrange(desc(Overall)) %>%
      head(20)  # Top 20 features
    
    # Visualise top features
    ggplot(top_features, aes(x = reorder(Feature, Overall), y = Overall)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      coord_flip() +
      labs(title = "Top 20 Features for Crash Cause Classification",
           x = "Feature", 
           y = "Importance") +
      theme_minimal()
  } else {
    cat("Warning: Could not identify 'Overall' importance column for plotting.\n")
  }
}

# Calculate class-specific metrics for deeper model evaluation
class_metrics <- data.frame(
  Class = levels(test_data[[target]]),
  Precision = best_cm$byClass[,"Precision"],
  Recall = best_cm$byClass[,"Sensitivity"],
  F1_Score = best_cm$byClass[,"F1"]
)

print(class_metrics)

# Visualise confusion matrix as a heatmap
cm_heatmap <- melt(best_cm$table)
names(cm_heatmap) <- c("Reference", "Prediction", "Count")

ggplot(cm_heatmap, aes(x = Reference, y = Prediction, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count)) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix Heatmap") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 8. SAVE THE MODEL FOR DEPLOYMENT
saveRDS(best_model, "crash_cause_classification_model.rds")

# 10. FINAL VISUALISATION AND INSIGHTS
# Create a chart showing changes in crash causes over time
# This helps identify historical trends
cause_by_decade <- aero %>%
  group_by(Decade, Cause_Of_Crash) %>%
  summarise(Count = n(), .groups = "drop") %>%
  group_by(Decade) %>%
  mutate(Percentage = Count / sum(Count) * 100)

ggplot(cause_by_decade, aes(x = Decade, y = Percentage, fill = Cause_Of_Crash)) +
  geom_bar(stat = "identity") +
  labs(title = "Evolution of Crash Causes Over Time",
       x = "Decade", 
       y = "Percentage") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Visualize cause distribution by severity
ggplot(aero, aes(x = Cause_Of_Crash, fill = Crash_Severity)) +
  geom_bar(position = "fill") +
  labs(title = "Crash Severity Distribution by Cause",
       x = "Cause of Crash", 
       y = "Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Create dynamic range visualization of fatality rates by cause
aero %>%
  group_by(Cause_Of_Crash) %>%
  summarise(
    Mean_Fatalities = mean(Fatalities, na.rm = TRUE),
    Median_Fatalities = median(Fatalities, na.rm = TRUE),
    Max_Fatalities = max(Fatalities, na.rm = TRUE),
    Min_Fatalities = min(Fatalities, na.rm = TRUE)
  ) %>%
  ggplot(aes(x = Cause_Of_Crash, y = Mean_Fatalities)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(aes(ymin = Min_Fatalities, ymax = Max_Fatalities), width = 0.2) +
  labs(title = "Fatality Statistics by Crash Cause",
       x = "Cause of Crash", 
       y = "Mean Fatalities (with range)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Summary of insights

cat("Top 3 Crash Causes:\n")
cause_counts <- aero %>% 
  count(Cause_Of_Crash) %>% 
  arrange(desc(n))
print(cause_counts)

cat("\nModel Performance Summary:\n")
print(model_summary)

cat("\nClassification Report for Best Model:\n")
print(class_metrics)

cat("\nAnalysis complete. Model saved as 'crash_cause_classification_model.rds'\n")