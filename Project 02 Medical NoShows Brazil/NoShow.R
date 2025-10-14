# Machine Learning Model for Medical No-Shows
# Author: James Fothergill


# Load required libraries
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(caret)
library(xgboost)
library(randomForest)
library(MASS)
library(pROC)
library(corrplot)
library(skimr)

# Set script current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read dataset
noShow <- read.csv("KaggleV2-May-2016.csv")
message("Dataset loaded with ", nrow(noShow), " rows and ", ncol(noShow), " columns")

# Remove unnecessary ID columns
noShow <- noShow %>% 
  dplyr::select(-PatientId, -AppointmentID)

# Rename columns for UK context
noShow <- noShow %>%
  rename(
    Hypertension = Hipertension,
    Disability = Handcap,
    WelfareBenefits = Scholarship
  )

# Basic dataset exploration
str(noShow)
summary(noShow)
skim(noShow)

# Convert date columns to proper date format
noShow <- noShow %>%
  mutate(
    AppointmentDay = as.Date(AppointmentDay),
    ScheduledDay = as.POSIXct(ScheduledDay)
  )

# Create waiting days variables with more detailed categorisation
noShow <- noShow %>%
  mutate(
    WaitingDays = as.numeric(difftime(AppointmentDay, 
                                      date(ScheduledDay), units = "days")),
    
    # More nuanced waiting time categories
    WaitingDays_cat = factor(
      case_when(
        WaitingDays == 0 ~ "Same day",
        WaitingDays == 1 ~ "Next day",
        WaitingDays <= 3 ~ "2-3 days",
        WaitingDays <= 7 ~ "Within week",
        WaitingDays <= 14 ~ "Within fortnight",
        WaitingDays <= 30 ~ "Within month",
        TRUE ~ "Over month"
      ),
      ordered = TRUE,
      levels = c("Same day", "Next day", "2-3 days", "Within week", 
                 "Within fortnight", "Within month", "Over month")
    )
  )

# Add time-of-day feature
noShow <- noShow %>%
  mutate(
    AppointmentHour = hour(ScheduledDay),
    
    # Create time of day category
    TimeOfDay = factor(
      case_when(
        AppointmentHour < 10 ~ "Morning",
        AppointmentHour < 14 ~ "Midday",
        AppointmentHour < 17 ~ "Afternoon",
        TRUE ~ "Evening"
      ),
      levels = c("Morning", "Midday", "Afternoon", "Evening")
    )
  )

# Add weekday and weekend features
noShow <- noShow %>%
  mutate(
    WeekDay = factor(
      wday(AppointmentDay),
      levels = 1:7,
      labels = c("Sunday", "Monday", "Tuesday", "Wednesday", 
                 "Thursday", "Friday", "Saturday")
    ),
    IsWeekend = WeekDay %in% c("Saturday", "Sunday")
  )

# Remove weekend appointments (as they are closed weekends)
noShow <- noShow %>%
  filter(!IsWeekend)

# Add month and season features
noShow <- noShow %>%
  mutate(
    Month = month(AppointmentDay),
    MonthName = month(AppointmentDay, label = TRUE),
    
    Season = factor(
      case_when(
        Month %in% c(12, 1, 2) ~ "Winter",
        Month %in% c(3, 4, 5) ~ "Spring",
        Month %in% c(6, 7, 8) ~ "Summer",
        TRUE ~ "Autumn"  # British English: Autumn instead of Fall
      ),
      levels = c("Winter", "Spring", "Summer", "Autumn")
    )
  )

# Create the NHS age groups
noShow <- noShow %>%
  mutate(
    AgeGroup = factor(
      cut(Age, 
          breaks = c(-Inf, 15, 24, 44, 64, 84, Inf),
          labels = c("Children", "Young People", "Working Age", 
                     "Middle Age", "Elderly", "Very Elderly")
      ),
      ordered = TRUE
    )
  )

# Calculate and add neighbourhood risk
neighbourhood_rates <- noShow %>%
  group_by(Neighbourhood) %>%
  summarise(
    noshow_rate = mean(No.show == "Yes"),
    appointment_count = n(),
    .groups = 'drop'
  )

noShow <- noShow %>%
  left_join(neighbourhood_rates, by = "Neighbourhood") %>%
  mutate(
    # Create neighbourhood risk categories
    NeighbourhoodRisk = factor(
      case_when(
        noshow_rate <= quantile(neighbourhood_rates$noshow_rate, 0.25) ~ "Very Low Risk",
        noshow_rate <= quantile(neighbourhood_rates$noshow_rate, 0.50) ~ "Low Risk",
        noshow_rate <= quantile(neighbourhood_rates$noshow_rate, 0.75) ~ "Moderate Risk",
        TRUE ~ "High Risk"
      ),
      levels = c("Very Low Risk", "Low Risk", "Moderate Risk", "High Risk"),
      ordered = TRUE
    ),
    
    # Create a volume category for neighbourhoods
    NeighbourhoodVolume = factor(
      case_when(
        appointment_count <= quantile(neighbourhood_rates$appointment_count, 0.25) ~ "Low Volume",
        appointment_count <= quantile(neighbourhood_rates$appointment_count, 0.75) ~ "Medium Volume",
        TRUE ~ "High Volume"
      ),
      levels = c("Low Volume", "Medium Volume", "High Volume")
    )
  ) %>%
  dplyr::select(-noshow_rate, -appointment_count)  # Remove intermediate columns

# Combine health conditions into a single feature
noShow <- noShow %>%
  mutate(
    # Create health conditions variable
    HealthConditions = case_when(
      Diabetes == 1 & Hypertension == 1 & Alcoholism == 1 ~ "Multiple Conditions",
      Diabetes == 1 & Hypertension == 1 ~ "Diabetes & Hypertension",
      Diabetes == 1 & Alcoholism == 1 ~ "Diabetes & Alcoholism",
      Hypertension == 1 & Alcoholism == 1 ~ "Hypertension & Alcoholism",
      Diabetes == 1 ~ "Diabetes",
      Hypertension == 1 ~ "Hypertension",
      Alcoholism == 1 ~ "Alcoholism",
      TRUE ~ "None"
    ),
    
    # Convert to factor with meaningful order
    HealthConditions = factor(
      HealthConditions,
      levels = c("None", "Diabetes", "Hypertension", "Alcoholism", 
                 "Diabetes & Hypertension", "Diabetes & Alcoholism",
                 "Hypertension & Alcoholism", "Multiple Conditions")
    ),
    
    # Create a count of health conditions
    HealthConditionCount = Diabetes + Hypertension + Alcoholism
  )

# Create interaction features between key variables
noShow <- noShow %>%
  mutate(
    # Young people with long waiting times
    YoungWithLongWait = (AgeGroup %in% c("Children", "Young People")) & 
      (WaitingDays_cat %in% c("Within month", "Over month")),
    
    # Elderly with SMS notifications
    ElderlyWithSMS = (AgeGroup %in% c("Elderly", "Very Elderly")) & (SMS_received == 1),
    
    # Health conditions with welfare benefits
    HealthConditionsWithBenefits = (HealthConditionCount > 0) & (WelfareBenefits == 1)
  )

# Remove extreme ages
noShow <- noShow %>%
  filter(Age <= 100)

# Select and prepare relevant features
model_data <- noShow %>%
  # Select features for modelling
  dplyr::select(
    # Target variable
    No.show,
    
    # Demographics
    Gender, AgeGroup,
    
    # Health-related
    HealthConditions, Disability, HealthConditionCount,
    
    # Appointment details
    WaitingDays_cat, WeekDay, TimeOfDay, Season,
    
    # Location
    NeighbourhoodRisk, NeighbourhoodVolume,
    
    # Notifications
    SMS_received,
    
    # Social factors
    WelfareBenefits,
    
    # Interaction terms
    YoungWithLongWait, ElderlyWithSMS, HealthConditionsWithBenefits
  ) %>%
  # Convert all character columns to factors
  mutate(
    across(where(is.character), as.factor),
    
    # Convert ordered factors to regular factors
    across(where(is.ordered), ~factor(., ordered = FALSE)),
    
    # Ensure target variable is properly formatted
    No.show = factor(No.show, levels = c("No", "Yes"))
  )

# Check for missing values
missing_values <- colSums(is.na(model_data))
if (sum(missing_values) > 0) {
  message("Warning: Missing values detected in the following columns:")
  print(missing_values[missing_values > 0])
  
  # Define a mode function
  mode <- function(x) {
    ux <- unique(na.omit(x))
    ux[which.max(tabulate(match(x, ux)))]
  }
  
  # Impute missing values
  message("Imputing missing values...")
  model_data <- model_data %>%
    mutate(across(everything(), ~ifelse(is.na(.), mode(.), .)))
}

# Check class balance
class_distribution <- table(model_data$No.show)
print("Class distribution:")
print(class_distribution)
print(prop.table(class_distribution))

# Split data first to avoid data leakage
set.seed(42)  # Set seed for reproducibility
train_index <- createDataPartition(model_data$No.show, p = 0.75, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

message("Original class distribution in training data:")
print(table(train_data$No.show))

# Implement random upsampling for class balance
# Find out which is the minority class
minority_class <- names(which.min(table(train_data$No.show)))
majority_class <- names(which.max(table(train_data$No.show)))

# Separate minority and majority classes
minority_samples <- train_data[train_data$No.show == minority_class, ]
majority_samples <- train_data[train_data$No.show == majority_class, ]

# Calculate how many samples to upsample
n_majority <- nrow(majority_samples)
n_minority <- nrow(minority_samples)

# Random upsampling of minority class with replacement
set.seed(42)
upsampled_minority <- minority_samples[sample(1:n_minority, n_majority, replace = TRUE), ]

# Combine upsampled minority class with majority class
balanced_train_data <- rbind(majority_samples, upsampled_minority)

# Shuffle the data
balanced_train_data <- balanced_train_data[sample(1:nrow(balanced_train_data)), ]

message("Balanced class distribution after random upsampling:")
print(table(balanced_train_data$No.show))

# Define resampling methods in training control
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

message("Training models with balanced data. This may take some time.")

# Train XGBoost model on balanced data
xgb_grid <- expand.grid(
  nrounds = c(100),
  max_depth = c(3, 6),
  eta = c(0.1),
  gamma = 0,
  colsample_bytree = c(0.8),
  min_child_weight = c(1),
  subsample = 0.8
)

xgb_model <- train(
  No.show ~ .,
  data = balanced_train_data,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  metric = "ROC",
  verbose = FALSE
)

message("XGBoost training complete.")

# Train Random Forest model on balanced data
rf_model <- train(
  No.show ~ .,
  data = balanced_train_data,
  method = "rf",
  trControl = train_control,
  metric = "ROC",
  ntree = 100
)

message("Random Forest training complete.")

# Train a Logistic Regression model on balanced data
glm_model <- train(
  No.show ~ .,
  data = balanced_train_data,
  method = "glm",
  trControl = train_control,
  metric = "ROC",
  family = "binomial"
)

message("Logistic Regression training complete.")

# Generate predictions on test data
xgb_preds <- predict(xgb_model, test_data, type = "prob")[,"Yes"]
rf_preds <- predict(rf_model, test_data, type = "prob")[,"Yes"]
glm_preds <- predict(glm_model, test_data, type = "prob")[,"Yes"]

# Simple weighted average ensemble
ensemble_preds <- (xgb_preds * 0.4) + (rf_preds * 0.4) + (glm_preds * 0.2)

# Calculate ROC curves for all models
xgb_roc <- roc(test_data$No.show == "Yes", xgb_preds)
rf_roc <- roc(test_data$No.show == "Yes", rf_preds)
glm_roc <- roc(test_data$No.show == "Yes", glm_preds)
ensemble_roc <- roc(test_data$No.show == "Yes", ensemble_preds)

# Print AUC values
message("Model Performance (AUC):")
print(paste("XGBoost AUC:", round(xgb_roc$auc, 4)))
print(paste("Random Forest AUC:", round(rf_roc$auc, 4)))
print(paste("Logistic Regression AUC:", round(glm_roc$auc, 4)))
print(paste("Ensemble AUC:", round(ensemble_roc$auc, 4)))

# Plot ROC curves
plot(xgb_roc, col = "blue", main = "ROC Curve Comparison")
lines(rf_roc, col = "red")
lines(glm_roc, col = "purple")
lines(ensemble_roc, col = "black", lwd = 2)
legend("bottomright", 
       legend = c(paste("XGBoost:", round(xgb_roc$auc, 3)),
                  paste("Random Forest:", round(rf_roc$auc, 3)),
                  paste("Logistic Regression:", round(glm_roc$auc, 3)),
                  paste("Ensemble:", round(ensemble_roc$auc, 3))),
       col = c("blue", "red", "purple", "black"),
       lwd = c(1, 1, 1, 2))

# Find optimal threshold based on F1 score
thresholds <- seq(0.1, 0.9, by = 0.05)
results <- data.frame(
  threshold = thresholds,
  accuracy = numeric(length(thresholds)),
  precision = numeric(length(thresholds)),
  recall = numeric(length(thresholds)),
  f1_score = numeric(length(thresholds))
)

# Calculate metrics at each threshold
for (i in 1:length(thresholds)) {
  threshold <- thresholds[i]
  pred_class <- ifelse(ensemble_preds > threshold, "Yes", "No")
  pred_factor <- factor(pred_class, levels = levels(test_data$No.show))
  
  # Create confusion matrix
  cm <- confusionMatrix(pred_factor, test_data$No.show, positive = "Yes")
  
  # Store metrics
  results$accuracy[i] <- cm$overall["Accuracy"]
  results$precision[i] <- cm$byClass["Pos Pred Value"]
  results$recall[i] <- cm$byClass["Sensitivity"]
  results$f1_score[i] <- 2 * (results$precision[i] * results$recall[i]) / 
    (results$precision[i] + results$recall[i])
}

# Find optimal threshold based on F1 score
optimal_threshold <- results$threshold[which.max(results$f1_score)]
message("Optimal threshold based on F1 score: ", optimal_threshold)

# Create final predictions using optimal threshold
final_predictions <- ifelse(ensemble_preds > optimal_threshold, "Yes", "No")
final_predictions <- factor(final_predictions, levels = levels(test_data$No.show))

# Final confusion matrix with optimal threshold
final_cm <- confusionMatrix(final_predictions, test_data$No.show, positive = "Yes")
print("Final model performance with optimal threshold:")
print(final_cm)

# Extract feature importance from XGBoost
xgb_importance <- varImp(xgb_model)

# Plot feature importance
plot(xgb_importance, top = 20, main = "XGBoost Feature Importance")

# Save feature importance to CSV
xgb_importance_df <- data.frame(
  Feature = rownames(xgb_importance$importance),
  Importance = xgb_importance$importance$Overall
)
xgb_importance_df <- xgb_importance_df[order(-xgb_importance_df$Importance),]
write.csv(xgb_importance_df, "feature_importance.csv", row.names = FALSE)

# Save the trained models
saveRDS(xgb_model, "xgb_model.rds")
saveRDS(rf_model, "rf_model.rds")
saveRDS(glm_model, "glm_model.rds")

# Save the optimal threshold
write.csv(data.frame(optimal_threshold = optimal_threshold), 
          "threshold_results.csv", row.names = FALSE)

message("All models saved successfully.")

# function that can be used to predict on new data
predict_noshow <- function(new_data) {
  # Ensure new_data has the same structure as training data
  required_cols <- setdiff(names(model_data), "No.show")
  missing_cols <- setdiff(required_cols, names(new_data))
  
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }
  
  # Generate predictions from each model
  xgb_pred <- predict(xgb_model, new_data, type = "prob")[,"Yes"]
  rf_pred <- predict(rf_model, new_data, type = "prob")[,"Yes"]
  glm_pred <- predict(glm_model, new_data, type = "prob")[,"Yes"]
  
  # Create ensemble prediction (weighted average)
  ensemble_pred <- (xgb_pred * 0.4) + (rf_pred * 0.4) + (glm_pred * 0.2)
  
  # Apply optimal threshold
  result <- data.frame(
    probability = ensemble_pred,
    prediction = ifelse(ensemble_pred > optimal_threshold, "Yes", "No")
  )
  
  return(result)
}

message("Prediction function created. You can now use predict_noshow() with new data.")
message("Analysis complete!")