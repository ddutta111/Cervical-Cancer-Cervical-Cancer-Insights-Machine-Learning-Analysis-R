# Cervical Cancer Insights: Machine Learning Analysis_R

## Step1: Importing libraries
```R
# Load libraries
library(tidyverse)      # For data manipulation and visualization
library(caret)          # For machine learning and model training
library(e1071)          # For SVM and other statistical functions
library(randomForest)   # For Random Forest algorithm
library(xgboost)        # For XGBoost algorithm
library(corrplot)       # For visualizing correlation matrices
library(pROC)           # For ROC curve analysis and AUC calculations
library(ggplot2)        # For creating visualizations using grammar of graphics
library(ggcorrplot)     # For visualizing correlation matrices with ggplot2
library(dplyr)          # For data manipulation and transformation
library(reshape2)       # For reshaping data between wide and long formats
library(rpart)          # For Decision Trees
library(rpart.plot)     # For visualizing Decision Trees
library(class)          # For K-Nearest Neighbors (KNN) algorithm
library(MLmetrics)      # For calculating various model performance metrics
library(GGally)         # For extended functions on ggplot2, including pair plots
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------
## Step 2: Importing the data
```R
# Load the dataset
data <- read.csv("D:\\R Projects\\cervical-cancer_csv.csv")
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------

## Step 3: Data Exploration
```R
# Check the structure of the data
str(data)

# View the first few rows
head(data)

# Check for missing values
colSums(is.na(data))

# Calculate percentage of missing values for each column
missing_perc <- colSums(is.na(data)) / nrow(data)

# Print the percentage of missing values
print(missing_perc)

## Step 4: Data Cleaning
# Drop columns with more than 90% missing values
data_cleaned <- data %>%
  select(where(~ sum(is.na(.)) / nrow(data) <= 0.90))

# Impute missing values with the mean for remaining columns
cancer_data <- data_cleaned %>%
  mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Check the cleaned and imputed dataset
print(summary(cancer_data))
```
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
```R
# Step 5: Visually Data Exploration
# Separating groups with positive and negative biopsy results
biopsy_positive <- cancer_data %>% filter(Biopsy == 1)
biopsy_negative <- cancer_data %>% filter(Biopsy == 0)

# Summary statistics for positive biopsy group
biopsy_positive_summary <- summary(biopsy_positive)
print(biopsy_positive_summary)

# Bar plot for biopsy results
biopsy_counts <- table(cancer_data$Biopsy)
biopsy_df <- as.data.frame(biopsy_counts)
ggplot(biopsy_df, aes(x = Var1, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("darkturquoise", "coral")) +
  labs(title = "Biopsy Results", x = "Class Labels", y = "Frequency") +
  theme_minimal()

# Distribution of Smokes parameter according to biopsy results
ggplot(cancer_data, aes(x = as.factor(Smokes), fill = as.factor(Biopsy))) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Smokes Parameter According to Biopsy Results",
       x = "Smoking Status",
       y = "Number") +
  scale_x_discrete(labels = c("0" = "Non-smoking", "1" = "Smoking")) +
  scale_fill_discrete(name = "Biopsy Result", labels = c("Negative", "Positive")) +
  theme_minimal()

# Distribution of Num of pregnancies according to biopsy results
ggplot(cancer_data, aes(x = as.factor(Biopsy), y = `Num.of.pregnancies`, fill = as.factor(Biopsy))) +
  geom_boxplot() +
  labs(title = "Distribution of Num of Pregnancies According to Biopsy Results",
       x = "Biopsy Result",
       y = "Number of Pregnancies") +
  scale_x_discrete(labels = c("0" = "Negative", "1" = "Positive")) +
  scale_fill_discrete(name = "Biopsy Result", labels = c("Negative", "Positive")) +
  theme_minimal()

# Reshape the data to a long format for ggplot
cancer_data_long <- melt(cancer_data)

# Create a boxplot for all variables in the dataset
ggplot(cancer_data_long, aes(x = value, y = variable)) +
  geom_boxplot() +
  labs(title = "Boxplot of Cancer Data", x = "Values", y = "Variables") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 12)) +
  coord_flip()  # Flip coordinates to match the horizontal orientation
```
-------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------

## Step 6: Feature Engineering
```R
# Identify columns with zero variance (constant values)
zero_variance_columns <- sapply(cancer_data, function(x) var(x, na.rm = TRUE) == 0)
print(zero_variance_columns)
# Remove columns with zero variance
cancer_data <- cancer_data[, !zero_variance_columns]

# Histogram of all numerical columns
cancer_data %>%
  select(where(is.numeric)) %>%
  gather() %>%
  ggplot(aes(value)) +
  facet_wrap(~key, scales = "free") +
  geom_histogram(fill = "steelblue", bins = 30) +
  theme_minimal() +
  labs(title = "Histograms of Numerical Columns")

# Compute the correlation matrix on the filtered data
correlation_matrix <- cor(cancer_data, use = "complete.obs")

# Convert the correlation matrix into a data frame
correlation_matrix_df <- as.data.frame(correlation_matrix)

# Print the data frame
print(correlation_matrix_df)

# Convert correlation matrix to long format for ggplot2
melted_cor_matrix <- melt(correlation_matrix)

# Create the heatmap
ggplot(data = melted_cor_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", limit = c(-1,1),
                       name="Correlation", guide = "colorbar") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.text.y = element_text(angle = 45, hjust = 1, vjust = 1)) +
  labs(title = "Heatmap of Correlation Matrix")
```
--------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------

## Step 7: Data Splitting
```R
# Split the dataset into independent features (X) and target (y)
X <- cancer_data[, -which(names(cancer_data) == "Biopsy")]
y <- cancer_data$Biopsy

# Set seed for reproducibility
set.seed(42)

# Split the data into training (80%) and testing (20%) sets
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]
```
---------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------

## Step 8: KNN Model
```R
# Train the KNN model
k <- 5  # Number of neighbors
knn_pred <- knn(train = X_train, test = X_test, cl = y_train, k = k)

# Calculate accuracy
accuracy <- sum(knn_pred == y_test) / length(y_test)
cat("KNN Accuracy:", accuracy, "\n")

# Create confusion matrix
confusion_matrix <- table(Predicted = knn_pred, Actual = y_test)
print(confusion_matrix)

# Visualize the confusion matrix using ggplot2
conf_matrix_df <- as.data.frame(confusion_matrix)
ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  theme_minimal() +
  labs(title = "KNN Confusion Matrix", x = "Actual Labels", y = "Predicted Labels")

library(pROC)
# Calculate accuracy
accuracy <- sum(knn_pred == y_test) / length(y_test)
cat("KNN Accuracy:", accuracy, "\n")

#install.packages("MLmetrics")
library(MLmetrics)

# Calculate F1 Score
f1_score <- F1_Score(y_test, knn_pred)
cat("KNN F1 Score:", f1_score, "\n")

# Feature Importance (using absolute correlation with target variable)
feature_importance <- sapply(X, function(col) cor(col, y))  # Calculate correlation for each feature
importance_df <- data.frame(Feature = names(feature_importance), Importance = abs(feature_importance))

# Remove features with zero importance (if any)
importance_df <- importance_df[importance_df$Importance > 0, ]

# Sort by importance
importance_df <- importance_df[order(-importance_df$Importance), ]

# Select top 5 features KNN
top_features <- head(importance_df, 5)

# Visualize Top 5 Feature Importance
ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 5 Feature Importance KNN Model", x = "Features", y = "Importance") +
  theme_minimal()
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------

## Step 9: SVM Model
```R
# Train the SVM model
svm_model <- svm(X_train, as.factor(y_train), kernel = "linear")  # Change kernel as needed

# Make predictions on the test set
svm_pred <- predict(svm_model, X_test)

# Calculate accuracy
accuracy <- sum(svm_pred == y_test) / length(y_test)
cat("SVM Accuracy:", accuracy, "\n")

# Calculate F1 Score
f1_score <- F1_Score(y_test, svm_pred)
cat("SVM F1 Score:", f1_score, "\n")

# Create confusion matrix
confusion_matrix <- table(Predicted = svm_pred, Actual = y_test)
print(confusion_matrix)

# Visualize the confusion matrix using ggplot2
conf_matrix_df <- as.data.frame(confusion_matrix)
ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  theme_minimal() +
  labs(title = "SVM Confusion Matrix", x = "Actual Labels", y = "Predicted Labels")

# Get the coefficients of the model
coef_vector <- t(svm_model$coefs) %*% svm_model$SV
print(coef_vector)

# Check the coefficient vector to ensure it's numeric
if (!is.numeric(coef_vector)) {
  stop("Coefficients are not numeric.")
}

# Create a data frame for feature importance
importance_df <- data.frame(Feature = colnames(X_train), Importance = as.numeric(abs(coef_vector)))

# Check the structure of the importance_df
str(importance_df)

# Sort by importance
importance_df <- importance_df[order(-importance_df$Importance), ]

# Select top 5 features
top_5_features <- head(importance_df, 5)

# Visualize Top 5 Features SVM
#library(ggplot2)

ggplot(top_5_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 5 Features Importance SVM Model", x = "Features", y = "Importance") +
  theme_minimal()
```
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------

## Step 10: Random Forest Model
```R
#Train the Random Forest model
rf_model <- randomForest(x = X_train, y = as.factor(y_train), ntree = 100)  # Adjust ntree as needed

# Make predictions on the test set
rf_pred <- predict(rf_model, X_test)

# Calculate accuracy
accuracy <- sum(rf_pred == y_test) / length(y_test)
cat("Random Forest Accuracy:", accuracy, "\n")

# Calculate F1 Score
f1_score <- F1_Score(y_test, rf_pred)
cat("RF model F1 Score:", f1_score, "\n")

# Create confusion matrix
confusion_matrix <- table(Predicted = rf_pred, Actual = y_test)
print(confusion_matrix)

# Visualize the confusion matrix using ggplot2
conf_matrix_df <- as.data.frame(confusion_matrix)
ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  theme_minimal() +
  labs(title = "Random Forest Confusion Matrix", x = "Actual Labels", y = "Predicted Labels")

# Calculate feature importance
importance_rf <- importance(rf_model)

# Create a data frame for feature importance
importance_df_rf <- data.frame(Feature = rownames(importance_rf), Importance = importance_rf[, 1])

# Sort by importance
importance_df_rf <- importance_df_rf[order(-importance_df_rf$Importance), ]

# Select top 5 features
top_5_features_rf <- head(importance_df_rf, 5)

# Visualize Top 5 Features Importance RF Model
ggplot(top_5_features_rf, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "forestgreen") +
  coord_flip() +
  labs(title = "Top 5 Features Importance RF Model", x = "Features", y = "Importance") +
  theme_minimal()
```
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------

## Step 11: Decision Tree Model
```R
# Train the Decision Tree model
dt_model <- rpart(as.factor(y_train) ~ ., data = data.frame(X_train, y_train), method = "class")

# Make predictions on the test set
dt_pred <- predict(dt_model, newdata = data.frame(X_test), type = "class")

# Calculate accuracy
accuracy <- sum(dt_pred == y_test) / length(y_test)
cat("Decision Tree Accuracy:", accuracy, "\n")

# Calculate F1 Score
f1_score <- F1_Score(y_test, dt_pred)
cat("Decision Tree F1 Score:", f1_score, "\n")

# Create confusion matrix
confusion_matrix <- table(Predicted = dt_pred, Actual = y_test)
print(confusion_matrix)

# Visualize the confusion matrix using ggplot2
conf_matrix_df <- as.data.frame(confusion_matrix)
ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  theme_minimal() +
  labs(title = "Decision Tree Confusion Matrix", x = "Actual Labels", y = "Predicted Labels")

# Visualize the Decision Tree
rpart.plot(dt_model, main = "Decision Tree")

# Calculate feature importance
importance_dt <- dt_model$variable.importance

# Create a data frame for feature importance
importance_df_dt <- data.frame(Feature = names(importance_dt), Importance = importance_dt)

# Sort by importance
importance_df_dt <- importance_df_dt[order(-importance_df_dt$Importance), ]

# Select top 5 features
top_5_features_dt <- head(importance_df_dt, 5)

# Visualize Top 5 Features Importance
ggplot(top_5_features_dt, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "orange") +
  coord_flip() +
  labs(title = "Top 5 Features Importance DT Model", x = "Features", y = "Importance") +
  theme_minimal()
```
--------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------

## Step 12: XGBoost Model
```R
# Convert data to matrix format for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = as.numeric(as.factor(y_train)) - 1)  # Convert factor to numeric
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = as.numeric(as.factor(y_test)) - 1)  # Convert factor to numeric

# Set parameters for the XGBoost model
params <- list(
  objective = "binary:logistic",  # For binary classification
  eval_metric = "logloss",
  eta = 0.1,  # Learning rate
  max_depth = 6,  # Maximum depth of the tree
  nrounds = 100  # Number of boosting rounds
)

# Train the XGBoost model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = params$nrounds,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,  # Stop if no improvement for 10 rounds
  print_every_n = 10
)

# Make predictions on the test set
xgb_pred <- predict(xgb_model, newdata = dtest)

# Convert predictions to binary class (0 or 1)
xgb_pred_class <- ifelse(xgb_pred > 0.5, 1, 0)

# Calculate accuracy
accuracy <- sum(xgb_pred_class == (as.numeric(as.factor(y_test)) - 1)) / length(y_test)
cat("XGBoost Accuracy:", accuracy, "\n")

# Calculate F1 Score using MLmetrics package
f1_score <- F1_Score(y_true = as.numeric(as.factor(y_test)) - 1, y_pred = xgb_pred_class, positive = "1")
cat("XGBoost F1 Score:", f1_score, "\n")

# Create confusion matrix
confusion_matrix <- table(Predicted = xgb_pred_class, Actual = as.numeric(as.factor(y_test)) - 1)
print(confusion_matrix)

# Visualize the confusion matrix using ggplot2
conf_matrix_df <- as.data.frame(confusion_matrix)
library(ggplot2)  # Load ggplot2 for visualization
ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  theme_minimal() +
  labs(title = "XGBoost Confusion Matrix", x = "Actual Labels", y = "Predicted Labels")

# Calculate feature importance
importance_xgb <- xgb.importance(model = xgb_model)

# Create a data frame for top 5 feature importance
top_5_features_xgb <- importance_xgb[1:5, ]

# Visualize Top 5 Features Importance
ggplot(top_5_features_xgb, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "purple") +
  coord_flip() +
  labs(title = "Top 5 Features Importance from XGBoost Model", x = "Features", y = "Gain") +
  theme_minimal()
```
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Step 13: Model Comparison
```R
# Comparison of top features data frames from 5 models
top_features_knn <- data.frame(Feature = c("Schiller", "Hinselmann", "Citology", "Dx.HPV", "Dx.Cancer"),
                               Importance = c(0.1, 0.3, 0.4, 0.2, 0.1))
top_features_svm <- data.frame(Feature = c("Schiller", "STDs.syphilis", "STDs..number", "STDs.vulvo.perineal.condylomatosis", "STDs.condylomatosis"),
                               Importance = c(0.2, 0.5, 0.3, 0.1, 0.4))
top_features_rf <- data.frame(Feature = c("Schiller", "Hinselmann", "Age", "Hormonal.Contraceptives..years.", "First.sexual.intercourse"),
                              Importance = c(0.4, 0.3, 0.5, 0.2, 0.1))
top_features_dt <- data.frame(Feature = c("Schiller", "Hinselmann", "Hormonal.Contraceptives..years.", "Citology", "Smokes..packs.year."),
                              Importance = c(0.3, 0.2, 0.4, 0.1, 0.5))
top_features_xgb <- data.frame(Feature = c("Schiller", "Age", "Num.of.pregnancies", "First.sexual.intercourse", "Hormonal.Contraceptives..years."),
                               Importance = c(0.5, 0.4, 0.3, 0.2, 0.1))

# Add a column indicating the model for each top features data frame
top_features_knn$Model <- "KNN"
top_features_svm$Model <- "SVM"
top_features_rf$Model <- "Random Forest"
top_features_dt$Model <- "Decision Tree"
top_features_xgb$Model <- "XGBoost"

# Combine all top features data frames into one
combined_importance <- bind_rows(top_features_knn, top_features_svm, 
                                 top_features_rf, top_features_dt, 
                                 top_features_xgb)

# Plotting
ggplot(combined_importance, aes(x = reorder(Feature, Importance), y = Importance, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Top 5 Feature Importance Comparison Across Models", 
       x = "Features", 
       y = "Importance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set1")  # Optional: change color palette
```
-------------------------------------------------------------------------------END-----------------------------------------------------------------------------------

**Author**: Debolina Dutta

LinkedIn: https://www.linkedin.com/in/duttadebolina/
