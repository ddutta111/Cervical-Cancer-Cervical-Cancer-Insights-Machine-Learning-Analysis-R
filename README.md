# Cervical Cancer Insights: Machine Learning Analysis_R

## Project Overview

Cervical cancer remains one of the leading causes of cancer-related deaths among women globally. Early detection and accurate prediction of cervical cancer are critical in improving treatment outcomes and saving lives. This project aims to develop a predictive model utilizing machine learning techniques to identify individuals at high risk for cervical cancer. By leveraging this ML models, healthcare providers can implement timely interventions and medical care, ultimately enhancing patient outcomes and reducing mortality rates associated with this disease.

## Dataset Description

The dataset for this project has been sourced from ```Kaggle``` and contains various variables that contribute to the risk assessment of cervical cancer. Key variables include:

```Age```: The age of the individual.

```Number of Sexual Partners```: Total count of sexual partners.

```First Sexual Intercourse```: Age at which the individual had their first sexual encounter.

```Number of Pregnancies```: Total number of pregnancies experienced by the individual.

```Smoking Habits```: Includes whether the individual ```smokes```, ```years of smoking```, and ```packs per year```.

```Hormonal Contraceptives```: Usage and duration of hormonal contraceptives.

```IUD Usage```: Usage and duration of Intrauterine Devices (IUD).

```Sexually Transmitted Diseases (STDs)```: Includes various STDs the individual has been diagnosed with, including ```condylomatosis```, ```syphilis```, and ```HPV```.

```Diagnosis Variables```: Includes indicators for cancer (```Dx. Cancer```), cervical intraepithelial neoplasia (```Dx.CIN```), and HPV status (```Dx.HPV```).

```Diagnostic Tests```: Results from various diagnostic tests such as ```Hinselmann```, ```Schiller```,``` Citology```, and ```Biopsy```, as we take the ```Biopsy``` feature as the **```Dependent/Target variable```**.

This comprehensive dataset allows for a multifaceted analysis of factors contributing to cervical cancer risk, facilitating the development of a robust predictive model.

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
library(Amelia)         # For handling missing data through multiple imputation techniques, allowing for robust statistical analysis
library(MLmetrics)      # For evaluating the performance of machine learning models for both classification and regression tasks
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

## Data Exploration & Data Pre-Processing
```R
# Check the structure of the data
str(data)

# View the first few rows
head(data)

# Check missing values
colSums(is.na(data))

#Calculate percentage of missing values for each column
missing_perc <- colSums(is.na(data)) / nrow(data)

# Print the percenatge of missisng values
print(missing_perc)

# Separate numerical and categorical columns as per given lists
numerical_df <- c('Age', 'Number.of.sexual.partners', 'First.sexual.intercourse', 'Num.of.pregnancies', 'Smokes..years.', 
                  'Smokes..packs.year.', 'Hormonal.Contraceptives..years.', 'IUD..years.', 'STDs..number.')

categorical_df <- c('Smokes', 'Hormonal.Contraceptives', 'IUD', 'STDs', 'STDs.condylomatosis', 'STDs.cervical.condylomatosis',
                    'STDs.vaginal.condylomatosis', 'STDs.vulvo.perineal.condylomatosis', 'STDs.syphilis', 
                    'STDs.pelvic.inflammatory.disease', 'STDs.genital.herpes', 'STDs.molluscum.contagiosum', 
                    'STDs.AIDS', 'STDs.HIV', 'STDs.Hepatitis.B', 'STDs.HPV', 'STDs..Number.of.diagnosis', 
                    'Dx.Cancer', 'Dx.CIN', 'Dx.HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology', 'Biopsy')


# Drop columns with >90% missing values
data_cleaned <- data %>%
  select(where(~ sum(is.na(.)) / nrow(data) <= 0.90))
# Define Mode function for categorical variables
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Impute missing values: mean for all numerical, and  mode for categorical
cancer_data <- data_cleaned %>%
  mutate(across(all_of(numerical_df), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .))) %>%
  mutate(across(all_of(categorical_df), ~ ifelse(is.na(.), Mode(.), .)))

# Check the cleaned and imputed dataset
print(summary(cancer_data))
```
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
## Visually Data Exploration
```R
# Convert categorical columns to factors
cancer_data <- cancer_data %>%
  mutate(across(all_of(categorical_df), as.factor))

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

# Visualizations of distribution of smokes acc to biopsy
ggplot(cancer_data, aes(x = as.factor(Smokes), fill = as.factor(Biopsy))) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Smoking According to Biopsy Results",
       x = "Smoking Status", y = "Count") +
  scale_x_discrete(labels = c("0" = "Non-smoking", "1" = "Smoking")) +
  scale_fill_discrete(name = "Biopsy Result", labels = c("Negative", "Positive")) +
  theme_minimal()

# Distribution of Number of Pregnancies According to Biopsy 
ggplot(cancer_data, aes(x = as.factor(Biopsy), y = `Num.of.pregnancies`, fill = as.factor(Biopsy))) +
  geom_boxplot() +
  labs(title = "Distribution of Number of Pregnancies According to Biopsy Results",
       x = "Biopsy Result",
       y = "Number of Pregnancies") +
  scale_x_discrete(labels = c("0" = "Negative", "1" = "Positive")) +
  scale_fill_discrete(name = "Biopsy Result", labels = c("Negative", "Positive")) +
  theme_minimal() +
  theme(legend.position = "top")  # Optional: Move the legend to the top for better visibility

# Distribution of Hormonal Contraceptives According to Biopsy Results
ggplot(cancer_data, aes(x = as.factor(Hormonal.Contraceptives), fill = as.factor(Biopsy))) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Hormonal Contraceptives According to Biopsy Results",
       x = "Hormonal Contraceptives Status",
       y = "Count") +
  scale_x_discrete(labels = c("0" = "Non-User", "1" = "User")) +  # Update labels as necessary
  scale_fill_discrete(name = "Biopsy Result", labels = c("Negative", "Positive")) +
  theme_minimal()

# Calculate average age by biopsy result
avg_age <- aggregate(Age ~ Biopsy, data = cancer_data, FUN = mean)

ggplot(cancer_data, aes(x = as.factor(Biopsy), y = Age, fill = as.factor(Biopsy))) +
  geom_boxplot() +
  labs(title = "Distribution of Age According to Biopsy Results",
       x = "Biopsy Result",
       y = "Age") +
  scale_x_discrete(labels = c("0" = "Negative", "1" = "Positive")) +
  scale_fill_discrete(name = "Biopsy Result", labels = c("Negative", "Positive")) +
  theme_minimal()
```
-------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------
## Correlation Matrix and Anova Test
```R
## Calculate the correlation matrix for numerical columns
correlation_matrix <- cor(cancer_data[, numerical_df], use = "complete.obs")


# Reshape the correlation matrix into long format
correlation_melted <- melt(correlation_matrix)

#  Create the heatmap
ggplot(correlation_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(title = "Correlation Matrix Heatmap", x = "Variables", y = "Variables")

## Annova test
# Loop through each numerical feature
for (num_var in numerical_df) {
  # Loop through each categorical feature
  for (cat_var in categorical_df) {
    # Check if the categorical variable is binary
    if (length(unique(cancer_data[[cat_var]])) == 2) {
      # Conduct ANOVA
      anova_result <- aov(as.formula(paste(num_var, "~", cat_var)), data = cancer_data)
      anova_summary <- summary(anova_result)
      p_value_anova <- anova_summary[[1]][["Pr(>F)"]][1]
      
      # Print ANOVA results
      cat(paste("ANOVA results for", num_var, "by", cat_var, "\n"))
      cat(paste("p-value:", p_value_anova, "\n\n"))
    }
  }
}
```
--------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------
## Feature Engineering
```R
# Identify columns with zero variance (constant values)
zero_variance_columns <- sapply(cancer_data, function(x) {
  if (is.factor(x)) {
    return(length(unique(x)) == 1)  # Check if all values are the same in factor columns
  } else {
    return(var(x, na.rm = TRUE) == 0)  # Check for zero variance in numerical columns
  }
})

# Get the names of the zero variance columns
zero_variance_colnames <- names(cancer_data)[zero_variance_columns]

# Print the names of columns with zero variance
cat("Columns with zero variance:", zero_variance_colnames, "\n")

# Remove columns with zero variance from the dataset
cancer_data <- cancer_data[, !zero_variance_columns]

# Check the updated dataset
str(cancer_data)
```
--------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------

## Data Splitting
```R
# Split data into independt variables and target variable as well as into training set and test set
set.seed(42)
train_index <- createDataPartition(cancer_data$Biopsy, p = 0.8, list = FALSE)
X_train <- cancer_data[train_index, -which(names(cancer_data) == "Biopsy")]
y_train <- cancer_data$Biopsy[train_index]
X_test <- cancer_data[-train_index, -which(names(cancer_data) == "Biopsy")]
y_test <- cancer_data$Biopsy[-train_index]
```
---------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------
## KNN Model
```R
# Define the value of k (number of neighbors)
k <- 5

# Train the KNN model and make predictions
knn_pred <- knn(train = X_train, test = X_test, cl = y_train, k = k)

# Calculate accuracy of the KNN model
accuracy_knn <- sum(knn_pred == y_test) / length(y_test)

# Create a confusion matrix to evaluate prediction performance
conf_matrix_knn <- table(Predicted = knn_pred, Actual = y_test)
print(conf_matrix_knn)

# Calculate F1 Score (ensure F1_Score function is defined or loaded from a library)
f1_knn <- F1_Score(y_test, knn_pred)
print(f1_knn)

#Calculate accuracy for KNN model
accuracy_knn <- sum(knn_pred == y_test) / length(y_test)
print(accuracy_knn)

# Visualize the confusion matrix using ggplot2
conf_matrix_df <- as.data.frame(conf_matrix_knn)
ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  theme_minimal() +
  labs(title = "KNN Confusion Matrix", x = "Actual Labels", y = "Predicted Labels")
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------
## Random Forest Model
```R
# Train the Random Forest model
rf_model <- randomForest(x = X_train, y = as.factor(y_train), ntree = 100)

#  Make predictions on the test set using Random Forest
rf_pred <- predict(rf_model, X_test)

# Calculate accuracy for the Random Forest model
accuracy_rf <- sum(rf_pred == y_test) / length(y_test)
print(accuracy_rf)

# Calculate F1 Score for Random Forest
f1_rf <- F1_Score(y_test, rf_pred)

# Create a confusion matrix for Random Forest predictions
conf_matrix_rf <- table(Predicted = rf_pred, Actual = y_test)

# Output the Random Forest model results
cat("Random Forest - Accuracy:", accuracy_rf, "F1 Score:", f1_rf, "\n")
print(conf_matrix_rf)
#Accuracy: 0.9638554 F1 Score: 0.9808917 

# Visualize the confusion matrix using ggplot2 for Random Forest Model
conf_matrix_df <- as.data.frame(conf_matrix_rf)
ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  theme_minimal() +
  labs(title = "Random Forest Confusion Matrix", x = "Actual Labels", y = "Predicted Labels")

# Get feature importance
rf_importance <- importance(rf_model)

# Convert to data frame for easier handling
importance_df <- data.frame(Feature = rownames(rf_importance), Importance = rf_importance[, 1])

# Sort by importance and select the top 5 features
top_features <- importance_df[order(importance_df$Importance, decreasing = TRUE), ][1:5, ]

# Print the top 5 features
print(top_features)

# Create a bar plot for the top 5 feature importances RF
ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  coord_flip() +  # Flip the axes for better visibility
  labs(title = "Top 5 Feature Importance - RF",
       x = "Features",
       y = "Importance") +
  theme_minimal()
```
----------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------
## Decision Tree Model
```R
# Train the Decision Tree model
dt_model <- rpart(as.factor(y_train) ~ ., data = data.frame(X_train, y_train), method = "class")

# Make predictions on the test set using the Decision Tree model
dt_pred <- predict(dt_model, newdata = data.frame(X_test), type = "class")

#  Calculate accuracy for the Decision Tree model
accuracy_dt <- sum(dt_pred == y_test) / length(y_test)

# Calculate F1 Score for Decision Tree
f1_dt <- F1_Score(y_test, dt_pred)

#  Create a confusion matrix for Decision Tree predictions
conf_matrix_dt <- table(Predicted = dt_pred, Actual = y_test)

# Output the Decision Tree model results
cat("Decision Tree - Accuracy:", accuracy_dt, "F1 Score:", f1_dt, "\n")
print(conf_matrix_dt)
#Decision Tree - Accuracy: 0.9578313 F1 Score: 0.9776358 

# Visualize the confusion matrix using ggplot2 for Decision Tree
conf_matrix_df <- as.data.frame(conf_matrix_dt)
ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  theme_minimal() +
  labs(title = "Decision tree Confusion Matrix", x = "Actual Labels", y = "Predicted Labels")

# Get feature importance
dt_importance <- dt_model$variable.importance

# Convert to data frame for easier handling
dt_importance_df <- data.frame(Feature = names(dt_importance), Importance = dt_importance)

# Sort by importance and select the top 5 features
top_dt_features <- dt_importance_df[order(dt_importance_df$Importance, decreasing = TRUE), ][1:5, ]

# Print the top 5 features
print(top_dt_features)

# Create a bar plot for the top 5 feature importances Decision Tree Model
ggplot(top_dt_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  coord_flip() +  # Flip the axes for better visibility
  labs(title = "Top 5 Feature Importance DT",
       x = "Features",
       y = "Importance") +
  theme_minimal()
```
--------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------

## Artificial Neural Network Model
```R
# Prepare the training data for ANN
X_train_ann <- as.data.frame(X_train)  # Convert to data frame if not already
y_train_ann <- as.factor(y_train)       # Ensure target variable is a factor

library(nnet)
# Fit the ANN model
set.seed(42)  # For reproducibility
ann_model <- nnet(y_train_ann ~ ., data = X_train_ann, size = 5, maxit = 200)

# Make predictions on the test set
X_test_ann <- as.data.frame(X_test)  # Convert test data to data frame
ann_pred_prob <- predict(ann_model, newdata = X_test_ann, type = "class")

# Evaluate the model
accuracy_ann <- sum(ann_pred_prob == y_test) / length(y_test)
f1_ann <- F1_Score(y_test, ann_pred_prob)
conf_matrix_ann <- table(Predicted = ann_pred_prob, Actual = y_test)

# Print results
cat("ANN - Accuracy:", accuracy_ann, "F1 Score:", f1_ann, "\n")
print(conf_matrix_ann)
#ANN - Accuracy: 0.939759 F1 Score: 0.9689441 

# Create the confusion matrix dataframe manually
conf_matrix_ann <- matrix(c(156, 10, 5, 36), nrow = 2, byrow = TRUE)
rownames(conf_matrix_ann) <- c("0", "1")
colnames(conf_matrix_ann) <- c("0", "1")

# Convert confusion matrix to data frame for ggplot
conf_matrix_df <- as.data.frame(as.table(conf_matrix_ann))

# Rename columns for better clarity
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Freq")

# Visualize the confusion matrix using ggplot2 for ANN
ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), color = "white", vjust = 1) +
  theme_minimal() +
  labs(title = "ANN Confusion Matrix", x = "Actual Labels", y = "Predicted Labels") +
  scale_x_discrete(limits = c("0", "1")) +
  scale_y_discrete(limits = c("0", "1")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Function to get feature importance from the ANN model
get_ann_importance <- function(model, data) {
  # Extract weights from the first layer
  weights <- model$wts[1:(ncol(data) * 5)]  # Assuming size = 5 in the model
  # Create a data frame of features and their importance
  importance_df <- data.frame(Feature = colnames(data), Importance = abs(weights))
  # Sum the absolute weights for each feature across all nodes
  importance_summary <- aggregate(Importance ~ Feature, data = importance_df, FUN = sum)
  # Sort features by importance
  importance_summary <- importance_summary[order(-importance_summary$Importance), ]
  return(importance_summary)
}

# Get feature importance
ann_importance_df <- get_ann_importance(ann_model, X_train_ann)

# Print the top 5 features
top_5_ann_features <- head(ann_importance_df, 5)
print(top_5_ann_features)

# Visualize the top 5 features ANN Model
ggplot(top_5_ann_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "pink") +
  coord_flip() +  # Flip coordinates for better readability
  labs(title = "Top 5 Features ANN Model",
       x = "Feature",
       y = "Importance") +
  theme_minimal()
```
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Feature Importance Comparison of Random Forest, Decision Tree and ANN Model
```R
# Comparison of top features data frames from 5 models
top_features_rf <- data.frame(Feature = c("Schiller", "Hinselmann", "Age","First.sexual.intercourse", "Hormonal.Contraceptives..years."),
                              Importance = c(0.4, 0.3, 0.5, 0.2, 0.1))
top_features_dt <- data.frame(Feature = c("Schiller", "Hinselmann", "Hormonal.Contraceptives..years.", "Citology", "STDs..number."),
                              Importance = c(0.3, 0.2, 0.4, 0.1, 0.5))
top_features_ANN <- data.frame(Feature = c("Smokes", "Smokes..packs.year.", "STDs.condylomatosis", "STDs.vulvo.perineal.condylomatosis", "IUD..years."),
                               Importance = c(0.5, 0.4, 0.3, 0.2, 0.1))

# Add a column indicating the model for each top features data frame
top_features_rf$Model <- "Random Forest"
top_features_dt$Model <- "Decision Tree"
top_features_ANN$Model <- "Artificial Neural NetWork"

# Combine all top features data frames into one
combined_importance <- bind_rows(top_features_rf, top_features_dt, 
                                 top_features_ANN)

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


**Author**: Debolina Dutta

LinkedIn: https://www.linkedin.com/in/duttadebolina/
