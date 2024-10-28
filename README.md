# Cervical Cancer Analysis & Insight: ML Modelling R

## Project Overview

Cervical cancer remains one of the leading causes of cancer-related deaths among women globally. Early detection and accurate prediction of cervical cancer are critical in improving treatment outcomes and saving lives. This project aims to develop a predictive model utilizing machine learning techniques to identify individuals at high risk for cervical cancer. By leveraging this ML models, healthcare providers can implement timely interventions and medical care, ultimately enhancing patient outcomes and reducing mortality rates associated with this disease.

## Project Objective

1. The objective of this analysis is to develop predictive models for cervical cancer biopsy results using a comprehensive dataset. The steps involved in this process include:

2. Dataset Description: Provide an overview of the dataset, including its structure, features, and any missing values.

3. Data Preprocessing and Exploratory Data Analysis (EDA): Clean and prepare the data for analysis by handling missing values and encoding categorical variables as needed. Conduct EDA to understand the distribution of variables and identify patterns, relationships, and potential outliers within the data.

4. Model Selection: Choose appropriate machine learning models, including K-Nearest Neighbors (KNN), Random Forest (RF), Decision Tree (DT), and Artificial Neural Network (ANN) for predicting biopsy results.

5. Model Training and Validation: Split the data into training and testing sets, train the selected models, and validate their performance using appropriate metrics such as accuracy and F1 score.

6. Feature Importance Analysis: Assess the importance of different features in making predictions, focusing on the models that support this analysis.

7. Comparison and Recommendation: Compare the performance of the models based on their accuracy, F1 score, and interpretability, and provide recommendations for the best-performing model for clinical application.

## Dataset Description

The dataset ```cervical_cancer_csv``` for this project has been sourced from ```Kaggle``` and contains various variables that contribute to the risk assessment of cervical cancer. Key variables include:

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

## Importing libraries
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

## Importing the data
```R
# Load the dataset
data <- read.csv("D:\\R Projects\\cervical-cancer_csv.csv")
```

## Data Pre-Processing & Exploratory Data Analysis
> Data Summary Check
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
```
The dataset comprises 835 observations and 36 variables, with varying degrees of missing values. Notable missing counts include Num.of.pregnancies (56), IUD (112), and a significant STDs..Time.since.first.diagnosis (764). The percentage of missing values ranges from 0.84% for First.sexual.intercourse to 91.50% for STDs..Time.since.first.diagnosis, indicating considerable data loss in several columns. This missing data must be addressed before further analysis or modeling can proceed.

> Identifying numerical and Categorical features
```R
# Separate numerical and categorical columns as per given lists
numerical_df <- c('Age', 'Number.of.sexual.partners', 'First.sexual.intercourse', 'Num.of.pregnancies', 'Smokes..years.', 
                  'Smokes..packs.year.', 'Hormonal.Contraceptives..years.', 'IUD..years.', 'STDs..number.')

categorical_df <- c('Smokes', 'Hormonal.Contraceptives', 'IUD', 'STDs', 'STDs.condylomatosis', 'STDs.cervical.condylomatosis',
                    'STDs.vaginal.condylomatosis', 'STDs.vulvo.perineal.condylomatosis', 'STDs.syphilis', 
                    'STDs.pelvic.inflammatory.disease', 'STDs.genital.herpes', 'STDs.molluscum.contagiosum', 
                    'STDs.AIDS', 'STDs.HIV', 'STDs.Hepatitis.B', 'STDs.HPV', 'STDs..Number.of.diagnosis', 
                    'Dx.Cancer', 'Dx.CIN', 'Dx.HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology', 'Biopsy')
```
> Update dataset: Taking care of misssing values thrugh dropping variables & replacement of values and Taking care of categorical fetaures
```R
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

# Convert categorical columns to factors
cancer_data <- cancer_data %>%
  mutate(across(all_of(categorical_df), as.factor))

# Check the cleaned and imputed dataset
print(summary(cancer_data))
```

## Visually Data Exploration
> Histogram of Numerical features distribution
```R
# Reshape the data for plotting
melted_data <- cancer_data %>%
  select(all_of(numerical_df)) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

# Create histograms for each numerical column
ggplot(melted_data, aes(x = Value)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black", alpha = 0.7) +
  facet_wrap(~ Variable, scales = "free_x") +
  labs(title = "Histograms of Numerical Columns", x = "Value", y = "Frequency") +
  theme_minimal()
```
![Plot histograms of Numerical features](https://github.com/user-attachments/assets/02ab92dd-5da0-46e6-9f84-45d7f83e76f2)

These histograms provide valuable insights into the distribution of numerical features in the cancer data. They can help identify potential outliers, identify relationships between variables, and inform further analysis. For example, the right-skewed distributions of age, sexual activity, and smoking habits suggest that these factors might play a role in cancer development. The bimodal distribution of hormonal contraceptive usage suggests that there might be two distinct groups of patients with different risk factors.

By understanding the distribution of these features, we can make informed decisions about feature engineering, and model selection for further analysis and prediction.

> Separating and Summarizing Biopsy Results by Group
```R
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
```
We explore here demographic and health variables for groups with positive and negative biopsy results:

- Group Separation: Filters data into biopsy_positive (Biopsy = 1) and biopsy_negative (Biopsy = 0) for focused analysis.
- Positive Biopsy Summary: Provides statistics for age (16–52, median 28), number of sexual partners (avg. ~2.5), smoking status (10 smoke, 44 don’t), and STD indicators (mostly zero values). This summary highlights key characteristics and patterns within the biopsy-positive group, which could be relevant to biopsy outcomes.

![Biopsy plot](https://github.com/user-attachments/assets/d0740956-08e1-47e0-8c05-987ebfbf395b)

The graph shows the distribution of biopsy results. It appears that the majority of the biopsies (around 800) were negative (class label 0), while only a small number (around 50) were positive (class label 1). This suggests that the majority of the patients in this study did not have the condition being tested for.

> Visual analysis of effects of Smokes acc. to Biopsy results
```R
# Visualizations of distribution of smokes acc to biopsy
ggplot(cancer_data, aes(x = as.factor(Smokes), fill = as.factor(Biopsy))) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Smoking According to Biopsy Results",
       x = "Smoking Status", y = "Count") +
  scale_x_discrete(labels = c("0" = "Non-smoking", "1" = "Smoking")) +
  scale_fill_discrete(name = "Biopsy Result", labels = c("Negative", "Positive")) +
  theme_minimal()
```
![Smoke vs  Biopsy Distribution](https://github.com/user-attachments/assets/bc921c41-1269-4db4-a816-fd279a615ea2)

This bar chart shows the distribution of smoking status among individuals with positive and negative biopsy results.

Non-smoking: The majority of individuals with negative biopsy results are non-smokers (high red bar), while a smaller portion with positive biopsy results also falls in this group (small blue bar).
Smoking: Fewer individuals smoke, with a higher count of negative biopsy results (red) compared to positive results (blue) among smokers.
Overall, most individuals are non-smokers, regardless of biopsy result, though there is a noticeable drop in positive cases among smokers compared to non-smokers.

> Visual Exploration of Number of pregnancies acc. to Biopsy
```R
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
```
> ![Number of preg vs Biopsy](https://github.com/user-attachments/assets/95a63b98-d12a-4897-a43c-06a84455879a)

The graph shows how the number of pregnancies is distributed among patients who had negative and positive biopsy results:

- Median: The median number of pregnancies is slightly higher for patients with positive biopsy results compared to those with negative results.
- Range: There is a wider range of pregnancies among patients with negative biopsy results, indicating more variability.
- Outliers: There are a few outliers with a high number of pregnancies in both groups.
Overall, the graph suggests that there might be a slight association between a higher number of pregnancies and a positive biopsy result. It's not clear, so we need further advance techniques o analyse it.

> Visual Exploration of Hormonal Contraceptives According to Biopsy Results
```R
# Distribution of Hormonal Contraceptives According to Biopsy Results
ggplot(cancer_data, aes(x = as.factor(Hormonal.Contraceptives), fill = as.factor(Biopsy))) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Hormonal Contraceptives According to Biopsy Results",
       x = "Hormonal Contraceptives Status",
       y = "Count") +
  scale_x_discrete(labels = c("0" = "Non-User", "1" = "User")) +  # Update labels as necessary
  scale_fill_discrete(name = "Biopsy Result", labels = c("Negative", "Positive")) +
  theme_minimal()
```
![Hormonal Contraceptive vs Biopsy](https://github.com/user-attachments/assets/690650e7-3c60-48ec-a706-2374bf401cd6)

The graph shows how the use of hormonal contraceptives is distributed among patients who had negative and positive biopsy results:

- Non-Users: The majority of patients in both groups are non-users of hormonal contraceptives.
- Users: There is a higher proportion of hormonal contraceptive users among patients with negative biopsy results compared to those with positive results.
Overall, the graph suggests that there might be a slight association between the use of hormonal contraceptives and a negative biopsy result, but further analysis is required.

> Visual Exploration of Age distribution According to Biopsy Result
```R
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
![Age vs Biopsy distribution](https://github.com/user-attachments/assets/0d69057b-dd2e-4a52-a149-692c96bf5ada)

This boxplot displays the age distribution for individuals with positive and negative biopsy results.

- Negative Biopsy Results (red): Ages are primarily between 20 and 40, with a few outliers reaching up to 80.
- Positive Biopsy Results (blue): Ages also mostly fall between 20 and 40, with a slightly higher median age compared to the negative group, but a similar overall range.
- The comparison suggests no significant age difference between the two groups, though the positive group shows a slightly higher central tendency.
-------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------
## **Correlation Matrix and Anova Test**

> Correlation Matrix Calculation for the Numerical Features
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
```
![heatmap correlation matrix of numerical features](https://github.com/user-attachments/assets/b76bed6e-eeca-4d04-82a0-0fca714976e6)

Key Observations from the Correlation Heatmap: -

1. Strong Positive Correlations: (Strong red colour)

- ```STDs.number``` and ```IUD.years```: Using an IUD might be associated with a higher risk of STDs.
- ```IUD.years``` and ```Hormonal.Contraceptives.years```: There might be a tendency to use both methods.
- ```Smokes.packs.year``` and ```Smokes.years```: The more years of smoking, the more packs smoked per year.

2. Moderate Positive Correlations: (Lighter red Colour)

- ```STDs.number``` and``` Hormonal.Contraceptives.years```: Hormonal contraceptives might be linked to a slightly higher risk of STDs.

3. Strong Negative Correlations: (Blue Colour)

- ```First.sexual.intercourse``` and ```Num.of.pregnancies```: Earlier sexual debut might be associated with more pregnancies.

4. No Significant Correlations: (White Colour)

-```Age``` and other variables: Age doesn't seem to have a strong relationship with other variables.

> Annova test
```R
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
The ANOVA test results indicate significant associations between various health and lifestyle variables and demographic factors such as age and smoking habits, highlighting the following key findings:

- ```Age```: Strongly significant associations with ```IUD usage``` (p < 0.001), ```cancer diagnosis``` (p = 0.0017), ```HPV diagnosis``` (p = 0.0039), and ```Schiller``` Test results (p = 0.0034) suggest that age influences these health-related outcomes.
- ```Number of Sexual Partners```: A significant correlation with smoking status (p < 0.001) implies that smoking behavior may relate to an individual's sexual history.
- ```First Sexual Intercourse Age```: ```Smoking``` (p = 0.0002) and certain ```STDs (syphilis```: p = 0.0045; ```vaginal condylomatosis```: p = 0.033) are significantly associated with the age at which individuals first engage in sexual activity.
- ```Number of Pregnancies```: Significant relationships are observed with ```IUD usage``` (p < 0.001), ```hormonal contraceptives``` (p = 0.0027), and ```STDs (syphilis```: p = 0.00003), indicating these factors may affect pregnancy frequency.
- ```Smoking History```: Both smoking duration (years) and intensity (packs/year) show a strong association with smoking status (p < 0.001), along with significant correlations with certain ```STDs (HIV```: p = 0.0095; ```Hepatitis``` B: p = 0.0042) and ```Schiller``` Test results (p = 0.0064).
-```Biopsy``` Results: In contrast, the ANOVA results indicate that none of the analyzed demographic and health factors exhibit statistically significant correlations with biopsy results, as all p-values exceed the conventional significance threshold of 0.05. The only borderline case is the association between smoking years and biopsy (p = 0.0759), which suggests a potential trend but does not meet the criteria for statistical significance. Therefore, overall, the analysis underscores significant correlations between ```age```, ```sexual behavior```, and ```smoking``` with various health variables, suggesting behavioral and demographic influences on health conditions. However, the lack of significant correlations with biopsy results indicates that further research is needed to explore these relationships more deeply, particularly regarding smoking history.

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
We conduct feature engineering where columns with zero variance (i.e., columns where all values are the same) are identified and removed from the dataset. The updated cancer_dataset no longer includes these uninformative columns, this has been done in oreder to reduce dimensionality and to improving model efficiency before training models in our next steps.

## Data Splitting 
```R
# Split data into independent variables and target variable as well as into training set and test set
set.seed(42)
train_index <- createDataPartition(cancer_data$Biopsy, p = 0.8, list = FALSE)
X_train <- cancer_data[train_index, -which(names(cancer_data) == "Biopsy")]
y_train <- cancer_data$Biopsy[train_index]
X_test <- cancer_data[-train_index, -which(names(cancer_data) == "Biopsy")]
y_test <- cancer_data$Biopsy[-train_index]
```

## Models Selection: 

We selected the K-Nearest Neighbors (KNN), Random Forest (RF), Decision Tree, and Artificial Neural Network (ANN) models for the analysis and prediction of biopsy results in cervical cancer data due to their diverse strengths and suitability for handling categorical variables. KNN is effective for its simplicity and ability to capture local patterns, making it useful for classification tasks in high-dimensional spaces. The Random Forest model, an ensemble learning method, excels in reducing overfitting and enhancing predictive accuracy by aggregating multiple decision trees, making it robust for varied data characteristics. Decision Trees provide interpretability, allowing us to visualize decision-making processes and understand variable importance in classification. Lastly, the ANN model leverages complex relationships in the data through multiple layers, making it well-suited for capturing intricate patterns that may exist in the dataset. 

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
![KNN confusion matrix](https://github.com/user-attachments/assets/809dee21-de61-448e-bce4-8a4fce24f2e2)

The confusion matrix shows the performance of the K-Nearest Neighbors (KNN) model in predicting biopsy results:

- True Negatives (TN): 156 cases were correctly predicted as negative (0).
- False Positives (FP): 10 cases were incorrectly predicted as negative, but were actually positive.
- False Negatives (FN): 0 cases were incorrectly predicted as positive, though they were negative.
- True Positives (TP): 0 cases were correctly predicted as positive.
- F1 Score (96.8%): This score balances precision and recall, indicating strong predictive performance in identifying the negative class, as there were no true positives. The high F1 score reflects high precision and recall for classifying negatives correctly.
- Accuracy (93.8%): This score reflects the proportion of correct predictions among all predictions, with the model accurately predicting the majority negative class in this dataset.

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
```
![Random Forest confusion matrix](https://github.com/user-attachments/assets/7dce5ba7-0bae-49a8-8673-79666e00c8a5)

The Random Forest confusion matrix shows:

- True Negatives: 153 (correctly predicted negatives)
- False Positives: 4 (incorrectly predicted negatives)
- False Negatives: 3 (incorrectly predicted positives)
- True Positives: 6 (correctly predicted positives)
- Accuracy (95.8%) indicates that most predictions are correct, and F1 Score (97.8%) shows strong balance between precision and recall, particularly in identifying positives effectively.

> Feature Importance RF Model
```R
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
![RF feature importance](https://github.com/user-attachments/assets/dcc4b0c1-ef52-4db4-a68b-2863da96db4f)

The graph shows the top 5 features that have the highest importance in a Random Forest model prediction:

- ```Schiller``` and ```Hinselmann```: These features have the highest importance, suggesting they are the strongest predictors of the outcome.
- ```Age```: Age also plays a significant role in the prediction.
- ```Hormonal Contraceptives``` and ```First Sexual Intercourse```: These features have lower importance compared to the top 3.
Overall, the graph suggests that the Random Forest model relies heavily on the Schiller and Hinselmann features, along with age, to make accurate predictions.

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
```
![Decision Tree confusion matrix](https://github.com/user-attachments/assets/b7e8fb1d-73a1-4e59-938c-cef3d1a36665)

The confusion matrix of DT model provides a summary of the model's predictions compared to the actual outcomes:

- True Positives (TP): 6 (Correctly predicted positive cases)
- True Negatives (TN): 153 (Correctly predicted negative cases)
- False Positives (FP): 4 (Incorrectly predicted as positive)
- False Negatives (FN): 3 (Incorrectly predicted as negative)
- Accuracy: 95.78% of the predictions were correct, indicating strong overall performance.
- F1 Score: 0.9776 shows excellent precision and recall balance, highlighting the model's effectiveness in identifying positive biopsy cases while minimizing false positives and negatives.

> Feature Importance Decision Tree Model
```R
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
![DT model Feature Importance](https://github.com/user-attachments/assets/c293ed12-d48b-464c-a885-4b5845206758)

The graph shows the top 5 features that have the highest importance in a Decision Tree model for predicting biopsy result: 

- ```Schiller and Hinselmann```: These features have the highest importance, suggesting they are the strongest predictors of the biopsy result.
- ```Hormonal Contraceptives```, ```Cytology```, and ```Smoking```: These features have lower importance, indicating a lesser influence on the prediction.
Overall, the graph suggests that the Decision Tree model primarily relies on the Schiller and Hinselmann features to make accurate predictions about biopsy results.

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
```

![ANN Confusion Matrix](https://github.com/user-attachments/assets/13035d17-7df2-407b-b141-20ba3b525d92)

The confusion matrix summarizes the model's predictions against the actual outcomes:

- True Positives (TP): The model correctly predicted 36 positive cases.
- True Negatives (TN): The model correctly predicted 156 negative cases.
- False Positives (FP): The model incorrectly predicted 5 negative cases as positive.
- False Negatives (FN): The model incorrectly predicted 10 positive cases as negative.
- Accuracy (0.939759): This indicates that the model correctly predicted 93.97% of the cases.
- F1 Score (0.9689441): This is a measure of the model's precision and recall. A high F1 score suggests that the model has a good balance between precision and recall.
Overall, the ANN model shows strong performance in predicting biopsy results, with high accuracy and F1 score. 

> Feature importance ANN Model
```R
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
![ANN feature importance](https://github.com/user-attachments/assets/308bae9b-b36b-4dc3-baa8-db10ebc1217e)

The graph shows the top 5 features that have the highest importance in an Artificial Neural Network (ANN) model in predicting the outcome variable:

- ```Smokes```: This feature has the highest importance, suggesting it is the strongest predictor of the outcome.
- ```Smokes.packs.year```: This feature is also highly important, indicating that the number of packs smoked per year is a significant factor.
- ```STDs.condylomatosis``` and ```STDs.vulvo.perineal.condylomatosis```: These features related to sexually transmitted diseases have moderate importance.
- ```IUD.years```: This feature has the lowest importance among the top 5.
Overall, the graph suggests that the ANN model relies heavily on smoking-related features to make accurate predictions.


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
  scale_fill_brewer(palette = "Set1")  
```
![Top features Comparison](https://github.com/user-attachments/assets/0e85b7b4-d04c-427b-9656-4acb41b43c10)

The graph compares the top 5 features that have the highest importance in three different models: Artificial Neural Network (ANN), Decision Tree, and Random Forest. This means these features are the most useful in predicting the biopsy result.

Key Observations:

- ```Schiller, STDs.number```: These features are consistently among the top 5 important features across all three models.
- ```Smokes, Smokes.packs.year```: These features are also highly important in all models, indicating that smoking-related factors are strong predictors of the biopsy result.
- ```Hinselmann, STDs.condylomatosis```: These features are moderately important in all models.
- ```Cytology, IUD.years, First sexual.intercourse```: These features have lower importance in all models.
Overall, the graph suggests that all three models rely heavily on smoking-related features and the Schiller test to make accurate predictions about biopsy results. However, the ANN model seems to give more weight to STDs.number compared to the other two models (by visually comparing the height of the bars for "STDs.number" in the ANN model to the other models, we can see that it appears to be relatively higher. This suggests that the ANN model might be assigning more importance to this feature).

Note: *Feature importance was not calculated for the K-Nearest Neighbors (KNN) model because KNN is a non-parametric algorithm that makes predictions based on the distance between data points rather than a learned model structure. Unlike models such as Random Forest or Decision Trees, KNN does not inherently provide a mechanism to assess the contribution of individual features to its predictions, making feature importance analysis impractical for this algorithm.*

## **Conclusion & Recommendations**

Based on performance analysis of the models above, we come to concludion that the Random Forest (RF) model emerges as the best-performing model for cervical cancer prediction based on biopsy results. Its high accuracy (95.8%) and F1 score (97.8%) suggest robust performance across both classes, making it effective in identifying positive and negative biopsy cases. The feature importance analysis indicates that it leverages the most predictive features effectively, specifically Schiller and Hinselmann, which are likely clinically relevant. 

While the Decision Tree (DT) also performs comparably with a slightly lower accuracy and F1 score, its interpretability allows for clear understanding of decision pathways. The Artificial Neural Network (ANN) provides strong sensitivity in predicting positive cases, but its overall performance in terms of accuracy lags behind RF and DT. 

In contrast, the K-Nearest Neighbors (KNN) model, despite its high precision for negative cases, lacks the ability to identify positive cases, limiting its applicability in clinical decision-making where detecting positive biopsy results is crucial.

Based on the performance analysis, it is recommended to utilize the Random Forest (RF) model for predicting cervical cancer biopsy results. Its high accuracy and F1 score indicate a strong capability to effectively identify both positive and negative cases, making it a reliable tool for clinical decision-making. Additionally, its emphasis on key predictive features such as Schiller and Hinselmann provides valuable insights that can enhance diagnostic processes. Incorporating the RF model into clinical practice could improve early detection and treatment outcomes for cervical cancer.

**Author**: Debolina Dutta

LinkedIn: https://www.linkedin.com/in/duttadebolina/
