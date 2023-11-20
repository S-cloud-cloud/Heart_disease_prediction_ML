# Install required packages
install.packages(c("dplyr", "ggplot2", "forcats", "rsample", "tidyverse", "tidymodels", "gridExtra", "pROC"))
# libraries are used throughout the project for tidying the data, randomly 
#splitting the data, creating data visualizations, formulating summary tables, 
#and evaluating the model.
# Load libraries (all open source packages)
library(dplyr) #data manipulation
library(ggplot2)  #plotting complex data
library(forcats)  #dealing with factors and changing the order of levels,re-factoring,fct()
library(rsample)  #used along with tidyverse ,estimating distributions and assessing model performance
library(tidyverse) #Pckg that helps to transform and better present data. 
#It assists with data import, tidying, manipulation, and data visualization. 
library(tidymodels) # collection of packages for modeling and machine learning using tidyverse principles. 
library(gridExtra)  #provides useful extensions to the grid system, with 
#an emphasis on higher-level functions to work with grid graphic objects, 
library(pROC) #analyze and compare ROC curves #ROC classification error metric

# Set working directory
print(getwd())
setwd("C:/Users/KIIT/OneDrive/Desktop/codes/R_PROJECT/heart_disease")
print(getwd())

# Read data and explore
cleveland <- read.csv("processed.cleveland.data", header = FALSE, fileEncoding = "UTF-8")
#don't use the first row as names of variables
glimpse(cleveland)

# Data tidying
names <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "heart_disease")
colnames(cleveland) <- names
#(pipe) operator from the dplyr package. manipulating a dataset related to heart disease.
#mutate is a function that is used to add new variables (columns) to a data frame or modify 
cleveland <- cleveland %>%
  mutate(sex = case_when(sex == 0 ~ "female", sex == 1 ~ "male")) %>%  #creates a new variable/column called "sex" s
  mutate(cp = case_when(cp == 1 ~ "typical angina", cp == 2 ~ "atypical angina", cp == 3 ~ "non-anginal pain", cp == 4 ~ "asymptomatic")) %>%
  mutate(fbs = case_when(fbs == 1 ~ "high", fbs == 0 ~ "low")) %>% 
  mutate(exang = case_when(exang == 0 ~ "no", exang == 1 ~ "yes")) %>%
  mutate(heart_disease = case_when(heart_disease == 0 ~ "absence", TRUE ~ "presence"))
#more interpretable and human-friendly labels
cleveland <- cleveland %>%
  mutate(across(c(sex, cp, fbs, exang, heart_disease), as.factor))
#convert categorial variables into factors for ease of analysis.
cleveland <- cleveland %>%
  select(age, sex, cp, trestbps, chol, fbs, thalach, exang, heart_disease) %>%
  rename("max_hr" = "thalach", "exercise_angina" = "exang") %>%
  drop_na()

glimpse(cleveland)

# Exploratory Analysis
age.plot <- ggplot(cleveland, aes(x = age, fill = heart_disease)) +
  geom_histogram() +
  facet_wrap(vars(heart_disease)) +
  labs(title = "Prevalence of Heart Disease Across Age", x = "Age (years)", y = "Count", fill = "Heart Disease")
#facet_wrap to create separate panels for each level of "heart_disease." 
#labs function to set the plot title and axis labels,
cp.plot <- ggplot(cleveland, aes(x = heart_disease, fill = cp)) +
  geom_bar(position = "dodge") +
  labs(title = "Prevalence of Heart Disease for Different Chest Pain Types", x = "Heart Disease", y = "Count", fill = "Chest Pain Type")
#The geom_bar function in ggplot2 is used to create bar charts. It is often used 
#to visualize the distribution of a categorical variable by counting the occurrences of each category.
sex.plot <- ggplot(cleveland, aes(x = sex, fill = heart_disease)) +
  geom_bar(position = "fill") +
  labs(x = "Sex", y = "Proportion", fill = "Heart Disease")

fbs.plot <- ggplot(cleveland, aes(x = fbs, fill = heart_disease)) +
  geom_bar(position = "fill") +
  labs(x = "Fasting Blood Sugar", y = "Proportion", fill = "Heart Disease") +
  scale_x_discrete(labels = c("low", "high"))
#customize x-axis variables for discrete values

exang.plot <- ggplot(cleveland, aes(x = exercise_angina, fill = heart_disease)) +
  geom_bar(position = "fill") +
  labs(x = "Exercise Induced Angina", y = "Proportion", fill = "Heart Disease")

grid.arrange(sex.plot, fbs.plot, exang.plot, nrow = 2)
#spanning of row / cols

# Cholesterol and Blood Pressure
trestbps.plot <- ggplot(cleveland, aes(x = trestbps, y = heart_disease)) +
  geom_boxplot() +
  labs(x = "Resting Blood Pressure (mm Hg)", y = "Heart Disease")

chol.plot <- ggplot(cleveland, aes(x = chol, y = heart_disease)) +
  geom_boxplot() +
  labs(x = "Serum Cholesterol (mg/dl)", y = "Heart Disease")

maxhr.plot <- ggplot(cleveland, aes(x = max_hr, y = heart_disease)) +
  geom_boxplot() +
  labs(x = "Maximum Heart Rate (bpm)", y = "Heart Disease")

grid.arrange(trestbps.plot, chol.plot, maxhr.plot, nrow = 2)

# Data splitting for training and testing (75% & 25% respectively)
heart.split <- initial_split(cleveland)
#tidymodels ecosystem, particularly with the r_sample package. 
#The initial_split function is used to create an initial split of a data_set into training and testing sets.
heart.train <- training(heart.split)
heart.test <- testing(heart.split)

# Logistic Regression model using all 8 predictors
#glm:generalized linear model
heart.full <- glm(heart_disease ~ ., data = heart.train, family = "binomial")
summary(heart.full)

#The Estimate column shows the estimated coefficients for each predictor.
#The Std. Error column shows the standard errors of the coefficients.
#The z value is the z-statistic, and Pr(>|z|) is the p-value associated with the z-statistic. 
#These are used to test the null hypothesis that the corresponding coefficient is equal to zero (no effect).
#The difference between null and residual deviance is used in the likelihood ratio test.
#The Wald statistics test the hypothesis that each coefficient is equal to zero. 
#It's a z-test for each individual coefficient.
# Logistic regression model with age, fasting blood sugar, and cholesterol removed
heart_model <- logistic_reg() %>%
  set_engine("glm")
#using the tidymodels framework to specify and configure a logistic regression model in R

# recipe function from the tidymodels framework in R to create a 
#data preprocessing recipe for a logistic regression model.
heart_recipe <- recipe(heart_disease ~ ., data = heart.train) %>%
  step_rm(fbs, age, chol) %>%
  step_zv(all_predictors())
#This step removes the variables specified in the 
#step_rm function from the dataset. In this case, it removes the predictors fbs, age, and chol.
#This step removes variables with zero variance, indicating that they have no variability in the training data.

#A workflow in tidymodels is a container that holds the modeling information, 
#including the model and the preprocessing steps.
heart_wflow <- workflow() %>%
  add_model(heart_model) %>%
  add_recipe(heart_recipe)

#This line fits the entire workflow (heart_wflow) to your training data (heart.train). 
#The fit function is used to train the model and apply the preprocessing steps defined in the workflow.
heart_fit <- heart_wflow %>%
  fit(data = heart.train)
# the tidy function is used to extract a tidy summary of the model coefficients.
tidy(heart_fit)

# Receiver Operating Characteristic Technique (ROC)
heart.train.pred <- predict(heart_fit, new_data = heart.train)
#comparison data frame (traincomp) between the actual responses 
#(heart.train$heart_disease) and the predicted responses (heart.train.pred).
traincomp <- data.frame(heart.train$heart_disease, heart.train.pred)
colnames(traincomp) <- c("train.response", "train.prediction")
traincomp <- traincomp %>%
  mutate(across(c(train.response, train.prediction), factor)) 

heart.roc <- roc(response = ordered(traincomp$train.response), predictor =ordered(traincomp$train.prediction))

plot(heart.roc, print.thres = "best", main = "Receiver Operating Characteristic Technique Plot")

# AUC calculation
print(auc(heart.roc))

# 5-fold cross-validation
set.seed(470)
folds <- vfold_cv(heart.train, v = 5)
heart_fit_rs <- heart_wflow %>%
  fit_resamples(folds)

metrics <- data.frame(collect_metrics(heart_fit_rs, summarize = FALSE))
metrics <- metrics %>%
  select(-.config)
colnames(metrics) <- c("Fold", "Metric", "Estimator", "Estimate")
metrics

# Generating predictions on testing data
heart_disease_pred <- predict(heart_fit, new_data = heart.test) %>%
  bind_cols(heart.test %>% select(heart_disease))

# Evaluate model on testing data
test_accuracy <- accuracy(heart_disease_pred, truth = heart_disease, estimate = .pred_class)
test_specificity <- spec(heart_disease_pred, truth = heart_disease, estimate = .pred_class)
test_sensitivity <- sens(heart_disease_pred, truth = heart_disease, estimate = .pred_class)

test.values <- data.frame(test_accuracy$.estimate, test_sensitivity$.estimate, test_specificity$.estimate)
colnames(test.values) <- c("Test set Accuracy", "Test set Sensitivity", "Test set Specificity")
test.values

# Conclusion
# The model has a test set accuracy of 0.80, indicating 80% correct classification of patients.
# The true positive rate on the test set is 84.44%, and the true negative rate is 73.33%.
# The high true positive rate is promising for predicting individuals with heart disease.
# The high true negative rate corresponds to a low false positive rate, indicating accurate classification
# of individuals without heart disease.
