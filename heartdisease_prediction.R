install.packages("dplyr")
install.packages("ggplot2")
install.packages("forcats")
install.packages("rsample")
install.packages("tidyverse")
install.packages("tidymodels")
install.packages("gridExtra")
install.packages("pROC")


library(dplyr)
library(ggplot2)
library(forcats)
library(rsample)
library(tidyverse)
library(tidymodels)
library(gridExtra)
library(pROC)

#
print(getwd())
setwd("C:/Users/KIIT/OneDrive/Desktop/codes/R_PROJECT/heart_disease")
print(getwd())
cleveland <- read.csv("processed.cleveland.data", header = FALSE, fileEncoding = "UTF-8")
glimpse(cleveland)

#data_tyding
names = c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "heart_disease")
colnames(cleveland) <- names

cleveland <- cleveland %>%
  mutate(sex = case_when(sex == 0 ~ "female",
                         sex == 1 ~ "male")) %>%
  mutate(cp = case_when(cp == 1 ~ "typical angina",
                        cp == 2 ~ "atypical angina", 
                        cp == 3 ~ "non-anginal pain",
                        cp == 4 ~ "asymptomatic")) %>%
  mutate(fbs = case_when(fbs == 1 ~ "high",
                         fbs == 0 ~ "low")) %>% 
  mutate(exang = case_when(exang == 0 ~ "no",
                           exang == 1 ~ "yes")) %>%
  mutate(heart_disease = case_when(heart_disease == 0 ~ "absence",
                                   TRUE ~ "presence"))

cleveland <- cleveland %>%
  mutate(sex = as.factor(sex)) %>%
  mutate(cp = as.factor(cp)) %>%
  mutate(fbs = as.factor(fbs)) %>%
  mutate(exang = as.factor(exang)) %>%
  mutate(heart_disease = as.factor(heart_disease))

cleveland <- cleveland %>%
  select(age, sex, cp, trestbps, chol, fbs, thalach, exang, heart_disease) %>%
  rename("max_hr" = "thalach",
         "exercise_angina" = "exang") %>%
  drop_na()

glimpse(cleveland)

#Exploratory Analysis
age.plot <- ggplot(cleveland, mapping = aes(x = age, fill = heart_disease)) +
  geom_histogram() +
  facet_wrap(vars(heart_disease)) +
  labs(title = "Prevelance of Heart Disease Across Age", x = "Age (years)", y = "Count", fill = "Heart Disease")

age.plot

#prevelance of heart diseases for diff chest pains: 
cp.plot <- ggplot(cleveland, mapping = aes(x=heart_disease, fill = cp)) +
  geom_bar(position = "dodge") +
  labs(title = "Prevelance of Heart Disease for Different Chest Pain Types", x = "Heart Disease", y = "Count", fill = "Chest Pain Type")

cp.plot

#asymptomatic chest pain type has the highest count for the presence of heart disease, 
#while typical angina pain has the lowest count. There is a higher count of people 
#without heart disease that have atypical or typical angina chest pain compared to 
#people with heart disease. Angina is listed as one of the most common symptoms of 
#heart attack and so this result is skeptical 
#and needs further investigation, but we will assume it is correct for the current analysis.

sex.plot <- ggplot(cleveland, mapping = aes(x = sex, fill = heart_disease)) +
  geom_bar(position = "fill") +
  labs(x = "Sex", y = "Proportion", fill = "Heart Disease") +
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), 
        axis.title.y = element_text(size = 12), axis.text.y = element_text(size = 12))

fbs.plot <- ggplot(cleveland, mapping = aes(x=fbs, fill=heart_disease)) +
  geom_bar(position = "fill") +
  labs(x = "Fasting Blood Sugar", y = "Proportion", fill = "Heart Disease") +
  scale_x_discrete(labels = c("low", "high"))+
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), 
        axis.title.y = element_text(size = 12), axis.text.y = element_text(size = 12))

exang.plot <- ggplot(cleveland, mapping = aes(x = exercise_angina, fill = heart_disease)) +
  geom_bar(position = "fill") +
  labs(x = "Exercise induced angina", y = "Proportion", fill = "Heart Disease") +
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12))

grid.arrange(sex.plot, fbs.plot, exang.plot, nrow=2)

#cholestrol and BP :
trestbps.plot <- ggplot(cleveland, mapping = aes(x=trestbps, y=heart_disease)) +
  geom_boxplot() +
  labs(x = "Resting Blood Pressure (mm Hg)", y = "Heart Disease") +
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), 
        axis.title.y = element_text(size = 12), axis.text.y = element_text(size = 12))

chol.plot <- ggplot(cleveland, mapping = aes(x=chol, y=heart_disease)) +
  geom_boxplot() +
  labs(x = "Serum Cholestoral (mg/dl)", y = "Heart Disease") +
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), 
        axis.title.y = element_text(size = 12), axis.text.y = element_text(size = 12))

maxhr.plot <- ggplot(cleveland, mapping = aes(x = max_hr, y = heart_disease)) +
  geom_boxplot() +
  labs(x = "Maximum Heart Rate (bpm)", y = "Heart Disease") +
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), 
        axis.title.y = element_text(size = 12), axis.text.y = element_text(size = 12))

grid.arrange(trestbps.plot, chol.plot, maxhr.plot, nrow=2)


#data splitting for training and testing : 75% & 25% respectively
heart.split <- initial_split(cleveland)
heart.train <- training(heart.split)
heart.test <- testing(heart.split)

#Logistic Regression model using all 8 predictors
heart.full <- glm(heart_disease~., data = heart.train, family = "binomial")
summary(heart.full)

#Logistic regression model with age, fasting blood sugar, and cholesterol removed
# set engine
heart_model <- logistic_reg() %>%
  set_engine("glm")

# create recipe
heart_recipe <- recipe(heart_disease ~., data = heart.train) %>%
  step_rm(fbs) %>%
  step_rm(age) %>%
  step_rm(chol) %>%
  step_zv(all_predictors())

# build work flow
heart_wflow <- workflow() %>%
  add_model(heart_model) %>%
  add_recipe(heart_recipe)

# fit training data through the work flow 
heart_fit <- heart_wflow %>%
  fit(data = heart.train)
tidy(heart_fit)

#
#Receiver Operating Characteritic Technique (ROC):
heart.train.pred = predict(heart_fit, new_data = heart.train)

traincomp <- data.frame(heart.train$heart_disease, heart.train.pred)
colnames(traincomp) <- c("train.response", "train.prediction")
traincomp <- traincomp %>%
  mutate(train.response = factor(case_when(train.response == "absence" ~ 0,
                                           train.response == "presence" ~ 1))) %>%
  mutate(train.prediction = factor(case_when(train.prediction == "absence" ~ 0,
                                             train.prediction == "presence" ~ 1)))

heart.roc <- roc(response = ordered(traincomp$train.response), predictor = ordered(traincomp$train.prediction))

plot(heart.roc, print.thres = "best", main = "Receiver Operating Characteritic Technique Plot")

#
#
print(auc(heart.roc))

#Our model for prediction of heart disease has an AUC value of 0.784; thus, 
#the model will correctly predict a heart disease diagnosis from a negative 
#diagnosis 78.4% of the time, given new data. The optimal threshold for the model is 0.500, 
#which means observations with predicted probabilities < 0.500 will be classified as 
#not having heart disease and observations with predicted probabilities > 0.500 will be 
#classified as having heart disease. The specificity at the optimal threshold is 0.815, 
#which corresponds to a 81.5% true negative rate and 18.5% false positive rate. 
#The sensitivity at the optimal threshold is 0.752, 
#which corresponds to a 75.2% true positive rate and 24.8% false negative rate.


##
#Perform 5-fold cross validation
set.seed(470)
folds <- vfold_cv(heart.train, v=5)

heart_fit_rs <- heart_wflow %>%
  fit_resamples(folds)

metrics <- data.frame(collect_metrics(heart_fit_rs, summarize = FALSE))

metrics <- metrics %>%
  select(-.config)
colnames(metrics) <- c("Fold", "Metric", "Estimator", "Estimate")
metrics

#
#Generating predictions on testing data
heart_disease_pred <- predict(heart_fit, new_data = heart.test) %>%
  bind_cols(heart.test %>% select(heart_disease))

test_accuracy <- accuracy(heart_disease_pred, truth = heart_disease, estimate = .pred_class)
test_specificity <- spec(heart_disease_pred, truth = heart_disease, estimate = .pred_class)
test_sensitivity <- sens(heart_disease_pred, truth = heart_disease, estimate = .pred_class)

test.values <- data.frame(test_accuracy$.estimate, test_sensitivity$.estimate, test_specificity$.estimate)
colnames(test.values) <- c("Test set Accuracy", "Test set Sensitivity", "Test set Specificity")
test.values

#conclusion
##The model has a test set accuracy of 0.80, which indicates 80% of the patients in the 
#test set are correctly classified as having heart disease or not. The true positive rate 
#on the test set is 84.44% and the true negative rate is 73.33%. The high true positive rate 
#of our model is promising, and indicates a high predictive ability of our model to correctly 
#classify individuals that have heart disease. The high true negative rate of our model 
#corresponds to a low false positive rate, which means 
#individuals without heart disease will largely be classified as not having heart disease.