# This file is for the Amazon Employee Access Kaggle competition.
# This is the primary analysis file

# Libraries -----------------------------------------------
library(tidymodels)
library(vroom)
# library(ggmosaic)
library(tidyverse)
library(embed)
library(ranger)
library(discrim)

# Read in the data ------------------------------------
base_folder <- "AmazonEmployeeAccess/"
access_train <- vroom(paste0(base_folder, "train.csv"))
access_test <- vroom(paste0(base_folder, "test.csv"))

glimpse(access_train)
glimpse(access_test)

# Exploratory Data Analysis --------------------------
# # Create dataset for exploration
# access_explore <- access_train
# access_explore$ACTION <- factor(access_explore$ACTION)
# glimpse(access_explore)
# 
# # Write a function to assign any category with less than p% as "Other"
# condense_as_other <- function(this_data, x, p){
#   # This function condenses the column in this_data specified by "x" according to percent p.
#   # Inputs:
#   # this_data - a tibble
#   # x - a string containing the name of the column you want to reference
#   # p - a decimal representing the percent required to label a category as "other"
#   #     (e.g. if p = 0.01, then any category with fewer than 1% of the total observations gets labeled as "Other")
#   n <- length(this_data[[x]])
#   resource_freq <- 1:n
#   for (i in 1:n) {
#     resource_freq[i] = sum(this_data[[x]][i] == this_data[[x]])/n
#     if (resource_freq[i] < p) {
#       this_data[[x]][i] = "Other"
#     }
#   }
#   
#   # Return the data
#   this_data
# }
# 
# # Condense all predictors
# for (this_name in names(access_explore)[2:length(names(access_explore))]) {
#   access_explore <- condense_as_other(access_explore, this_name, 0.01)
# }
# 
# access_explore |> group_by(RESOURCE) |> summarize()
# access_explore |> group_by(MGR_ID) |> summarize()
# access_explore |> group_by(ROLE_ROLLUP_1) |> summarize()
# access_explore |> group_by(ROLE_ROLLUP_2) |> summarize()
# access_explore |> group_by(ROLE_DEPTNAME) |> summarize()
# access_explore |> group_by(ROLE_TITLE) |> summarize()
# access_explore |> group_by(ROLE_FAMILY_DESC) |> summarize()
# access_explore |> group_by(ROLE_FAMILY) |> summarize()
# access_explore |> group_by(ROLE_CODE) |> summarize()
# 
# # Plot access ACTION for various condensed predictors
# # ROLE_DEPTNAME
# ggplot(data = access_explore) + 
#   geom_mosaic(mapping = aes(x = product(ROLE_DEPTNAME), fill = ACTION))
# ggsave(paste0(base_folder, "Department Name Access Plot.png"))
# # ROLE_TITLE
# ggplot(data = access_explore) + 
#   geom_mosaic(mapping = aes(x = product(ROLE_TITLE), fill = ACTION))
# ggsave(paste0(base_folder, "Role Title Access Plot.png"))

# Recipes ---------------------------------------

# Make sure the response variable is categorical
access_train$ACTION <- as.factor(access_train$ACTION)

# Set the threshold percent to use for making a category "other"
threshold_percent <- 0.001

# Apply a recipe that condenses infrequent data values into "other" categories
access_recipe <- recipe(ACTION ~ ., data = access_train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> # turns all numeric features into factors
  step_other(all_nominal_predictors(), threshold = threshold_percent) |> # condenses categorical values that are less than 1% into an "other" category
  step_dummy(all_nominal_predictors()) # encode to dummy variables

# prepped_access_recipe <- prep(access_recipe)
# baked_access <- bake(prepped_access_recipe, new_data = access_train)
# glimpse(baked_access) # Check how many columns there are; should be 112


# Recipe for penalized logsitic regression
penalized_logistic_recipe <- recipe(ACTION ~ ., data = access_train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_other(all_nominal_predictors(), threshold = threshold_percent) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

# Recipe for random forest
tree_recipe <- recipe(ACTION ~ ., data = access_train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_other(all_nominal_predictors(), threshold = threshold_percent) # |> 
  # step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

# Recipe for Naive Bayes
naive_recipe <- recipe(ACTION ~ ., data = access_train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> # turns all numeric features into factors
  step_other(all_nominal_predictors(), threshold = threshold_percent) # condenses categorical values that are less than 1% into an "other" category
  # step_dummy(all_nominal_predictors()) # encode to dummy variables

# Recipe for K Nearest Neighbors
knn_recipe <- recipe(ACTION ~ ., data = access_train) |> 
  # step_other(all_numeric_predictors(), threshold = threshold_percent) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |> 
  step_normalize(all_numeric_predictors())

# Set the R^2 threshold for PCA
threshold_value <- 0.9

# Principal Component Analysis Recipes
pca_knn_recipe <- recipe(ACTION ~ ., data = access_train) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_normalize(all_predictors()) |> 
  step_pca(all_predictors(), threshold = threshold_value)

pca_naive_recipe <- recipe(ACTION ~ ., data = access_train) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_normalize(all_predictors()) |> 
  step_pca(all_predictors(), threshold = threshold_value)

# Recipe for Support Vector Machine (SVM)
svm_recipe <- recipe(ACTION ~ ., data = access_train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_other(all_numeric_predictors(), threshold = threshold_percent) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_pca(all_predictors(), threshold = threshold_value)

# Logistic Regression Model -----------------------
# logistic_mod <- logistic_reg() |> set_engine("glm")
# 
# logistic_amazon_wf <- workflow() |> 
#   add_recipe(access_recipe) |> 
#   add_model(logistic_mod) |> 
#   fit(data = access_train)
# 
# logistic_amazon_pred <- predict(logistic_amazon_wf, 
#                                 new_data = access_test,
#                                 type = "prob")
# logistic_amazon_pred
# logistic_amazon_export <- data.frame("id" = 1:length(logistic_amazon_pred$.pred_1),
#                                      "Action" = logistic_amazon_pred$.pred_1)
# 
# Penalized Logistic Regression Model -----------------------
# penalized_logistic_mod <- logistic_reg(mixture = tune(), penalty = tune()) |> 
#   set_engine("glmnet")
# 
# penalized_amazon_wf <- workflow () |>
#   add_recipe(penalized_logistic_recipe) |> 
#   add_model(penalized_logistic_mod)
# 
# # Set the tuning grid
# amazon_logistic_tuning_grid <- grid_regular(penalty(),
#                                             mixture(),
#                                             levels = 5)
# 
# # Set up the CV
# penalized_amazon_folds <- vfold_cv(access_train, v = 10, repeats = 1)
# 
# # Run the CV
# penalized_CV_results <- penalized_amazon_wf |> 
#   tune_grid(resamples = penalized_amazon_folds,
#             grid = amazon_logistic_tuning_grid,
#             metrics = metric_set(roc_auc)) #, f_meas, sens, recall, spec,
#                                  # precision, accuracy))
# 
# # Find out the best tuning parameters
# best_tune <- penalized_CV_results |> select_best("roc_auc")
# best_tune
# 
# # Use the best tuning parameters for the model
# final_penalized_wf <- penalized_amazon_wf |> 
#   finalize_workflow(best_tune) |> 
#   fit(data = access_train)
# 
# # Predictions
# penalized_logistic_preds <- final_penalized_wf |> 
#   predict(new_data = access_test, type = "prob")
# 
# # Prepare export
# penalized_export <- data.frame("id" = 1:length(penalized_logistic_preds$.pred_1),
#                                "Action" = penalized_logistic_preds$.pred_1)
# 
# 
# Random Forest (Classification) -----------------------------
# forest_amazon <- rand_forest(mtry = tune(),
#                              min_n = tune(),
#                              trees = 500) |>
#   set_engine("ranger") |>
#   set_mode("classification")
# 
# # Create a workflow using the model and recipe
# forest_amazon_wf <- workflow() |>
#   add_model(forest_amazon) |>
#   add_recipe(tree_recipe)
# 
# # Set up the grid with the tuning values
# forest_amazon_grid <- grid_regular(mtry(range = c(1, (length(access_train)-1))), min_n())
# 
# # Set up the K-fold CV
# forest_amazon_folds <- vfold_cv(data = access_train, v = 10, repeats = 1)
# 
# # Find best tuning parameters
# forest_cv_results <- forest_amazon_wf |>
#   tune_grid(resamples = forest_amazon_folds,
#             grid = forest_amazon_grid,
#             metrics = metric_set(roc_auc))
# 
# # Finalize the workflow using the best tuning parameters and predict
# # The best parameters were mtry = 9 and min_n = 2
# 
# # Find out the best tuning parameters
# best_forest_tune <- forest_cv_results |> select_best("roc_auc")
# 
# # Use the best tuning parameters for the model
# forest_final_wf <- forest_amazon_wf |>
#   finalize_workflow(best_forest_tune) |>
#   fit(data = access_train)
# 
# forest_amazon_predictions <- predict(forest_final_wf,
#                                      new_data = access_test,
#                                      type = "prob")
# forest_amazon_predictions
# 
# forest_export <- data.frame("id" = 1:length(forest_amazon_predictions$.pred_class),
#                                "Action" = forest_amazon_predictions$.pred_class)

# Naive Bayes Method ------------------------
# 
# # Set the model
# naive_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |>
#   set_mode("classification") |>
#   set_engine("naivebayes")
# 
# # Set workflow
# naive_wf <- workflow() |>
#   add_recipe(naive_recipe) |>
#   add_model(naive_model)
# 
# # Tuning
# # Set up the grid with the tuning values
# naive_grid <- grid_regular(Laplace(), smoothness())
# 
# # Set up the K-fold CV
# naive_folds <- vfold_cv(data = access_train, v = 10, repeats = 1)
# 
# # Find best tuning parameters
# naive_cv_results <- naive_wf |>
#   tune_grid(resamples = naive_folds,
#             grid = naive_grid,
#             metrics = metric_set(roc_auc))
# 
# # Select best tuning parameters
# naive_best_tune <- naive_cv_results |> select_best("roc_auc")
# naive_final_wf <- naive_wf |>
#   finalize_workflow(naive_best_tune) |>
#   fit(data = access_train)
# 
# # Make predictions
# naive_predictions <- predict(naive_final_wf, new_data = access_test, type = "prob")
# naive_predictions
# 
# # Prepare data for export
# naive_export <- data.frame("id" = 1:length(naive_predictions$.pred_class),
#                            "Action" = naive_predictions$.pred_class)

# KNN --------------------------------------------
# knn_model <- nearest_neighbor(neighbors = tune()) |>
#   set_mode("classification") |>
#   set_engine("kknn")
# 
# knn_wf <- workflow() |>
#   add_recipe(knn_recipe) |>
#   add_model(knn_model)
# 
# # Set up the tuning grid
# knn_grid <- grid_regular(neighbors())
# 
# # Set up the K-fold CV
# knn_folds <- vfold_cv(data = access_train, v = 3, repeats = 1)
# 
# # Find best tuning parameters
# knn_cv_results <- knn_wf |>
#   tune_grid(resamples = knn_folds,
#             grid = knn_grid,
#             metrics = metric_set(roc_auc))
# 
# # Select best parameters for model
# knn_best_tune <- knn_cv_results |> select_best("roc_auc")
# knn_final_wf <- knn_wf |>
#   finalize_workflow(knn_best_tune) |>
#   fit(data = access_train)
# 
# # Make predictions
# knn_predictions <- predict(knn_final_wf, new_data = access_test, type = "prob")
# knn_predictions
# 
# # Prepare data for export
# knn_export <- data.frame("id" = 1:length(knn_predictions$.pred_class),
#                            "Action" = knn_predictions$.pred_class)

# PCA KNN--------------------------------------
# 
# # Set the model
# pca_knn_model <- nearest_neighbor(neighbors = tune()) |> 
#   set_mode("classification") |> 
#   set_engine("kknn")
# 
# # Set the workflow
# pca_knn_wf <- workflow() |> 
#   add_recipe(pca_knn_recipe) |> 
#   add_model(pca_knn_model)
# 
# # Set up the tuning grid
# pca_knn_grid <- grid_regular(neighbors())
# 
# # Set up the K-fold CV
# pca_knn_folds <- vfold_cv(data = access_train, v = 10, repeats = 1)
# 
# # Find best tuning parameters
# pca_knn_cv_results <- pca_knn_wf |> 
#   tune_grid(resamples = pca_knn_folds,
#             grid = pca_knn_grid,
#             metrics = metric_set(roc_auc))
# 
# # Select best parameters for model
# pca_knn_best_tune <- pca_knn_cv_results |> select_best("roc_auc")
# pca_knn_final_wf <- pca_knn_wf |> 
#   finalize_workflow(pca_knn_best_tune) |> 
#   fit(data = access_train)
# 
# # Make predictions
# pca_knn_predictions <- predict(pca_knn_final_wf, new_data = access_test, type = "prob")
# pca_knn_predictions
# 
# # Prepare data for export
# pca_knn_export <- data.frame("id" = 1:length(pca_knn_predictions$.pred_class),
#                          "Action" = pca_knn_predictions$.pred_class)

# PCA Naive Bayes ----------------------------------
# 
# # Set the model
# pca_naive_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |>
#   set_mode("classification") |>
#   set_engine("naivebayes")
# 
# # Set workflow
# pca_naive_wf <- workflow() |>
#   add_recipe(pca_naive_recipe) |>
#   add_model(pca_naive_model)
# 
# # Tuning
# # Set up the grid with the tuning values
# pca_naive_grid <- grid_regular(Laplace(), smoothness())
# 
# # Set up the K-fold CV
# pca_naive_folds <- vfold_cv(data = access_train, v = 10, repeats = 1)
# 
# # Find best tuning parameters
# pca_naive_cv_results <- pca_naive_wf |>
#   tune_grid(resamples = pca_naive_folds,
#             grid = pca_naive_grid,
#             metrics = metric_set(roc_auc))
# 
# # Select best tuning parameters
# pca_naive_best_tune <- pca_naive_cv_results |> select_best("roc_auc")
# pca_naive_final_wf <- pca_naive_wf |>
#   finalize_workflow(pca_naive_best_tune) |>
#   fit(data = access_train)
# 
# # Make predictions
# pca_naive_predictions <- predict(pca_naive_final_wf, new_data = access_test, type = "prob")
# pca_naive_predictions
# 
# # Prepare data for export
# pca_naive_export <- data.frame("id" = 1:length(pca_naive_predictions$.pred_class),
#                            "Action" = pca_naive_predictions$.pred_class)

# SVM ------------------------------------------------

# Set the model
svm_radial_model <- svm_rbf(rbf_sigma = tune(), cost = tune()) |>
  set_mode("classification") |>
  set_engine("kernlab")

# Set workflow
svm_wf <- workflow() |>
  add_recipe(svm_recipe) |>
  add_model(svm_radial_model)

# Tuning
# Set up the grid with the tuning values
svm_grid <- grid_regular(rbf_sigma(), cost())

# Set up the K-fold CV
svm_folds <- vfold_cv(data = access_train, v = 10, repeats = 1)

# Find best tuning parameters
svm_cv_results <- svm_wf |>
  tune_grid(resamples = svm_folds,
            grid = svm_grid,
            metrics = metric_set(roc_auc))

# Select best tuning parameters
svm_best_tune <- svm_cv_results |> select_best("roc_auc")
svm_final_wf <- svm_wf |>
  finalize_workflow(svm_best_tune) |>
  fit(data = access_train)

# Make predictions
svm_predictions <- predict(svm_final_wf, new_data = access_test, type = "prob")
svm_predictions

# Prepare data for export
svm_export <- data.frame("id" = 1:length(svm_predictions$.pred_class),
                               "Action" = svm_predictions$.pred_class)

# Write the data ---------------------------------
# vroom_write(logistic_amazon_export, paste0(base_folder, "logistic.csv"), delim = ",")
# vroom_write(penalized_export, paste0(base_folder, "penalized_logistic.csv"), delim = ",")
vroom_write(forest_export, paste0(base_folder, "random_forest_classification.csv"), delim =",")
vroom_write(naive_export, paste0(base_folder, "naive_bayes.csv"), delim = ",")
vroom_write(knn_export, paste0(base_folder, "knn.csv"), delim = ",")
vroom_write(pca_knn_export, paste0(base_folder, "pca_knn.csv"), delim = ",")
vroom_write(pca_naive_export, paste0(base_folder, "pca_naive.csv"), delim = ",")
# vroom_write(svm_export, paste0(base_folder, "svm.csv"), delim = ",")