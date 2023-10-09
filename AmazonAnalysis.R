# This file is for the Amazon Employee Access Kaggle competition.
# This is the primary analysis file

# Libraries -----------------------------------------------
library(tidymodels)
library(vroom)
library(ggmosaic)
library(tidyverse)

# Read in the data -----------------------------------
base_folder <- "C:/Users/BYU Rental/STAT348/AmazonEmployeeAccess/"
access_train <- vroom(paste0(base_folder, "train.csv"))
access_test <- vroom(paste0(base_folder, "test.csv"))

glimpse(access_train)
glimpse(access_test)

# Exploratory Data Analysis --------------------------
# Create dataset for exploration
access_explore <- access_train
access_explore$ACTION <- factor(access_explore$ACTION)
glimpse(access_explore)

# Write a function to assign any category with less than p% as "Other"
condense_as_other <- function(this_data, x, p){
  # This function condenses the column in this_data specified by "x" according to percent p.
  # Inputs:
  # this_data - a tibble
  # x - a string containing the name of the column you want to reference
  # p - a decimal representing the percent required to label a category as "other"
  #     (e.g. if p = 0.01, then any category with fewer than 1% of the total observations gets labeled as "Other")
  n <- length(this_data[[x]])
  resource_freq <- 1:n
  for (i in 1:n) {
    resource_freq[i] = sum(this_data[[x]][i] == this_data[[x]])/n
    if (resource_freq[i] < p) {
      this_data[[x]][i] = "Other"
    }
  }
  
  # Return the data
  this_data
}

# Condense all predictors
for (this_name in names(access_explore)[2:length(names(access_explore))]) {
  access_explore <- condense_as_other(access_explore, this_name, 0.01)
}

access_explore |> group_by(RESOURCE) |> summarize()
access_explore |> group_by(MGR_ID) |> summarize()
access_explore |> group_by(ROLE_ROLLUP_1) |> summarize()
access_explore |> group_by(ROLE_ROLLUP_2) |> summarize()
access_explore |> group_by(ROLE_DEPTNAME) |> summarize()
access_explore |> group_by(ROLE_TITLE) |> summarize()
access_explore |> group_by(ROLE_FAMILY_DESC) |> summarize()
access_explore |> group_by(ROLE_FAMILY) |> summarize()
access_explore |> group_by(ROLE_CODE) |> summarize()

# Plot access ACTION for various condensed predictors
# ROLE_DEPTNAME
ggplot(data = access_explore) + 
  geom_mosaic(mapping = aes(x = product(ROLE_DEPTNAME), fill = ACTION))
ggsave(paste0(base_folder, "Department Name Access Plot.png"))
# ROLE_TITLE
ggplot(data = access_explore) + 
  geom_mosaic(mapping = aes(x = product(ROLE_TITLE), fill = ACTION))
ggsave(paste0(base_folder, "Role Title Access Plot.png"))

# Recipes -----------------------------------------------
# Apply a recipe that condenses infrequent data values into "other" categories
access_train$ACTION <- as.factor(access_train$ACTION)
access_recipe <- recipe(ACTION ~ ., data = access_train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> # turns all numeric features into factors
  step_other(all_nominal_predictors(), threshold = 0.01) |> # condenses categorical values that are less than 1% into an "other" category
  step_dummy(all_nominal_predictors()) # encode to dummy variables

prepped_access_recipe <- prep(access_recipe)
baked_access <- bake(prepped_access_recipe, new_data = access_train)
glimpse(baked_access) # Check how many columns there are; should be 112



# Logistic Regression Model -----------------------
logistic_mod <- logistic_reg() |> set_engine("glm")

logistic_amazon_wf <- workflow() |> 
  add_recipe(access_recipe) |> 
  add_model(logistic_mod) |> 
  fit(data = access_train)

logistic_amazon_pred <- predict(logistic_amazon_wf, 
                                new_data = access_test,
                                type = "prob")
logistic_amazon_pred
logistic_amazon_export <- data.frame("id" = 1:length(logistic_amazon_pred$.pred_1),
                                     "Action" = logistic_amazon_pred$.pred_1)


# Write the data
vroom_write(logistic_amazon_export, paste0(base_folder, "logistic.csv"), delim = ",")
