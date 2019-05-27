#' A blueconic Function
#'
#' This function allows you to predict prodcut type using blueconic data.
#' @param data formatted as Blueconic data from Hackthon 2019
#' @keywords blueconic
#' @export
#' @examples
#' blueconic()
blueconic <- function(input){
  list.of.packages <- c("dplyr", "xgboost", "caret")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)
  library(dplyr)

#Drop columns with more than 10% of missing values
#-----------------
   #input can either be csv file or data	
  df <- if(is.character(input) && file.exists(input)){
    read.csv(input)
  } else {
    as.data.frame(input)
  }
  df = df[,!sapply(df, function(x) mean(is.na(x)))>0.5]
#Remove useless or repetitive variables
drop <- c("profileid","aov_restoretokenurl","aov_token","entrypage","gclid",
          "interactions_viewed","mr_geo_city_name","geo_subdivision_1_name","geo_subdivision_2_name",
          "browserversion","currentresolution","geo_lat","geo_long","inkomen")
df <- df[,!(names(df) %in% drop)]
# Add NA as a factor
addNA(df)

#Variable selection
#-----------------
keep <- c("adblock_detected","aov_care_score","aov_do_score","aov_see_score",
          "aov_think_score","browsername","churn_risk","clickcount","engagement_score",
          "frequency","funnel_busy_with","funnel_laststep","recency","prospect_status",
          "product_category_interest",
          "recent_intensity")
df <- df[,(names(df) %in% keep)]

#Convert certain columns to numeric
#-----------------
gsub("€ ", "", df$income, ignore.case = TRUE)
tmp <- c("engagement_score")
df[tmp] <- lapply(df[tmp], as.numeric)

#Convert certain columns' date format
#-----------------
df$year <- format(as.Date(df$dummy_date_of_birth, format="%Y-%m-%d"),"%Y")
df$year <- as.numeric(df$year)
df <- select(df, -matches("dummy_date_of_birth"))

# Create target variable
# -----------------
df$target <- as.factor(df$funnel_laststep)
df <- subset(df, select=-c(funnel_laststep))
df$target <- as.numeric(df$target)
levels(hr_data$left) <- c("stayed", "left")

save(df, file = "df.Rdata")
load(file = "df.Rdata")
#===============================================================
# Build predictive model
#===============================================================
load("df.Rdata")
pTraining     <- 0.8
nTraining     <- round(pTraining*nrow(df))
obsTraining <- sample(1:nrow(df), nTraining, 
                      replace=FALSE)
dfTrain <- df[obsTraining, ]
dfTest  <- df[-obsTraining, ]


# -----------------
# https://rpubs.com/zxs107020/368478
library("xgboost")
library(caret)
# Create a training and validation sets

trainObs <- sample(nrow(dfTrain), .8 * nrow(dfTrain), replace = FALSE)
valObs <- sample(nrow(dfTrain), .2 * nrow(dfTrain), replace = FALSE)

train_dat <- dfTrain[trainObs,]
val_dat <- dfTrain[valObs,]

# Create numeric labels with one-hot encoding
train_labs <- as.numeric(train_dat$target)-1 
val_labs <- as.numeric(val_dat$target)-1

new_train <- model.matrix(~. + 0, data = train_dat[, -16])
new_val <- model.matrix(~. + 0, data = dfTrain[valObs, -16])

# Prepare matrices
xgb_train <- xgb.DMatrix(data = new_train, label = train_labs)
xgb_val <- xgb.DMatrix(data = new_val, label = val_labs)

# Set parameters(default)
params <- list(booster = "gbtree", objective = "multi:softprob", num_class = 11, eval_metric = "mlogloss")

# Calculate # of folds for cross-validation
xgbcv <- xgb.cv(params = params, data = xgb_train, nrounds = 100, nfold = 5, showsd = TRUE, stratified = TRUE, print.every.n = 10, early_stop_round = 20, maximize = FALSE, prediction = TRUE)


# -----------------
# Function to compute classification error
classification_error <- function(conf_mat) {
  conf_mat = as.matrix(conf_mat)
  error = 1 - sum(diag(conf_mat)) / sum(conf_mat)
  return (error)
}

# Mutate xgb output to deliver hard predictions
library(dplyr)
xgb_train_preds <- data.frame(xgbcv$pred) %>% mutate(max = max.col(., ties.method = "last"), label = train_labs + 1)

# Examine output
head(xgb_train_preds)

# -----------------
# Confustion Matrix
xgb_conf_mat <- table(true = train_labs + 1, pred = xgb_train_preds$max)

# Error 
cat("XGB Training Classification Error Rate:", classification_error(xgb_conf_mat), "\n")


# Automated confusion matrix using "caret"
xgb_conf_mat_2 <- confusionMatrix(factor(xgb_train_preds$label),
                                  factor(xgb_train_preds$max),
                                  mode = "everything")

print(xgb_conf_mat_2)
return(xgb_train_preds, error)
}


