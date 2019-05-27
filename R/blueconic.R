#' A blueconic Function
#'
#' This function allows you to express your love of cats.
#' @param data formatted as Blueconic data from Hackthon 2019
#' @keywords blueconic
#' @export
#' @examples
#' blueconic()

blueconic <- function(df){
#==========================
# Feature Selection 
#==========================

keep <- c("adblock_detected","aov_care_score","aov_do_score","aov_see_score",
          "aov_think_score","browsername","churn_risk","clickcount","engagement_score",
          "frequency","funnel_busy_with","funnel_laststep","recency","prospect_status",
          "product_category_interest",
          "recent_intensity")
df <- df[,(names(df) %in% keep)]
tmp <- c("engagement_score")
df[tmp] <- lapply(df[tmp], as.numeric)

#============================
# Create target variable
#============================
df$target <- as.factor(df$funnel_laststep)
df <- subset(df, select=-c(funnel_laststep))


#=============================
# Model 
#=============================
library(caret)
library(xgboost)
df2 <- df
df2 <- df2[complete.cases(df2),]
toBeRemoved <- c("AOV - 4 Aanvragen (deel B)","DBLG - 0 Start","DBLG - 1 Uw gegevens",
                 "DBLG - 2 Pensioenuitkering","ORV - 2 Uw aanvraag","ORV - 3 Slotvragen",
                 "ORV - 4 Aanvraag verzonden","SME - 2 Offerte samenstellen","SME - 3 Gegevens",
                 "SME - 4 Overzicht","SME - 5 Offerte aanvraag verzonden","360x640")
df2 <- df2[!df2$target %in% toBeRemoved, ]
df2 <- droplevels.data.frame(df2)

df <- df2

library("xgboost")
# Create a training and validation sets
trainObs <- sample(nrow(dfTrain), .8 * nrow(dfTrain), replace = FALSE)
valObs <- sample(nrow(dfTrain), .2 * nrow(dfTrain), replace = FALSE)

train_dat <- dfTrain[trainObs,]
val_dat <- dfTrain[valObs,]

# Create numeric labels with one-hot encoding
train_labs <- as.numeric(train_dat$target)-1 
val_labs <- as.numeric(val_dat$target)-1

new_train <- model.matrix(~ . + 0, data = train_dat[, -16])
new_val <- model.matrix(~ . + 0, data = dfTrain[valObs, -16])

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


# -----------------
# Create the model
xgb_model <- xgb.train(params = params, data = xgb_train, nrounds = 100)

# Predict for validation set
xgb_val_preds <- predict(xgb_model, xgb_val)


xgb_val_out <- matrix(xgb_val_preds, nrow = 11, ncol = length(xgb_val_preds) / 11) %>% 
  t() %>%
  data.frame() %>%
  mutate(max = max.col(., ties.method = "last"), label = val_labs + 1) 

# -----------------
# Confustion Matrix
xgb_val_conf <- table(true = val_labs + 1, pred = xgb_val_out$max)

cat("XGB Validation Classification Error Rate:", classification_error(xgb_val_conf), "\n")

# Automated confusion matrix using "caret"
xgb_val_conf2 <- confusionMatrix(factor(xgb_val_out$label),
                                 factor(xgb_val_out$max),
                                 mode = "everything")

print(xgb_val_conf2)
return(xgb_val_preds)
}


  
