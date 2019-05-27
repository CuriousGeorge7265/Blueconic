#' A blueconic Function
#'
#' This function allows you to express your love of cats.
#' @param data formatted as Blueconic data from Hackthon 2019
#' @keywords blueconic
#' @export
#' @examples
#' blueconic()
blueconic <- function(df){
  list.of.packages <- c("dplyr", "xgboost", "caret")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)
  library(dplyr)
  
  #Drop columns with more than 10% of missing values
  #-----------------
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
  newdata <- df
  
    #blueconic_model is included with the package
  # Predict 
  library("xgboost")
  library(caret)
    newdata$prediction <- as.vector(predict(blueconic_model, newdata))
    return(newdata)
  }

  
