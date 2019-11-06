# Time: 09/02/2018
# Author: Yuan Gao
# This script is an advanced version to "NaiveBayes_Part1A.R"
# added function of dealling with missing value

library(plyr)

data_prepro <- function(all_data){
  # This function is used to replace 0 values in attribute
  # 3,4,6,8 to NA 
  # all_data: data.frame
  # return a dataframe with same dim
  broken_col <- c(3,4,6,8)
  all_data[,broken_col][all_data[,broken_col] == 0] <- NA
  return(all_data)
}

param_estimate <- function(data,y){
  #this function is used to estimate the distribution of x given y
  #default distribution: normal distribution
  #data: matrix, dim(data) == n*2
  #y: integer
  #return a list of "avg" and "sigma"
  avg <- mean(data[data[,2]==y,1],na.rm = TRUE)
  variance <- var(data[data[,2]==y,1],na.rm = TRUE)
  para <- list(mean=avg,variance=variance)
  return(para)
}

freq_cal <- function(input){
  # this function is used to calculate the frequency of 
  # each unique element in a vector
  # input: a n*1 vector
  # return a dataframe contains frequency of each unique element
  res <- count(input)
  return(res)
}

data_split <- function(all_data,p){
  #this function is used to split dataset to train dataset and test dataset
  #data: original data, which is dataframe
  #p: proportion of test dataset, which is double
  #return a list contain "train "and "test"
  
  n_rows <- dim(all_data)[1] - 1
  n_test <- floor(n_rows * p)
  index_test <- sample(1:n_rows,n_test)
  index_train <- setdiff(1:n_rows,index_test)
  test_data <- all_data[index_test,]
  train_data <- all_data[index_train,]
  data_splited <- list(train=train_data,test=test_data)
  return(data_splited)
}

model_building <- function(train,y_stat){
  y_cate <- y_stat$x # category of y
  y_freq <- y_stat$freq # frequency corresponding to each category
  
  param_given_y <- list()
  for(y in y_cate){
    param <- list()
    for(i in 2:dim(train)[2]-1){
      x_i <- cbind(train[,i],train[,ncol(train)])
      norm_para <- param_estimate(x_i,y)
      param <- cbind(param,norm_para)
    }
    param_given_y[[toString(y)]] <- param
  }
  return(param_given_y)
}

model_pred <- function(test,param_given_y,y_freq){
  n_features = ncol(test) - 1
  x_test <- test[,1:n_features]
  pred_res <- data.frame(y_true=integer(),
                         y_pred=double())
  for(n in 1:dim(x_test)[1]){
    temp_prob <- c()
    
    #caculate prob for each record in test set
    for(y in names(param_given_y)){
      y_index <- 1
      prob <- 0
      for(x in 1:n_features){
        if(is.na(x_test[n,x])){
          # jump features with data missing
          next
        }
        avg <- as.numeric(param_given_y[[y]][1,x])
        sigma <- sqrt(as.numeric(param_given_y[[y]][2,x]))
        prob <- prob + log(pnorm(x_test[n,x],mean=avg,sd=sigma))
      }
      prob <- prob + log(y_freq[y_index])
      y_index <- y_index + 1
      temp_prob <- c(temp_prob,prob)
    }
    pred <- names(param_given_y)[which.max(temp_prob)]
    pred_res[n,1] <- test$Outcome[n]
    pred_res[n,2] <- as.numeric(pred)
  }
  return(pred_res)
}

performance_eval <- function(pred_res){
  # accuracy
  n_right <- dim(pred_res[pred_res$y_true == pred_res$y_pred,])[1]
  acc <- n_right/dim(pred_res)[1]
  print(paste0("accuracy is :",acc))
  # confusion_matrix
  print("=====Confusion Matrix=====")
  confusion_mat <- matrix(0,2,2)
  confusion_mat[1,1] <- nrow(pred_res[pred_res$y_true == 1 & pred_res$y_pred==1,])
  confusion_mat[1,2] <- nrow(pred_res[pred_res$y_true == 0 & pred_res$y_pred==1,])
  confusion_mat[2,1] <- nrow(pred_res[pred_res$y_true == 1 & pred_res$y_pred==0,])
  confusion_mat[2,2] <- nrow(pred_res[pred_res$y_true == 0 & pred_res$y_pred==0,])
  confusion_mat <- data.frame(confusion_mat,row.names = c("1","0"))
  colnames(confusion_mat) <- c("1", "0")
  print(confusion_mat)
  
  return(acc)
}

single_run <- function(all_data){
  p <- 0.2 #proportion of test dataset
  
  #split data to train set and test set
  data_splited <- data_split(all_data,p)
  train <- data_splited$train  # class(train) == "data.frame"
  test <- data_splited$test  # class(test) == "data.frame"
  
  # count frequency of y
  y_stat <- freq_cal(train$Outcome)
  y_cate <- y_stat$x # category of y
  y_freq <- y_stat$freq # frequency corresponding to each category
  y_freq <- y_freq/sum(y_freq)
  # model training and parameter estimation
  param_given_y <- model_building(train,y_stat)
  
  # make prediction in test dataset
  pred_res <- model_pred(test,param_given_y,y_freq)
  
  # performance evaluation
  acc <- performance_eval(pred_res)
}

main <- function(n_run){
  
  # load data set
  root_path <- "D:\\ѧϰ\\Graduate\\applied machine learning\\HW1\\problem1"
  file_path = paste0(root_path,"\\diabetes.csv")
  all_data <- read.csv(file_path,header = TRUE)
  all_data <- data_prepro(all_data)# data preprocessing
  
  acc <- vector(mode="numeric", length=n_run)
  for(i in 1:n_run){
    print(paste("run times:",i))
    acc[i] <- single_run(all_data)
  }
  print(paste("average prediction accuracy is",mean(acc)))
  plot(acc)
}

n_run <- 10
main(n_run)