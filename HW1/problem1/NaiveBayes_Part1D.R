# Time: 09/03/2018
# Author: Yuan Gao
library(klaR)
data_split <- function(all_data,p){
  #this function is used to split dataset to train dataset and test dataset
  #guarenttee the balance of distribution of Y in each train set and test set
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
  
  x<-svmlight(Outcome ~ ., data=train)
  pred <- predict(x,test)
  pred_class <- as.numeric(as.character(pred$class))# this is a vector
  pred_res <- data.frame(y_true=test$Outcome,
                         y_pred=pred_class)
  
  # performance evaluation
  acc <- performance_eval(pred_res)
  return(acc)
}

main <- function(n_run){
  
  # load data set
  root_path <- "D:\\ѧϰ\\Graduate\\applied machine learning\\HW1\\problem1"
  file_path = paste0(root_path,"\\diabetes.csv")
  all_data <- read.csv(file_path,header = TRUE)
  single_run(all_data)
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
