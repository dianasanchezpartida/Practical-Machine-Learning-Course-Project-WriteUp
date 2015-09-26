# Practical-Machine-Learning-Course-Project-WriteUp
This is an Course Project Writeup for the Practical Machine Learning Course
Created by Diana SP
September 26th, 2015 Puebla, Mexico

Instructions:

Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 



Data 


The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 

Reproducibility 

Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates. Please be sure that if they download the repo, they will be able to view the compiled HTML version of your analysis. 
#Practical Machine Learning Class - Project Write-up

#In this project write-up, I have used the data from Human Activity Recognition (HAR). 
#The aim was to train a model based on the data of various sensor values,
#which could later be used to predict the Classe variable, 
#that is the manner in which the participants of HAR did the exercise.

#After having examined the data briefly using the Rattle GUI, 
#I have realized that some columns have a lot of missing (NA) values. 
#Instead of trying to model them, I have decided to remove them from the data set. 
#So the first step, after having loaded the required caret library 
#(I've skipped the demonstration of Rattle GUI, since, after all, it was an interactive session with GUI part),
#was to detect and eliminate columns with a lot of missing values:
  
#Loading the library
library(caret)

# Load the training data set
trainingAll <- read.csv("pml-training.csv",na.strings=c("NA",""))
str(trainingAll)
#'data.frame':	19622 obs. of  160 variables:
#$ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
#$ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
#$ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
#$ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
#$ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
#$ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
#$ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
# So on...

# Remove columns with NA´s
NAs <- apply(trainingAll, 2, function(x) { sum(is.na(x)) })
trainingValid <- trainingAll[, which(NAs == 0)]
str(trainingValid)
#data.frame':	19622 obs. of  60 variables:
#$ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
#$ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
#$ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
#$ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
#$ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
#$ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
#$ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
#$ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
#so on...

#I select a probability of data training in 60%, so the test data will be 40%, 
#because are typical sizes for the training and test sets.
# Create a subset of trainingValid data set
trainAllPredictors <- createDataPartition(y = trainingValid$classe, p=0.6,list=FALSE)
trainData <- trainingValid[trainAllPredictors,]

# Remove useless predictors  like timestamps, the X column, user_name, and new_window
removeUselessPre <- grep("timestamp|X|user_name|new_window", names(trainData))
trainData <- trainData[, -removeUselessPre]
str(trainData)
#'data.frame':	11776 obs. of  54 variables:
#$ num_window          : int  11 11 12 12 12 12 12 12 12 12 ...
#$ roll_belt           : num  1.41 1.42 1.48 1.48 1.42 1.42 1.43 1.45 1.45 1.42 ...
#$ pitch_belt          : num  8.07 8.07 8.05 8.07 8.09 8.13 8.16 8.17 8.18 8.21 ...
#$ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
#$ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
#So on...

#Based on the suggestion of the slides I use 5-fold cross validation 
#in order to have smaller K = more bias, less variance
# Configure the train control for cross-validation
tc = trainControl(method = "cv", number = 5)

# Fit the model using Random Forests algorithm
modelFit <- train(trainData$classe ~.,
                data = trainData,
                method="rf",
                trControl = tc,
                prox = TRUE,
                allowParallel = TRUE)
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.

#I expected relatively good model performance, and a relatively low out of sample error rate:
print(modelFit)
## Random Forest
##
## 11776 samples
##   53 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E'
##
## No pre-processing
## Resampling: Cross-Validated (5 fold)
##
## Summary of sample sizes: 9421, 9420, 9422, 9421, 9420 
##
## Resampling results across tuning parameters:
##
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2    0.9921027  0.9900095  0.0026272210  0.003323988
##  27    0.9968580  0.9960257  0.0008276806  0.001046769
##  53    0.9944802  0.9930178  0.0012389239  0.001566650
##
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
print(modelFit$finalModel)
##
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE,      allowParallel = TRUE)
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
##
##         OOB estimate of  error rate: 1.55%
## Confusion matrix:
## A    B    C    D    E  class.error
## A 3347    0    0    0    1 0.0002986858
## B   10 2266    3    0    0 0.0057042563
## C    0    6 2048    0    0 0.0029211295
## D    0    0    6 1924    0 0.0031088083
## E    0    0    0    8 2157 0.0036951501

#After having fit the model with training data, 
#I have used it for predictions on test data. 
#I've applied the same removal of columns to the testing data as I have done for the training data set:
# Loading test data
testingAll = read.csv("pml-testing.csv",na.strings=c("NA",""))
# Only take the columns of testingAll that are also in trainData
testing <- testingAll[ , which(names(testingAll) %in% names(trainData))]

# Running the prediction
prediction <- predict(modelFit, newdata = testing)
print(prediction)
#[1] B A B A A E D B A A B C B A E E A B B B

# Utility function provided by the instructor
#Function to generate files with predictions to submit for assignment
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(prediction)
