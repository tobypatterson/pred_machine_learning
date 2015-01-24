---
title: "Machine Learning 010 Project Writeup"
author: "Toby Patterson"
date: "January 23, 2015"
output:
  html_document: default
---



## Executive Summary

In this analysis we will try to predict 20 observations based on data from personal fitness devices. Using the _randomForest_ algorithm on a subset of important predictors (31 of total), we can predict the outcome on a training dataset with about 99% accuracy.

## Exploratory Analysis

The data for our analysis has been partitioned into training and test datasets already. Our first step will be to load the data, and then remove any of observations that are not complete.  We'll also remove the first five datapoints which are not device measurements.


```r
training   = read.csv("pml-training.csv", stringsAsFactors=F)
validation = read.csv("pml-testing.csv", stringsAsFactors=F)
completeCasesByVariables = complete.cases(t(training))
completeCasesByVariables[1:7] = F
training   = training[,completeCasesByVariables]
validation = validation[,completeCasesByVariables]
```




The training set now contains 19622 and variables 86 variables, but some of these contain no variation or are covarient with other variables, and thus should be removed. Note that we are also removing the variables from the validation set.


```r
nzvVariables = nearZeroVar(training, uniqueCut=10)
training     = training[,-nzvVariables]
validation   = validation[,-nzvVariables]

M = abs(cor(training[,-dim(training)[2]]))
diag(M) = 0
corVariables = findCorrelation(M, .7)
training   = training[,-corVariables]
validation = validation[,-corVariables]
```



The processed training dataset now contains 19622 observations and variables 31 variables. 

## Data Analysis

We'll treat the provided test set as our _validation_ dataset, and partition the provided training data into our own training and test datasets.


```r
inTrain = createDataPartition(y=training$classe, p=.75, list=F)
trainingSet = training[inTrain,]
testingSet  = training[-inTrain,]
```

Next will train the model using all remaining variables using the _randomForest_ algorithm with the default options, and build a prediction based on our testing dataset.


```r
trainingSet$classe = as.factor(trainingSet$classe)
resultColumn = dim(testingSet)[2]
fittedModel = randomForest(classe ~ ., data = trainingSet)
predictResult = predict(fittedModel, testingSet[,-resultColumn])
modelAccuracy = confusionMatrix(testingSet[,resultColumn], predictResult)
modelAccuracy$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9900082      0.9873586      0.9868115      0.9925991      0.2858891 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

Our model provides a 0.9900082% accuracy, which is pleasantly sufficient, and predicts most variables very well.


```r
correctPredictions = testingSet$classe == predictResult
table(predictResult, testingSet$classe)
```

```
##              
## predictResult    A    B    C    D    E
##             A 1394    7    1    0    0
##             B    1  939   11    0    0
##             C    0    3  841   18    1
##             D    0    0    0  785    4
##             E    0    0    2    1  896
```

Let's have a look at our 5 most important predictors.

```r
variableImportance = varImp(fittedModel)
rownames(variableImportance)[order(variableImportance, decreasing = T)][1:15]
```

```
##  [1] "magnet_dumbbell_z"    "magnet_belt_z"        "pitch_forearm"       
##  [4] "roll_forearm"         "roll_dumbbell"        "gyros_belt_z"        
##  [7] "roll_arm"             "total_accel_dumbbell" "yaw_dumbbell"        
## [10] "gyros_dumbbell_y"     "accel_forearm_x"      "accel_forearm_z"     
## [13] "magnet_forearm_z"     "magnet_arm_x"         "magnet_belt_x"
```
## Conclusion

In this analysis, we showed that it is possible to predict with the outcome of a testing dataset 0.9900082% accuracy using simple machine learning techniques and the _randomForest_ algorithm. The total execution time of this script was under one minute using 19622 observations and 31 variables, demonstrating the usefulness of this easily applied method.

## Appendix

Plot of must important predictors and their cross-validation importance.

```r
varImpPlot(fittedModel, n.var = 10)
```

<img src="figure/unnamed-chunk-10-1.png" title="plot of chunk unnamed-chunk-10" alt="plot of chunk unnamed-chunk-10" style="display: block; margin: auto;" />

Plot the error rate over various outcomes and iterations of the tree.

```r
plot(fittedModel, log="y")
legend("topright", colnames(fittedModel$err.rate),col=1:5,fill=1:5)
```

<img src="figure/unnamed-chunk-11-1.png" title="plot of chunk unnamed-chunk-11" alt="plot of chunk unnamed-chunk-11" style="display: block; margin: auto;" />

Examine how our top three predictors relate to each other

```r
variableImportance = varImp(fittedModel)
tenMostImportantVariables = rownames(variableImportance)[order(variableImportance, decreasing = T)][1:10]

testingSet$classe = as.factor(testingSet$classe)
featurePlot(x=testingSet[, tenMostImportantVariables[1:3]], y=testingSet$classe, plot="pairs")
```

<img src="figure/unnamed-chunk-12-1.png" title="plot of chunk unnamed-chunk-12" alt="plot of chunk unnamed-chunk-12" style="display: block; margin: auto;" />

General information about accuracy, confidence, and error.


```r
modelAccuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    1    0    0    0
##          B    7  939    3    0    0
##          C    1   11  841    0    2
##          D    0    0   18  785    1
##          E    0    0    1    4  896
## 
## Overall Statistics
##                                           
##                Accuracy : 0.99            
##                  95% CI : (0.9868, 0.9926)
##     No Information Rate : 0.2859          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9874          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9943   0.9874   0.9745   0.9949   0.9967
## Specificity            0.9997   0.9975   0.9965   0.9954   0.9988
## Pos Pred Value         0.9993   0.9895   0.9836   0.9764   0.9945
## Neg Pred Value         0.9977   0.9970   0.9946   0.9990   0.9993
## Prevalence             0.2859   0.1939   0.1760   0.1609   0.1833
## Detection Rate         0.2843   0.1915   0.1715   0.1601   0.1827
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9970   0.9924   0.9855   0.9952   0.9977
```

## Reference

Dataset and citations: http://groupware.les.inf.puc-rio.br/har
