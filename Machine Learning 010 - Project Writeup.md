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
##      0.9902121      0.9876162      0.9870433      0.9927746      0.2862969 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

Our model provides a 0.9902121% accuracy, which is pleasantly sufficient, and predicts most variables very well.


```r
correctPredictions = testingSet$classe == predictResult
table(predictResult, testingSet$classe)
```

```
##              
## predictResult    A    B    C    D    E
##             A 1395    9    0    0    0
##             B    0  934    9    0    0
##             C    0    4  843   18    0
##             D    0    0    2  784    1
##             E    0    2    1    2  900
```

Let's have a look at our 5 most important predictors.

```r
variableImportance = varImp(fittedModel)
rownames(variableImportance)[order(variableImportance, decreasing = T)][1:15]
```

```
##  [1] "magnet_dumbbell_z"    "pitch_forearm"        "magnet_belt_z"       
##  [4] "roll_dumbbell"        "roll_forearm"         "gyros_belt_z"        
##  [7] "roll_arm"             "total_accel_dumbbell" "gyros_dumbbell_y"    
## [10] "yaw_dumbbell"         "magnet_forearm_z"     "magnet_belt_x"       
## [13] "accel_forearm_x"      "magnet_arm_x"         "accel_forearm_z"
```
## Conclusion

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
##          A 1395    0    0    0    0
##          B    9  934    4    0    2
##          C    0    9  843    2    1
##          D    0    0   18  784    2
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9902         
##                  95% CI : (0.987, 0.9928)
##     No Information Rate : 0.2863         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9876         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9936   0.9905   0.9746   0.9962   0.9945
## Specificity            1.0000   0.9962   0.9970   0.9951   0.9997
## Pos Pred Value         1.0000   0.9842   0.9860   0.9751   0.9989
## Neg Pred Value         0.9974   0.9977   0.9946   0.9993   0.9988
## Prevalence             0.2863   0.1923   0.1764   0.1605   0.1845
## Detection Rate         0.2845   0.1905   0.1719   0.1599   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9968   0.9933   0.9858   0.9957   0.9971
```

## Reference

Dataset and citations: http://groupware.les.inf.puc-rio.br/har
