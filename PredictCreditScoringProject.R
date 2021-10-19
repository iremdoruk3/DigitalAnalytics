###only code
# install.packages("ggplot2")
# install.packages("Hmisc")
# install.packages("e1071")
# install.packages("dplyr")
# install.packages("rms")
# install.packages("survival")
# install.packages("reprex")
#install.packages("tidyverse")
install.packages("ROCR")
install.packages("randomForest")
install.packages("partykit")
library(partykit)
library(ggplot2)
library(Hmisc)
library(e1071)
library(randomForest)
library(reshape2)
library(survival)
library(rms)
library(tidyverse)
library(reprex)
library(dplyr)
library(caret)
library(ggplot2)
library(lattice)
library(ROCR)
library(pROC)
setwd("/Users/iremdoruk3/Desktop/FS/Lectures/2021-2022/1st semester/Digital Analytics/Project")
credit_data_initial <- read.csv('Creditscoring.csv') 
credit_data_dropped_number <- credit_data_initial[, -1]
credit_data_cleaned_days_past<-credit_data_dropped_number[!(credit_data_dropped_number$NumberOfTime60.89DaysPastDueNotWorse > 95 | credit_data_dropped_number$NumberOfTime30.59DaysPastDueNotWorse > 95 | credit_data_dropped_number$NumberOfTimes90DaysLate > 95),]
credit_data_cleaned_Utilization<-subset( credit_data_cleaned_days_past, credit_data_cleaned_days_past$RevolvingUtilizationOfUnsecuredLines < 13)
credit_data_cleaned_age<-credit_data_cleaned_Utilization[!(credit_data_cleaned_Utilization$age <1),]
credit_data_cleaned_age<-credit_data_cleaned_age[!(credit_data_cleaned_age$age > 99),]
credit_data_cleaned_age<-credit_data_cleaned_age[!(credit_data_cleaned_age$age > 99),]
credit_data_cleaned_DebtRatio<-credit_data_cleaned_age[!(credit_data_cleaned_age$DebtRatio >3490),]
credit_data_cleaned_DebtRatio$NumberOfDependents[is.na(credit_data_cleaned_DebtRatio$NumberOfDependents)] <- 0
credit_data_cleaned_DebtRatio$MonthlyIncome[is.na(credit_data_cleaned_DebtRatio$MonthlyIncome)] <- median(credit_data_cleaned_DebtRatio$MonthlyIncome, na.rm=TRUE)
credit_data_cleaned <- credit_data_cleaned_DebtRatio
credit_data_cleaned_deskewed <- sqrt(credit_data_cleaned)
credit_data_cleaned_deskewed['IncomePerPerson'] = credit_data_cleaned_deskewed['MonthlyIncome']/(credit_data_cleaned_deskewed['NumberOfDependents']+1)
credit_data_cleaned_deskewed['MonthlyDebt'] = credit_data_cleaned_deskewed['DebtRatio']*credit_data_cleaned_deskewed['MonthlyIncome'] 
credit_data_cleaned_deskewed['NumOfOpenCreditLines'] = credit_data_cleaned_deskewed['NumberOfOpenCreditLinesAndLoans']-credit_data_cleaned_deskewed['NumberRealEstateLoansOrLines']
credit_data_cleaned_deskewed$SeriousDlqin2yrs <- as.factor(credit_data_cleaned_deskewed$SeriousDlqin2yrs)
levels(credit_data_cleaned_deskewed$SeriousDlqin2yrs)
credit_data_cleaned_deskewed$SeriousDlqin2yrs<- relevel(credit_data_cleaned_deskewed$SeriousDlqin2yrs, ref = "1")

#partition
set.seed(1)
index <- createDataPartition(credit_data_cleaned_deskewed$SeriousDlqin2yrs, p=0.8, list=FALSE)
training <- credit_data_cleaned_deskewed[index,]
test<- credit_data_cleaned_deskewed[-index, ]
TControl <- trainControl(method="cv", number=10)
report <- data.frame(Model=character(), Acc.Train=numeric(), Acc.Test=numeric()) 

#knn
#Checking if knn is suitable
set.seed(1)
library(class)
knnmodel <- knn(train=training[,-1], test=test[,-1], cl=training$SeriousDlqin2yrs, k=11)
knnmodel
summary(knnmodel) 
num.correct.labels <- sum(knnmodel == test$SeriousDlqin2yrs)
accuracy <- num.correct.labels / length(test$SeriousDlqin2yrs)
accuracy 

# best model is k=11
knnmodel_tr <- knn(train=training[,-1], test=training[,-1], cl=training$SeriousDlqin2yrs, k=11)
knnmodel <- knn(train=training[,-1], test=test[,-1], cl=training$SeriousDlqin2yrs, k=11, prob=TRUE)

set.seed(1)
knnmodel <- train(SeriousDlqin2yrs~., data=training, method="knn", trControl=TControl)
knnmodel 
prediction.train <- predict(knnmodel, training[,-1],type="raw")
prediction.test <- predict(knnmodel, test[,-1],type="raw")
acctr <- confusionMatrix(prediction.train, training[,1])
acctr$table 
acctr$overall['Accuracy'] 
accte <- confusionMatrix(prediction.test, test[,1]) 
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="k-NN", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

precision <- posPredValue(prediction.test, test$SeriousDlqin2yrs, positive="1")
recall <- sensitivity(prediction.test, test$SeriousDlqin2yrs, positive="1")
F1 <- (2 * precision * recall) / (precision + recall) 

pr<- prediction(as.numeric(prediction.test), as.numeric(test$SeriousDlqin2yrs))
prf<- performance(pr, "tpr", "fpr")
prf2<- performance(pr, "auc")
plot1<- plot(prf, colorize=TRUE, print.cutoffs.at=seq(0,3, by=0.3), text.adj=c(0, 3))

roc_obj<- roc(prediction.test, as.numeric(test$SeriousDlqin2yrs)) 
auc(roc_obj)
plot(roc_obj) 

##random forest
set.seed(1)
rformodel <- randomForest(SeriousDlqin2yrs ~., data=training, method="rf", trControl=TControl)
rformodel
prediction.train <- predict(rformodel, training[,-1], type="response")
prediction.test <- predict(rformodel, test[,-1], type="response")
acctr <- confusionMatrix(prediction.train, training[,1])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,1])
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="Random Forest", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

precision <- posPredValue(prediction.test, test$SeriousDlqin2yrs, positive="1")
recall <- sensitivity(prediction.test, test$SeriousDlqin2yrs, positive="1")
F1 <- (2 * precision * recall) / (precision + recall) 

pr<- prediction(as.numeric(prediction.test), as.numeric(test$SeriousDlqin2yrs))
prf<- performance(pr, "tpr", "fpr")
prf2<- performance(pr, "auc")
plot1<- plot(prf, colorize=TRUE, print.cutoffs.at=seq(0,3, by=0.3), text.adj=c(0, 3))

roc_obj<- roc(prediction.test, as.numeric(test$SeriousDlqin2yrs)) 
auc(roc_obj)
varImpPlot(rformodel, type=2)

train2<- subset(training, select=c(RevolvingUtilizationOfUnsecuredLines, SeriousDlqin2yrs, DebtRatio))
ct<- ctree(SeriousDlqin2yrs~., data=train2)
plot(ct, type="simple")

##logistic regression
set.seed(1)
lrmodel <- train(SeriousDlqin2yrs~., data=training, method="glm", trControl=TControl)
lrmodel
prediction.train <- predict(lrmodel, training[,-1], type="raw")
prediction.test <- predict(lrmodel, test[,-1], type="raw")
acctr <- confusionMatrix(prediction.train, training[,1])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,1])
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="Logistic Regression", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

precision <- posPredValue(prediction.test, test$SeriousDlqin2yrs, positive="1")
recall <- sensitivity(prediction.test, test$SeriousDlqin2yrs, positive="1")
F1 <- (2 * precision * recall) / (precision + recall) 

pr<- prediction(as.numeric(prediction.test), as.numeric(test$SeriousDlqin2yrs))
prf<- performance(pr, "tpr", "fpr")
prf2<- performance(pr, "auc")
plot1<- plot(prf, colorize=TRUE, print.cutoffs.at=seq(0,3, by=0.3), text.adj=c(0, 3))

roc_obj<- roc(prediction.test, as.numeric(test$SeriousDlqin2yrs)) 
auc(roc_obj, equals = smooth)
