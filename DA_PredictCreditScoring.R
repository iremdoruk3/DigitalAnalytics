###############################################################
############## PROJECT - PREDICT CREDIT SCORING ###############

#Install packages and load libraries
install.packages("ggplot2")
install.packages("Hmisc")
install.packages("e1071")
install.packages("dplyr")
install.packages("rms")
install.packages("survival")
install.packages("reprex")
install.packages("tidyverse")
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

#Set working directory
setwd("/Users/iremdoruk3/Desktop/FS/Lectures/2021-2022/1st semester/Digital Analytics/Project")

# 1. Read csv
credit_data_initial <- read.csv('Creditscoring.csv') 

# 1.1 Exploratory Data Analysis
head(credit_data_initial, 10) 
dim(credit_data_initial) 
str(credit_data_initial) 
summary(credit_data_initial)  

# 2.Visualize and fix the data feature by feature

# 2.0 The output variable 

summary(credit_data_initial$SeriousDlqin2yrs)
table(credit_data_initial$SeriousDlqin2yrs) 

# The dataset is extremely disbalanced 1: 10026 (6.684%) 0 : 139974 (93.316%)
# This must be in our mind all the time while cleaning !!! We drop/impute ... only if it does not further
# destroy the balance or if we see that the rows with 1 are actually fake rows (damaged data).

# 2.1. Number Feature

summary(credit_data_initial$Number)

aa = data.frame(feature = "Number", value = c(credit_data_initial$Number))

ggplot(aa, aes(x=feature, y=value, fill=feature)) +
  geom_boxplot()+
  theme(panel.background = element_rect(fill = "white"))+ 
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) 

# We see that this is a simple numerator feature. Since R already creates an index doing the same in a dataframe
# We can just drop this feature. It has no predictive qualities.

credit_data_dropped_number <- credit_data_initial[, -1]

# 2.2 NumberOfTimes90DaysLate, NumberOfTime60-89DaysPastDueNotWorse, NumberOfTime30-59DaysPastDueNotWorse

#They all have max value as 98. Let's visualize and see further. 

summary(credit_data_dropped_number$NumberOfTimes90DaysLate) 
summary(credit_data_dropped_number$NumberOfTime30.59DaysPastDueNotWorse)
summary(credit_data_dropped_number$NumberOfTime60.89DaysPastDueNotWorse)

# Immediately from the summary we see issues. Based on the data there are people late 98 times in all categories
# This is impossible. Even if we take the shortest time period of 30 days and multiply with 98 times, this would mean
# that in the period of 2 years the person was late 2940 days. 2940 days is more than 8 years. Hence these features
# need to be cleaned of outliers. We perform boxplots to see better how many of these rows are there and their nature.

a = data.frame(numofdayslate = "30-59", value = c(credit_data_dropped_number$NumberOfTime30.59DaysPastDueNotWorse))
b = data.frame(numofdayslate = "60-89", value = c(credit_data_dropped_number$NumberOfTime60.89DaysPastDueNotWorse))
c = data.frame(numofdayslate = "90", value = c(credit_data_dropped_number$NumberOfTimes90DaysLate))
# This function will bind or join the rows. See plot below
plot.data = rbind(a,b,c) 

ggplot(plot.data, aes(x=numofdayslate, y=value, fill=numofdayslate))+
  geom_boxplot()+
  theme(panel.background = element_rect(fill = "white"))+ 
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) 

# We see a lot of issues with these 3 features. They do not have any values between circa 20 and circa 95.
# They have all outliers however at 96 and 98. We drop these outliers from all three features.

credit_data_cleaned_days_past<-credit_data_dropped_number[!(credit_data_dropped_number$NumberOfTime60.89DaysPastDueNotWorse > 95 | credit_data_dropped_number$NumberOfTime30.59DaysPastDueNotWorse > 95 | credit_data_dropped_number$NumberOfTimes90DaysLate > 95),]

# We lost in total 269 rows which is not bad at all. Just checking quickly did we disbalance the dataset additionally.

summary(credit_data_cleaned_days_past$SeriousDlqin2yrs)

# The proportion is now 6.598% so we made the dataset 0,0086% more disbalanced but we dropped 269 rows that made no sense.

#2.3 RevolvingUtilizationOfUnsecuredLines

# Defined as ratio of the total amount of money owed to total credit limit.

d = data.frame(feature = "RevolvingUtilizationOfUnsecuredLines", value = c(credit_data_cleaned_days_past$RevolvingUtilizationOfUnsecuredLines))
ggplot(d, aes(x=feature, y=value, fill=feature)) +
  geom_boxplot()+
  theme(panel.background = element_rect(fill = "white"))+ 
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) 

# The boxplot shows outliers but widely spread. We need to pick an exact point where we will cut.
# We will do this based on the proportion of SeriousDlqin2yrs. We do not want to lose a huge proportion of 1s.

length(which(credit_data_cleaned_days_past$SeriousDlqin2yrs == 1 & credit_data_cleaned_days_past$RevolvingUtilizationOfUnsecuredLines > 13 )) #14

# We see there are 14 defaulters where the RevolvingUtilization > 13 How many rows are there in total

length(which(credit_data_cleaned_days_past$RevolvingUtilizationOfUnsecuredLines > 13 )) #238

# There are 238 samples in total. This means that the proportion of defaulters in this group is 5.58%.
# The proportion is actually lower than in the whole population. This means we can cut outliers above this point.
# Already from 10 we are cutting too many rows. From 17 we are missing 1 datapoint so 13 seems to be the perfect point.

#We drop RevolvingUtilization larger than 13.

credit_data_cleaned_Utilization<-subset( credit_data_cleaned_days_past, credit_data_cleaned_days_past$RevolvingUtilizationOfUnsecuredLines < 13)

summary(credit_data_cleaned_days_past$SeriousDlqin2yrs)

#2.4 Age

e = data.frame(Feature = "Age", value = c(credit_data_cleaned_Utilization$age))
ggplot(e, aes(x=Feature, y=value, fill=Feature)) +
  geom_boxplot()+
  theme(panel.background = element_rect(fill = "white"))+ 
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) 

# There is one definetely eronous entry age = 0 which we will def. clean, but there are also ages above 100. 
# We apply the same logic as before, what is the default % in this group. Will we disbalance the dataframe further by 
# dropping them.

length(which(credit_data_cleaned_Utilization$SeriousDlqin2yrs == 1 & credit_data_cleaned_Utilization$age > 99 )) #1
length(which(credit_data_cleaned_Utilization$age > 99 ))  #13

# There are 13 entries and 1 is a defaulter, around 7.11% . What about young people?

length(which(credit_data_cleaned_Utilization$SeriousDlqin2yrs == 1 & credit_data_cleaned_Utilization$age < 30 )) #972
length(which(credit_data_cleaned_Utilization$age < 30 )) #8666

# There are 972 rows that default younger than 30. All in total 8666 people. This is almost 12%!
# We see that younger people are proner to default than older(specially the super old). This makes us feel comfortable
# about dropping the 0 and the very old (>99)

#We drop age equal to 0 and larger than 99.

credit_data_cleaned_age<-credit_data_cleaned_Utilization[!(credit_data_cleaned_Utilization$age <1),]
credit_data_cleaned_age<-credit_data_cleaned_age[!(credit_data_cleaned_age$age > 99),]

summary(credit_data_cleaned_days_past$SeriousDlqin2yrs)

#2.5  DebtRatio

f = data.frame(feature = "DebtRatio", value = c(credit_data_cleaned_age$DebtRatio))
ggplot(f, aes(x=feature, y=value, fill=feature)) +
  geom_boxplot()+
  theme(panel.background = element_rect(fill = "white"))+ 
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) 

length(which(credit_data_cleaned_age$DebtRatio > 3490 )) #3736
# Massive amounts of outliers again, are these real data rows or not we will need to check.
# There are a whooping 3736 entries that have debt ratio >3490 ! We need to check are these "normal" rows.
# We will check it with the usual procedure, by checking the % of 1s in this group but also we will check are these rows
# the same rows that have N/A in the monthly income, because they look suspicious.

length(which(credit_data_cleaned_age$DebtRatio > 3490 & credit_data_cleaned_age$MonthlyIncome > 0)) #12
# Only 12 rows have any income given out of the 3736 !! Immediately we suspect these rows even more

length(which(credit_data_cleaned_age$SeriousDlqin2yrs == 1 & credit_data_cleaned_age$DebtRatio > 3490)) #240
# The proportion of defaulters in these extremely suspicious rows is the population average, 6.68%. We can just drop
# these rows because they are highly suspicious and do not contribute in their balance to the dataset further.

#We drop DebtRatio larger than 3490.

credit_data_cleaned_DebtRatio<-credit_data_cleaned_age[!(credit_data_cleaned_age$DebtRatio >3490),]

summary(credit_data_cleaned_days_past$SeriousDlqin2yrs)

# 2.6 NumberOfOpenCreditLinesandLones and NumberofRealestateLoansandLines

g = data.frame(feature = "Number of Open Credit Loans/Lines", value = c(credit_data_cleaned_DebtRatio$NumberOfOpenCreditLinesAndLoans))
h = data.frame(feature = "Number of Real Estate Loans/Lines", value = c(credit_data_cleaned_DebtRatio$NumberRealEstateLoansOrLines))
plot.data2 = rbind(g,h) 

ggplot(plot.data2, aes(x=feature, y=value, fill=feature)) +
  geom_boxplot()+
  theme(panel.background = element_rect(fill = "white"))+ 
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) 

# Again we see outliers but much less pronounced than in other features.
# Let's observe the usual statistics about these potential outliers, specially do they have income and proportion of 1s
# in output variable.

length(which(credit_data_cleaned_DebtRatio$NumberOfOpenCreditLinesAndLoans > 20 )) #3728 
length(which(credit_data_cleaned_DebtRatio$NumberOfOpenCreditLinesAndLoans > 20 &  credit_data_cleaned_DebtRatio$MonthlyIncome > 1)) # 3444 
length(which(credit_data_cleaned_DebtRatio$NumberOfOpenCreditLinesAndLoans > 20 &  credit_data_cleaned_DebtRatio$SeriousDlqin2yrs == 1)) # 258

#We see that apart for 3728 - 3443 = 285 people all of them have income. We also see that the proportion
# of defaulters is quite high in this group hence we cannot just drop all of them out, they are valid datarows.

# Now let's see for the realestate lines. 
# They look quite good but just in any case let's repeat the usual procedure for the outlier check.

length(which(credit_data_cleaned_DebtRatio$NumberRealEstateLoansOrLines > 10 )) #80 
length(which(credit_data_cleaned_DebtRatio$NumberRealEstateLoansOrLines > 10 &  credit_data_cleaned_DebtRatio$MonthlyIncome > 1)) # All have income
length(which(credit_data_cleaned_DebtRatio$NumberRealEstateLoansOrLines > 10 &  credit_data_cleaned_DebtRatio$SeriousDlqin2yrs == 1)) # 20 !

# We see that there is a 25% default rate in people with income and high number of lines. These lines look great , no need to drop anything

# 2.7 Monthly Income and Number of Dependents
# 2.7.1 Number of Dependents

#After dropping the outliers, we check how many NA's do we still have in these two features.

sum(is.na(credit_data_cleaned_DebtRatio$NumberOfDependents)) #3647
sum(is.na(credit_data_cleaned_DebtRatio$NumberOfDependents) & is.na(credit_data_cleaned_DebtRatio$MonthlyIncome)) #3647
sum(is.na(credit_data_cleaned_DebtRatio$NumberOfDependents) & is.na(credit_data_cleaned_DebtRatio$MonthlyIncome) & credit_data_cleaned_DebtRatio$SeriousDlqin2yrs == 1) #155

# For Number of Dependents the mode (most frequent value) which is 0. better like this than mode(..) so there is no implicit char conversion

credit_data_cleaned_DebtRatio$NumberOfDependents[is.na(credit_data_cleaned_DebtRatio$NumberOfDependents)] <- 0

# 2.7.2 Monthly Income

# For income we use median to input

credit_data_cleaned_DebtRatio$MonthlyIncome[is.na(credit_data_cleaned_DebtRatio$MonthlyIncome)] <- median(credit_data_cleaned_DebtRatio$MonthlyIncome, na.rm=TRUE)

summary(credit_data_cleaned_days_past$SeriousDlqin2yrs)

#We control the NA's to be sure.
na_count <-sapply(credit_data_cleaned_DebtRatio, function(y) sum(length(which(is.na(y))))) 
na_count <- data.frame(na_count) 

i = data.frame(feature = "Monthly Income", value = c(credit_data_cleaned_DebtRatio$MonthlyIncome))

ggplot(i, aes(x=feature, y=value, fill=feature)) +
  geom_boxplot()+
  theme(panel.background = element_rect(fill = "white"))+ 
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) 

# Here the big positive outliers are fine, just very rich people. 

j = data.frame(group = "Number of Dependends", value = c(credit_data_cleaned_DebtRatio$NumberOfDependents))

ggplot(j, aes(x=group, y=value, fill=group)) +
  geom_boxplot()+
  theme(panel.background = element_rect(fill = "white"))+ 
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) 

# This all looks pretty much good at this point , all the features were cleaned. After the first models are fit,
# We can do additional cleaning by revisiting any of these points.

#We save the variable to a new variable name to avoid errors.
credit_data_cleaned <- credit_data_cleaned_DebtRatio

# 3. Skewness/Scaling/Preparing for model

hist.data.frame(credit_data_cleaned) 

# Features are extremely skewed we can see it easily on the histograms, except age pretty much all of them.
# Some features like Number of Dependents have almost all 0 (which is ok) but might cause problem with logs.
# Let's quantify the skew using library e1071 and the skewness function.

skew <-sapply(credit_data_cleaned, function(y) skewness(y))
skew <- data.frame(skew) 

# The skew is massive. We need to pick which way to solve it . Possibilities are 
# 1) Log , might be problematic because of 0 . We either add small values per feature(like 0.00001) bla bla but sounds stupid 
# Solution : We use the fact we have no negative values in the dataframe, which means the square root is perfect.
# It solves skewnes quite well https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45

credit_data_cleaned_deskewed <- sqrt(credit_data_cleaned)

skew2 <-sapply(credit_data_cleaned_deskewed, function(y) skewness(y))
skew2 <- data.frame(skew2)

# The effectiveness is easily observeed by looking at the skew2 and skew tables

hist.data.frame(credit_data_cleaned_deskewed) 

# We control the NA's once more.  
na_count2 <-sapply(credit_data_cleaned_deskewed, function(y) sum(length(which(is.na(y)))))
na_count2 <- data.frame(na_count)
sapply(credit_data_cleaned_deskewed, class)

# 3.1 Correlation Matrix 

#We save the final cleaned dataset to a new variable name when building corrmatrix.
for_corr<- credit_data_cleaned_deskewed
#We shorten the column names just for visual reasons.
colnames(for_corr)<- c("NumOfDep", "NumOf6089", "NumOfReal", "NumOf90", "NumOfOpen", "MonthlyInc", "DebtRatio", "NumOf3059", "Age", "RevolUtil", "SerDlq2yrs")
#We use the cor function to calculate the correlation and saving it to variable cormat.
cormat<- cor(for_corr)
#We inspect the data.
head(cormat) 

#The package reshape is required to melt the correlation matrix :
melted_cormat <- melt(cormat)
#We inspect the data.
head(melted_cormat) 

#The function geom_tile()[ggplot2 package] is used to visualize the correlation matrix :
ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()


# 4.0 Feature Engineering
credit_data_cleaned_deskewed['IncomePerPerson'] = credit_data_cleaned_deskewed['MonthlyIncome']/(credit_data_cleaned_deskewed['NumberOfDependents']+1)
credit_data_cleaned_deskewed['MonthlyDebt'] = credit_data_cleaned_deskewed['DebtRatio']*credit_data_cleaned_deskewed['MonthlyIncome'] 
credit_data_cleaned_deskewed['NumOfOpenCreditLines'] = credit_data_cleaned_deskewed['NumberOfOpenCreditLinesAndLoans']-credit_data_cleaned_deskewed['NumberRealEstateLoansOrLines']

#From the correlation matrix we built after engineering all features, we saw that the below feature is correlated negatively with other engineered features. Therefore, in the models we do not engineer this feature. 
#credit_data_cleaned_deskewed['MonthlyBalance'] = credit_data_cleaned_deskewed['MonthlyIncome']-credit_data_cleaned_deskewed['MonthlyDebt']

# 4.1 Correlation Matrix after Feature Engineering

#We save the final cleaned dataset to a new variable name when building corrmatrix.
for_corr_feature_engineer<- credit_data_cleaned_deskewed
#We shorten the column names for visual reasons.
colnames(for_corr_feature_engineer)<- c("SerDlq2yrs", "RevolvUtiliz", "Age", "NumOf3059" , "DebtRatio" , "MonthlyInc","NumOfOpen" ,"NumOf90" , "NumOfReal", "NumOf6089", "NumOfDep", "IncpPerson", "MonthlyDebt", "NumOfCredLines")
#We use the cor function to calculate the correlation and saving it to variable cormat.
cormat2<- cor(for_corr_feature_engineer)
#We inspect the data
head(cormat2)

#The package reshape is required to melt the correlation matrix :
melted_cormat2 <- melt(cormat2)
#Inspecting the data
head(melted_cormat2)

ggplot(data = melted_cormat2, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

#Everything seems to be alright!
#Ready for partitioning.

#One final step. Convert the target variable as factor variable.
credit_data_cleaned_deskewed$SeriousDlqin2yrs <- as.factor(credit_data_cleaned_deskewed$SeriousDlqin2yrs)

#Examine the levels of the target variable and relevel if needed. We came across with this when we trained and printed the models, saw the positive class as "0". Then, we used relevel function to indicate that the positive class is indeed "1" to the model.
levels(credit_data_cleaned_deskewed$SeriousDlqin2yrs)
credit_data_cleaned_deskewed$SeriousDlqin2yrs<- relevel(credit_data_cleaned_deskewed$SeriousDlqin2yrs, ref = "1")

#Controling the proportion
summary(credit_data_cleaned_deskewed$SeriousDlqin2yrs)

#Check for NA's one last time
colSums(is.na(credit_data_cleaned_deskewed))


# 5. Partition the Data

#We use the createDataPartition function from Caret library to partition the data. We are using the most common 80/20 split.
#We set seed so that same sample can be reproduced in future.

set.seed(1)
index <- createDataPartition(credit_data_cleaned_deskewed$SeriousDlqin2yrs, p=0.8, list=FALSE)
training <- credit_data_cleaned_deskewed[index,]
test<- credit_data_cleaned_deskewed[-index, ]

#We indicate here the 10 fold cross-validation we will be using in our models. 
TControl <- trainControl(method="cv", number=10)

#We create a report data frame to save the results of the models.
report <- data.frame(Model=character(), Acc.Train=numeric(), Acc.Test=numeric()) 

#We inspect the target variable in train and test sets.
summary(training$SeriousDlqin2yrs)
summary(test$SeriousDlqin2yrs)


## 6. MODELS
#**********************
# 6.1 k-Nearest Neighbors
#**********************
#After several trials, we found the optimal k that gives the best accuracy. (k=11)

set.seed(1)
library(class)
knnmodel <- knn(train=training[,-1], test=test[,-1], cl=training$SeriousDlqin2yrs, k=11)
knnmodel
summary(knnmodel) 
num.correct.labels <- sum(knnmodel == test$SeriousDlqin2yrs)
accuracy <- num.correct.labels / length(test$SeriousDlqin2yrs)
accuracy 

# best model is k=11
knnmodel_tr <- knn(train=training[,-1], test=training[,-1], cl=training$SeriousDlqin2yrs, k=11, prob = TRUE)
knnmodel <- knn(train=training[,-1], test=test[,-1], cl=training$SeriousDlqin2yrs, k=11, prob=TRUE)

#We use the train function from caret library to fit and evaluate the model in order to apply the cross validation we set in partitioning part. 

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

#With the ROCR library, we calculate the F1 score without manually calculating the necessary metrices. 
precision <- posPredValue(prediction.test, test$SeriousDlqin2yrs, positive="1") 
recall <- sensitivity(prediction.test, test$SeriousDlqin2yrs, positive="1")
F1 <- (2 * precision * recall) / (precision + recall) 

#One way of calculating the Area Under the Curve(AUC) with a pre-built function. 
#With the pROC library and roc function, we can find and plot the AUC of the model.  

roc_obj<- roc(prediction.test, as.numeric(test$SeriousDlqin2yrs))
auc(roc_obj) 
plot(roc_obj, smoothed = TRUE)

#Creating the precision-recall curve. (pROC library)

# 1) Creat this special prediction object.
pred <- prediction(as.numeric(prediction.test), test$SeriousDlqin2yrs) 
# 2) Function to calculate precission and recall.
RP.perf <- performance(pred, "prec", "rec") 
# 3) Plot the precission recall curve
plot(RP.perf) 


#***********APPLYING kNN with scaling***************#
# We are scaling the values only for the knn algorithm. 
# We are also scaling the train and the test datasets separately so that there's no information leakage between the two datasets.

# Creating separate data frames to avoid scaling the target variable
serious_training<- as.data.frame(training[1])
serious_test<- as.data.frame(test[1])

# We are using the scale function to scale the values from training and test sets besides the target variable.
preProcValues <- scale(training[c(2:14)])
preProcValues_test <- scale(test[c(2:14)])

# We are converting the scaled values to data frames from matrix.
preProcValues <- data.frame(preProcValues)
preProcValues_test <- data.frame(preProcValues_test)

# We are combining the data frames again. 
new_training<- cbind(serious_training, preProcValues)
new_test<- cbind(serious_test, preProcValues_test)

# We apply our model with the scaled values.
set.seed(1)
knnmodel <- train(SeriousDlqin2yrs~., data=new_training, method="knn", trControl=TControl)
knnmodel 

prediction.train <- predict(knnmodel, new_training[,-1],type="raw")
acctr <- confusionMatrix(prediction.train, new_training[,1])
acctr$table 
acctr$overall['Accuracy'] 

prediction.test <- predict(knnmodel, new_test[,-1],type="raw")
accte <- confusionMatrix(prediction.test, new_test[,1]) 
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="k-NN", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

precision <- posPredValue(prediction.test, new_test$SeriousDlqin2yrs, positive="1")
recall <- sensitivity(prediction.test, new_test$SeriousDlqin2yrs, positive="1")
F1 <- (2 * precision * recall) / (precision + recall) 

roc_obj<- roc(prediction.test, as.numeric(new_test$SeriousDlqin2yrs))
auc(roc_obj) 
plot(roc_obj, smoothed = TRUE)

#**********************
# 6.2 Logistic Regression
#**********************
#We use library caret for the logistic regression model and pick the method as generalized linear models. 

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

#With the ROCR library, we can calculate the F1 score without manually calculating the necessary metrices. 

precision <- posPredValue(prediction.test, test$SeriousDlqin2yrs, positive="1")
recall <- sensitivity(prediction.test, test$SeriousDlqin2yrs, positive="1")
F1 <- (2 * precision * recall) / (precision + recall) 

#One way of calculating the Area Under the Curve(AUC) with a pre-built function. 
#With the pROC library and roc function, we can find and plot the AUC of the model. 

roc_obj<- roc(prediction.test, as.numeric(test$SeriousDlqin2yrs))
auc(roc_obj) 
plot(roc_obj, smoothed = TRUE)

#Creating the precision-recall curve. (pROC library)

# 1) Creat this special prediction object.
pred <- prediction(as.numeric(prediction.test), test$SeriousDlqin2yrs) 
# 2) Function to calculate precission and recall.
RP.perf <- performance(pred, "prec", "rec") 
# 3) Plot the precission recall curve
plot(RP.perf) 

#**********************************
# 6.3 Random Forest
#**********************************
#We used randomForest library to train the Random Forest model. 
#After trying multiple hyperparameters (changed the number of trees, maxnodes etc.), we got the best accuracy from the default ntree=500 parameter. 

set.seed(1)
rformodel <- randomForest(SeriousDlqin2yrs ~., data=training, method="rf", importance=TRUE)
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

#With the ROCR library, we can calculate the F1 score without manually calculating the necessary metrices. 

precision <- posPredValue(prediction.test, test$SeriousDlqin2yrs, positive="1")
recall <- sensitivity(prediction.test, test$SeriousDlqin2yrs, positive="1")
F1 <- (2 * precision * recall) / (precision + recall) 

#One way of calculating the Area Under the Curve(AUC) with a pre-built function. 
#With the pROC library and roc function, we can find and plot the AUC of the model. 

roc_obj<- roc(prediction.test, as.numeric(test$SeriousDlqin2yrs))
auc(roc_obj) 
plot(roc_obj, smoothed = TRUE) 

#Creating the precision-recall curve. (pROC library)

# 1) Creat this special prediction object.
pred <- prediction(as.numeric(prediction.test), test$SeriousDlqin2yrs) 
# 2) Function to calculate precission and recall.
RP.perf <- performance(pred, "prec", "rec") 
# 3) Plot the precission recall curve
plot(RP.perf) 

#Visualizing the Feature Importance
varImpPlot(rformodel, type=2)

#Used partykit library.
#Visuzaling the split of the trees.	Type -> either 1 or 2, specifying the type of importance measure (1=mean decrease in accuracy, 2=mean decrease in node impurity).
train2<- subset(training, select=c(RevolvingUtilizationOfUnsecuredLines, SeriousDlqin2yrs, DebtRatio))
ct<- ctree(SeriousDlqin2yrs~., data=train2)
plot(ct, type="simple")


#**********************************
# 6.4 Support Vector Machine
#**********************************
#We fitted SVM but the model split into extreme numbers of kernel vectors.
library(e1071)
svmfit = svm(SeriousDlqin2yrs ~ ., data = train_svm)
print(svmfit)

pred_svm = predict(svmfit, test_svm,type = "response")

cm = table(test_svm[,1], pred_svm)
cm
## From CM you can get manually F1 , Prec, Recall , accuracy 
library(pROC)
roc_obj <-roc(pred_svm, as.numeric(test_svm$SeriousDlqin2yrs))
auc(roc_obj)
plot(roc_obj)


#install.packages("kernlab")
library(kernlab)
letter_classifier <- ksvm(SeriousDlqin2yrs ~ ., data = train_svm,kernel = "vanilladot")

