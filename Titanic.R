#Install titanic data set following the below steps
install.packages("titanic")
library("titanic")
data<-titanic_train
str(data)

#change data type variables
data$Survived<-as.factor(data$Survived)
table(data$Pclass)
data$Sex<-as.factor(data$Sex)
data$Pclass<-as.factor(data$Pclass)
data$SibSp<-as.factor(data$SibSp)
data$Parch<-as.factor(data$Parch)


###Check NA Values.
data.frame(colSums(is.na(data)))



#Filling out unknown categorical variables by their most frequent category
data[is.na(data$Age),6]=mean(data$Age,na.rm=T)




#run the below packages
library(caret)
library(ggplot2)
library(MASS)
library(car)
library(mlogit)
library(sqldf)
library(Hmisc)
library(aod)
library(BaylorEdPsych)
library(ResourceSelection)
library(pROC)
library(ROCR)
library(caTools)
library(gmodels)
library(dummies)


# Data Splitting
set.seed(200)
split <- sample.split(data$Survived, SplitRatio = 0.8)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)

str(train)

# Model Building - Taking all the predictors
model<-glm(Survived~PassengerId+Pclass+Sex+Age+SibSp+Parch+Fare+Cabin+Embarked,data = train, family = binomial)
summary(model)
model1<-glm(Survived~PassengerId+Pclass+Sex+Age+SibSp+Parch+Fare+Embarked,data = train, family = binomial)
summary(model1)
final_model<-glm(Survived~I(Pclass=="2")+I(Pclass=="3")+I(Sex=="male")+Age+I(SibSp=="3"),data = train, family = binomial)
summary(final_model)
vif(final_model)


wald.test(b=coef(final_model), Sigma = vcov(final_model),Terms = 1:12)
FMchi<- final_model$null.deviance - final_model$deviance
FMdf<- final_model$df.null - final_model$df.residual
chi_pro<- 1- pchisq(FMchi, FMdf)
PseudoR2(final_model)
#Hoshmer Lamshow test ##
HL<- FMchi/final_model$null.deviance
## Cox and Snell R square #
Cox_Snell<- 1 - exp ((final_model$deviance - final_model$null.deviance) /nrow(train))
## HL test statistic calculation ##
hl <- hoslem.test(as.integer(train$Survived), fitted(final_model), g=10)
## importance of the model ##
varImp(final_model)



# in-sample accuracy at 0.5
pred <- predict(final_model, type = "response")
table(train$Survived, pred >= 0.5)
# out-sample accuracy(Validation)
pred_test <- predict(final_model, newdata = test, type = "response") 
table(test$Survived, pred_test >= 0.5)

# ROC Curve
install.packages("ROCR")
library(ROCR)
ROCRpred <- prediction(pred, train$Survived)
ROCRperf <- performance(ROCRpred, "tpr", "fpr")
plot(ROCRperf, colorize = TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))
auc = as.numeric(performance(ROCRpred, "auc")@y.values)
auc


# in-sample accuracy at 0.6
pred_train1 <- predict(final_model, type = "response")
table(train$Survived, pred_train1 >= 0.6)
# out-sample accuracy(Validation)
pred_test1 <- predict(final_model, newdata = test, type = "response") 
table(test$Survived, pred_test1 >= 0.6)

# ROC Curve
ROCRpred1 <- prediction(pred_train1, train$Survived)
ROCRperf1 <- performance(ROCRpred1, "tpr", "fpr")
plot(ROCRperf, colorize = TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))
auc = as.numeric(performance(ROCRpred1, "auc")@y.values)
auc


# in-sample accuracy at 0.4
pred_train2 <- predict(final_model, type = "response")
table(train$Survived, pred_train2 >= 0.4)
# out-sample accuracy(Validation)
pred_test2 <- predict(final_model, newdata = test, type = "response") 
table(test$Survived, pred_test2 >= 0.4)

# ROC Curve
ROCRpred2 <- prediction(pred_train2, train$Survived)
ROCRperf2 <- performance(ROCRpred2, "tpr", "fpr")
plot(ROCRperf, colorize = TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))
auc = as.numeric(performance(ROCRpred2, "auc")@y.values)
auc



#final_model will be best at accuracy of 0.5
#Test model is 74.71% accurate.
####-------xxxx-----------xxx--------------xxx---------------------xxxx-----------xxx------------------------------###





