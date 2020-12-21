
# Code
# Yichao, Ivan, DAI
# Regression Analysis on the Wine Classification and Quality

# Package Needed ------------------------------------------------

library(ggplot2)
library(reshape2)
library(gridExtra)
library(statsr)
library(corrplot)
library(tidyr); library(dplyr); library(ggplot2)
library(caret)
library(caTools)
library(rpart)
library(rpart.plot)
library(ROCR)
library(olsrr)
library(glmnet)

# Data Import and combine ---------------------------------------------------

red = read.csv('winequality-red.csv', sep = ';')
white = read.csv('winequality-white.csv', sep = ';')
names(red) == names(white)
wine = rbind(red, white)
wine$type = c(rep('Red', 1599), rep('White', 4898))
wine = wine[sample.int(6497, 6497),]
wine = wine[sample.int(6497, 6497),]

# Basic Exploration   ---------------------------------------------------

# Boxplot
boxplot(wine$pH ~ wine$type)
str(wine)
table(wine$quality)
d <- melt(wine, id.vars="type")
d
ggplot(d, aes(x = type, y = value, fill = type))+
        geom_boxplot(alpha = 0.7)+
        facet_wrap(.~variable, nrow = 3, scales = 'free_y')+
        theme_bw() +
        labs(x = '', y = '')


a = ggplot(wine, aes(x = type, y = alcohol, fill = type))+
        geom_boxplot(alpha = 0.7)+ theme_bw()+
        labs(x = '', y = '')
b = ggplot(wine, aes(x = alcohol, fill = type))+
        geom_density(alpha = 0.7)+ theme_bw()+
        labs(x = '', y = '')
grid.arrange(a, b, ncol = 2)

# Standard deviation
sd(white$alcohol)
sd(red$alcohol)


# Correlation plot
corrplot(cor(wine[,-c(12,13)]),method = 'square',type = 'lower',diag = F)
corMatrix = as.data.frame(cor(wine[,-c(12,13)]))
corMatrix$var1 = rownames(corMatrix)
corMatrix %>%
        gather(key=var2,value=r,1:11)%>%
        arrange(var1,desc(var2))%>%
        ggplot(aes(x=var1,y=reorder(var2, order(var2,decreasing=F)),fill=r))+
        geom_tile()+
        geom_text(aes(label=round(r,2)),size=3)+
        scale_fill_gradientn(colours = c('#d7191c','#fdae61','#ffffbf','#a6d96a','#1a9641'))+
        theme(axis.text.x=element_text(angle=75,hjust = 1))+xlab('')+ylab('')


# Data Splite --------------------------------------------------------

set.seed(1031)
split = sample.split(wine$type,SplitRatio = 0.7)
train = wine[split,]
test = wine[!split,]

# Classification ------------- ---------------------------------------

# Maximum Tree
par(mfrow = c(1,1))
wine$type = factor(wine$type)
levels(wine$type) = c(0,1)
wine$type = as.integer(as.character(wine$type))
classTree1 = rpart(type~.,data=train,method='class', cp=0)
## Here we use the cp, which stands for complexity parameter, the smaller the cp is
## the more splits the tree model will have. 

# To visualize the tree
rpart.plot(classTree1)
prp(classTree1)

# Confusion matrix of maximum tree on test set
pred = predict(classTree1,newdata = test,type='class')
cm = confusionMatrix(factor(test$type),pred)
cm
fourfoldplot(cm$table, margin = c(1, 2), space = 0.2, main = NULL,
             mfrow = NULL, mfcol = NULL)

# Tuning tree to find the best cp.
trControl = trainControl(method='cv',number = 5)
tuneGrid = expand.grid(.cp = seq(from = 0.001,to = 0.05,by = 0.0005))

set.seed(617) # to make sure the repeated result. 
cvModel = train(factor(type)~.,
                data=train,
                method="rpart",
                trControl = trControl,
                tuneGrid = tuneGrid)
ggplot(data=cvModel$results, aes(x=cp, y=Accuracy))+
        geom_line(size=0.5,alpha=0.2)+
        geom_point(color='brown')+
        theme_bw()
cvModel$bestTune$cp

# The best tree with the best complexity parameter
classTree1 = rpart(type~.,data=train,method='class', cp=cvModel$bestTune$cp)
rpart.plot(classTree1)
prp(classTree1)

# Predtion on the test set. 
pred = predict(classTree1,newdata = test,type='class')
cm = confusionMatrix(factor(test$type),pred)
cm
fourfoldplot(cm$table, margin = c(1, 2), space = 0.2, main = NULL,
             mfrow = NULL, mfcol = NULL)

# Use the ROC and AUC to evaluate the model performance
pred = predict(classTree1,newdata=test,type='prob')
ROCRpred = prediction(pred[,2],test$type)
ROCRperf = performance(ROCRpred,"tpr","fpr")
plot(ROCRperf,
     colorize=TRUE,
     print.cutoffs.at=seq(0,1, 0.8),
     text.adj=c(-0.3,2),
     xlab="1 - Specificity",
     ylab="Sensitivity") 
as.numeric(performance(ROCRpred,"auc")@y.values)
# the number show the area under curve. We can see the area is over 0.95, which shows
# a great model power to predict the wine classification. 

# Evaluations Function ------------------------------------------------

# Compute R^2 and RMSE from true and predicted values
eval_results <- function(true, predicted, df) {
        SSE <- sum((predicted - true)^2)
        SST <- sum((true - mean(true))^2)
        R_square <- 1 - SSE / SST
        RMSE = sqrt(SSE/nrow(df))

        data.frame(
                RMSE = RMSE,
                Rsquare = R_square
        )
        
}

# ols Regression model white -------------------------------------------------------- 

train = wine[split,]
test = wine[!split,]
model = lm(quality~.,data=train[train$type == 1,-13])
summary(model)
par(mfrow = c(2,2), mar = c(5,7,5,5))
plot(model)
par(mfrow = c(1,1))

# Delete the outlier
train = train[rownames(train) != 4381, ]
model = lm(quality~.,data=train[train$type == 1,-13])
summary(model)
# SHow the residual and Leverage again 
plot(model, which =5)

# Prediction on the trainset 
x = model.matrix(quality~.-1,data=train[train$type == 1,-13])
y = train$quality[train$type == 1]
predictions_train <- predict(model, newx = x)
sqrt(mean((predictions_train - y)^2))
eval_results(y, predictions_train, train[train$type == 1,])

# Prediction on the test set 
x_test = as.data.frame(model.matrix(quality~.-1,data=test[test$type == 1,-13]))
y_test = test$quality[test$type == 1]
predictions_test <- predict(model, newdata = x_test)
sqrt(mean((predictions_test - y_test)^2))
eval_results(y_test, predictions_test, test[test$type == 1,])


# ------------ AIC Selection Regression model white

# Set up the start model and end model 
start_mod = lm(quality~.,data=train[train$type == 1,-13])
empty_mod = lm(quality~1,data=train[train$type == 1,-13])
full_mod = lm(quality~.,data=train[train$type == 1,-13])
backwardStepwise = step(start_mod,
                        scope=list(upper=full_mod,lower=empty_mod),
                        direction='backward')
summary(backwardStepwise)

# Show the more accurate process and selection parameter
ols_step_backward_p(model, details = TRUE)


x = model.matrix(quality~.-1,data=train[train$type == 1,-13])
y = train$quality[train$type == 1]
predictions_train <- predict(backwardStepwise, newx = x)
sqrt(mean((predictions_train - y)^2))
eval_results(y, predictions_train, train[train$type == 1,])

# Prediction on the test set
x_test = as.data.frame(model.matrix(quality~.-1,data=test[test$type == 1,-13]))
y_test = test$quality[test$type == 1]
predictions_test <- predict(backwardStepwise, newdata = x_test)
sqrt(mean((predictions_test - y_test)^2))
eval_results(y_test, predictions_test, test[test$type == 1,])


# ----------- Ridge Model


x = model.matrix(quality~.-1,data=train[train$type == 1,-13])
y = train$quality[train$type == 1]
ridgeModel = glmnet(x,y, alpha=0) # Alpha = 0 is the ridge model and 1 is LASSO model.

# Ridge model to show the coeficients with lambda and MSE
plot(ridgeModel,xvar='lambda',label=T)
plot(ridgeModel,xvar='dev',label=T)

# Make sure the repeated result for each run
set.seed(617)
cv.ridge = cv.glmnet(x,y,alpha=0) # 10-fold cross-validation
plot(cv.ridge)
coef(cv.ridge)

# Best 
ridge_best <-cv.ridge$lambda.min 
ridge_best
# Build the Ridge model with the best lambda we select.
ridge_model <- glmnet(x, y, alpha = 0, lambda = ridge_best)


# Predict on the train set 
x = model.matrix(quality~.-1,data=train[train$type == 1,-13])
y = train$quality[train$type == 1]
predictions_train <- predict(ridge_model, newx = x)
sqrt(mean((predictions_train - y)^2))
eval_results(y, predictions_train, train[train$type == 1,])

# Use the formula to turn the R^2 to adjusted R square, becuase we Ridge method use
# sampling method, also because everytime the script runs, the train set will different
# so I comment on it. For the report, the value I calculated is 0.2971642. 
# 1 - ((1 - 0.2971642)*(3428 - 1)/(3428 - 11-1))

# Predictions on the test set. 
x_test = model.matrix(quality~.-1,data=test[test$type == 1,-13])
y_test = test$quality[test$type == 1]
predictions_test <- predict(ridge_model, newx = x_test)
sqrt(mean((predictions_test - y_test)^2))
eval_results(y_test, predictions_test, test[test$type == 1,])

# ----------- LASSO Model

x = model.matrix(quality~.-1,data=train[train$type == 1,-13])
y = train$quality[train$type == 1]
lassoModel = glmnet(x,y, alpha=1) 

# Visualize the coeficients with its lambda and MSE
plot(lassoModel,xvar='lambda',label=T)
plot(lassoModel,xvar='dev',label=T)

# To make sure the repeated result the script run, set a seed
set.seed(617)
cv.lasso = cv.glmnet(x,y,alpha=1) # 10-fold cross-validation
plot(cv.lasso)
coef(cv.lasso)

# Best 
lambda_best <-cv.lasso$lambda.min 
lambda_best
lasso_model <- glmnet(x, y, alpha = 1, lambda = lambda_best) # find the model 

# Model prediction on the train set 
x = model.matrix(quality~.-1,data=train[train$type == 1,-13])
y = train$quality[train$type == 1]
predictions_train <- predict(lasso_model, newx = x)
sqrt(mean((predictions_train - y)^2))
eval_results(y, predictions_train, train[train$type == 1,])

# Model prediction on the test set 
x_test = model.matrix(quality~.-1,data=test[test$type == 1,-13])
y_test = test$quality[test$type == 1]
predictions_test <- predict(lasso_model, newx = x_test)
sqrt(mean((predictions_test - y_test)^2))
eval_results(y_test, predictions_test, test[test$type == 1,])

# EN Model ---------------------------------------

set.seed(123)
elastic <- train(
        quality~ ., data = train[,-13], method = "glmnet",
        trControl = trainControl("cv", number = 10),
        tuneLength = 10
)
elastic$bestTune
elastic$bestTune$lambda

# Visaulize the model selection process above:
plot(elastic)

# Model coefficients
coef(elastic$finalModel, elastic$bestTune$lambda)
# Make predictions
predictions <- elastic %>% predict(train[, -c(12,13)])
# Model prediction performance
data.frame(
        RMSE = RMSE(predictions,train$quality),
        Rsquare = R2(predictions, train$quality)
)
## Use the following process to find the adjsuted R square, the time I run is 0.3055
## 1 - ((1 - 0.3055)*(3428 - 1)/(3428 - 11-1))

predictions <- elastic %>% predict(test[, -c(12,13)])
# Model prediction performance
data.frame(
        RMSE = RMSE(predictions,test$quality),
        Rsquare = R2(predictions, test$quality)
)


