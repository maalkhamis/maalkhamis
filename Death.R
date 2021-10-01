##install.packages("memisc")
library(memisc)
rm(list = ls())

library("randomForest");library("caret");library("pROC");library("ROCR");library("plyr");library("missForest")
library("gbm");library("pdp");library("ggplot2"); library("iml"); library("Boruta");library("dplyr")

# load functions
setwd("/Users/malkhamis/Desktop/OneDrive2/COAST/Analyses/")
source("functions.R")
source("functionsCV.R")

###1. Pre-processing
###1.1. What to do with missing data?

data <- read.csv("CoastDeath.csv", row.names=1)
head(data)
data<-data[complete.cases(data), ]
dim(data)
data<-data[c(-21)]
head(data)
####1.2 Selecting relevant features

colnames(data)[1] <- "Class"

X = data[-which(names(data) == "Class")]
Y <- data$Class

set.seed(124)#this is another stochastic machine learning model
ImpVar <- Boruta(X, Y, doTrace = 0, maxRuns = 999)
print(ImpVar)

#plot with x axis label vertical
plot(ImpVar, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(ImpVar$ImpHistory),function(i)
  ImpVar$ImpHistory[is.finite(ImpVar$ImpHistory[,i]),i])
names(lz) <- colnames(ImpVar$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(ImpVar$ImpHistory), cex.axis = 0.7)
     
#only choose the variables that are relevant for subsequent models.
set <- getSelectedAttributes(ImpVar, withTentative = TRUE)#keep tentative
dataReduced <- data[which(names(data) %in% set)]
data<- cbind(data[1], dataReduced);

###2. Training the models
###2.1 Create partitions

inTrain= createDataPartition(y=data$Class, p = .8, list = FALSE) #.8 of the dataset for training is reccomended 

data.train.descr <- data[inTrain,-1]
data.train.class <- data[inTrain,1]
data.test.descr <- data[-inTrain,-1]
data.test.class <- data[-inTrain,1]

# ## create folds for CV
set.seed(123)
myFolds <- createMultiFolds(y=data.train.class,k=10,times=10)

# create folds for CV, performing downsample of majority class
set.seed(130)
myFoldsDownSample <- CreateBalancedMultiFolds(y=data.train.class,k=10,times=10) 
length(unique(unlist(myFoldsDownSample)))

#downsampled version

myControlDownSample <- trainControl(## 10-fold CV
    method = "repeatedcv",
    number = 10,
   repeats = 10,
   index = myFoldsDownSample,
   savePredictions=TRUE,        
   classProbs = TRUE,
   summaryFunction = twoClassSummary,  
   allowParallel=TRUE,
   selectionFunction = "best")
   
library(doParallel) 
registerDoParallel(cores = 12)

### 2.2 Running the models
#------------------------------------------------------------------
#################Random Forests #################
#------------------------------------------------------------------

rf.grid <- expand.grid(.mtry=(3:6)) #number of predictors (mtry) to test.

set.seed(125) #again this is a stochastic model so set the seed to reproducibility.

rf.fit <- train(data.train.descr,data.train.class,
                method = "rf", #this is where we select which model to use for prediction (out of the 237 available)
                metric = "ROC",# Reciever opperating characteristic used to test model performance     (sensitivity/specificity)
                verbose = FALSE,
                trControl = myControlDownSample,
                tuneGrid = rf.grid, # Optimized parameter tuning is done usig the 'rf.grid' set up

                verboseIter=TRUE)

plot(rf.fit)
rf.fit
oof <- rf.fit$pred # check if worked
head(oof)
oof <- oof[oof$mtry==rf.fit$bestTune[,'mtry'],]# consider only the best tuned model to use

repeats<-rf.fit$control$repeats

rf.performance.cv <- EstimatePerformanceCV(oof=oof,repeats=repeats) #Matthew's correlation coefficient (MCC)
rf.performance.cv 
rf.error.cv <- EstimateErrorRateCV(oof=oof,repeats=repeats)
rf.error.cv
rm(oof,repeats)
####save(rf.fit,file="Death RF.RData")

#------------------------------------------------------------------
################# Support vector machine #################
#------------------------------------------------------------------
# 
# 
set.seed(123)
svm.fit <- train(Class ~ .,data=cbind(Class=data.train.class,data.train.descr),
                 method = "svmRadial",
                 tuneLength = 9,
                 metric="ROC",
                 trControl = myControlDownSample)
plot(svm.fit)
svm.fit
# 
# # estimate model performance in terms of a confusion matrix from repeated cross-validation
oof <- svm.fit$pred
# # consider only the best tuned model
oof <- oof[intersect(which(oof$sigma==svm.fit$bestTune[,'sigma']),which(oof$C==svm.fit$bestTune[,'C'])),]
repeats <- svm.fit$control$repeats
svm.performance.cv <- EstimatePerformanceCV(oof=oof,repeats=repeats)
svm.performance.cv 
# # estimate error rates from repeated cross-validation
svm.error.cv <- EstimateErrorRateCV(oof=oof,repeats=repeats)
rm(oof,repeats)
# save the model
###save(svm.fit, file="Death SVM.RData")

#------------------------------------------------------------------
################# Gradient Boosting #################
#------------------------------------------------------------------
# 
# #set up GBM tuning paramters
 gbm.grid <-  expand.grid(interaction.depth = c(1,3,5,7,9),
                          n.trees = (1:30)*10,
                          shrinkage = 0.5,
                          n.minobsinnode = c(10))# will stop when is 10 onservation in terminal node

 nrow(gbm.grid)
set.seed(123)
 gbm.fit <- train(data.train.descr, data.train.class,
                  method = "gbm",
                  metric = "ROC",
                  verbose = FALSE,
                  trControl = myControlDownSample,
                  ## Now specify the exact models 
                  ## to evaludate:
                  tuneGrid = gbm.grid)
# 
plot(gbm.fit)
gbm.fit
###save(gbm.fit,file="Death GBMAll.RData")
# # estimate model performance in terms of a confusion matrix from repeated cross-validation
oof <- gbm.fit$pred
# # consider only the best tuned model
oof <- oof[intersect(which(oof$n.trees==gbm.fit$bestTune[,'n.trees']),which(oof$interaction.depth==gbm.fit$bestTune[,'interaction.depth'])),]
repeats <- gbm.fit$control$repeats
gbm.performance.cv <- EstimatePerformanceCV(oof=oof,repeats=repeats)
gbm.performance.cv
# # estimate error rates from repeated cross-validation
gbm.error.cv <- EstimateErrorRateCV(oof=oof,repeats=repeats)
gbm.error.cv 

## #------------------------------------------------------------------
## ################# Interpreting the best model #################
## #------------------------------------------------------------------

X <-data[-which(names(data) == "Class")] #load data again for the visualization
Y <- data$Class

# create the iml object

mod <-Predictor$new(rf.fit, data = X, y = Y) #create predictor object. Add your model object name here (GLM, RF, GBM or SVM)
set.seed(123)
imp <-FeatureImp$new(mod, loss = "ce", compare='ratio', n.repetitions = 1) 
imp.dat<- imp$results
plot(imp)+ theme_bw()#plot results

#plot pd plots for the top predictors (any number appropriate for the data). Cateforical features don't plot properly

top5<- imp.dat$feature[1:5]#n = number of predictors you want to display

ice_curves <- lapply(top5, FUN = function(x) {
  cice <- partial(rf.fit, pred.var = x, center = TRUE, ice = TRUE, which.class="Positive",
                  prob = T) #note that we center values in the plotso these are centered ICE plots (cICE)
  autoplot(cice, rug = TRUE, train = dataReduced, alpha = 0.1) +
    theme_bw() +
    ylab("c-ICE")
  })

grid.arrange(grobs = c(ice_curves), ncol = 2)

Categorical <- "HF"
ice_curves1 <- lapply(Categorical, FUN = function(x) {
  ice <- partial(rf.fit, pred.var =  'HF', ice = TRUE, center = FALSE, which.class="Positive",
                  prob = T)
  ggplot(ice, rug=T, train = data, aes(x=HF, y = yhat, group = HF)) +
    geom_boxplot()+theme_bw() 
})
#put them together
grid.arrange(grobs = c(ice_curves, ice_curves1), ncol = 2)

#' ###4.3 Interactions using Friedman's H index
#' 
#' Calculating Friedman's H index provides a robust way to assess the importance of interactions in shaping risk across models. The interactions identified can then be visualized using PD plots.
#' 
## ------------------------------------------------------------------------

set.seed(345)
mod <- Predictor$new(rf.fit, data = X, y=Y, type='prob', class='Positive') #we just want  positive class results now

interact <- Interaction$new(mod)
plot(interact)+ theme_bw()

interact1 <- Interaction$new(mod, feature = "Hemoglobin")
plot(interact1)+theme_bw()

pdp.obj <-  FeatureEffect$new(mod, feature = c("Age","Hemoglobin"), method='pdp')
plot(pdp.obj)+ scale_fill_gradient(low = "white", high = "red")+ theme_bw()

#' To better understand model predictions, the last step is to use a game theory and specifically Shapely values to understand how the model is applied to individual observations (see main text for more details).
#' 
## ------------------------------------------------------------------------

shapley <- Shapley$new(mod, x.interest = X[2001,]);shapley$plot()+ theme_bw()
results <- shapley$results #for each instance you can view these results as a table
head(results, n = 7L)
sum(results$phi)# <0 indicate model prediction was negative, > 0 model prediction was postive.

Y[2641] #see if observation was negative postive
#negative = 2307

