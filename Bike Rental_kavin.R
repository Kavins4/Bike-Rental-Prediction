#clear the r environment
rm(list=ls())


#Load Libraries
library(rpart)
library(MASS)
library(ggplot2)
library(randomForest)
library(C50)


#setting the working directory
setwd("D:/Kavin/Edwisor/Dataset/Bike rental")

#verifying the directory
getwd()


#loading the train and test data
df_train = read.csv("day.csv",  header = T , na.strings = c(" ", "", "NA"))

##Copy the original data set to new set as backup
df_train_bkp=df_train


#Data Exploration
summary(df_train)


############Missing value analysis######################
#Finding the missing values in train dataset
missing_val_Train = data.frame(missing_val_Train=apply(df_train,2,function(x){sum(is.na(x))}))
missing_val_Train = sum(missing_val_Train)
missing_val_Train



numeric_index = sapply(df_train,is.numeric) #selecting only numeric variables
numeric_data = df_train[,numeric_index]
#View(df_train)
#View(numeric_data[2:15])

#numeric_data
cnames = colnames(numeric_data[2:15])
#Santander_train[,numeric_index[3:202]]
#View(numeric_data[2:201])
#View(cnames)

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(df_train))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "Green" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="cnt")+
           ggtitle(paste("Box plot of count for",cnames[i])))
}
gridExtra::grid.arrange(gn1,gn2,ncol=3)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn9,gn10,gn11,ncol=3)

# #loop to remove from all variables
for(i in cnames){
  #print(i)
  val = df_train[,i][df_train[,i] %in% boxplot.stats(df_train[,i])$out]
  #print(length(val))
  df_train = df_train[which(!df_train[,i] %in% val),]
}  


#imputation of the outliers we are using the capping function for Training dataset
x = as.data.frame(df_train[cnames])
caps = data.frame(apply(df_train[cnames],2, function(x){
  quantiles = quantile(x, c(0.25, 0.75))
  x[x < quantiles[1]] = quantiles[1]
  x[x > quantiles [2]] = quantiles[2]
}))



# #Standardisation For Train dataset
for(i in cnames){
  #print(i)
  df_train[,i] = (df_train[,i] - mean(df_train[,i]))/
    sd(df_train[,i])
}

#View(df_train[,c(3:14)])
#Correlations in Train dataset
Train_dataset_correlations=cor(df_train[,c(3:16)])
View(Train_dataset_correlations)

## Dimension Reduction####

df_train = subset(df_train,select = -c(temp,holiday,mnth,hum))



set.seed(123)
train_index = sample(1:nrow(df_train), 0.8 * nrow(df_train))
train = df_train[train_index,]
test = df_train[-train_index,]

###########Decision tree regression  #################
# ##rpart for regression
fit = rpart(cnt  ~ ., data = train[,3:12], method = "anova")

#View(train[,3:14])
#Predict for new test cases
predictions_DT = predict(fit, test[,3:11])


#############Random Forest Model##########################
RF_model = randomForest(cnt ~ ., train[,3:12], importance = TRUE, ntree = 200)
predictions_RF = predict(RF_model, test[,3:11])
plot(RF_model)

#View(test[,3:11])

#############Linear Regression Model##########################

#Linear regression model making
LR_model = lm(cnt ~., data = train[,3:12])
predictions_LR = predict(LR_model,test[,3:11])
summary(LR_model)
#summary(LR_model)$r.squared 

#############Evaluating Model##########################
#MAPE
#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test[,12], predictions_DT)

MAPE(test[,12], predictions_RF)

MAPE(test[,12], predictions_LR)


#Final submission

submission = data.frame(test, Cnt_Predicted = predictions_RF)

View(submission)
