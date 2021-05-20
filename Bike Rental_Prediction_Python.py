import os
import warnings
#import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fancyimpute import KNN 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')
random.seed(10)
#Set working directory

os.chdir("D:\Kavin\Edwisor\Dataset\Bike rental")
df_train = pd.read_csv("day.csv")


###########################################################
#Basic Structure of Train and Test Dataset
df_train.info()

###########################################################
#Missing Value Analysis
df_train.isnull().sum()
df_train.isna().sum()

###########################################################
#Outlier Analysis
df_train_bkp=df_train.copy()

#fetching only the Numerical values
Numeric_Variables=list(df_train.columns[2:16])

#Show box plot in set group
plt.show(df_train[Numeric_Variables].boxplot(column =Numeric_Variables,figsize=(15,8)))


Independent_Variables=list(df_train.columns[2:15])


#Checking the  outliers % using the chauvenet way:
def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.    
    return prob < criterion       # Use boolean array outside this function

from scipy.special import erfc
Outilers = dict()
for col in [col for col in Independent_Variables]:
    Outilers[col] = df_train[chauvenet(df_train[col].values)].shape[0]
df_train_Outilers = pd.Series(Outilers)

print('Total number of outliers in  dataset: {} ({:.2f}%)'.format(sum(df_train_Outilers.values), (sum(df_train_Outilers.values) / df_train.shape[0]) * 100))
print('   \n     ')    
#Removing Outliers using IQR range
for i in Independent_Variables:
    #print(i)
##Detect and replace with NA
#Extract quartiles
    q75, q25 = np.percentile(df_train[i], [75 ,25])

#Calculate IQR
    iqr = q75 - q25

#Calculate inner and outer fence
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)

#Replace with NA
    
    if minimum != 0.0 and maximum != 0.0:#Replace NA only to inner and outer fence !=0
        df_train.loc[df_train[i] < minimum,i] = np.nan
        df_train.loc[df_train[i] > maximum,i] = np.nan
        
print('Impute NAN data with KNN \n     ')
##Impute NAN data with KNN
df_train = pd.DataFrame(KNN(k = 3).fit_transform(df_train[Independent_Variables]), columns = df_train[Independent_Variables].columns)
print('   \n     ')      
#########################################Feature Selection#########################################

## Feature Selection
#Correlation plot
df_corr = df_train
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(10, 8))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax,annot = True,fmt='.2g')
ax.set_yticklabels(corr.columns, rotation = 0)
ax.set_xticklabels(corr.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

#Function to find the correaltion values between Variables
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=3):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
print('   \n     ')
print("Top 10 Absolute Correlations in Training Dataset")
print(get_top_abs_correlations(df_train[Independent_Variables], 10))
print('     \n         ')
print('Maximum correlation between independent variables in Training DataSet : {}'.format(get_top_abs_correlations(df_train[Independent_Variables], 1)))
print('   \n     ')
df_train=pd.concat([df_train,df_train_bkp.iloc[:,15].reindex(df_train.index)], axis=1)


##################Information gain###################
#Information gain of Independent variables related to Depenedent variables#based on this we can remove one variable among 
# two variables with high correlation
print('   \n     ')
print('Information gain to identify the strength of Independent varible that caries information about target variable')
import warnings
from sklearn.feature_selection import mutual_info_classif
for inde,column in enumerate(Independent_Variables):
    c=mutual_info_classif(df_train[column].values.reshape((731 , 1 )), df_train['cnt'].values.reshape((731 , 1 )), discrete_features='auto', n_neighbors=3, copy=True, random_state=0)
    print(column,c)
warnings.filterwarnings("ignore")
print('\n')
print('As "atemp" independent variables carries more information about Targert variable compared to "temp" variable,so we\nare removing temp data')
#print('\n')
print('And Removing the variable "mnth","holiday","hum" as it doesn''t cary any information about target\n')


#As atemp independent variables carries more information about Targert variable compared to temp variable we removing temp data
#droping corelated variable (as per the dark green box as middle in above plot)
print('\n')
df_train = df_train.drop(['temp','mnth','holiday','hum'], axis=1)


#################Splitting Train and Testing Data set#######################
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_train, test_size=0.2)

##################Linear Regression Algorithm#########################
from sklearn.linear_model import LinearRegression
model_LR = LinearRegression()
model_LR.fit(train.iloc[:,0:9], train.iloc[:,9])
pred_LR = model_LR.predict(test.iloc[:,0:9])
#######################DecisionTree###############################

DT = DecisionTreeRegressor().fit(train.iloc[:,0:9], train.iloc[:,9])
predictions_DT = DT.predict(test.iloc[:,0:9])

####################RandomForestRegressor#########################
RFmodel = RandomForestRegressor(n_estimators = 200).fit(train.iloc[:,0:9], train.iloc[:,9])
RF_Predictions = RFmodel.predict(test.iloc[:,0:9])

################Validating Performance in Regression Model#############
print('RSquare Validation\n ',)
r_sq = model_LR.score(df_train.iloc[:,0:9], df_train.iloc[:,9])
print('coefficient of determination R^2 :', r_sq)
print('   \n     ')
#Calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape
#print('   \n     ')



print('MAPE for Linear Regression :',MAPE(test.iloc[:,9], pred_LR))
print('MAPE for Decision Tree :',MAPE(test.iloc[:,9], predictions_DT))
print('MAPE for Random Forest :',MAPE(test.iloc[:,9], RF_Predictions))

#####################Submissions###########################

Submission=pd.DataFrame(test.iloc[:,0:9])
Submission['pred_cnt'] = (RF_Predictions)
Submission.to_csv("Random forest output Prediction Python.csv",index=False)
###########################################################
###########################################################
