import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor 

# Display all the columns of the dataset
pd.set_option('display.max_columns', None)

# Read the csv file
df_train = pd.read_csv('Training_Data_Set.csv')
df_test = pd.read_csv('Test_Data_Set.csv')

# Dropping the columns that has very less data
cols_missingvalues = ['body_type']
df_train = df_train.drop(cols_missingvalues, axis=1)
df_test = df_test.drop(cols_missingvalues, axis=1)

# Label encoding multiple column
cols_label = ['Maker', 'model', 'Location']
label_encoder = preprocessing.LabelEncoder()
for col in cols_label:
   label_encoder.fit(df_train[col])
   df_train[col] = label_encoder.transform(df_train[col])
   df_test[col] = label_encoder.transform(df_test[col])

# One hot encoding multiple columns
dummies_train = []
dummies_test = []
cols_categ = ['Owner Type', 'transmission', 'fuel_type']
for col in cols_categ:
   dummies_train.append(pd.get_dummies(df_train[col]))
   dummies_test.append(pd.get_dummies(df_test[col]))
cardetails_dummies_train = pd.concat(dummies_train, axis=1)
df_train = pd.concat((df_train,cardetails_dummies_train), axis=1)
cardetails_dummies_test = pd.concat(dummies_test, axis=1)
df_test = pd.concat((df_test,cardetails_dummies_test), axis=1)

# Drop the actual columns after one hot encoding
df_train = df_train.drop(cols_categ, axis=1)
df_test = df_test.drop(cols_categ, axis=1)

# Drop the  correlated columns
cols_corr = ['manufacture_year']
df_train = df_train.drop(cols_corr, axis=1)
df_test = df_test.drop(cols_corr, axis=1)

# Change columns with type object to numeric
cols_with_objecttype = ['door_count', 'seat_count']
df_train[cols_with_objecttype] = df_train[cols_with_objecttype].apply(pd.to_numeric, errors='coerce', axis=1)
df_test[cols_with_objecttype] = df_test[cols_with_objecttype].apply(pd.to_numeric, errors='coerce', axis=1)

# Replace missing values with mean
df_train = df_train.apply(lambda x: x.fillna(x.mean()),axis=0)
df_test = df_test.apply(lambda x: x.fillna(x.mean()),axis=0)

# Replace outliers with median
Q1_train = df_train['Distance '].quantile(0.50)
Q1_test = df_train['Distance '].quantile(0.50)
Q2_train = df_train['Distance '].quantile(0.95)
Q2_test = df_test['Distance '].quantile(0.95)
df_train['Distance '] = np.where(df_train['Distance '] > Q2_train, Q1_train, df_train['Distance '])
df_test['Distance '] = np.where(df_test['Distance '] > Q2_test, Q1_test, df_test['Distance '])

# Copy all the predictor variables into X dataframe. Since 'Price' is dependent variable and Id is index, drop them
X_train = df_train.drop(['Id','Price'], axis=1)

# Copy the 'Price' column alone into the y dataframe. This is the dependent variable
y_train = df_train['Price']

X_test = df_test.drop(['Id'],axis=1)

rng = np.random.RandomState(1)

regr_1 = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=17), n_estimators=400, random_state=rng   )

regr_1.fit(X_train, y_train)

pred_train = regr_1.predict(X_train)
pred_test = regr_1.predict(X_test)
MSE = mean_squared_error(y_train, pred_train)
RMSE = math.sqrt(MSE)
print(RMSE)

res = pd.DataFrame(pred_test)
res.index = df_test["Id"]
res.columns = ["Price"]
res.to_csv("prediction_results.csv")
