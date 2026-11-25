import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor 
from sklearn.preprocessing import LabelEncoder 

dataframe= pd.read_csv("D:\\sales_prediction\\sales_data.csv")

# print(dataframe.head())
# print(dataframe.tail())
# print(dataframe.describe())
# print(dataframe.shape)
# print(dataframe.info())
# print(dataframe.isnull().sum())
# print(dataframe.isnull().sum().sum())
# print(type(XGBRegressor))

encoder= LabelEncoder()

dataframe["Item_Identifier"]= encoder.fit_transform(dataframe["Item_Identifier"])
dataframe["Outlet_Identifier"]= encoder.fit_transform(dataframe["Outlet_Identifier"])

x= dataframe.drop(["Item_Outlet_Sales", "s_no"], axis= 1)
y= dataframe["Item_Outlet_Sales"]

# print(x.value_counts())
# print(y.value_counts())

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=2) 

model= XGBRegressor()

model.fit(x_train, y_train)

train_data_score= model.predict(x_train)
test_data_score= model.predict(x_test)

print(metrics.r2_score(y_train, train_data_score))
print(metrics.r2_score(y_test, test_data_score))


## This model is heavily overfit as it might have learned the data-pattern and not the assigned data itself. 