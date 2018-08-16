import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import Imputer

# Imputation
# from sklearn.impute import SimpleImputer
#Tao ra mo hinh xac suat Bayes thong qua thu vien
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# #Tao ra mo hinh xac suat Bayes thong qua thu vien
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

#Doc du lieu tu file
dataset = pd.read_csv("Heart_Disease_Data.csv",na_values="?", low_memory = False)

# doi cac gia tri 1, 2, 3, 4 ve 1
dataset["pred_attribute"].replace(inplace=True, value=[1, 1, 1, 1], to_replace=[1, 2, 3, 4])
#Chuyen doi gia tri kieu chu sang kieu so
# le = preprocessing.LabelEncoder()
# dataset = dataset.apply(le.fit_transform)
#np_dataset = np.asarray(dataset)
# 13 dataset features
feature13 = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slop','ca','thal']

# print dataset.isnull().sum()

# Load data
# Load data
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values # = dataset.iloc[:, 13].values
my_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
my_imputer = my_imputer.fit(X[:,0:13])
X[:, 0:13] = my_imputer.transform(X[:, 0:13])

print (X)
# print y

# data = dataset[dataset.columns[:13]]
# print data
#outcome = dataset['pred_attribute']

#Chon du lieu da tach theo nghi thuc hold-out
# X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=1/3.0)
X_train,X_test,y_train,y_test= train_test_split(X,y)

# #Xay dung mo hinh Bayes voi 2 tap du lieu X_train va y_train
model.fit(X_train, y_train)
#
# #Thuc hien doan nhan cho tap du lieu X con lai va luu nhan cua chung vao vien thucte de doi chieu
#dubao = model.predict(X_test)
# np.array([545,180,183,2,0,1,1,0,0,0,1,1]).reshape(1,12)
dubao = model.predict(np.array([63,1,1,145,233,1,2,150,0,2.3,3,0,6]).reshape(1,13))
thucte = y_test

print (dubao)
# print ("Do chinh xac tong the: ",accuracy_score(thucte,dubao))
