import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import Imputer
#Tao ra mo hinh xac suat Bayes thong qua thu vien
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# #Tao ra mo hinh xac suat Bayes thong qua thu vien
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

model = GaussianNB()
#Doc du lieu tu file
dataset = pd.read_csv("Heart_Disease_Data.csv",na_values="?", low_memory = False)
# doi cac gia tri 1, 2, 3, 4 ve 1
dataset["pred_attribute"].replace(inplace=True, value=[1, 1, 1, 1], to_replace=[1, 2, 3, 4])
# # 13 dataset features
# feature13 = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slop','ca','thal']
# print dataset.isnull().sum()
# Load data
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values # = dataset.iloc[:, 13].values
my_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
my_imputer = my_imputer.fit(X[:,0:13])
X[:, 0:13] = my_imputer.transform(X[:, 0:13])
scaler = StandardScaler()
X = scaler.fit_transform(X)
#Chon du lieu da tach theo nghi thuc hold-out
X_train,X_test,y_train,y_test= train_test_split(X,y)
# #Xay dung mo hinh Bayes voi 2 tap du lieu X_train va y_train
print(X)
#
# #Thuc hien doan nhan cho tap du lieu X con lai va luu nhan cua chung vao vien thucte de doi chieu
# dubao = model.predict(np.array([56,0,2,140,294,0,2,153,0,1.3,2,0,3]).reshape(1,13))
# thucte = y_test
# print (dubao)
def ReadData(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slop,ca,thal):
    model.fit(X_train, y_train)
    dubao = model.predict(np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slop,ca,thal]).reshape(1,13))
    return dubao[0]

print (ReadData(65,0,4,150,225,0,2,114,0,1,2,3,7))

# print ("Do chinh xac tong the: ",accuracy_score(thucte,dubao))
#### KNN
##to choose the right K we build a loop witch examen all the posible values for K.
