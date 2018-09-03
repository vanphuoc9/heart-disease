
# Import environment tools
import re
import itertools
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


import sklearn
from sklearn.preprocessing import Imputer
from sklearn import decomposition, preprocessing, svm
# Imputation
# from sklearn.impute import SimpleImputer
# Import relevant machine learning models
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# #Tao ra mo hinh xac suat Bayes thong qua thu vien
from sklearn.naive_bayes import GaussianNB
# Instance Based
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import VotingClassifier
#Import Dataset
dataset = pd.read_csv("Heart_Disease_Data.csv",na_values="?", low_memory = False)
# doi cac gia tri 1, 2, 3, 4 ve 1
dataset["pred_attribute"].replace(inplace=True, value=[1, 1, 1, 1], to_replace=[1, 2, 3, 4])
# np_dataset = np.asarray(dataset)
# print (np_dataset)


# np_dataset = np.asarray(dataset)
feature13 = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slop','ca','thal']
# #Review all features
# columns=dataset.columns[:13]
# plt.subplots(figsize=(20,15))
# length=len(columns)
# for i,j in itertools.zip_longest(columns,range(length)):
#     plt.subplot((length/2),3,j+1)
#     plt.subplots_adjust(wspace=0.2,hspace=1.0)
#     dataset[i].hist(bins=20,edgecolor='black')
#     plt.title(i)
# plt.show()
#
# #Only heart disease partipants
# dataset_copy=dataset[dataset['pred_attribute']==1]
# columns=dataset.columns[:13]
# plt.subplots(figsize=(20,15))
# length=len(columns)
# for i,j in itertools.zip_longest(columns,range(length)):
#     plt.subplot((length/2),3,j+1)
#     plt.subplots_adjust(wspace=0.2,hspace=1.0)
#     dataset_copy[i].hist(bins=20,edgecolor='black')
#     plt.title(i)
# plt.show()

#Review continuous features
features_continuous=["age", "trestbps", "chol", "thalach", "oldpeak", "pred_attribute"]
#features_continuous=["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# dataset = dataset.drop(columns = ["age", "trestbps", "chol", "thalach", "oldpeak"])
# sns.pairplot(dataset,hue='pred_attribute',diag_kind='kde')
#plt.gcf().set_size_inches(20,15)
# sns.boxplot(data=dataset[features_continuous])
# plt.gcf().set_size_inches(20,15)
# plt.show()
# sns.pairplot(data=dataset[features_continuous],diag_kind='kde', hue = 'pred_attribute')
#
# #plt.gcf().set_size_inches(20,15)
# plt.show()
# # # print (dataset[features_continuous])
# # Diagnosis of heart disease (angiographic disease status): "pred_attribute"
# # Value 0 = diameter narrowing <= 50% (Healthy)
# # Value 1 = diameter narrowing > 50% (Sick)
# sns.countplot(x='pred_attribute',data=dataset)
# plt.show()
# # Sex: "sex"
# # Value 0 = female
# # Value 1 = male
# sns.countplot(x='sex',data=dataset)
# plt.show()

healthy = dataset[(dataset['pred_attribute'] ==0) ].count()[1]
sick = dataset[(dataset['pred_attribute'] ==1) ].count()[1]
print ("num of pepole without heart deacise: "+ str(healthy))
print ("num of pepole with chance for heart deacise: "+ str(sick))
X = dataset.iloc[:, :-1].values
# print (X)
y = dataset.iloc[:, -1].values # = dataset.iloc[:, 13].values
my_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
my_imputer = my_imputer.fit(X[:,0:13])
X[:, 0:13] = my_imputer.transform(X[:, 0:13])
#High variance has to be standardised
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)
