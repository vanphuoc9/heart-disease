
# ReadData(56,0,2,140,294,0,2,153,0,1.3,2,0,3)

from flask import Flask, request, jsonify
app = Flask(__name__)
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import sklearn
from sklearn.preprocessing import Imputer
#Tao ra mo hinh xac suat Bayes thong qua thu vien
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# #Tao ra mo hinh xac suat Bayes thong qua thu vien
from sklearn.naive_bayes import GaussianNB

#Chon du lieu da tach theo nghi thuc hold-out

# #Xay dung mo hinh Bayes voi 2 tap du lieu X_train va y_train
# def ReadData(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slop,ca,thal):
#     model.fit(X_train, y_train)
#     dubao = model.predict(np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slop,ca,thal]).reshape(1,13))
#     return dubao[0]





@app.route('/getResult',methods=['GET','POST'])

def index():
    model = GaussianNB()
    #Doc du lieu tu file
    dataset = pd.read_csv("Heart_Disease_Data.csv",na_values="?", low_memory = False)
    # doi cac gia tri 1, 2, 3, 4 ve 1
    dataset["pred_attribute"].replace(inplace=True, value=[1, 1, 1, 1], to_replace=[1, 2, 3, 4])
    # Load data
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values # = dataset.iloc[:, 13].values
    my_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    my_imputer = my_imputer.fit(X[:,0:13])
    X[:, 0:13] = my_imputer.transform(X[:, 0:13])
    X_train,X_test,y_train,y_test= train_test_split(X,y)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    #get data from json
    age = request.json['age']
    sex = request.json['sex']
    cp = request.json['cp']
    trestbps = request.json['trestbps']
    chol = request.json['chol']
    fbs = request.json['fbs']
    restecg = request.json['restecg']
    thalach = request.json['thalach']
    exang = request.json['exang']
    oldpeak = request.json['oldpeak']
    slop = request.json['slop']
    ca = request.json['ca']
    thal = request.json['thal']

    #read model_selection
    modeltest = joblib.load('model.pkl')
    dubao = modeltest.predict_proba(np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slop,ca,thal]).reshape(1,13))
     # persist model
    return jsonify({'result': round(dubao[0,1]), 'accuracy': round(model.score(X_train, y_train) * 100, 2)})
    # return jsonify({"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol":chol, "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak, "slop": slop, "ca": ca, "thal": thal})


if __name__ == "__main__":
    app.run(host='172.20.10.2',debug=True, port=5000)
