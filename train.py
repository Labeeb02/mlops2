from fileinput import filename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

def train(x):
    filename='dataset/breast_cancer-'+str(x)+'.csv'
    df = pd.read_csv(filename)

    df.head()

    y = df['target']

    X = df.drop('target', axis=1)

    # X

    # y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    
    # X_train

    # y_train

    # from sklearn import preprocessing

    # %%
    # scaler = preprocessing.MinMaxScaler()
    # scaler_obj = scaler.fit(X_train)

    # %%
    # X_train_scaled = scaler_obj.transform(X_train)

    # %%
    # X_train_scaled

    # %%
    # X_test

    # %%
    # X_test_scaled = scaler_obj.transform(X_test)

    # %%
    # X_test_scaled

    # %%
    # scaler_filename = 'scale.pkl'
    # joblib.dump(scaler_obj, scaler_filename)

    from sklearn.svm import SVC

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    svc_model = SVC()

    svc_model.fit(X_train, y_train)

    y_pred = svc_model.predict(X_test)

    model_filename = 'model'+str(x)+'.pkl'
    joblib.dump(svc_model, model_filename)

    y_pred

    cm = confusion_matrix(y_test,y_pred)

    print(classification_report(y_test, y_pred))

    print(accuracy_score(y_test, y_pred))

    TP=cm[1][1]
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]

    print(TP,TN,FP,FN)

    if(TP+FP==0):
        precision=0
    else:
        precision=TP/(TP+FP)
    if(TP+FN==0):
        recall=0
    else:
        recall=TP/(TP+FN)
    if(precision+recall==0):
        f1_score=0
    else:
        f1_score=2*(precision*recall)/(precision+recall)
    if(FP+TN==0):
        false_pos_rate=0
    else:
        false_pos_rate=FP/(FP+TN)
    
    accuracy=(TP+TN)/(TP+TN+FP+FN)

    return {"recall":recall,"false_pos_rate":false_pos_rate,"precision":precision,"accuracy":accuracy,"f1_score":f1_score}




