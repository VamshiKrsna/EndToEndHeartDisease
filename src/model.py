import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib 

# data is in : data/heart.csv
df = pd.read_csv('../data/heart.csv')

# print(df.head())

X = df.drop(columns=['target'])
y = df['target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = LogisticRegression()

model.fit(X_train,y_train)

y_preds = model.predict(X_test)

print(accuracy_score(y_preds, y_test) * 100)

joblib.dump(model, '../models/heart_disease_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

print("model and scaler have been saved to model dir")