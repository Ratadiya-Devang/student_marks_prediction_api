import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score



data = pd.read_csv("logistics_regration.csv")
data

fe = data[['StudyHours']]
ta = data['Pass']


#Train Test Split

fe_train,fe_test,ta_train,ta_test = train_test_split(fe,ta,random_state=12,test_size=0.20) 


model = LogisticRegression(random_state=42)
model.fit(fe_train,ta_train)

import joblib 

joblib.dump(model,"logestic.pkl")