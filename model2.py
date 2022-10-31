import pandas as pd
import numpy as np
import pickle 
data=pd.read_csv('hiring.csv')
data['experience']=data['experience'].fillna(0)
data['test_score'].mean()
data['test_score']=data['test_score'].fillna(data['test_score'].mean())
def convert_to_numeric(word):
    word_dic={0: 0,'one':1,'two':2,'three':3,'five':5,'seven':7,'ten':10,'eleven':11}
    return word_dic[word]
data['experience']=data['experience'].apply(lambda x: convert_to_numeric(x))
from sklearn.linear_model import LinearRegression
lns=LinearRegression()
x=data.iloc[:,:3]
y=data.iloc[:,-1]
lns.fit(x.values,y.values)
pickle.dump(lns,open('model2.pkl','wb'))
model2=pickle.load(open('model2.pkl','rb'))
print(model2.predict([[5,6,7]]))