import numpy as np
import pickle
from flask import Flask,request,render_template
app=Flask(__name__)
model=pickle.load(open('model3.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature_arr=[int(x) for x in request.form.values()]
    final_feat=[np.array(feature_arr)]
    prediction=model.predict(final_feat)
    output=round(prediction[0],2)
    return render_template('index.html',prediction_text='The outut is {}'.format(output))
if __name__=='__main__':
    app.run(debug=True)
