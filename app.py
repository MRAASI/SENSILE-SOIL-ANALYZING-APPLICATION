import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('.\model\crop_model.pkl', 'rb'))
model2=pickle.load(open('.\model\soil_model.pkl','rb'))
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    output=[]
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    input_features1 = [input_features[0],input_features[1],input_features[2]]
    features_value1 = [np.array(input_features1)]
    features_name = ['N','P','K','temperature','humidity','ph','rainfall']
    features_name1=['N','P','K']
    # for crop
    df = pd.DataFrame(features_value, columns=features_name)
    output1 = model.predict(df)
    # for soil condition
    df1=pd.DataFrame(features_value1, columns=features_name1)
    output2=model2.predict(df1)
    output=[output1,output2]
    
    #return render_template('main.html', prediction_text='The recommended crop is  {}'.format(output ))
    return render_template('main.html', prediction_text='The soil condition is ' + output[1] + ' and the recommended crop is ' + output[0] )
                        



if __name__ == "__main__":
    app.run(debug=True)
