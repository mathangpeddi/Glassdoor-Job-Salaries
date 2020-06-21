# Running the flask server
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#model1 = pickle.load(open('model1.pkl', 'rb')) # model1 is using Lasso Regression
model2 = pickle.load(open('model2.pkl', 'rb')) # model2 is using Linear Regression

@app.route('/') # root page or home page
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST']) # POST method where in we will be providing some features to the model.pkl file so that it will take those inputs and give the output
def predict(): # This is a web API
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model2.predict(final_features)

    output = round(prediction[0], 2) # get the output and round it off to 2 digits

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
