from flask import Flask, render_template, request, make_response, redirect
import pickle
import numpy as np
import json

from flask_restful import Resource, Api 

app = Flask(__name__)

api = Api(app) 

model = pickle.load(open("GBR_model.pkl","rb"))

## selected features for model are: ['smoker', 'bmi', 'age']

@app.route('/')
def insurance_form():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():    
    if request.method == 'POST':
        print(request.data)
        result = request.form
        #print(result)
        age = np.sqrt(int(result['age']))
        height = int(result['height'])
        weight = int(result['weight'])
        smoker = result['smoker']
        label_mapping = {'yes':1, 'no':0}
        smoker = label_mapping[smoker]
        bmi = weight/(height/100)**2
        if(bmi<16):
            bmi=16
        elif(bmi>53):
            bmi=53

        charges = model.predict(np.reshape([smoker, bmi, age],(1,-1)))
        x = {'charges' : charges[0]}
        
        response = make_response(json.dumps(x))
        response.content_type = 'application/json'
        return response

"""
@app.route('/charges',methods = ['POST', 'GET'])
def charges():
    if request.method == 'POST':
        data = request.get_json()
        try:
            age = np.sqrt(int(data['age']))
            height = int(data['height'])
            weight = int(data['weight'])
            smoker = int(data['smoker'])
        except:
            print("Error in fetched values")
            return None
            
        bmi = weight/(height/100)**2
        if(bmi<16):
            bmi=16
        elif(bmi>53):
            bmi=53
        
        charges = model.predict(np.reshape([smoker, bmi, age],(1,-1)))
        x = {'charges' : int(charges[0])} 
        response = make_response(json.dumps(x))
        response.content_type = 'application/json'
        print("Success")
        return response
"""
# making a class for a particular resource 
# the get, post methods correspond to get and post requests 
# they are automatically mapped by flask_restful. 
# other methods include put, delete, etc. 
class Charges(Resource): 
    # Corresponds to POST request 
    def post(self):
        data = request.get_json()
        try:
            age = np.sqrt(int(data['age']))
            height = int(data['height'])
            weight = int(data['weight'])
            smoker = int(data['smoker'])
            print("Combined works!!")
        except:
            print("Error in fetched values")
            return None
            
        bmi = weight/(height/100)**2
        if(bmi<16):
            bmi=16
        elif(bmi>53):
            bmi=53
        
        charges = model.predict(np.reshape([smoker, bmi, age],(1,-1)))
        x = {'charges' : int(charges[0])} 
        response = make_response(json.dumps(x))
        response.content_type = 'application/json'
        print("Success")
        return response
  
# adding the defined resources along with their corresponding urls 
api.add_resource(Charges, '/charges') 

if __name__ == "__main__":        
    app.run(host="0.0.0.0", port=5000, debug=True)