# using flask_restful 
from flask import Flask, jsonify, request, make_response 
from flask_restful import Resource, Api 
import pickle
import numpy as np
import json
# creating the flask app 
app = Flask(__name__) 
# creating an API object 
api = Api(app) 

model = pickle.load(open("GBR_model.pkl","rb"))
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

  
  
# driver function 
if __name__ == '__main__':
    app.run(debug = True) 