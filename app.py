from flask import Flask, jsonify, request
import os
import os.path
# import torch
# from transformers import BertTokenizer
from flask_cors import CORS
from train import train
# from toxicClassifier import ToxicCommentClassifier
import joblib
import pandas as pd

# BERT_MODEL_NAME = "bert-base-cased"
# tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
# CLASSES=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# model = torch.load("final_model.pkl")
# model.eval()
import pyrebase
config =  {
  "apiKey": "AIzaSyDe8a1lmQZw62myaYiMfwUVsYoYo0g8i9k",
  "authDomain": "toxic-mlops.firebaseapp.com",
  "databaseURL": "https://toxic-mlops-default-rtdb.firebaseio.com",
  "projectId": "toxic-mlops",
  "storageBucket": "toxic-mlops.appspot.com",
  "messagingSenderId": "671461749137",
  "appId": "1:671461749137:web:56b19d17b308622d892a30",
  "measurementId": "G-P6LY2BNB0P"
}
firebase=pyrebase.initialize_app(config)
db=firebase.database()


app = Flask(__name__)
CORS(app)

version=0

while True:
    model_filename = 'model'+str(version+1)+'.pkl'
    model_exists = os.path.exists(model_filename)
    if(not model_exists):
        break
    version+=1



# @app.route('/predict', methods=['POST'])
# def predict():

#     req = request.get_json()
    
#     input_data = req['data']

#     test=input_data

#     encoding = tokenizer.encode_plus(
#     test,
#     add_special_tokens=True,
#     max_length=128,
#     return_token_type_ids=False,
#     padding="max_length",
#     truncation=True,
#     return_attention_mask=True,
#     return_tensors="pt"
#     )
#     model.eval()
#     _, preds = model(encoding["input_ids"], encoding["attention_mask"])
#     preds = preds.flatten().detach().numpy()
#     predictions = []
#     for idx, label in enumerate(CLASSES):
#         if preds[idx] > 0.5:
#             predictions.append((label, round(preds[idx]*100, 2)))
    
#     if(predictions==[]):
#         return jsonify({"predictions": [("Non toxic", 0.0)]})
#     else:
#         return jsonify({"predictions": predictions})

@app.route('/predict', methods=['POST'])
def predict():

    req = request.get_json()
    
    model_version = request.args.get('version')

    model_filename = 'model'+str(model_version)+'.pkl'
    model_exists = os.path.exists(model_filename)
    if(not model_exists):
        return jsonify({"error": "Model not found"})
    model = joblib.load(model_filename)

    input_data = req['data']
    input_data_df = pd.DataFrame.from_dict(input_data)
    
    # scale_obj = joblib.load('scale.pkl')

    # input_data_scaled = scale_obj.transform(input_data_df)

    # print(input_data_scaled)

    prediction = model.predict(input_data_df)

    if prediction[0] == 1.0:
        cancer_type = 'Malignant Cancer'
    else:
        cancer_type = 'Benign Cancer'
        
    return jsonify({'output':{'cancer_type':cancer_type},'version':model_version})

@app.route('/training')
def training():
    global version
    version=version+1
    ls=train(version)
    db.child("version").child(str(version)).set(ls)
    return "Model Version "+str(version)+" now Online"

@app.route('/metrics')
def metrics():

    model_version = int(request.args.get('version'))
    metrics=db.child("version").get()
    if((len(metrics.val())-1)<model_version):
        return jsonify({"error": "Model not found"})
    return jsonify(metrics.val()[model_version])


@app.route('/')
def home():
    return "Welcome to Toxicity Detection"


if __name__=='__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', '3000'))
