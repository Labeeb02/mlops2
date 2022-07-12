from ensurepip import version
import pyrebase
from train import train
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

# data = {"accuracy": 0.09,"status": "ready"}
# db.child("version").child("1").set(data)
version=db.child("version").get()
# print(len(version.val()))

print(version.val()[2]["f1_score"])
# users = db.child("users").get()
# print(users.val()['Morty']['name']) # {"Morty": {"name": "Mortimer 'Morty' Smith"}, "Rick": {"name": "Rick Sanchez"}}