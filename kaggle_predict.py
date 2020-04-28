from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import pickle

def Binarize(value):
    if value >= 0.7:
        return 1
    return 0

with open("X_test.pickle", 'rb') as file:
    X = pickle.load(file)

model = load_model("saved_model")

result = model.predict(X)

df = pd.read_csv("gender_submission.csv")
df["Survived"] = list(map(Binarize, result))
df.set_index("PassengerId", inplace=True)

df.to_csv("results.csv")
