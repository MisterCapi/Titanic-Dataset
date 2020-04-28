import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import time

with open("X.pickle", 'rb') as file:
    X = pickle.load(file)

with open("y.pickle", 'rb') as file:
    y = pickle.load(file)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = 4e-4
dense1 = 70
dense2 = 25
dropout1 = 0.15
dropout2 = 0.2

model = Sequential()

model.add(Dense(dense1, activation="relu"))
model.add(Dropout(dropout1))

model.add(Dense(dense2, activation="relu"))
model.add(Dropout(dropout2))

model.add(Dense(1, activation="sigmoid"))

model.compile(Adam(lr=lr), loss="binary_crossentropy", metrics=["accuracy"])

log_dir = "logs\\fit\\" + f"lr={lr} dense ({dense1}, {dense2}) drop ({dropout1}, {dropout2}) " + str(int(time.time()))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
model_saver = tf.keras.callbacks.ModelCheckpoint('saved_model', monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

model.fit(X, y, epochs=150, callbacks=[model_saver])

print(model.evaluate(X, y))
