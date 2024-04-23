import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

# Loading IMDB dataset with the most frequent 10,000 words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences to ensure uniform length
X_train = pad_sequences(X_train, maxlen=10000)
X_test = pad_sequences(X_test, maxlen=10000)

# Consolidating data for exploratory data analysis (EDA)
data = np.concatenate((X_train, X_test), axis=0)
label = np.concatenate((y_train, y_test), axis=0)

# Creating train and test dataset
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=1)

# Creating sequential model
model = Sequential()
model.add(Dense(50, activation="relu", input_shape=(10000, )))
model.add(Dropout(0.3))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(50, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Summary of the model
model.summary()

# For early stopping
callback = EarlyStopping(monitor='loss', patience=3)

# Compiling the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Training the model
results = model.fit(
    X_train, y_train,
    epochs=2,
    batch_size=500,
    validation_data=(X_test, y_test),
    callbacks=[callback]
)

# Evaluating the model
score = model.evaluate(X_test, y_test, batch_size=500)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plotting training history
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
