import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import models
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN,Input
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, GRU
from keras.layers import Embedding
from matplotlib import pyplot as plt
from keras.layers import Conv1D, LSTM, MaxPooling1D
from keras.layers import GlobalMaxPooling1D
import warnings
warnings.filterwarnings('ignore')
from MongoDatabase_onderdeel import laad_data

if __name__ == "__main__":
    df = laad_data("cleaned_data")

max_words = 10000  # Maximaal aantal woorden in de vocabulaire
max_sequence_length = 100  # Maximale lengte van elke review

X = df['review'].astype(str)  # Gebruik de 'review'-kolom als tekstuele gegevens
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

model_cnn = Sequential()
model_cnn.add(Embedding(max_words, 100, input_length=max_sequence_length))
model_cnn.add(Conv1D(128, 5, activation='relu'))
model_cnn.add(MaxPooling1D(5))
model_cnn.add(Conv1D(128, 5, activation='relu'))
model_cnn.add(MaxPooling1D(5))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dense(1, activation='sigmoid'))

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_cnn = model_cnn.fit(X_train_pad, y_train, epochs=5, batch_size=128, validation_data=(X_test_pad, y_test))

score = model_cnn.evaluate(X_test_pad, y_test)
print("Test loss (CNN):", score[0])
print("Test accuracy (CNN):", score[1])

# Voor voorspellingen
predictions = model_cnn.predict(X_test_pad)

plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])
plt.title('CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model_cnn.save(r'C:\Users\thijs\Desktop\assignment 2\opgelsagen neural networks\model_cnn.h5')

model_rnn = Sequential()
model_rnn.add(Embedding(max_words, 100, input_length=max_sequence_length))
model_rnn.add(SimpleRNN(128))  # Simple RNN-laag met 128 eenheden
model_rnn.add(Dense(1, activation='sigmoid'))

# Compileer en train het RNN-model
model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_rnn = model_rnn.fit(X_train_pad, y_train, epochs=5, batch_size=128, validation_data=(X_test_pad, y_test))

# Evalueer het RNN-model en gebruik het voor voorspellingen
score_rnn = model_rnn.evaluate(X_test_pad, y_test)
print("Test loss (RNN):", score_rnn[0])
print("Test accuracy (RNN):", score_rnn[1])

predictions_rnn = model_rnn.predict(X_test_pad)

model_rnn.save(r'C:\Users\thijs\Desktop\assignment 2\opgelsagen neural networks\model_rnn.h5')


plt.plot(history_rnn.history['accuracy'])
plt.plot(history_rnn.history['val_accuracy'])
plt.title('RNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model_gru = Sequential()
model_gru.add(Embedding(max_words, 100, input_length=max_sequence_length))
model_gru.add(GRU(128))  # Voeg een GRU-laag toe met 128 eenheden
model_gru.add(Dense(1, activation='sigmoid'))

# Compileer en train het GRU-model
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_gru = model_gru.fit(X_train_pad, y_train, epochs=5, batch_size=128, validation_data=(X_test_pad, y_test))

# Evalueer het GRU-model en gebruik het voor voorspellingen
score_gru = model_gru.evaluate(X_test_pad, y_test)
print("Test loss (GRU):", score_gru[0])
print("Test accuracy (GRU):", score_gru[1])

model_gru.save(r'C:\Users\thijs\Desktop\assignment 2\opgelsagen neural networks\model_gru.h5')


# Voeg een LSTM-laag toe aan het RNN-model
model_lstm = Sequential()
model_lstm.add(Embedding(max_words, 100, input_length=max_sequence_length))
model_lstm.add(LSTM(128))  # Voeg een LSTM-laag toe met 128 eenheden
model_lstm.add(Dense(1, activation='sigmoid'))

# Compileer en train het LSTM-model
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_lstm = model_lstm.fit(X_train_pad, y_train, epochs=5, batch_size=128, validation_data=(X_test_pad, y_test))

# Evalueer het LSTM-model en gebruik het voor voorspellingen
score_lstm = model_lstm.evaluate(X_test_pad, y_test)
print("Test loss (LSTM):", score_lstm[0])
print("Test accuracy (LSTM):", score_lstm[1])

model_lstm.save(r'C:\Users\thijs\Desktop\assignment 2\opgelsagen neural networks\model_lstm.h5')

# Plot de nauwkeurigheid voor de verschillende modellen
plt.plot(history_rnn.history['accuracy'], label='RNN')
plt.plot(history_gru.history['accuracy'], label='GRU')
plt.plot(history_lstm.history['accuracy'], label='LSTM')
plt.title('RNN, GRU, en LSTM Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()