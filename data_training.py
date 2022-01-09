import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.layers.recurrent import LSTM

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.layers import Dropout, BatchNormalization

df = pd.read_csv('Modified.csv')

training_sentences = []
training_labels = []
ignore_letters = ['?', '!', '.', ',']
for sent in df['Combined']:
    training_sentences.append(sent)
for i in df['Title']:
    training_labels.append(i)
labels = []
for label in training_labels:
    if label not in labels:
        labels.append(label)
num_classes = len(labels)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 15000
embedding_dim = 512
max_len = 4000
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)


# def get_model():
#     inputs = keras.Input(shape=(4000,))
#
#     x = Embedding(vocab_size, embedding_dim)(inputs)
#     x=GlobalAveragePooling1D()(x)
#     x = Dropout(0.75)(x)
#
#     # x = LSTM(64)(x)
#     # x = Dropout(0.6)(x)
#
#     x = Dense(32)(x)
#     x = BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = Dropout(0.3)(x)
#
#     x = Dense(10)(x)
#     x = BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = Dropout(0.3)(x)
#
#     outputs = Dense(num_classes, activation='sigmoid')(x)
#
#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model
#
#
# model = get_model()
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.4))
# model.add(LSTM(32))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.35))
# model.add(Dense(3000, activation='relu'))
# model.add(Dense(5000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

epochs = 50
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs, batch_size=16, verbose=1)

model.save("chat_model")

import pickle

# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
