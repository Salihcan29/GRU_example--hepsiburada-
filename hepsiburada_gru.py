# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 19:18:19 2020

@author: Salihcan
"""

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,GRU,Embedding,CuDNNGRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

data = pd.read_csv("hepsiburada.csv")

x = data["Review"]
y = data["Rating"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

num_words = 10000 # Max vocab length

tokenizer = Tokenizer(num_words = num_words)

tokenizer.fit_on_texts(x_train)

x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)

max_tokens = [len(tokens) for tokens in x_train_tokens+x_test_tokens]
max_tokens = int(np.mean(max_tokens)+np.std(max_tokens)*2)

x_train_pad = pad_sequences(x_train_tokens,maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens,maxlen=max_tokens)

idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(),idx.keys()))

def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token != 0]
    text = ' '.join(words)
    return text

# %%
model = Sequential()
embedding_size = 50

model.add(Embedding(input_dim=num_words,
                    output_dim = embedding_size,
                    input_length =max_tokens,
                    name = "embedding_layer"))

model.add(GRU(units=16,return_sequences=True))
model.add(GRU(units=8,return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1,activation = "sigmoid"))

model.compile(optimizer = "adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(x_train_pad,y_train)

def tahmin(text):
    text = [text]
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text,maxlen = max_tokens)
    print(model.predict(text))


