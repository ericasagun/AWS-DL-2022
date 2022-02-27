import pandas as pd
import numpy as np
import os
from numpy import random

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf

from preprocess import create_spacy_docs, get_pos


print('Importing datasets')

# featured_df = pd.read_csv('./datasets/featured_df.csv')

## Training datasets
# df_recipe = pd.read_csv('./datasets/RecipeNLG_dataset.csv')
df_action = pd.read_csv('./datasets/action-items dataset.csv')

# Pos data contains sentences with action items. Neg otherwise
# pos_data = clean_instructions(df_recipe['directions'])

print('Preprocessing data...')
pos_data = list(df_action[df_action['label'] == 1]['sentences'])
neg_data = list(df_action[df_action['label'] == 0]['sentences'])

num_pos = len(pos_data)
num_neg = len(neg_data)
num_per_class = num_pos if num_pos < num_neg else num_neg

random.shuffle(pos_data)
random.shuffle(neg_data)

lines = []
for l in pos_data[:num_per_class]:
    lines.append((l, 'pos'))
for l in neg_data[:num_per_class]:
    lines.append((l, 'neg'))

docs = create_spacy_docs(lines)
featured_df = get_pos(docs)

featured_df.to_csv("./datasets/featured_df.csv", index=False)

X = featured_df.iloc[:, 1:]
y = featured_df.iloc[:, :1]

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y.values.ravel())

NUM_EPOCHS = 100
BATCH_SIZE = 50

print('Initializing model')
model = Sequential()
model.add(Dense(128, input_dim=len(X.columns), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print('Model initialized!')

print('Compiling model')
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])
print('Model compiled')

print('Training model')
model.fit(X, encoded_y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,)
print('Training done!')

model.save('./models/action_item.h5')
print('Model saved!')
