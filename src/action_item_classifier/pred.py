import pandas as pd
import numpy as np
import nltk

if not dir(nltk.punkt):
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from action_item_classifier.preprocess import create_spacy_docs, get_pos, featurize
import tensorflow as tf
import os
os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

columns = pd.read_csv('./datasets/featured_df.csv', nrows=1).columns.tolist()

print('Importing model')
model = tf.keras.models.load_model('./models/action_item.h5')
print('Model imported!')

def pred_sent(sentence):
    """
    Predicts action items per sentence

    Args:
        string: sentence

    Returns:
        array: probability
    """
    print('Predicting...')
    results = model.predict(sentence)
    return results

    
def pred_action_items(text):
    """
    Returns action items per summary

    Args:
        string: summary

    Returns:
        list: action items
    """
    print('Preprocessing data')
    sentences = sent_tokenize(text)
    sentence_list = []
    for sent in sentences:
        sentence_list.append((sent, 'p'))
    docs = create_spacy_docs(sentence_list)
    df = get_pos(docs, False, columns[1:])

    results = pred_sent(df)
    index = list(np.where(np.round(results) == 1)[0])

    action_items = []
    for i in index:
        action_items.append(sentence_list[i][0])

    return action_items
