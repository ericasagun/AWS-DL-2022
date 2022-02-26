import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from action_item_classifier.preprocess import create_spacy_docs, get_pos, featurize
import tensorflow as tf
import os
os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

columns = pd.read_csv('./datasets/featured_df.csv', nrows=1).columns.tolist()

def pred_sent(df):
    print('Importing model')
    model = tf.keras.models.load_model('./models/action_item.h5')
    results = model.predict(df)
    return results

    
def pred_action_items(text):
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




    #     if pred_sent:
    #         action_items.appent(sent)
    # return action_items