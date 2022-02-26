import re
from collections import defaultdict
from enum import Enum, auto
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# Features as defined in the introduction
class FeatureName(Enum):
    VERB = auto()
    FOLLOWING_POS = auto()
    FOLLOWING_POSTAG = auto()
    CHILD_DEP = auto()
    PARENT_DEP = auto()
    CHILD_POS = auto()
    CHILD_POSTAG = auto()
    PARENT_POS = auto()
    PARENT_POSTAG = auto()

def clean_instructions(df):
    """
    Separate each instructions into different rows

    Args:
        dataset (pd.DataFrame): downloaded dataset

    Returns:
        df (pd.DataFrame): processed dataset
    """
    df = df.str.replace("[", "", regex=True)
    df = df.str.replace("]", "", regex=True)
    
    instructions = []
    for instruction in df:
        tmp_list = instruction.split("\"")
        tmp_list = [x for x in tmp_list if len(x) > 2]
        instructions.append(tmp_list)
    
    instructions = [item for items in instructions for item in items]
    return instructions

def create_spacy_docs(ll):
    dd = [(nlp(l[0]), l[1]) for l in ll]
    return dd

def add_tag(tag_list, tag_dir):
    for tag in tag_list:
        tag_dir.append(tag) if tag not in tag_dir else tag_dir
    return tag_dir

def featurize(d):
    s_features = defaultdict(int)
    for idx, token in enumerate(d):
        if re.match(r'VB.?', token.tag_) is not None:
            s_features[FeatureName.VERB.name] += 1
            
            next_idx = idx + 1
            
            if next_idx < len(d):
                s_features[f'{FeatureName.FOLLOWING_POS.name}_{d[next_idx].pos_}'] += 1
                s_features[f'{FeatureName.FOLLOWING_POSTAG.name}_{d[next_idx].tag_}'] += 1
                
            if (token.head is not token):
                s_features[f'{FeatureName.PARENT_DEP.name}_{token.head.dep_.upper()}'] += 1
                s_features[f'{FeatureName.PARENT_POS.name}_{token.head.pos_}'] += 1
                s_features[f'{FeatureName.PARENT_POSTAG.name}_{token.head.tag_}'] += 1
                
            for child in token.children:
                s_features[f'{FeatureName.CHILD_DEP.name}_{child.dep_.upper()}'] += 1
                s_features[f'{FeatureName.CHILD_POS.name}_{child.pos_}'] += 1
                s_features[f'{FeatureName.CHILD_POSTAG.name}_{child.tag_}'] += 1
    return dict(s_features)

def get_pos(docs, training=True, cols=[]):
    feature_df = pd.DataFrame()

    for i in range(len(docs)):
        features = featurize(docs[i][0])
        
        if training:
            feature_df.at[i, 'label'] = docs[i][1]
            keys = features.keys()
            for key in keys:
                feature_df.at[i, key] = features[key]
        else:
            train_keys = cols
            test_keys = features.keys()
            for train_key in train_keys:
                if train_key in test_keys:
                    feature_df.at[i, train_key] = features[train_key]
                else:
                    feature_df.at[i, train_key] = 0
        
    feature_df = feature_df.fillna(0)
    return feature_df