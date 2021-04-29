#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


def build_adult_data(dataset,sens_attribute = 'sex',load_data_size=None):
    """Build the Adult dataset.
    Source: UCI Machine Learning Repository.
    All Binary Mappings are defined in Lohaus' Too Relaxed To Be Fair research.
    We have kept the exact same mapping to make for clean comparisons. 

    Parameters
    ----------
    load_data_size: int
      The number of points to be loaded. If None, returns all data points unshuffled.
      If other than None, returns load_data_size shuffled.

    Returns
    ---------
    X: numpy array
      The feature input matrix after a binary mapping of attributes.
      Shape=(number_data_points, number_features)
    y: numpy array
      The classification labels (matrix) after a binary mapping of attributes.
      Shape=(number_data_points,).
    s: numpy array
      The sensitive feature vector after a binary mapping of attributes. 
      Shape=(number_data_points,).
    """
  
    def binary_mapping(tuple):
        # 'age'- Binary Cut-off: 37
        tuple['age'] = 1 if tuple['age'] > 37 else 0
        # 'workclass'- Binary Translation to Private/NonPrivate
        tuple['workclass'] = 'NonPrivate' if tuple['workclass'] != 'Private' else 'Private'
        # 'education-num'- Binary Cut-off: 9
        tuple['education-num'] = 1 if tuple['education-num'] > 9 else 0
        # 'maritial-status'- Binary Translation to Married-civ-spouse/nonMarriedcivspouse
        tuple['marital-status'] = "Marriedcivspouse" if tuple['marital-status'] == "Married-civ-spouse" else "nonMarriedcivspouse"
        # 'occupation'- Binary Translation to Craft-repair/NonCraftrepair
        tuple['occupation'] = "Craftrepair" if tuple['occupation'] == "Craft-repair" else "NonCraftrepair"
        # 'relationship'- Binary Translation to InFamily/Not-in-family
        tuple['relationship'] = "NotInFamily" if tuple['relationship'] == "Not-in-family" else "InFamily"
        # 'race'- Binary Translation to White/NonWhite
        tuple['race'] = 'NonWhite' if tuple['race'] != "White" else "White"
        # 'sex'- Binary Translation to Male/Female
        tuple['sex'] = 'Female' if tuple['sex'] != "Male" else 'Male'
        # 'hours-per-week'- Binary Cut-off: 40
        tuple['hours-per-week'] = 1 if tuple['hours-per-week'] > 40 else 0
        # 'native-country'- Binary Translation to United-States/NonUS
        tuple['native-country'] = "US" if tuple['native-country'] == "United-States" else "NonUS"

        return tuple

    df = dataset
    df = df.apply(binary_mapping, axis=1)

    # Convert Binary Mapping of Sensitive Attribute to {1,-1}
    if sens_attribute == 'sex':
        sensitive_attr_map = {'Male': 1, 'Female': -1}
        x_vars = ['age','workclass','education-num','marital-status','occupation','relationship','race','hours-per-week','native-country']
    elif sens_attribute == 'race':
        sensitive_attr_map = {'White': 1, 'NonWhite': -1}
        x_vars = ['age','workclass','education-num','marital-status','occupation','relationship','sex','hours-per-week','native-country']
  
    s = df[sens_attribute].map(sensitive_attr_map).astype(int)

    # Convert Binary Mapping of Label Attribute to {1,-1}
    label_map = {'>50K': 1, '<=50K': -1}
    y = df['income'].map(label_map).astype(int)

    # Build Input Matrix (Feature Set) as a proper DataFrame
    x = pd.DataFrame(data=None)
    for x_var in x_vars:
        x = pd.concat([x, pd.get_dummies(df[x_var],prefix=x_var, drop_first=False)], axis=1)

    # Return as numpy objects: Matrix/Vectors
    X = x.to_numpy()
    s = s.to_numpy()
    y = y.to_numpy()

    if load_data_size is not None:
    # Shuffle the data only if data_size is specified (Random detail from Paper code)
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm][:load_data_size]
        y = y[perm][:load_data_size]
        s = s[perm][:load_data_size]

    # X = X[:, (X != 0).any(axis=0)]

    return X, y, s

def normalize(x):
    # scale to [-1, 1]
    x_ = (x - x.min()) / (x.max() - x.min()) * 2 - 1
    return x_




