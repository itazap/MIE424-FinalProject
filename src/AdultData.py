#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


def build_adult_data(dataset,sens_attribute = 'sex',load_data_size=None):
    """Building the Adult dataset [Machine Learning Repository] for future use.
    Lohaus' Too Relaxed To Be Fair paper provides further explainations and definitions of each binary mapping.
    The mappings have been kept identical in order to make comparing easier.

    Parameters:
    
        load_data_size: int
            Loaded number of points: if an existing value, it will return the shuffled load_data_size; otherwise, if none type, it will
            return all the data points in an unshuffled manner.

    Returns:
    
        X: numpy.array
            Shape is going to be: (num_data_points, num_features).
            A matrix of the returned feature inputs upon being mapped with binary attributes

        y: numpy.array
            Shape is going to be: (num_data_points, ).
            A matrix of the returned classifications labels upon being mapped with binary attributes

        s: numpy.array
            Shape is going to be: (num_data_points, ).
            A vector of the returned sensitive features upon being mapped with binary attributes. 
    """
  
    def binary_mapping(tuple):
        #   Cutting off the age feature (>37 and <=37) to be a binary attribute.
        tuple['age'] = 1 if tuple['age'] > 37 else 0
        
        #   Transforming the workclass attribute to be binary attribute of Private / NonPrivate
        tuple['workclass'] = 'NonPrivate' if tuple['workclass'] != 'Private' else 'Private'
        
        #   Cutting off the education-num feature (>9 and <=9) to be a binary attribute.
        tuple['education-num'] = 1 if tuple['education-num'] > 9 else 0
        
        #   Transforming the marital-status attribute to be binary attribute of MarriedCivSpouse / NonMarriedCivSpouse
        tuple['marital-status'] = "Marriedcivspouse" if tuple['marital-status'] == "Married-civ-spouse" else "nonMarriedcivspouse"
        
        #   Transforming the occupation attribute to be binary attribute of CraftRepair / NonCraftRepair
        tuple['occupation'] = "Craftrepair" if tuple['occupation'] == "Craft-repair" else "NonCraftrepair"
        
        #   Transforming the relationship attribute to be binary attribute of NotInFamily / InFamily
        tuple['relationship'] = "NotInFamily" if tuple['relationship'] == "Not-in-family" else "InFamily"
        
        #   Transforming the race attribute to be binary attribute of White / NonWhite
        tuple['race'] = 'NonWhite' if tuple['race'] != "White" else "White"
        
        #   Transforming the sex attribute to be binary attribute of Male / Female
        tuple['sex'] = 'Female' if tuple['sex'] != "Male" else 'Male'
        
        #   Cutting off the hours-per-week feature (>40 and <=40) to be a binary attribute.
        tuple['hours-per-week'] = 1 if tuple['hours-per-week'] > 40 else 0
        
        #   Transforming the native-country attribute to be binary attribute of US / NonUS
        tuple['native-country'] = "US" if tuple['native-country'] == "United-States" else "NonUS"

        return tuple

    df = dataset
    df = df.apply(binary_mapping, axis=1)

    #   Conversion of the sensitive attributes' binary mapping to 1 or -1
    if sens_attribute == 'sex':
        sensitive_attr_map = {'Male': 1, 'Female': -1}
        x_vars = ['age','workclass','education-num','marital-status','occupation','relationship','race','hours-per-week','native-country']
    elif sens_attribute == 'race':
        sensitive_attr_map = {'White': 1, 'NonWhite': -1}
        x_vars = ['age','workclass','education-num','marital-status','occupation','relationship','sex','hours-per-week','native-country']
  
    s = df[sens_attribute].map(sensitive_attr_map).astype(int)

    #   Conversion of the label attributes' binary mapping to 1 or -1
    label_map = {'>50K': 1, '<=50K': -1}
    y = df['income'].map(label_map).astype(int)

    #   Building a dataframe object for the input matrix (which is the feature set).
    x = pd.DataFrame(data=None)
    for x_var in x_vars:
        x = pd.concat([x, pd.get_dummies(df[x_var],prefix=x_var, drop_first=False)], axis=1)

    #   Setting to matrices and vectors to be returned as numpy objects.
    X = x.to_numpy()
    s = s.to_numpy()
    y = y.to_numpy()

    #   If: data_size is specified, then: shuffle the data
    if load_data_size is not None:
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm][:load_data_size]
        y = y[perm][:load_data_size]
        s = s[perm][:load_data_size]

    # X = X[:, (X != 0).any(axis=0)]

    return X, y, s

def normalize(x):
    # Normalization to transform between 1 and -1.
    x_ = (x - x.min()) / (x.max() - x.min()) * 2 - 1
    return x_




