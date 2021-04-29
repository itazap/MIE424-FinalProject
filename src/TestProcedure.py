#!/usr/bin/env python
# coding: utf-8

from src.AdultData import build_adult_data, normalize
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time
import numpy as np
from random import shuffle

class TestProcedure():
    def __init__(self,model):

        self.model = model

    def BuildDataset(self,sens_attribute,dataset,train_size = 1200):
        self.sens_attribute = sens_attribute

        # Adult Dataset creation and building into x,y,s
        x_data, y_data, s_data = build_adult_data(dataset=dataset,sens_attribute=sens_attribute,load_data_size=None)
        
        # Split size of training and testing data
        train_size = 1200
        
        # Split data
        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x_data, y_data, s_data, train_size=train_size, shuffle=True)
        
        self.X_train = x_train
        self.y_train = y_train

        self.X_test = x_test
        self.y_test = y_test

        self.s_train = s_train
        self.s_test = s_test
                
    def BuildModel(self):
        
        start_time = time.time()
        
        self.model.fit(self.X_train,self.y_train,self.s_train)
        
        end_time = time.time()
        build_time = end_time - start_time
        
        return build_time
        
    def RunTest(self,sens_attribute,dataset):
        self.BuildDataset(sens_attribute,dataset)
        build_time = self.BuildModel()
        predictions = self.model.predict(self.X_test)
        prediction_accuracy = np.equal(self.y_test, predictions).mean()
        
        ddp,deo = self.compute_fairness_measures(predictions, self.y_test ,self.s_test)
        results = {"BuildTime":build_time,"PredictionAccuracy":prediction_accuracy,"DDP":ddp,"DEO":deo}
        self.PrintResults(results)
        return results
        
    def compute_fairness_measures(self, y_predicted, y_true, sens_attr):
        positive_rate_prot = self.get_positive_rate(y_predicted[sens_attr==-1], y_true[sens_attr==-1])
        positive_rate_unprot = self.get_positive_rate(y_predicted[sens_attr==1], y_true[sens_attr==1])
        true_positive_rate_prot = self.get_true_positive_rate(y_predicted[sens_attr==-1], y_true[sens_attr==-1])
        true_positive_rate_unprot = self.get_true_positive_rate(y_predicted[sens_attr==1], y_true[sens_attr==1])
        DDP = positive_rate_unprot - positive_rate_prot
        DEO = true_positive_rate_unprot - true_positive_rate_prot

        return DDP, DEO

    def get_positive_rate(self, y_predicted, y_true):
        tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_predicted.astype(int)).ravel()
        pr = (tp+fp) / (tp+fp+tn+fn)
        return pr

    def get_true_positive_rate(self, y_predicted, y_true):
        tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_predicted.astype(int)).ravel()
        tpr = tp / (tp+fn)
        return tpr
        
    def PrintResults(self,results):
        print("Sensitive Attribute:",self.sens_attribute)
        print("Kernel Type:",self.model.kernel)
        print("Loss Func:",self.model.loss_name)
        print("Run Time:",round(results['BuildTime'],4),"seconds")
        print("Prediction Accuracy:",str(round(results['PredictionAccuracy']*100,4)),"%")
        print("DDP Score:",str(round(results['DDP'],4)))
        print("DEO Score:",str(round(results['DEO'],4)))




