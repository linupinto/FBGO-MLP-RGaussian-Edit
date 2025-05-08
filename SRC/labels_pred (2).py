# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 19:16:50 2025

@author: HP
"""
import numpy as np

#--------- Prediction Label Dry Beans (7Class)------------
def labels_7(prediction):
    y_predict=np.zeros(len(prediction.T))
    y_predict = np.array(y_predict)
    for i in range(0,len(prediction.T)):
        if prediction[0,i] < 0.07:
            y_predict[i] = 0
        elif prediction[0,i] < 0.21:
            y_predict[i] = 1
        elif prediction[0,i] < 0.35:
            y_predict[i] = 2
        elif prediction[0,i] < 0.49:
            y_predict[i] = 3
        elif prediction[0,i] < 0.64:
            y_predict[i] = 4
        elif prediction[0,i] < 0.78:
            y_predict[i] = 5
        else:
            y_predict[i] = 6
    return y_predict

#------------- Prediction Label CIFAR ( 10 Classes)----------
def labels_10(prediction):
    y_predict=np.zeros(len(prediction.T))
    y_predict = np.array(y_predict)
    for i in range(0,len(prediction.T)):
        if prediction[0,i] < 0.05:
            y_predict[i] = 0
        elif prediction[0,i] < .15:
            y_predict[i] = 1
        elif prediction[0,i] < 0.25:
            y_predict[i] = 2
        elif prediction[0,i] < 0.35:
            y_predict[i] = 3
        elif prediction[0,i] < 0.45:
            y_predict[i] = 4
        elif prediction[0,i] < 0.55:
            y_predict[i] = 5
        elif prediction[0,i] < 0.65:
            y_predict[i] = 6
        elif prediction[0,i] < 0.75:
            y_predict[i] = 7
        elif prediction[0,i] < 0.85:
            y_predict[i] = 8
        else:
            y_predict[i] = 9
    return y_predict




