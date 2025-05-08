# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:55:39 2025

@author: HP
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#------- confusin matrix for Iris data set --- 3 class ---
def confmat_iris(x_true, x_predict):
    y_predict = np.zeros(len(x_true))
    y_predict = np.array(y_predict)
    print(type(y_predict))
    for i in range(0,len(x_true)):
        if x_predict[i] < 0.5:
            y_predict[i] = 0
        elif x_predict[i] < 1.5:
            y_predict[i] = 1
        
        else:
            y_predict[i] = 2
     
         
    # Confusion matrix and related metrics
    cm = confusion_matrix(x_true, y_predict)
    cm_df = pd.DataFrame(cm,
                         index = ['class 0', 'class 1','class 2'],
                         columns = ['class 0', 'class 1', 'class 2'])
    
    # Plotting Confusion matrix
    plt.figure(1,figsize=(15,10))
    ax=sns.heatmap(cm_df, annot=True, fmt='d',cmap='Reds', annot_kws={"size": 16, "weight": "bold"})
    plt.title('Confusion Matrix',fontsize=14)
    plt.ylabel('Actual Values',fontsize=14)
    plt.xlabel('Predicted Values',fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=13)
    return cm
#--------------Confusion matrix CIFAR ----- 10 class---- 
def confmat_CIFAR(x_true, x_predict):
    y_predict = np.zeros(len(x_true))
    y_predict = np.array(y_predict)
    print(type(y_predict))
    for i in range(0,len(x_true)):
        if x_predict[i] < 0.05:
            y_predict[i] = 0
        elif x_predict[i] < .15:
            y_predict[i] = 1
        elif x_predict[i] < 0.25:
            y_predict[i] = 2
        elif x_predict[i] < 0.35:
            y_predict[i] = 3
        elif x_predict[i] < 0.45:
            y_predict[i] = 4
        elif x_predict[i] < 0.55:
            y_predict[i] = 5
        elif x_predict[i] < 0.65:
            y_predict[i] = 6
        elif x_predict[i] < 0.75:
            y_predict[i] = 7
        elif x_predict[i] < 0.85:
            y_predict[i] = 8
        else:
            y_predict[i] = 9
 

        x_true=x_true*10
     
# Confusion matrix and related metrics

        cm = confusion_matrix(x_true, y_predict)
        cm_df = pd.DataFrame(cm,
                     index = ['class 0', 'class 1','class 2', 'class 3', 'class 4', 'class 5', 'class 6','class 7', 'class 8', 'class 9'],
                     columns = ['class 0', 'class 1','class 2', 'class 3', 'class 4', 'class 5', 'class 6','class 7', 'class 8', 'class 9'])

        # Plotting Confusion matrix
        plt.figure(1,figsize=(20,10))
        ax=sns.heatmap(cm_df, annot=True, fmt='d',cmap='Reds', annot_kws={"size": 16, "weight": "bold"})
        plt.title('Confusion Matrix',fontsize=14)
        plt.ylabel('Actual Values',fontsize=14)
        plt.xlabel('Predicted Values',fontsize=14)
        
        ax.tick_params(axis='both', which='major', labelsize=13)
        return cm  
#--------------- Confusion matrix Dry Beans and glass---- 7 class---------------
def confmat_Drybeans(x_true, x_predict):
    y_predict = np.zeros(len(x_true))
    y_predict = np.array(y_predict)
    print(type(y_predict))
    for i in range(0,len(x_true)):
        if x_predict[i] < 0.07:
            y_predict[i] = 0
        elif x_predict[i] < 0.21:
            y_predict[i] = 1
        elif x_predict[i] < 0.35:
            y_predict[i] = 2
        elif x_predict[i] < 0.49:
            y_predict[i] = 3
        elif x_predict[i] < 0.64:
            y_predict[i] = 4
        elif x_predict[i] < 0.78:
            y_predict[i] = 5
        else:
            y_predict[i] = 6
     
    x_true=x_true*7
    
    

    x_true = np.round(x_true).astype(int)
    #print(x_true.T)
    y_predict = np.round(y_predict).astype(int)
    print(y_predict)
    print(x_true.T)
    print("Unique values in true labels (x_true):", np.unique(x_true))
    print("Unique values in predicted labels (y_predict):", np.unique(y_predict))
    
    
    # Specify all possible classes (0 to 6)
    all_classes = [0, 1, 2, 3, 4, 5, 6]

    # --------Confusion matrix and related metrics------------
    cm = confusion_matrix(x_true, y_predict,labels=all_classes)
    cm_df = pd.DataFrame(cm,
                         index = ['class 0', 'class 1','class 2', 'class 3', 'class 4', 'class 5', 'class 6'],
                         columns = ['class 0', 'class 1','class 2', 'class 3', 'class 4', 'class 5', 'class 6'])
    
    # Plotting Confusion matrix
    plt.figure(1,figsize=(20,10))
    ax=sns.heatmap(cm_df, annot=True, fmt='d',cmap='Reds', annot_kws={"size": 16, "weight": "bold"})
    plt.title('Confusion Matrix',fontsize=14)
    plt.ylabel('Actual Values',fontsize=14)
    plt.xlabel('Predicted Values',fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=13)
    return cm