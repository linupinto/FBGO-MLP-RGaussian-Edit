# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:49:05 2025

@author: HP
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv


# import data using pandas
path=r"C:\Users\HP\Desktop\D folder\Linu_savitha_2\Main pgms\Gitshub"
dataset = pd.read_csv(f"{path}\\Dataset\std_cifar.csv")
# Select the input features as x output features as y
X = dataset.iloc[:, 0:3072].values
Y = dataset.iloc[:, 3072].values[:, np.newaxis]
Y=Y/10


# split data into train and test set
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=2)
n = train_X.shape[1]  # no of input features (n- no of columns)
m = train_X.shape[0]  # no of training samples (m- no of rows)


# specify the no of layers and neurons in each layer
layer_dims = [n,6, 5, 1]
L = len(layer_dims) - 1  # Total layers other than input layer

# specify the learning rate
learningRate =0.1
ep=10**-5 # Epislon value
momentum = 0.9  # Momentum for moving averages

# Initializing  weights and dimension in ech layer
parameters = {}
activation = {}
running_mean = {}
running_var = {}

activation[f"A{0}"] = train_X
for l in range(1, L + 1):
    parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1])
    parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    
    running_mean[f"mean{l}"] = np.zeros((1, layer_dims[l]))
    running_var[f"var{l}"] = np.ones((1, layer_dims[l]))    
    parameters[f"mean{l}"] = np.ones((1, layer_dims[l]))
    parameters[f"gamma{l}"] = np.ones((layer_dims[l], 1)).T
    parameters[f"beta{l}"] = np.zeros((layer_dims[l], 1)).T
    activation[f"Z{l}"] = np.random.randn(train_X.shape[0], layer_dims[l])
    activation[f"Z*{l}"] = np.random.randn(train_X.shape[0], layer_dims[l])

for l in range(1, L):
    activation[f"A{l}"] = np.random.randn(train_X.shape[0], layer_dims[l])
        
# Forward propagation of weight
def forwardPropagation(X, Y, parameters,training=True):
    for l in range(1, L + 1):     
        Z = np.dot(activation[f"A{l-1}"],parameters[f"W{l}"].T) +  (np.kron(np.ones((1,m)),parameters[f"b{l}"])).T
        
        if training:
            mean = np.mean(Z, axis=0, keepdims=True)
            var = np.var(Z, axis=0, keepdims=True)

            running_mean[f"mean{l}"] = momentum * running_mean[f"mean{l}"] + (1 - momentum) * mean
            running_var[f"var{l}"] = momentum * running_var[f"var{l}"] + (1 - momentum) * var
        else:
            mean = running_mean[f"mean{l}"]
            var = running_var[f"var{l}"]
            
        parameters[f"mean{l}"]=running_mean[f"mean{l}"]
        parameters[f"var{l}"]=np.sqrt(running_var[f"var{l}"]+ep)
        activation[f"Z{l}"] = (Z - parameters[f"mean{l}"]) / parameters[f"var{l}"]     
        G1 = np.kron(np.ones((m,1)),parameters[f"gamma{l}"])
        B1 = np.kron(np.ones((m,1)),parameters[f"beta{l}"])
        activation[f"Z*{l}"]= np.multiply(activation[f"Z{l}"],G1)+B1
        A1 = np.zeros(activation[f"Z*{l}"].shape)
        for i in range(len(A1)):
            for j in range(len(A1[0])):
                if activation[f"Z*{l}"][i, j] > 0:
                    A1[i, j] = activation[f"Z*{l}"][i, j]
                else:
                    A1[i, j] = 0       
        activation[f"A{l}"] = A1        
    return activation


change = {}
def backwardPropagation(X, Y, parameters, activation):
    change[f"dA{L}"] = activation[f"Z*{L}"]
    ED = (train_Y - activation[f"Z*{L}"]) / m
    f1=parameters[f"gamma{L}"]/parameters[f"var{L}"]
    G2=activation[f"Z{L}"]-(np.kron(np.ones((m,1)),parameters[f"mean{L}"]))
    G3=np.kron(np.ones((m,1)),parameters[f"var{L}"])
    change[f"dgamma{L}"]=np.dot(ED.T, (G2/G3))
    change[f"dbeta{L}"] = np.dot(ED.T, np.ones((m, 1)))    
    change[f"db{L}"] = f1*np.dot( np.ones((1,m)),ED)
    change[f"dW{L}"]=f1*np.dot(ED.T,activation[f"A{L-1}"])
    
    # change in weights for layers l 10 1       
    for l in range(L-1, 0,-1): # l t0 1   
        B = np.zeros(activation[f"A{l}"].shape)
        for i in range(len(B)):
            for j in range(len(B[0])):
                if activation[f"A{l}"][i, j] > 0:
                    B[i, j] = 1
                else:
                    B[i, j] = 0
        change[f"dA{l}"] = B
       
    C=np.ones((1,m))
    DG1=np.multiply( np.kron(C.T, parameters[f"W{L}"]), change[f"dA{L-1}"])
    DG2=(activation[f"Z{L-1}"]-(np.kron(C.T,parameters[f"mean{L-1}"])))/(np.kron(C.T,parameters[f"var{L-1}"]))
    DG3=np.multiply(DG1,DG2)
    DG4=f1*np.dot(ED.T,DG3)
    change[f"dgamma{L-1}"]=DG4 #l
    
    # for l-1 for gamma
        
    DG5=np.kron(C.T, (parameters[f"gamma{L-1}"]/ parameters[f"var{L-1}"]))
    DG6=np.multiply(DG1,DG5)
    DG=np.dot(DG6, parameters[f"W{L-1}"])
    h1=(activation[f"Z{L-2}"]-np.kron(C.T, parameters[f"mean{L-2}"]))/np.kron(C.T, parameters[f"var{L-2}"]) 
    h2=f1*np.dot(ED.T, np.multiply(np.multiply(DG,change[f"dA{L-2}"]),h1))
    change[f"dgamma{L-2}"]=h2    
    #beta
    change[f"dbeta{L-1}"]=f1*np.dot(ED.T,DG1) # l
    change[f"dbeta{L-2}"]=f1*np.dot(ED.T, np.multiply(DG,change[f"dA{L-2}"]) )# l-1
    
    
    # l-2 to 1
    for l in range(L-3,0,-1):
        h3=np.korn(C.T,(parameters[f"gamma{l+1}"]/parameters[f"var{l+1}"]))
        h4=DG*change[f"dA{l+1}"]*h3
        DG=np.dot(h4,parameters[f"W{l+1}"])
        
        g1=np.kron(C.T, parameters[f"mean{l}"])
        g2=(activation[f"Z{l}"]-g1)/np.kron(C.T,parameters[f"var{l}"])
        g3=DG*change[f"dA{l}"]*g2
        g4=f1*np.dot(ED.T,g3)
        change[f"dgamma{l}"]=g4        
        # for beta
        change[f"dbeta{l}"]=f1*np.dot(ED.T,(DG*change[f"dA{l}"]))
    
# for W and b values for l-1 t0 1 
    # for L-1=l
    DW1=np.multiply( change[f"dA{L-1}"].T, np.kron(C,(parameters[f"gamma{L-1}"]/parameters[f"var{L-1}"]).T))
    DW=np.multiply(np.kron(C,parameters[f"W{L}"].T),DW1) 
    DW2=np.kron(np.ones((1,layer_dims[L-2])),ED)
    DW3= np.multiply( DW2, activation[f"A{L-2}"])
    change[f"dW{L-1}"]=np.multiply(f1,np.dot(DW,DW3))                                
    change[f"db{L-1}"]=np.multiply(f1,np.dot(DW,ED))
    
    #  L-2=l-1 to 1
    for l in range(L-2,0,-1):
        DW4=np.multiply(change[f"dA{l}"].T, np.kron(C,(parameters[f"gamma{l}"]/parameters[f"var{l}"]).T))
        DW=np.multiply(np.dot(parameters[f"W{l+1}"].T, DW),DW4)
        h5=np.ones((1,layer_dims[l-1]))
        change[f"dW{l}"]=np.multiply(f1,np.dot(DW,np.multiply(np.kron(h5,ED),activation[f"A{l-1}"])))
        change[f"db{l}"]=np.multiply(f1,np.dot(DW,ED))
    return change
     

def updateParameters(parameters, change, learningRate):
    for l in range(1, L + 1):
        parameters[f"W{l}"] = parameters[f"W{l}"] + learningRate * change[f"dW{l}"]
        parameters[f"gamma{l}"] = parameters[f"gamma{l}"] +learningRate * change[f"dgamma{l}"]
        parameters[f"b{l}"] = parameters[f"b{l}"] + learningRate * change[f"db{l}"]
        parameters[f"beta{l}"] = parameters[f"beta{l}"] + learningRate * change[f"dbeta{l}"]
    return parameters


epoch = 200 
Error = np.zeros(epoch)
for i in range(epoch):
   print("epoch", i)
   activation = forwardPropagation(train_X, train_Y, parameters,training=True)
   change = backwardPropagation(train_X, train_Y, parameters, activation)
   parameters = updateParameters(parameters, change, learningRate)
   fro_norm = np.linalg.norm(activation[f"A{L}"], 'fro')
   Error[i] = (fro_norm ** 2) / m
   
# Testing of the Model
n= test_X.shape[1] # no of input features (n- no of columns)
m= test_X.shape[0] # no of testing samples (m- no of rows)
activation[f"A{0}"]= test_X
activation=forwardPropagation(test_X, test_Y, parameters,training=False)

prediction = activation[f"Z{L}"].T
test_Y = test_Y.T
result = np.vstack((prediction, test_Y))


# Calculate the error
error = test_Y - prediction
mod_error = np.abs(error)
mod_error_sum = np.nansum(mod_error, dtype=np.float64)
avg_error = mod_error_sum / test_X.shape[0]
print("Average error", avg_error)




     
