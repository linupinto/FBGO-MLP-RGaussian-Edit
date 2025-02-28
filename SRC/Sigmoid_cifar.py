
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

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
layer_dims = [n,6,5,1]

L = len(layer_dims) - 1  # Total layers other than input layer
# specify the learning rate
learningRate = 0.1

# Initializing  weights and dimension in ech layer
parameters = {}
activation = {}
activation[f"A{0}"] = train_X
for l in range(1, L + 1):
    parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1])
    parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    activation[f"Z{l}"] = np.random.randn(train_X.shape[0], layer_dims[l])
    

for l in range(1, L):
    activation[f"A{l}"] = np.random.randn(train_X.shape[0], layer_dims[l])
    
# Forward propagation of weights
def forwardPropagation(X, Y, parameters):
    for l in range(1, L + 1):
        K = np.repeat(parameters[f"b{l}"], activation[f"A{l - 1}"].shape[0], axis=1)
        Z = np.dot(activation[f"A{l - 1}"], parameters[f"W{l}"].T) + K.T
        activation[f"Z{l}"] = Z
        Z1 = np.exp(-1 * activation[f"Z{l}"])
        A1 = 1 / (1 + Z1)
        activation[f"A{l}"] = A1
    return activation

change = {}
for l in range(1, L + 1):
    change[f"dA{l}"] = np.random.randn(train_X.shape[0], layer_dims[l])
    change[f"dZ{l}"] = np.random.randn(train_X.shape[0], layer_dims[l])
    change[f"S3{l}"] = np.random.randn(layer_dims[l], train_X.shape[1])
    change[f"dW{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1])
    change[f"db{l}"] = np.zeros((layer_dims[l], 1))
  
def backwardPropagation(X, Y, parameters, activation):
    change[f"dA{L}"] = activation[f"A{L}"]
    ED = (train_Y - activation[f"Z{L}"])/m
    change[f"db{L}"] = np.dot(ED.T, np.ones((m, 1)))
    change[f"dW{L}"] = np.dot(ED.T, activation[f"A{L - 1}"])

    # activation change
    change[f"dA{L - 1}"] = np.multiply(activation[f"A{L - 1}"], (1 - activation[f"A{L - 1}"]))
    S0 = np.tile(parameters[f"W{L}"].T, (1, m))
    S1 = change[f"dA{L - 1}"].T
    change[f"S3{L - 1}"] = np.multiply(S0, S1)
    change[f"db{L - 1}"] = np.dot(change[f"S3{L - 1}"], ED)
    S4 = np.tile(ED, (1, activation[f"A{L - 2}"].shape[1]))
    S5 = np.multiply(S4, activation[f"A{L - 2}"])
    change[f"dW{L - 1}"] = np.dot(change[f"S3{L - 1}"], S5)


    for l in range(L - 2, 0, -1):
        # activation change
        change[f"dA{l}"] = np.multiply(activation[f"A{l}"], (1 - activation[f"A{l}"]))
        S0 = np.dot(parameters[f"W{l + 1}"].T, change[f"db{l + 1}"])
        S1 = np.tile(S0, (1, m))
        change[f"S3{l}"] = np.multiply(S1, change[f"dA{l}"].T)
        change[f"db{l}"] = np.dot(change[f"S3{l}"], ED)
        S4 = np.dot(ED.T, activation[f"A{l - 1}"])
        S5 = np.tile(S4, (m, 1))
    return change


def updateParameters(parameters, change, learningRate):
    for l in range(1, L + 1):
        parameters[f"W{l}"] = parameters[f"W{l}"] + learningRate * change[f"dW{l}"]
        parameters[f"b{l}"] = parameters[f"b{l}"] + learningRate * change[f"db{l}"]
    return parameters

epoch = 200
Error = np.zeros(epoch) 
for i in range(epoch):
    activation = forwardPropagation(train_X, train_Y, parameters)
    change = backwardPropagation(train_X, train_Y, parameters, activation)
    parameters = updateParameters(parameters, change, learningRate)
    fro_norm = np.linalg.norm(activation[f"A{L}"], 'fro')
    Error[i] = (fro_norm ** 2) / m
    print("epoch", i)
#------------------- Testing------------
activation[f"A{0}"] = test_X
for l in range(1, L + 1):
    K = np.repeat(parameters[f"b{l}"], activation[f"A{l - 1}"].shape[0], axis=1)
    Z = np.dot(activation[f"A{l - 1}"], parameters[f"W{l}"].T) + K.T
    activation[f"Z{l}"] = Z
    Z1 = np.exp(-1 * activation[f"Z{l}"])
    A1 = 1 / (1 + Z1)
    activation[f"A{l}"] = A1
  
prediction = activation[f"Z{L}"].T
test_Y = test_Y.T
result = np.vstack((prediction, test_Y))

# Calculate the error
error = test_Y - prediction
mod_error = np.abs(error)
mod_error_sum = np.nansum(mod_error, dtype=np.float64)
avg_error = mod_error_sum / test_X.shape[0]
print("Average error", avg_error)

    

