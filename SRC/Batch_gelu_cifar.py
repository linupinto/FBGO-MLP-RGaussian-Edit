

import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import csv
from math import sqrt,erf,pi,exp
from sklearn.metrics import precision_score, recall_score,f1_score,classification_report
from  noise_salt_pepper import add_salt_and_pepper_noise
from confmat import confmat_iris,confmat_CIFAR,confmat_Drybeans
from labels_pred import labels_7, labels_10


path=r"C:\Users\HP\Desktop\D folder\Linu_savitha_2\minibatch_pgm\dataset\CIFAR"
dataset = pd.read_csv(f"{path}\\std_data_cifar.csv")
X = dataset.iloc[:, 0:3072].values
Y = dataset.iloc[:, 3072].values[:, np.newaxis]
#X=add_salt_and_pepper_noise(X, noise_density=0.1)  # Adding salt pepper noise 
Y=Y/10
#------------------
n_samples=X.shape[0]
# 1. Shuffle the data
indices = np.random.permutation(n_samples)
X_shuffled = X[indices]
y_shuffled = Y[indices]

# 2. Compute split points
train_end=int(0.7*n_samples)
val_end=int(.85*n_samples)
#-----------------
train_X ,train_Y = X_shuffled[:train_end], y_shuffled[:train_end]
val_X , val_Y = X_shuffled[train_end:val_end], y_shuffled[train_end:val_end]
test_X , test_Y =  X_shuffled[val_end:], y_shuffled[val_end:]

#-----------------
def mse_loss(y_pred,y_true):
    return np.mean((y_pred-y_true)**2)

n = train_X.shape[1]  # no of input features (n- no of columns)
m = train_X.shape[0]  # no of training samples (m- no of rows)

# specify the no of layers and neurons in each layer
layer_dims = [n,6, 5,7, 1]

L = len(layer_dims) - 1  # Total layers other than input layer
# print("Total layers other than input layer", L)

# specify the learning rate
learningRate =0.01
ep=10**-5

# Initializing  weights and dimension in ech layer
parameters = {}
activation = {}
activation[f"A{0}"] = train_X
for l in range(1, L + 1):
    activation[f"A{0}"] = train_X
    parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1])
    parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    parameters[f"mean{l}"] = np.ones((layer_dims[l], 1)).T
    parameters[f"var{l}"] = np.ones((layer_dims[l], 1)).T
    parameters[f"v{l}"] = np.ones((layer_dims[l], 1)).T
    parameters[f"gamma{l}"] = np.ones((layer_dims[l], 1)).T
    parameters[f"beta{l}"] = np.zeros((layer_dims[l], 1)).T
    activation[f"Z{l}"] = np.random.randn(train_X.shape[0], layer_dims[l])
    activation[f"Z*{l}"] = np.random.randn(train_X.shape[0], layer_dims[l])

for l in range(1, L):
    activation[f"A{0}"] = train_X
    activation[f"A{l}"] = np.random.randn(train_X.shape[0], layer_dims[l])
    print(f"A{l} shape:", activation[f"A{l}"].shape)

#-----------------Min_batch Generation----------------
def create_mini_batches(X, Y, batch_size):
    m = X.shape[0]
    mini_batches = []
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    for i in range(0, m, batch_size):
        end = i + batch_size
        X_batch = X_shuffled[i:end]
        Y_batch = Y_shuffled[i:end]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches  
    
# Forward propagation of weight
def forwardPropagation(X, Y, parameters):
    for l in range(1, L + 1):
        m = X.shape[0]
        activation[f"A{0}"] = X
        Z = np.dot(activation[f"A{l-1}"],parameters[f"W{l}"].T) +  (np.kron(np.ones((1,m)),parameters[f"b{l}"])).T
        activation[f"Z{l}"] = Z
        parameters[f"mean{l}"] = np.matrix(np.mean(activation[f"Z{l}"], axis=0))
        parameters[f"v{l}"]=np.sqrt(np.matrix(np.var(activation[f"Z{l}"], axis=0)))
        parameters[f"var{l}"] = np.sqrt(np.matrix(np.var(activation[f"Z{l}"], axis=0))+ ep)
        activation[f"Z{l}"] = (Z - parameters[f"mean{l}"]) / parameters[f"var{l}"]     
        G1 = np.kron(np.ones((m,1)),parameters[f"gamma{l}"])
        B1 = np.kron(np.ones((m,1)),parameters[f"beta{l}"])
        activation[f"Z*{l}"]= np.multiply(activation[f"Z{l}"],G1)+B1
        A1 = np.zeros(activation[f"Z*{l}"].shape)
        for i in range(len(A1)):
            for j in range(len(A1[0])):
                x=activation[f"Z*{l}"][i, j] 
                phi=0.5*(1+erf(x/sqrt(2)))
                A1[i,j]=x*phi
        activation[f"A{l}"] = A1  
    return activation


change = {}
def backwardPropagation(X, Y, parameters, activation):
    m = X.shape[0]
    change[f"dA{L}"] = activation[f"Z*{L}"]
    ED = (Y - activation[f"Z*{L}"]) / m
    #f1 = np.divide(parameters[f"gamma{L}"], parameters[f"var{L}"])
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
                x=activation[f"Z*{l}"][i, j] 
                phi=0.5*(1+erf(x/sqrt(2)))
                phi_der=0.5*x*exp(-x**2/2)/(sqrt(2*pi))
                B[i,j]=phi+phi_der
        change[f"dA{l}"] = B
        #return change
       
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
        h3=np.kron(C.T,(parameters[f"gamma{l+1}"]/parameters[f"var{l+1}"]))
        h4= np.array(DG) * np.array(change[f"dA{l+1}"]) * np.array(h3)
        DG=np.dot(h4,parameters[f"W{l+1}"])
        
        g1=np.kron(C.T, parameters[f"mean{l}"])
        g2=(activation[f"Z{l}"]-g1)/np.kron(C.T,parameters[f"var{l}"])
        g3=np.array(DG) * np.array(change[f"dA{l}"]) * np.array(g2)
        g4=f1*np.dot(ED.T,g3)
        change[f"dgamma{l}"]=g4        
        # for beta
        change[f"dbeta{l}"]=f1*np.dot(ED.T,(np.array(DG)*np.array(change[f"dA{l}"])))
    
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
        parameters[f"gamma{l}"] = parameters[f"gamma{l}"] + learningRate * change[f"dgamma{l}"]
        parameters[f"b{l}"] = parameters[f"b{l}"] + learningRate * change[f"db{l}"]
        parameters[f"beta{l}"] = parameters[f"beta{l}"] + learningRate * change[f"dbeta{l}"]
    return parameters    
#---------------------Traing Starting-------------------------------------
epoch =100
batch_size = 1000
min_delta = 1e-4  # Minimum change in loss to count as an improvement

grad_norm=[]
Max={}
Min={}
C={}
Avg=[]
C_avg=[] 
train_losses = []
val_losses = []

# === 6. Training Loop ===
best_loss = float('inf')          # Start with a very high loss
patience = 10                     # Number of epochs to tolerate no improvement
convergence_start_epoch = None    # The epoch where convergence starts
convergence_reached = False       # Flag to check if convergence is already achieved
start_time = time.time() 
for i in range(1,epoch+1):
    #start_time_epoch=time.time() # start time for this epoch
    #print("epoch", i)
    
    Avg=[]      
    flattened_grads = []
    train_array=[]
    mini_batches = create_mini_batches(train_X, train_Y, batch_size)
    
    for X_batch, Y_batch in mini_batches:         
        activation[f"A{0}"] = X_batch      
        activation = forwardPropagation(X_batch, Y_batch, parameters) 
                
        # Compute training loss for this mini-batch
        train_loss=mse_loss(np.array(activation[f"Z*{L}"]), Y_batch)
        train_array+=[train_loss]          # Saving values of all mini batches
        
        
        change = backwardPropagation(X_batch, Y_batch, parameters, activation)
        parameters = updateParameters(parameters, change, learningRate) 
    
    train_avg=np.mean(train_array)         # Avg of all mini batches     
    # Compute validation loss after each epoch
    
    # === Check for Convergence if it hasn't been reached yet ===
    if not convergence_reached:            # Only check convergence if it hasn't been reached
       if best_loss - train_avg > min_delta:
          best_loss = train_avg
          patience_counter = 0
       else:
          patience_counter += 1
       # Check if convergence is detected   
       if patience_counter == patience and convergence_start_epoch is None:
             convergence_start_epoch = i - patience + 1
             converged_time = time.time() - start_time
             convergence_reached = True  # Set flag to stop further convergence checks
             print(f"Convergence detected at epoch {convergence_start_epoch}")
             
    Y_val_pred = forwardPropagation(val_X, val_Y,parameters)
    val_loss = mse_loss(np.array(Y_val_pred[f"Z*{L}"]), val_Y)
        
        
    # Store loss values
    train_losses.append(train_avg)  # You can use last mini-batch loss or average
    val_losses.append(val_loss)
    
    # Print progress
    print(f"Epoch {i}/{epoch}  | Training Losses: {train_avg:.4f} | Validation Loss: {val_loss:.4f}")
    
    for l in range(1, L+1): 
        M=np.max(activation[f"Z*{l}"],axis=0)  
        Max[f"M{l}"]=M
        m=np.min(activation[f"Z*{l}"],axis=0)  
        Min[f"m{l}"]=m
        C[f"C{l}"]=Max[f"M{l}"]-Min[f"m{l}"]
        average=np.mean(C[f"C{l}"])
        Avg.append(average.reshape(1,-1))
        flattened_grads.append(change[f"dW{l}"].flatten().T) # Flatten the gradient and add it to the list
        flattened_grads.append(change[f"db{l}"].flatten().T)
        
    # Concatenate all flattened gradients into a single 1D array (single row) 
    concatenated_C= np.concatenate(Avg)
    C_avg+=[ concatenated_C.T]
    concatenated_grads = np.concatenate(flattened_grads) 
    grad_norm += [np.linalg.norm(concatenated_grads)]
    
    
# Calculate total time after all epochs
end_time_total=time.time()                 # End time for total training
total_time=end_time_total-start_time       # Total time for all epochs

# Converegnce Result for Train Loss
print(f"\nFinal Train Loss: {train_losses[-1]:.6f}")
if convergence_reached:
    print(f"\nConverged at Epoch: {convergence_start_epoch}")
    print(f"\nTime to Convergence: {converged_time:.2f} seconds")
else:
    print("\nTrain loss did not converge within given patience.")
print(f"\nTotal Training Time: {total_time:.2f} seconds")
#----------------Testing of the Model---------------

n= test_X.shape[1] # no of input features (n- no of columns)
m= test_X.shape[0] # no of testing samples (m- no of rows)
activation[f"A{0}"]= test_X
activation=forwardPropagation(test_X, test_Y, parameters)


#-------------- Predicted Value---------------
prediction =  activation[f"Z*{L}"].T
test_Y = test_Y.T
result = np.vstack((prediction,test_Y ))

# ------- Calculate the error and plotting --------------------
error = test_Y - prediction
mod_error = np.abs(error)
mod_error_sum = np.nansum(mod_error, dtype=np.float64)
avg_error = mod_error_sum / test_X.shape[0]
print("Average error", avg_error)
       
     
#------Saving the validity_loss values------------------
base_path=r"C:\Users\HP\Desktop\D folder\Linu_savitha_2\minibatch_pgm\Batch_N\output_gelu_cifar"
vali = np.array(val_losses)
with open(f"{base_path}\\val_loss_cifar.csv", "w", newline='') as file:
    csv.writer(file).writerows([[v] for v in vali.tolist()])

#------Saving the prediction values------------------
T = np.hstack((test_Y.T, prediction.T))
V = T.tolist()
with open(f"{base_path}\\gelu_prediction_cifar.csv","w", newline='') as file:
     csv.writer(file).writerows(V) 

#------------Precision and Recall---------------       

true_labels=np.array(test_Y*10)
true_labels=true_labels.reshape(-1)     
#----------Prediction Back to classes-------------


pre_labels=labels_10(prediction) 
pre_labels=pre_labels.astype(int)     # Pre Labels
true_labels=true_labels.astype(int)   # True labels

# Macro-average
# Macro-average with zero_division set to 1 (handle undefined cases)
precision_macro = precision_score(true_labels, pre_labels, average='macro',zero_division=0)
recall_macro = recall_score(true_labels, pre_labels, average='macro',zero_division=0)

# Micro-average
# Micro-average with zero_division set to 1 (handle undefined cases)
precision_micro = precision_score(true_labels, pre_labels, average='micro',zero_division=0)
recall_micro = recall_score(true_labels, pre_labels, average='micro',zero_division=0)

# Print results
print(f"Macro Precision: {precision_macro}")
print(f"Macro Recall: {recall_macro}")
print(f"Micro Precision: {precision_micro}")
print(f"Micro Recall: {recall_micro}")  
#-------------------------------------------------------
# Per-class metrics and F1 scores
print("\nClassification Report:")
print(classification_report(true_labels, pre_labels, digits=4,zero_division=0))
# F1 Macro
f1_macro = f1_score(true_labels, pre_labels, average='macro',zero_division=0)

# F1 Micro
f1_micro = f1_score(true_labels, pre_labels, average='micro',zero_division=0)

print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Micro: {f1_micro:.4f}")    

f1_weighted = f1_score(true_labels, pre_labels, average='weighted',zero_division=0)        

#---------------Fig Epoch Vs C Values, Gradient, Training Loss-----------


# Convert to a (100, 3) array
C_array = np.squeeze(np.array(C_avg))  # shape becomes (100, 3)
E=range(1, epoch + 1)
plt.figure(2,figsize=(12, 8)) 
for i in range(L):  # Loop over the 3 layers
    plt.plot(E, C_array[:, i], linestyle='--', label=f'Average C - Layer {i+1}')

plt.plot(E, grad_norm,color='blue',linestyle='-.',label='Gradient Norm')
plt.plot(E, train_losses, label="Training Loss", linestyle='-')
plt.xlabel("Epoch",fontsize=16)
plt.ylabel("Metric Value",fontsize=16)
#plt.title("Importance of Modulation Parameter C over Epochs")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(f"{base_path}\\metric_jpeg", dpi =500)
plt.show()

#----------- Confusion matrix is-------------

confmat_CIFAR(test_Y.T,prediction.T)
plt.savefig(f"{base_path}\\confmat.jpeg", dpi =300)








 

 















