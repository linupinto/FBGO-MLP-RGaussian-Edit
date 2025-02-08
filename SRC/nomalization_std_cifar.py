
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

base_path=r"C:\Users\HP\Desktop\D folder\Adam_NN"
dataset = pd.read_csv(f"{base_path}\\CIFAR\cifar data.csv")

X = dataset.iloc[:, 0:3072].values
Y = dataset.iloc[:, 3072].values[:, np.newaxis]

# Standard Scalar
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

X_std  = np.column_stack((X_std ,Y))

