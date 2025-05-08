# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 10:01:21 2025

@author: HP
"""

import numpy as np
import pandas as pd

# Function to add salt and pepper noise
def add_salt_and_pepper_noise(X, noise_density=0.1):
    X_noisy = np.copy(X)
    total_elements = X.size
    num_noisy = int(noise_density * total_elements)

    # Flatten indices for salt and pepper
    flat_indices = np.random.choice(total_elements, num_noisy, replace=False)
    salt_indices = flat_indices[:num_noisy // 2]
    pepper_indices = flat_indices[num_noisy // 2:]

    # Convert flat indices to 2D indices
    salt_coords = np.unravel_index(salt_indices, X.shape)
    pepper_coords = np.unravel_index(pepper_indices, X.shape)

    # Apply salt and pepper
    max_val = np.max(X)
    min_val = np.min(X)
    X_noisy[salt_coords] = max_val  # salt
    X_noisy[pepper_coords] = min_val  # pepper

    return X_noisy
