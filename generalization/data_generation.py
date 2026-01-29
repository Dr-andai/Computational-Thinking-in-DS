import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

def data_generator(n_samples = 100, noise=0.3):
    x = np.random.uniform(-3, 3, n_samples)
    y = 0.5 * x**3 - 2 * x**2 + 1.5 * x + np.random.normal(0, noise, n_samples)
    return x, y

# Generate train and test data
x_train, y_train = data_generator(n_samples=100, noise=0.3)
x_test, y_test = data_generator(n_samples=200, noise=0.3)

# Convert to PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train).unsqueeze(1)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
x_test_tensor = torch.FloatTensor(x_test).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)