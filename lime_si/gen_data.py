import numpy as np


def generate_data(n, p):
    
    X_train = np.random.normal(0, 1, size=(n, p))
    true_beta = np.zeros(p)
    true_y = np.dot(X_train, true_beta)
    noise = np.random.normal(0, 1, size=n)
    y_train = true_y + noise

    return X_train, y_train
