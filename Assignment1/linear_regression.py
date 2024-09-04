import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
class LinearRegression():
    
    def __init__(self, learning_rate=0.001, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        # y = ax+b
        self.b = 0  # Intercept
        self.a = 0  # Slope

    def fit(self, X, y):
        m = len(y)
        for _ in range(self.n_iterations):
            y_pred = self.b + self.a * X
            d_b = (1/m) * np.sum(y_pred - y)
            d_a = (1/m) * np.sum((y_pred - y) * X)
            self.b -= self.learning_rate * d_b
            self.a -= self.learning_rate * d_a

    def predict(self, X):
        return self.b + self.a * X
