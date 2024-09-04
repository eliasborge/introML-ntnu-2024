import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
class LinearRegression():
    
    def __init__(self, learning_rate=0.001, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta_0 = 0  # Intercept
        self.theta_1 = 0  # Slope

    def fit(self, X, y):
        m = len(y)
        for _ in range(self.n_iterations):
            y_pred = self.theta_0 + self.theta_1 * X
            d_theta_0 = (1/m) * np.sum(y_pred - y)
            d_theta_1 = (1/m) * np.sum((y_pred - y) * X)
            self.theta_0 -= self.learning_rate * d_theta_0
            self.theta_1 -= self.learning_rate * d_theta_1

    def predict(self, X):
        return self.theta_0 + self.theta_1 * X
