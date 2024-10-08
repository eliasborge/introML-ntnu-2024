import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
class LinearRegression():

     #sources == forelesningsnotater

    #Mission 1
    def __init__(self, learning_rate=0.001, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        # y = ax+b
        self.b = 0  # Intercept
        self.a = 0  # Slope

    def fit(self, X, y, tolerance=1e-6):
        m = len(y)
        previous_cost = float('inf')
        for _ in range(self.n_iterations):
            y_pred = self.b + self.a * X
            d_b = (1/m) * np.sum(y_pred - y)
            d_a = (1/m) * np.sum((y_pred - y) * X)
            cost = (1/m) * np.sum((y_pred - y) ** 2)  # Mean squared error
            
            if abs(previous_cost - cost) < tolerance:  # Convergence check
                print(f"Converged at iteration {_} with cost {cost}")
                break
            
            previous_cost = cost
            self.b -= self.learning_rate * d_b
            self.a -= self.learning_rate * d_a
            print(f"Iteration {_}, cost: {cost}, a: {self.a}, b: {self.b}")
    

    def predict(self, X):
        return self.b + self.a * X
