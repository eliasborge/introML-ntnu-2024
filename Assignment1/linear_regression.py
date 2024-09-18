import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
class LinearRegression():
    
    #sources == forelesningsnotater
    
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []  # Track loss for each epoch
        self.train_accuracies = []  # Track training accuracy for each epoch

    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))

    def _compute_loss(self, y, y_pred):
        m = len(y)
        loss = -(1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def compute_gradients(self, x, y, y_pred):
        m = x.shape[0]
        grad_w = (1/m) * np.dot(x.T, (y_pred - y))  
        grad_b = (1/m) * np.sum(y_pred - y)  
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w  
        self.bias -= self.learning_rate * grad_b  

    def predict_proba(self, x):

        lin_model = np.matmul(x, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return y_pred

    def fit(self, x, y):
    # Initialize weights and bias
        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        # Gradient Descent
        for epoch in range(self.epochs):
            lin_model = np.matmul(self.weights, x.transpose()) + self.bias
            y_pred = self.sigmoid_function(lin_model)
            grad_w, grad_b = self.compute_gradients(x, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            accuracy = accuracy_score(y, pred_to_class)
            self.train_accuracies.append(accuracy)
            self.losses.append(loss)
        
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{self.epochs}: Loss = {loss}, Accuracy = {accuracy}')
    
    def predict(self, x):
        lin_model = np.dot(x, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)

        return [1 if _y > 0.5 else 0 for _y in y_pred]
