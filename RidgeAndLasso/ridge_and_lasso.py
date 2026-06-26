import numpy as np

class RegularizedLinearRegression:
    def __init__(self, learning_rate = 0.1, lambda_val = 0.1, reg_type = 'l2', n_epochs = 1000):
        self.learning_rate = learning_rate
        self.lambda_val = lambda_val
        self.reg_type = reg_type
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Generate parameters
        n_samples, n_features = X.shape #(n_samples x n_features)
        self.weights = np.zeros(n_features) #(1 x n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_epochs):
            # Predict value
            y_pred = X @ self.weights + self.bias #(n_samples, 1)
            if self.reg_type == 'l1':
                dw = (-2/n_samples) * (y-y_pred) @ X + self.lambda_val * np.sign(self.weights)
            elif self.reg_type == 'l2':
                dw = (-2/n_samples) * (y-y_pred) @ X + 2 * self.lambda_val * self.weights
            db = -(2/n_samples) * np.sum(y - y_pred)
        
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        y_pred = X @ self.weights + self.bias
        return y_pred
    
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 3 * X.flatten() + 4 + np.random.randn(100)
    
    # Create and train model
    model = RegularizedLinearRegression(learning_rate=0.1, n_epochs=1000, reg_type='l2')
    model.fit(X, y)

    # Make predictions
    predictions = model.predict([[5], [10], [15], [20]])
    
    # Print results
    print("Predictions:", predictions)
    print(f"Learned weights: {model.weights[0]:.4f}")
    print(f"Learned bias: {model.bias:.4f}")