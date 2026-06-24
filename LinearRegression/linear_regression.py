import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Gradient descent
        for _ in range(self.n_epochs):
            y_pred = X@self.weights + self.bias

            # Calculate gradient
            dw = (1 / n_samples)*(y_pred-y)@X
            db = (1/ n_samples)*np.sum((y_pred-y))

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate loss
            loss = np.mean(np.sum((y_pred-y)**2))
            self.loss_history.append(loss)

    def predict(self,X):
        X = np.array(X)
        y_pred = X@self.weights + self.bias
        return y_pred

    def plot_fitted_line(self, X, y):
        plt.figure(figsize=(12, 5))
        
        # Plot 1: The Data and the Fit
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, label='Data Points', color='blue', alpha=0.6)
        plt.plot(X, self.predict(X), color='red', label='Fitted Line')
        plt.title('Linear Regression Fit')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()

        # Plot 2: The Loss History
        plt.subplot(1, 2, 2)
        plt.plot(range(self.n_epochs), self.loss_history, color='green')
        plt.title('Loss Curve over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 3 * X.flatten() + 4 + np.random.randn(100)
    
    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_epochs=1000)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict([[5], [10], [15], [20]])
    
    # Print results
    print("Predictions:", predictions)
    print(f"Learned weights: {model.weights[0]:.4f}")
    print(f"Learned bias: {model.bias:.4f}")
    print(f"Loss: {model.loss_history[-1]:.4f}")
    
    # Plotting both the fit and the loss curve
    model.plot_fitted_line(X, y)

