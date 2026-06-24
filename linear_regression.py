import numpy as np

np.random.seed(1)
x = np.array([1,2,3,4,5])
error = np.random.normal(0,5,5)
y = 3*x + error
w = 0
b = 0

def predict(x,w,b):
    y_pred = w*x + b
    return y_pred

def update_parameters(x,y,w,b,alpha):
    dw = 1/5 * np.sum((w*x+b-y)*x)
    db = 1/5 * np.sum(w*x+b-y)
    w -= alpha*dw
    b -= alpha*db
    return w,b
hist_w = []
hist_b = []
hist_rmse = []
for i in range(1000):
    w, b = update_parameters(x,y,w,b,0.01)
    y_pred = predict(x, w, b)
    rmse = np.sqrt(np.sum((y_pred - y)**2) / 5)
    hist_w.append(w)
    hist_b.append(b)
    hist_rmse.append(rmse)
    if i % 100 == 0:
        print(f"Epoch {i}: w = {w:.4f}, b = {b:.4f}, RMSE = {rmse:.4f}")

print(f"\nFinal Result -> epoch:{i}, w: {w:.4f}, b: {b:.4f}")

