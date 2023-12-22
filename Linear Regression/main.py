import numpy
import pandas as pd
import matplotlib.pyplot as plt


def mean_squared_error(true, predicted):
    return numpy.sum((true - predicted) ** 2) / len(true)


data = pd.read_csv('linear_regression.train')
x_vals = numpy.array(data['x'], dtype=float)
y_vals = numpy.array(data['y'], dtype=float)
del data

x_max, y_max = max(x_vals), max(y_vals)
x_min, y_min = min(x_vals), min(y_vals)
for i in range(1000):
    x_vals[i] = (x_vals[i] - x_min) / (x_max - x_min)
    y_vals[i] = (y_vals[i] - y_min) / (y_max - y_min)


weight, bias = 4, 2
previous_loss = None
for i in range(1000):
    predicted_vals = weight * x_vals + bias

    weight_gradient = -2 * (sum((y_vals - predicted_vals) * x_vals) / 1000)
    bias_gradient = -2 * (sum(y_vals - predicted_vals) / 1000)

    # Updating weights and bias
    weight = weight - (0.1 * weight_gradient)
    bias = bias - (0.1 * bias_gradient)

predicted_vals = weight * x_vals + bias
print(f'error: {mean_squared_error(y_vals, predicted_vals)}')
print(f'weight: {weight}, bias: {bias}')

plt.scatter(x_vals, y_vals, color='red')
plt.plot([min(x_vals), max(x_vals)], [min(predicted_vals), max(predicted_vals)], color='blue')
plt.show()
