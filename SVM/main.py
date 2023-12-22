import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib
matplotlib.use('macosx')

data = pd.read_csv('Linearly_Sepa_Data.csv')
X_train = np.array(data[['X1', 'X2']])
y_train = np.where(data['y'] != 0, data['y'], -1)

weights = np.array([0.0, 0.0])
bias = 0
lambda_param = 0.01  # controls how strict we are about wrongly classified data
learning_rate = 0.002
for _ in range(200):
    for X, y in zip(X_train, y_train):
        predicted_correctly = y * (weights.T @ X - bias) >= 1

        if predicted_correctly:
            weights -= learning_rate * 2 * lambda_param * weights
        else:
            weights -= learning_rate * (2 * lambda_param * weights - y * X)
            bias -= learning_rate * y
    prediction = np.zeros([1000])
    for i in range(len(data)):
        prediction[i] = np.dot(X_train[i], weights.T) - bias

    accuracy = np.sum(y_train == np.sign(prediction)) / len(y_train)
    print("SVM classification accuracy", accuracy)

print(weights, bias)

prediction = np.zeros([1000])
for i in range(len(data)):
    prediction[i] = np.dot(X_train[i], weights.T) - bias

accuracy = np.sum(y_train == np.sign(prediction)) / len(y_train)
print("SVM classification accuracy", accuracy)


def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]


fig = plot.figure()
ax = fig.add_subplot(1, 1, 1)
plot.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = get_hyperplane_value(x0_1, weights, bias, 0)
x1_2 = get_hyperplane_value(x0_2, weights, bias, 0)

x1_1_m = get_hyperplane_value(x0_1, weights, bias, -1)
x1_2_m = get_hyperplane_value(x0_2, weights, bias, -1)

x1_1_p = get_hyperplane_value(x0_1, weights, bias, 1)
x1_2_p = get_hyperplane_value(x0_2, weights, bias, 1)

ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

x1_min = np.amin(X_train[:, 1])
x1_max = np.amax(X_train[:, 1])
ax.set_ylim([x1_min - 3, x1_max + 3])
plot.show()

