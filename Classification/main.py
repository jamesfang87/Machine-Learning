import numpy
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('macosx')


def loss(true, predicted):
    temp_loss = 0.0
    for i in range(len(true)):
        temp_loss += max(0, -true[i] * predicted[i])
    return temp_loss / 1000


def return_prediction(s, l):
    return numpy.sign(weights[0] * x1_vals[s: s + l] + weights[1] * x2_vals[s: s + l] + weights[2])


data = numpy.recfromcsv('Linearly_Sepa_Data.csv')
data = numpy.asarray(data)
random.shuffle(data)
data_len = len(data)
x1_vals = numpy.array([data[i][0] for i in range(data_len)], dtype=float)
x2_vals = numpy.array([data[i][1] for i in range(data_len)], dtype=float)
y_vals = numpy.array([data[i][2] for i in range(data_len)], dtype=float)
for i in range(len(y_vals)):
    if y_vals[i] == 0:
        y_vals[i] = -1
del data

weights = [5, 5, 0.1]
for i in range(10000):
    predicted_vals = return_prediction(0, 800)

    # print(loss(y_vals[:800], predicted_vals))

    w1_derivative = 0
    w2_derivative = 0
    bias_derivative = 0

    for j in range(800):
        if predicted_vals[j] != y_vals[j]:
            w1_derivative += y_vals[j] * x1_vals[j]
            w2_derivative += y_vals[j] * x2_vals[j]
            bias_derivative += y_vals[j]

    w1_derivative /= 800
    w2_derivative /= 800
    bias_derivative /= 800

    weights[0] += 0.01 * w1_derivative
    weights[1] += 0.01 * w2_derivative
    weights[2] += 0.01 * bias_derivative

predicted_vals = return_prediction(800, 200)
print(weights)
print(loss(y_vals[800:], predicted_vals))
ax = plt.axes()
for i in range(200):
    if predicted_vals[i] == -1:
        ax.scatter([x1_vals[800 + i]], [x2_vals[800 + i]], color='blue')
    else:
        ax.scatter([x1_vals[800 + i]], [x2_vals[800 + i]], color='red')

print(min(x1_vals), max(x1_vals))
print(min(x2_vals), max(x2_vals))
line = numpy.linspace(-15, 10, 100)
line1 = numpy.linspace(min(x1_vals), max(x1_vals), 100)
line2 = numpy.linspace(min(x2_vals), max(x2_vals), 100)
y_prediction = weights[0] * line + weights[1] * line + weights[2]
plt.plot(line, y_prediction, color='gold', linewidth=3)

plt.show()
