import matplotlib
import numpy
import random
import matplotlib.pyplot as plt
matplotlib.use('macosx')


def mean_squared_error(true, predicted):
    cost = numpy.sum((true - predicted) ** 2) / len(true)
    return cost


def regression(degree, csv_file_name, iterations, learning_rate):
    data = numpy.recfromcsv(csv_file_name)
    data = numpy.asarray(data)
    data_len = len(data)
    x1_vals = numpy.array([data[i][0] for i in range(data_len)], dtype=float)
    x2_vals = numpy.array([data[i][1] for i in range(data_len)], dtype=float)
    y_vals = numpy.array([data[i][2] for i in range(data_len)], dtype=float)
    del data

    # normalize data to [0, 1]
    y_min, y_max = min(y_vals), max(y_vals)
    x1_min, x1_max = min(x1_vals), max(x1_vals)
    x2_min, x2_max = min(x2_vals), max(x2_vals)
    for i in range(data_len):
        x1_vals[i] = (x1_vals[i] - x1_min) / (x1_max - x1_min)
        x2_vals[i] = (x2_vals[i] - x2_min) / (x2_max - x2_min)
        y_vals[i] = (y_vals[i] - y_min) / (y_max - y_min)

    previous_loss = None
    weights = [0.1 for _ in range(degree + 1)]
    bias_weight = 0.1
    degrees = [j for j in range(degree, -1, -1)]
    degrees_inverse = [j for j in range(degree + 1)]
    for i in range(iterations):
        predicted_vals = numpy.zeros([1000], dtype=float)
        for k in range(degree + 1):
            temp = weights[k] * x1_vals ** degrees[k] * x2_vals ** degrees_inverse[k]
            predicted_vals += temp
        predicted_vals += bias_weight
        loss = mean_squared_error(y_vals, predicted_vals)

        if previous_loss and abs(loss - previous_loss) <= 0.0000000001:
            break

        previous_loss = loss

        derivatives = [0 for _ in range(degree + 1)]
        bias_derivative = 0
        for k in range(degree + 1):
            derivatives[k] += 2 * sum((predicted_vals - y_vals) *
                                      (x1_vals ** degrees[k] * x2_vals ** degrees_inverse[k])) / 1000
        bias_derivative += 2 * sum(predicted_vals - y_vals) / 1000
        """    
        for j in range(1000):
            for k in range(degree + 1):
                derivatives[k] += 2 * (predicted_vals[j] - y_vals[j]) * \
                                  (x1_vals[j] ** degrees[k] * x2_vals[j] ** degrees_inverse[k])

            bias_derivative += 2 * (predicted_vals[j] - y_vals[j])
        """
        for k in range(degree + 1):
            # derivatives[k] /= 1000
            weights[k] -= learning_rate * derivatives[k]
        #bias_derivative /= 1000
        bias_weight -= learning_rate * bias_derivative

    weights.append(bias_weight)
    print(weights)
    predicted_vals = numpy.zeros([1000], dtype=float)
    for k in range(degree + 1):
        temp = weights[k] * x1_vals ** degrees[k] * x2_vals ** degrees_inverse[k]
        predicted_vals += temp
    predicted_vals += bias_weight
    print(mean_squared_error(y_vals, predicted_vals))
    ax = plt.axes(projection='3d')
    ax.scatter(x1_vals, x2_vals, y_vals, 'green')
    ax.scatter(x1_vals, x2_vals, predicted_vals, 'blue')
    plt.show()


regression(4, "Sample_Data2.csv", 30000, 0.01)
