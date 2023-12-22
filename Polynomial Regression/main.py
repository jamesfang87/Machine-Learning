import numpy
import matplotlib.pyplot as plt
import random


def mean_squared_error(true, predicted):
    cost = numpy.sum((true - predicted) ** 2) / len(true)
    return cost


def regression(degree, csv_file_name, l, iterations):
    # get all the data from csv file
    data = numpy.recfromcsv(csv_file_name)
    data = numpy.asarray(data)
    random.shuffle(data)
    data_len = len(data)
    eighty_percent = int(0.8 * data_len)
    x_vals = numpy.array([data[i][0] for i in range(data_len)], dtype=float)
    y_vals = numpy.array([data[i][1] for i in range(data_len)], dtype=float)
    del data

    # normalize data to [0, 1]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    for i in range(data_len):
        x_vals[i] = (x_vals[i] - x_min) / (x_max - x_min)
        y_vals[i] = (y_vals[i] - y_min) / (y_max - y_min)

    # regression
    weights = [1 for _ in range(degree + 1)]
    previous_loss = None
    training_data = numpy.array(x_vals[:eighty_percent])

    for i in range(iterations):
        predicted_vals = numpy.zeros([eighty_percent], dtype=float)
        for k in range(degree + 1):
            predicted_vals += weights[k] * training_data ** k
        loss = mean_squared_error(y_vals[:eighty_percent], predicted_vals)

        if previous_loss and abs(previous_loss - loss) <= 0.00000001:
            break
        previous_loss = loss

        derivatives = [0 for _ in range(degree + 1)]
        for j in range(eighty_percent):
            for k in range(degree + 1):
                derivatives[k] += -2 * (y_vals[j] - predicted_vals[j]) * x_vals[j] ** k

        for k in range(degree + 1):
            derivatives[k] /= eighty_percent
            weights[k] = weights[k] - (l * derivatives[k])

    predicted_vals = numpy.zeros([200], dtype=float)
    for k in range(degree + 1):
        predicted_vals += weights[k] * x_vals[eighty_percent:] ** k
    loss = mean_squared_error(y_vals[eighty_percent:], predicted_vals)

    # output data and graph
    print(f'Loss: {loss}')
    print(list(reversed(weights)))
    # print(f'a: {weights[3]}, b: {weights[2]}, c: {weights[1]}, bias: {weights[0]}')
    plt.figure(figsize=(8, 8))
    plt.scatter(x_vals[eighty_percent:], y_vals[eighty_percent:], marker='o', color='red')
    plt.scatter(x_vals[eighty_percent:], predicted_vals)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


regression(3, "cubic_regression.csv", 0.05, 100000)
