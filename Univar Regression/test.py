import numpy
import random
import matplotlib.pyplot as plt


def mean_squared_error(true, predicted):
    cost = numpy.sum((true - predicted) ** 2) / len(true)
    return cost


def regression(csv_file_name):
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

regression("../multivar regression/Sample_Data2.csv")
