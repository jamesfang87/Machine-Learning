import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm


class Data:
    def __init__(self, file_name):
        self.file_name = file_name
        self.X = None
        self.y = None
        self.all_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.data_len = 0

    def split(self):
        self.X_train = self.X[:int(self.data_len * 0.8)]
        self.y_train = self.y[:int(self.data_len * 0.8)]
        self.X_test = self.X[int(self.data_len * 0.8):]
        self.y_test = self.y[int(self.data_len * 0.8):]

    def get_mnist(self):
        with np.load(f"{pathlib.Path(__file__).parent.absolute()}/mnist.npz") as f:
            images, labels = f["x_train"], f["y_train"]
        images = images.astype("float64") / 255
        images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
        labels = np.eye(10)[labels]
        self.X = images
        self.y = labels
        self.data_len = len(images)
        self.split()

    def read_data(self):
        data = pd.read_csv('troll.csv')
        temp_X = data[['x']]
        temp_y = data['y']

        self.all_data = data
        self.data_len = len(self.all_data)
        self.X = np.array(temp_X)
        self.y = np.array(temp_y)

        self.split()

    @staticmethod
    def calc_accuracy(y_true, y_pred):
        s = 0
        for i in range(len(y_true)):
            if y_true[i] == int(np.argmax(y_pred[i])):
                s += 1
        return s / len(y_true)

    # important when the costs of falsely classifying something as positives is high
    # like in spam email protection where wrongly classifying an email  as spam could result in
    # the user losing important info
    def calc_precision(self, y_true, y_pred):
        num_true_positive = 0
        num_false_positive = 0
        for i in range(self.data_len):
            if y_true[i] == 1 and y_pred[i] == 1:
                num_true_positive += 1
            if y_true[i] == 0 and y_pred[i] == 1:
                num_false_positive += 1

        return num_true_positive / (num_true_positive + num_false_positive)

    # important when the costs of falsely classifying something as negative is high
    # like in identifying terrorists where a terrorist is wrongly identified as a normal
    # person could result in great loss of lives
    def calc_recall(self, y_true, y_pred):
        num_true_positive = 0
        num_false_negative = 0
        for i in range(self.data_len):
            if y_true[i] == 1 and y_pred[i] == 1:
                num_true_positive += 1
            if y_true[i] == 1 and y_pred[i] == 0:
                num_false_negative += 1

        return num_true_positive / (num_true_positive + num_false_negative)

    def calc_f1_score(self, y_true, y_pred):
        recall = self.calc_recall(y_true, y_pred)
        precision = self.calc_precision(y_true, y_pred)
        return 2 * recall * precision / (recall + precision)

    @staticmethod
    def convert_to_matrix(X, y):
        X.shape += (1, )
        y.shape += (1, )
        return X, y
