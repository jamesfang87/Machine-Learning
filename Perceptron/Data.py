import numpy as np
import pandas as pd


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

    def read_data(self):
        data = pd.read_csv('Linearly_Sepa_Data.csv')
        temp_X = data[['X1', 'X2']]
        temp_y = data['y']

        self.all_data = data
        self.X = temp_X
        self.y = temp_y
        self.data_len = len(self.all_data)

        self.X_train = np.array(self.X)[:int(self.data_len * 0.8)]
        self.y_train = np.array(self.y)[:int(self.data_len * 0.8)]
        self.X_test = np.array(self.X)[int(self.data_len * 0.8):]
        self.y_test = np.array(self.y)[int(self.data_len * 0.8):]

    @staticmethod
    def calc_accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    # important when the costs of falsely classifying something as positives is high
    # like in spam email protection where wrongly classifying an email  as spam could result in
    # the user losing important info
    @staticmethod
    def calc_precision(y_true, y_pred):
        num_true_positive = 0
        num_false_positive = 0
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                num_true_positive += 1
            if y_true[i] == 0 and y_pred[i] == 1:
                num_false_positive += 1

        return num_true_positive / (num_true_positive + num_false_positive)

    # important when the costs of falsely classifying something as negative is high
    # like in identifying terrorists where a terrorist is wrongly identified as a normal
    # person could result in great loss of lives
    @staticmethod
    def calc_recall(y_true, y_pred):
        num_true_positive = 0
        num_false_negative = 0
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                num_true_positive += 1
            if y_true[i] == 1 and y_pred[i] == 0:
                num_false_negative += 1

        return num_true_positive / (num_true_positive + num_false_negative)

    def calc_f1_score(self, y_true, y_pred):
        recall = self.calc_recall(y_true, y_pred)
        precision = self.calc_precision(y_true, y_pred)
        return 2 * recall * precision / (recall + precision)
