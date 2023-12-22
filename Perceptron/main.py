from Perceptron import Perceptron, sigmoid
from Data import Data
import numpy as np

df = Data('Linearly_Sepa_Data.csv')
df.read_data()
perceptron = Perceptron(0.01, 100, sigmoid, df)

perceptron.fit()
print(perceptron.weights)
predictions = np.zeros_like(df.y_test)
for i in range(len(df.X_test)):
    predictions[i] = perceptron.predict(df.X_test[i])

print("Accuracy", df.calc_accuracy(df.y_test, predictions))
print(f"F1: {df.calc_f1_score(df.y_test, predictions)}")
