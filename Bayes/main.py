import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import category_encoders as ce

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship',
                                 'race', 'sex', 'native'])
gnb = GaussianNB()
train_data = pd.read_csv("SalaryData_Train.csv")

X = train_data.drop(['Salary'], axis=1)
y = train_data['Salary']
X = encoder.fit_transform(X)

gnb.fit(X, y)
prediction = gnb.predict(X)
print(accuracy_score(y, prediction))