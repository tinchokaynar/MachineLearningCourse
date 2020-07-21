import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

predict_example = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [5, 2, 1, 5, 3, 6, 3, 2, 1], [5, 3, 4, 3, 1, 2, 5, 2, 1]])

prediction = clf.predict(predict_example)
# 2 = Benigno
# 4 = Maligno
print(f'Accuracy: {accuracy}')
print(f'Prediction: {prediction}')
