import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd
import random


style.use('fivethirtyeight')


dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_point = [5, 7]


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('Valor de K invalido')

    distances = []
    for group in data:
        for features in data[group]:
            euc_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euc_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    result = Counter(votes).most_common(1)
    confidence = result[0][1] / k
    return result[0][0], confidence


prediction = k_nearest_neighbors(dataset, new_point, k=3)
print(f'Prediction: {prediction}')

plt.scatter(np.array(dataset['k'])[:, 0], np.array(dataset['k'])[:, 1], color='g')
plt.scatter(np.array(dataset['r'])[:, 0], np.array(dataset['r'])[:, 1], color='b')
plt.scatter(new_point[0], new_point[1])
plt.show()

# Real example in the same dataset
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# Some of the values get "''", this solve it
full_data = df.astype(float).values.tolist()

random.shuffle(full_data)
test_size = 0.4
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print(f'Accuracy: {correct/total}')

predict_example = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])

prediction_breast = k_nearest_neighbors(train_set, predict_example, k=5)
print(f'Prediction: {prediction_breast}')