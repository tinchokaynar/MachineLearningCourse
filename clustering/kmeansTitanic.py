import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import re

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name', 'ticket', 'fare', 'home.dest'], 1, inplace=True)
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    columns = df.columns.values
    r = re.compile("([a-zA-Z]+)")

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        def convert_cabin(val):
            m = None
            if val:
                m = r.match(val)
            return m.group(0) if m else val

        if df[column].dtype != np.str and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            if column == 'cabin':
                df[column] = list(map(convert_cabin, df[column]))
                column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))
    return df


df = handle_non_numerical_data(df)
# print(df.head())

x = np.array(df.drop(['survived'], 1))
x = preprocessing.scale(x)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(x)

correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i])
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(f'Accuracy: {correct/len(x)}')
