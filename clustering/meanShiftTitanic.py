import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd
import re

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
# df.drop(['body', 'name', 'ticket'], 1, inplace=True)
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

clf = MeanShift()
clf.fit(x)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan
for i, label in enumerate(labels):
    original_df['cluster_group'].iloc[i] = label

n_clusters = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
