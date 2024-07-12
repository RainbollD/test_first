import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('C:\school_ege\students.csv', delimiter=',')[['Hair length', 'Shoe size', 'Sex']].dropna()

stand = StandardScaler()

stand.fit(df[['Hair length', 'Shoe size']].values)
arr = stand.transform(df[['Hair length', 'Shoe size']].values)

model = KN(n_neighbors=3)
model.fit(arr, y=df['Sex'].values)

df = pd.read_csv('C:\school_ege\students_test.csv', delimiter=',')[['Hair length', 'Shoe size', 'Sex']].dropna()

stand.fit(df[['Hair length', 'Shoe size']].values)
arr = stand.transform(df[['Hair length', 'Shoe size']].values)

df['New'] = model.predict(arr)

print(pd.crosstab(df['Sex'], df['New']))
print('No way')
print('New brach')
