import numpy as np
from sklearn import preprocessing, neighbors, cross_validation
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?', -99999, inplace=True)
#This is required or else totally un related data will be saved
df.drop(['id'], 1, inplace=True) 

X = np.array(df.drop(['class'], 1))
z = np.array(df['class'])

X_train, X_test, z_train, z_test = cross_validation.train_test_split(X,z,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, z_train)

accu = clf.score(X_test, z_test)
print(accu)


example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)

print(prediction)