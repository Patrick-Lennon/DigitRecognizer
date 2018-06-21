import matplotlib.pyplot as plot
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pan

"""
Using scikit-learn to predict what numbers the hand-written digits are

#author Patrick Lennon
"""

#panda data converted to a matrix so it's a 2d array
data = pan.read_csv('data/train.csv').values
clf = DecisionTreeClassifier()

#spliting data into 2: training and prediction
dataTrain = data[0:21000, 1:]
tLabel = data[0:21000, 0]

clf.fit(dataTrain, tLabel)

dataTest = data[21000:, 1:]
aLabel = data[21000:, 0]
"""
#taking the virst element (number) converts to a 28x28 matrix vector
d = dataTest[0]
d.shape=(28,28)
plot.imshow(255-d, cmap='Greens')
print(clf.predict( [dataTest[0]]) )
plot.show()
"""

#the prediction array
predict = clf.predict(dataTest)

#looping through the predictions and the vector images to see if they're a match
counter = 0
for i in range(0, 21000):
    if predict[i] == aLabel[i]:
        counter += 1
print('Accuracy = ', (counter/21000) * 100)