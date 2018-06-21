import matplotlib.pyplot as plot
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pan

data = pan.read_csv('data/train.csv')
print(data)