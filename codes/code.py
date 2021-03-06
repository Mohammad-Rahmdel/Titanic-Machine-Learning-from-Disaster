import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

from preprocessing import after_preprocessing

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


processed_data = after_preprocessing()
y_train = pd.read_csv("./datasets/train.csv")['Survived']

x_train = processed_data[0:891]
x_test = processed_data[891:]




n_neighbors = [6]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,10))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc")

gd.fit( x_train , y_train )
print (gd.score( x_train , y_train ))

y_pred = gd.predict(x_test)
y_test = pd.read_csv("./datasets/labeled_test_set.csv")
y_test = np.array(y_test.loc[:,'Survived']) 
print (gd.score( x_test , y_test ))




# model = Perceptron()
# model = SGDClassifier()
# model = LogisticRegression()
# model = RandomForestClassifier()
# model = KNeighborsClassifier(n_neighbors = 6)
# model = LinearSVC()
# model = DecisionTreeClassifier()
# model = GradientBoostingClassifier()

# model.fit( x_train , y_train )
# print (model.score( x_train , y_train ))

# y_pred = model.predict(x_test)
# y_test = pd.read_csv("./datasets/labeled_test_set.csv")
# y_test = np.array(y_test.loc[:,'Survived']) 
# print (model.score( x_test , y_test ))

g = np.subtract(y_test, y_pred)
g = abs(g)
m = 418
print("Test Accuracy = " + str(1 - (np.sum(g)) / m))

temp = pd.DataFrame(pd.read_csv("./datasets/test.csv")['PassengerId'])
temp['Survived'] = y_pred
temp.to_csv("./datasets/submission.csv", index = False)



"""
1-
sex - embarked(Q,S) - pclass 1,3 - age 4 bins fill nans using pclass - fare 3 bins - 
name(female - mr - rare - drop master) - ticket - cabin - isalone

siblings removed -> does not help
cabin helps
ticket seems useless
embarked helps a bit

RandomForestClassifier      78.7%
KNeighborsClassifier        78%
KNeighborsClassifier        79.9%
n_neighbors = 6
LogisticRegression          77%
Perceptron                  72%
SGDClassifier               awful
SVC                         77%
DecisionTreeClassifier      79%
GradientBoostingClassifier  78%



KNeighborsClassifier with grid search 84%
DecisionTreeClassifier 71%
GradientBoostingClassifier 76%
LogisticRegression 77%
RandomForestClassifier 76%
LinearSVC 76%
Perceptron 75%
"""