from featue_engineering import preprocessed_data


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb


def test_evaluation(Y_hat):
    data = pd.read_csv("./datasets/answer.csv")
    Y = np.array(data.loc[:,'survived']) 
    Y = Y[891:]
    accuracy = accuracy_score(Y, Y_hat)
    print("Test Accuracy = " + str(accuracy))


def plot_model_var_imp( model , x , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = x.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = False )
    imp[ : 35 ].plot( kind = 'barh' )
    # print (model.score( x , y ))


def plot_variable_importance( x , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( x , y )
    plot_model_var_imp( tree , x , y )


def find_best_RFC(model):
    model = RandomForestClassifier()
    # Set the model to the best combination of parameters
    parameters = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
            
    acc_scorer = make_scorer(accuracy_score)

    grid_obj = GridSearchCV(model, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(x_train, y_train)
    model = grid_obj.best_estimator_

    return model

def get_models():
    model1 = RandomForestClassifier()
    model2 = GradientBoostingClassifier()
    model3 = AdaBoostClassifier()
    model4 = ExtraTreesClassifier()

    model = [model1, model2, model3, model4]

    model5 = SVC()
    
    model6 = LogisticRegression()
    model7 = KNeighborsClassifier()
    model8 = GaussianNB()

    model.append(model5)
    model.append(model6)
    model.append(model7)
    model.append(model8)

    return model

def test_models(x_train, y_train, x_test, y_test):
    models = get_models()
    for model in models:
        model.fit( x_train , y_train )
        print ("Train : " + str(model.score( x_train , y_train )))
        print ("Test  : " + str(model.score( x_test , y_test )))
        print("")


x_train, y_train, x_test, y_test = preprocessed_data()

# plot_variable_importance(x_train, y_train)

# model = RandomForestClassifier()
# model.fit( x_train , y_train )

# print (model.score( x_train , y_train ))
# print (model.score( x_test , y_test ))



# plot_model_var_imp(model, x_train, y_train)
# plt.show()


# test_models(x_train, y_train, x_test, y_test)