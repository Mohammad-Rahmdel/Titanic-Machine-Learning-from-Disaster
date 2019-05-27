from featue_engineering import preprocessed_data, get_passengerID


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.feature_selection import RFECV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

import plotly.graph_objs as go
import plotly.offline as py


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


def run_kfold(clf, X_all, Y_all):
    kf = KFold(n_splits=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(X_all, Y_all):
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = Y_all.values[train_index], Y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 



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
    #  Perceptron()
    # LinearSVC()
    # SGDClassifier()
    # DecisionTreeClassifier()

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



def optimal_features(model, x_train, y_train, x_test, y_test):
    rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( 2 ) , scoring = 'accuracy' )
    rfecv.fit( x_train , y_train )

    print (rfecv.score( x_train , y_train ) , rfecv.score( x_test , y_test ))
    print( "Optimal number of features : %d" % rfecv.n_features_ )

    # # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel( "Number of features selected" )
    plt.ylabel( "Cross validation score (nb of correct classifications)" )
    plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )
    plt.show()



def logisticregression_coefficients(x_train, model):
    coeff_df = pd.DataFrame(x_train.columns)
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(model.coef_[0])
    print(coeff_df.sort_values(by='Correlation', ascending=False))


def first_level(x_train, y_train, x_test, j=4):
    # # j = number of models we want to stack
    models = get_models()
    y_trains = []
    y_tests = []
    for i in range(j):
        model = models[i]
        model.fit( x_train , y_train )
        train_y = model.predict( x_train )
        test_y = model.predict( x_test )
        y_tests.append(test_y)
        y_trains.append(train_y)

    base_predictions_train = pd.DataFrame()
    for i in range(j):
        base_predictions_train["C" + str(i+1)] = y_trains[i]

    data = [
        go.Heatmap(
            z= base_predictions_train.astype(float).corr().values ,
            x=base_predictions_train.columns.values,
            y= base_predictions_train.columns.values,
            colorscale='Viridis',
                showscale=True,
                reversescale = True
        )
    ]
    py.iplot(data, filename='labelled-heatmap')

    x_train = y_trains[0]
    x_test = y_tests[0]
    for i in range(1, j):
        x_train = np.column_stack([x_train, y_trains[i]])
        x_test = np.column_stack([x_test, y_tests[i]])

    return x_train, x_test



def second_level(x_train, y_train, x_test, y_test):
    gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,                        
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1)

    gbm.fit(x_train, y_train)

    print ("XGB train accuracy = " + str(gbm.score( x_train , y_train )))
    print ("XGB test accuracy = " + str(gbm.score( x_test , y_test )))

    y_hat = gbm.predict(x_test)
    
    return y_hat




def train_splitter(x_train, y_train, num_test=0.1):
    return train_test_split(x_train, y_train, test_size=num_test, random_state=23)



def make_csv(model, x_test, passenger_id):
    test_Y = model.predict( x_test )
    test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
    test.shape
    test.head()
    test.to_csv( 'titanic_pred.csv' , index = False )





x_train, y_train, x_test, y_test = preprocessed_data()

# x_train, x_cvs, y_train, y_cvs = train_test_split(x_train, y_train)


# plot_variable_importance(x_train, y_train)

# model = RandomForestClassifier()
# model = SVC()
model = LogisticRegression()
# model = LinearSVC()
# model = SGDClassifier()  # ***
# model = DecisionTreeClassifier() # ***
model.fit( x_train , y_train )

# logisticregression_coefficients(x_train, model)



# run_kfold(model, x_train, y_train)

print (model.score( x_train , y_train ))
# print (model.score( x_cvs , y_cvs ))
print (model.score( x_test , y_test ))

# print(x_train.head(2))





# plot_model_var_imp(model, x_train, y_train)
# plt.show()



# # TWO LEVEL MODEL / STACKING / ENSEMBELING
# x_train_stacked, x_test_stacked = first_level(x_train, y_train, x_test, 8)
# second_level(x_train_stacked, y_train, x_test_stacked, y_test)


# test_models(x_train, y_train, x_test, y_test)

# make_csv(model, x_test, get_passengerID())


