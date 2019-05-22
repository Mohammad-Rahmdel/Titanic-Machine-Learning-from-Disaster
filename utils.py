import numpy as np
import math
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Imputer 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignoring warnings


def ReLu(x):
    return np.maximum(0, x)

def sigmoid(z):
    x = tf.placeholder(tf.float32, name = 'x')
    results = tf.sigmoid(x)
    with tf.Session() as session:
        results = session.run(results, feed_dict={x: z})
    
    return results


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X of shape (input size, m)
    Y of shape (1, m)
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                
    mini_batches = []
    np.random.seed(seed)
    
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def normalizer(x):
    m = x.shape[0]
    x = x - np.mean(x)
    x = (x * m) / (np.sum(x ** 2))
    return x



def train_visulization():
    data = pd.read_csv("./datasets/train.csv")
    # data.info()
    sns.heatmap(data.isnull(),yticklabels=False, cbar=False, cmap='YlGnBu')

    sns.countplot(x='Survived',hue='Pclass',data=data)
    sns.distplot(data['Age'].dropna(), kde=False, bins=30)
    data['Fare'].plot.hist(bins=40, figsize=(10,4))
    sns.boxplot(x='Pclass', y='Age', data=data)


    def impute_age(cols):
        Age=cols[0]
        Pclass=cols[1]
        
        if pd.isnull(Age):
            if Pclass==1:
                return 37
            elif Pclass==2:
                return 29
            else:
                return 24
        else:
            return Age


    data['Age']=data[['Age', 'Pclass']].apply(impute_age, axis=1)
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
    
    data.drop('Cabin', axis=1, inplace=True)

    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')

    data.dropna(inplace=True) #Dropping null values


    sex=pd.get_dummies(data['Sex'], drop_first=True)
    embark=pd.get_dummies(data['Embarked'], drop_first=True)
    classes=pd.get_dummies(data['Pclass'])
    data=pd.concat([data,sex,embark,classes], axis=1)
    data.drop(['Pclass', 'Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
    # print(data.head())
    # sns.heatmap(data.corr(), annot=True)

    return data


def train_preprocessing():
    data = pd.read_csv("./datasets/train.csv")

    del data['Cabin']
    del data['Ticket']
    del data['PassengerId']

    data = data.replace({'Mrs.': 1, 'Miss.': 2}, regex=True)
    data = data.replace({'Mr.': 0}, regex=True)
    for i in range(len(data['Name'])):
        # if data['Name'][i] != 0 and data['Name'][i] != 1 and data['Name'][i] != 2 :
        if data.loc[i,'Name'] != 0 and data.loc[i,'Name'] != 1 and data.loc[i,'Name'] != 2 :
            data.loc[i,'Name'] = 3

    def impute_age(cols):
        Age=cols[0]
        Pclass=cols[1]
        
        if pd.isnull(Age):
            if Pclass==1:
                return 37
            elif Pclass==2:
                return 29
            else:
                return 24
        else:
            return Age


    data['Age']=data[['Age', 'Pclass']].apply(impute_age, axis=1)

    data.dropna(inplace=True) #Dropping null values

    sex=pd.get_dummies(data['Sex'])
    embark=pd.get_dummies(data['Embarked'])
    classes=pd.get_dummies(data['Pclass'])
    data=pd.concat([data, sex, embark, classes], axis=1)
    
    data.drop(['Pclass', 'Sex', 'Embarked'], axis=1, inplace=True)
    # data.drop(['Name'], axis=1, inplace=True)
    train = data
    return train



def test_visualization():
    test=pd.read_csv('./datasets/test.csv')
    sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
    sns.boxplot(x='Pclass', y='Age', data=test)

    def impute_age2(cols):
        Age=cols[0]
        Pclass=cols[1]
        
        if pd.isnull(Age):
            if Pclass==1:
                return 42
            elif Pclass==2:
                return 26
            else:
                return 24
        else:
            return Age
    test['Age']=test[['Age', 'Pclass']].apply(impute_age2, axis=1)


    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #fixing Fare missing values
    imputer = imputer.fit(test.iloc[:, 8:9])
    test.iloc[:, 8:9] = imputer.transform(test.iloc[:, 8:9])
    sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')

    test.drop('Cabin', axis=1, inplace=True)

    sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')


    sex2=pd.get_dummies(test['Sex'], drop_first=True)
    embark2=pd.get_dummies(test['Embarked'], drop_first=True)
    pclasses=pd.get_dummies(test['Pclass'])
    test=pd.concat([test,sex2,embark2,pclasses], axis=1)
    test.drop(['PassengerId', 'Pclass', 'Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
    




def test_preprocessing():
    test=pd.read_csv('./datasets/test.csv')

    def impute_age(cols):
        Age=cols[0]
        Pclass=cols[1]
        
        if pd.isnull(Age):
            if Pclass==1:
                return 42
            elif Pclass==2:
                return 26
            else:
                return 24
        else:
            return Age
    test['Age']=test[['Age', 'Pclass']].apply(impute_age, axis=1)


    # print(test.head())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean') #fixing Fare missing values
        imputer = imputer.fit(test.iloc[:, 8:9])
        test.iloc[:, 8:9] = imputer.transform(test.iloc[:, 8:9])
    

    del test['Cabin']
    del test['Ticket']
    # Passenger_id = test.iloc[:,0]
    del test['PassengerId']

    test = test.replace({'Mrs.': 1, 'Miss.': 2}, regex=True)
    test = test.replace({'Mr.': 0}, regex=True)
    for i in range(len(test['Name'])):
        if test.loc[i,'Name'] != 0 and test.loc[i,'Name'] != 1 and test.loc[i,'Name'] != 2 :
            test.loc[i,'Name'] = 3


    sex=pd.get_dummies(test['Sex'])
    embark=pd.get_dummies(test['Embarked'])
    classes=pd.get_dummies(test['Pclass'])
    test=pd.concat([test,sex,embark,classes], axis=1)

    test.drop(['Pclass', 'Sex', 'Embarked'], axis=1, inplace=True)
    # test.drop(['Name'], axis=1, inplace=True)
    
    return test


