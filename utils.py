import numpy as np
import math
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Imputer 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


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


def preprocessing():
    data = pd.read_csv("./datasets/train.csv")
    # data.info()
    del data['Cabin']
    del data['Ticket']
    del data['PassengerId']
    data = data.replace({'male': 0, 'female': 1})
    data = data.replace({'S': 0, 'C': 1, 'Q': 2})
    data = data.replace({'Mrs.': 1, 'Miss.': 2}, regex=True)
    data = data.replace({'Mr.': 0}, regex=True)
    for i in range(len(data['Name'])):
        if isinstance(data['Name'][i], str):
            data['Name'][i] = 3
    data = data.fillna(data.mean())

    # sns.heatmap(data.corr(), annot=True)

    # for i in range(1, data.shape[1]):
    #     data.iloc[:,i] = normalizer(data.iloc[:,i])

    X = data.iloc[:,1:9]
    # Y = data.iloc[:,0]
    Y = np.array([data.loc[:,'Survived']])
    
    # print(data.head(20))
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = np.transpose(X)

    return X, Y

# preprocessing()





def test_manipulation():
    data = pd.read_csv("./datasets/test.csv")
    # m = data.shape[0]
    del data['Cabin']
    del data['Ticket']
    del data['PassengerId']
    data = data.replace({'male': 0, 'female': 1})
    data = data.replace({'S': 0, 'C': 1, 'Q': 2})
    data = data.replace({'Mrs.': 1, 'Miss.': 2}, regex=True)
    data = data.replace({'Mr.': 0}, regex=True)
    for i in range(len(data['Name'])):
        if isinstance(data['Name'][i], str):
            data['Name'][i] = 3
    data = data.fillna(data.mean())
    X = data
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = np.transpose(X)
    return X

# test_manipulation()





def data_manipulation(dataset):
    data = pd.read_csv("./datasets" + dataset + ".csv")
    # data = data[0:864]
    # print(data.shape)
    # print(data.head())
    m = data.shape[0] #891
    if dataset == "train":
        y = np.array([data.loc[:,'Survived']])

    x1 = np.array(data.loc[:,'Pclass'])
    x2 = np.array(data.loc[:,'Sex'])
    x3 = np.array(data.loc[:,'Age'])  # has NAN
    x4 = np.array(data.loc[:,'SibSp'])
    x5 = np.array(data.loc[:,'Parch'])
    x6 = np.array(data.loc[:,'Fare'])

    if dataset == "test":
        f = pd.isnull(x6)
        c = []
        for i in range(len(f)) :
            if f[i] == True:
                c.append(i)
        for i in c:
            x6[i] = 0

    x7 = np.array(data.loc[:,'Name'])
    x8 = np.array(data.loc[:,'Embarked'])  # has NAN


    x1 = normalizer(x1)
    x4 = normalizer(x4)
    x5 = normalizer(x5)
    x6 = normalizer(x6)

    x2[x2 == 'male'] = 1
    x2[x2 == 'female'] = -1
    x2 = normalizer(x2)

    #solving unknown age problem
    #replacing NANs with mean of ages
    f = np.isnan(x3)
    cnt = 0
    for i in f:
        if i == True:
            cnt += 1
    x3 = np.nan_to_num(x3)
    mean = np.sum(x3) / (m - cnt)
    x3[x3 == 0] = mean
    x3 = normalizer(x3)


    x8[x8 == 'S'] = 1
    x8[x8 == 'C'] = 2
    x8[x8 == 'Q'] = 3

    f = pd.isnull(x8)
    c = []
    for i in range(len(f)) :
        if f[i] == True:
            c.append(i)
    for i in c:
        x8[i] = 0
    mean = np.sum(x8) / (m - len(c))
    x8[x8 == 0] = mean
    x8 = normalizer(x8)





    for i in range(len(x7)):
        if "Mr." in x7[i]:
            x7[i] = 1
        elif "Mrs." in x7[i]:
            x7[i] = 2
        elif "Miss." in x7[i]:
            x7[i] = 3
        elif "Master." in x7[i]:
            x7[i] = 4
        elif "Dr." in x7[i]:
            x7[i] = 5
        else :
            # print(x7[i])
            x7[i] = 6

    x7 = normalizer(x7)

    #x7 = Name{
    # Mr.
    # Mrs.
    # Miss.
    # Master.
    # Dr.
    # Lady. (Lucille Christiana Sutherland) ("Mrs Morgan")    #1
    # Ms. Encarnacion                                         #1
    # Sir. Cosmo Edmund ("Mr Morgan")                         #1
    # Don.                                                    #1
    # Rev.                                                    #6
    # Mme.                                                    #1
    # Ms.                                                     #1
    # Major.                                                  #2
    # Gordon.                                                 #2
    # Mlle.                                                   #2
    # Col.                                                    #2
    # Capt.                                                   #1
    # Countess.                                               #1
    # Jonkheer.                                               #1
    # }


    # x = np.vstack((x1, x2, x3, x4, x5, x6))
    x = np.vstack((x1, x2, x3, x4, x5, x6, x7, x8))
    # print(x.shape)
    if dataset == "train" :
        return x, y
    elif dataset == "test" :
        return x
    else :
        print("WRONG INPUT!")






def data_visulization():
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

    # del data['Cabin']
    # del data['Ticket']
    # del data['PassengerId']
    # data = data.replace({'male': 0, 'female': 1})
    # data = data.replace({'S': 0, 'C': 1, 'Q': 2})
    # data = data.replace({'Mrs.': 1, 'Miss.': 2}, regex=True)
    # data = data.replace({'Mr.': 0}, regex=True)
    # for i in range(len(data['Name'])):
    #     if isinstance(data['Name'][i], str):
    #         data['Name'][i] = 3
    # data = data.fillna(data.mean())

    return data

def test_pre():
    test=pd.read_csv('./datasets/test.csv')
    sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='YlGnBu')
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


    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(test.iloc[:, 8:9])
    test.iloc[:, 8:9] = imputer.transform(test.iloc[:, 8:9])
    sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')

    test.drop('Cabin', axis=1, inplace=True)

    sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')


    sex2=pd.get_dummies(test['Sex'], drop_first=True)
    embark2=pd.get_dummies(test['Embarked'], drop_first=True)
    pclasses=pd.get_dummies(test['Pclass'])
    test=pd.concat([test,sex2,embark2,pclasses], axis=1)
    Passenger_id = test.iloc[:,0]
    test.drop(['PassengerId', 'Pclass', 'Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
    return test


train = data_visulization()
test = test_pre()

x_train=train.iloc[:,2:]
y_train=train.iloc[:,0:1]
x_test=test.iloc[:,:]


logisticReg=LogisticRegression()
logisticReg.fit(x_train,y_train)

y_pred= logisticReg.predict(x_test)
accuracy = round(logisticReg.score(x_train, y_train) * 100, 2)
print(accuracy)


ranFor = RandomForestClassifier(n_estimators = 70)
ranFor.fit(x_train,y_train)
y_pred2= ranFor.predict(x_test)
accuracy2 =round(ranFor.score(x_train, y_train)*100,2)
print(accuracy2)



svc=SVC()
svc.fit(x_train, y_train)
y_pred3=svc.predict(x_test)
accuracy3=round(svc.score(x_train, y_train)*100,2)
print(accuracy3)

