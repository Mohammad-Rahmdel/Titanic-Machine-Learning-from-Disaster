import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

import warnings
warnings.filterwarnings('ignore')





def sex_extraction(data):
    sex = pd.get_dummies(data['Sex'], drop_first=True)
    return sex


def embarked_extraction(data):
    embarked = pd.get_dummies(data['Embarked'], drop_first=False, prefix='Embark') #fills nan with 0 0 0
    embarked.drop('Embark_S', axis=1, inplace=True)
    return embarked


def pclass_extraction(data):
    pclass = pd.get_dummies(data['Pclass'], drop_first=False, prefix='class')
    pclass.drop('class_2', axis=1, inplace=True)
    
    return pclass


def age_find_bounds(data):
    data['AgeBand'] = pd.qcut(data['Age'], 4)
    print(data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

    
def age_extraction(data):
    
    # STEP 1 IMPUTING (filling missing values)
    def age_handler1(x):
        x = x.fillna( x.mean() )
        return x
    

    # sns.boxplot(x='Pclass', y='Age', data=data)
    # plt.show()
    def age_handler2(cols):
        Age=cols[0]
        Pclass=cols[1]
        
        if pd.isnull(Age):
            if Pclass==1:
                return 39
            elif Pclass==2:
                return 29
            else:
                return 24
        else:
            return Age

    
    age = data.Age
    ## CHOOSE ONE OF THESE TWO METHODS
    # age = age_handler1(age)
    age = data[['Age', 'Pclass']].apply(age_handler2, axis=1)


    ## STEP 2  DISCRETIZATION

    def DISCRETIZATION(age):
        # bins = (0, 5, 12, 18, 25, 35, 60, 120)   #LR coefficient = -0.2
        # group_names = [0, 1, 2, 3, 4, 5, 6]
        bins = (0, 20, 28, 38, 80)   #LR coefficient = -0.29
        group_names = [0, 1, 2, 3]
        age = pd.cut(age, bins, labels=group_names)

        return age
    
    age = DISCRETIZATION(age)
    data['age'] = age  ## Solving naming issue

    return data['age']


def fare_find_bounds(data):
    data['FareBand'] = pd.qcut(data['Fare'], 4)
    print(data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))


def fare_extraction(data):
    fare = data.Fare.fillna( data.Fare.mean() )
    # print(train.Fare.describe())
    # plt.hist(fare, bins=100)
    # plt.show()

    # bins = (-1, 5, 15, 25, 31, 90, 513)
    # group_names = [0, 1, 2, 3, 4, 5]

    # bins = (-0.1, 7.91, 14.454, 31, 1000) #LR Coefficient = -0.1
    # group_names = [0, 1, 2, 3]

    # bins = (-1, 0, 8, 15, 31, 1000) #LR Coefficient = -0.04
    # group_names = [0, 1, 2, 3, 4]

    bins = (-1, 12, 31, 1000)   #LR Coefficient = -0.16
    group_names = [0, 1, 2]

    fare = pd.cut(fare, bins, labels=group_names)
    return fare



def name_extraction(data):
    name = data.Name
    name = name.map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
    name = name.replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    name = name.replace(['Mlle', 'Ms'], 'Miss')
    name = name.replace('Mme', 'Mrs')
    name = pd.get_dummies(name, drop_first=False)
    name.drop('Miss', axis=1, inplace=True)
    return name


def ticket_extraction(data):
    ticket = data.Ticket

    def ticket_handler1(ticket):
        ticket = ticket.replace( '.' , '' )
        ticket = ticket.replace( '/' , '' )
        ticket = ticket.split()
        ticket = map( lambda t : t.strip() , ticket )
        ticket = list(filter( lambda t : not t.isdigit() , ticket ))
        if len( ticket ) > 0:
            return ticket[0]
        else: 
            return 'XXX'

    def ticket_handler2(ticket):
        ticket = ticket.replace( '.' , '' )
        ticket = ticket.replace( '/' , '' )
        ticket = ticket.split()
        ticket = map( lambda t : t.strip() , ticket )
        ticket = list(filter( lambda t : not t.isdigit() , ticket ))
        if len( ticket ) > 0:
            return (ticket[0])[0]
        else:
            return 'X'

    ticket = ticket.map( ticket_handler2 )
    ticket = pd.get_dummies(ticket, drop_first=False, prefix='Ticket')
    # print(ticket.head(20))
    return ticket




def cabin_extraction(data):
    cabin = data.Cabin
    cabin = cabin.fillna( 'Without Cabin' )
    cabin = cabin.map( lambda c : c[0] )
    cabin = pd.get_dummies( cabin , prefix = 'Cabin' )
    return cabin



def family_extraction(data):
    siblings = data.SibSp
    parents = data.Parch
    size = siblings + parents + 1
    isAlone = size.map( lambda s : 1 if s == 1 else 0 )
    data['size'] = size
    data['alone'] = isAlone
    
    return siblings, parents, data['size'], data['alone']



def show_categorical_features(data):
    print(data.describe(include=['O']))

def show_numerical_featues(data):
    print(data.describe())

def show_data_types(data):
    print(full.info())


def analyze_by_pivoting(data, feature):
    print(data[[feature, 'Survived']].groupby([feature], as_index=False).mean().sort_values(by='Survived', ascending=False))

def correlating_numerical_features(data, feature):
    g = sns.FacetGrid(data, col='Survived')
    g.map(plt.hist, feature, bins=20)
    plt.show()

def correlating_numerical_and_ordinal_features(data, feature_A, feature_B):
    grid = sns.FacetGrid(train, col='Survived', row=feature_A, size=2.2, aspect=1.6)
    grid.map(plt.hist, feature_B, alpha=.5, bins=20)
    grid.add_legend()
    plt.show()

def correlating_categorical_features(data, feature_A, feature_B, feature_C):
    grid = sns.FacetGrid(data, row=feature_A, size=2.2, aspect=1.6)
    grid.map(sns.pointplot, feature_B, 'Survived', feature_C, palette='deep')
    grid.add_legend()
    plt.show()

def correlating_categorical_and_numerical_features(data, feature_A, feature_B, feature_C):
    grid = sns.FacetGrid(data, row=feature_A, col='Survived', size=2.2, aspect=1.6)
    grid.map(sns.barplot, feature_B, feature_C, alpha=.5, ci=None)
    grid.add_legend()
    plt.show()

train = pd.read_csv("./datasets/train.csv")
# fare_find_bounds(train)
# age_find_bounds(train)


# analyze_by_pivoting(train, 'Sex')
# correlating_numerical_features(train, 'Age')
# correlating_numerical_and_ordinal_features(train, 'Pclass', 'Age')
# correlating_categorical_features(train, 'Embarked', 'Pclass', 'Sex')
# correlating_categorical_and_numerical_features(train, 'Embarked', 'Sex', 'Fare')


y_train = train.Survived
train.drop('Survived', axis=1, inplace=True)
test = pd.read_csv("./datasets/test.csv")
passenger_id = test.PassengerId
full = train.append(test , ignore_index = True)
full.drop('PassengerId', axis=1, inplace=True)


# print(full.describe())




sex = sex_extraction(full)
embarked = embarked_extraction(full)
pclass = pclass_extraction(full)
age = age_extraction(full)
fare = fare_extraction(full)
name = name_extraction(full)
ticket = ticket_extraction(full)
cabin = cabin_extraction(full)
siblings, parents, size, isAlone = family_extraction(full)

# full = pd.concat([sex,embarked,pclass,age,fare,name,ticket,cabin,siblings,parents,size,isAlone], axis=1)
full = pd.concat([sex,embarked,pclass,age,fare,name,isAlone], axis=1)

# full = pd.concat([siblings,parents,size,isAlone], axis=1)
# sns.heatmap(full.corr(), annot=True)
# plt.show()

# print(full.head(2))

# sns.heatmap(full.isnull(),yticklabels=False, cbar=False, cmap='YlGnBu')
# plt.show()




def get_passengerID():
    return passenger_id


def preprocessed_data():
    x_train = full[0:891]
    x_test = full[891:]

    data = pd.read_csv("./datasets/answer.csv")
    Y = np.array(data.loc[:,'survived']) 
    y_test = Y[891:]

    return x_train, y_train, x_test, y_test