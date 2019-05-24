import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

import warnings
warnings.filterwarnings('ignore')


def sex_extraction(data):
    sex = pd.get_dummies(data['Sex'], drop_first=False)
    return sex


def embarked_extraction(data):
    embarked = pd.get_dummies(data['Embarked'], drop_first=False) #fills nan with 0 0 0
    return embarked


def pclass_extraction(data):
    pclass = pd.get_dummies(data['Pclass'], drop_first=False, prefix='class')
    
    return pclass


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
        bins = (0, 5, 12, 18, 25, 35, 60, 120)
        group_names = [0, 1, 2, 3, 4, 5, 6]
        age = pd.cut(age, bins, labels=group_names)

        return age
    
    age = DISCRETIZATION(age)
    data['age'] = age  ## Solving naming issue

    return data['age']



def fare_extraction(data):
    fare = data.Fare.fillna( data.Fare.mean() )
    # print(np.max(fare))
    # print(np.min(fare))
    # plt.hist(fare, bins=100)
    # plt.show()
    bins = (-1, 5, 15, 25, 31, 90, 513)
    group_names = [0, 1, 2, 3, 4, 5]
    fare = pd.cut(fare, bins, labels=group_names)
    return fare



def name_extraction(data):
    name = data.Name
    name = name.map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
    name = name.replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    name = name.replace(['Mlle', 'Ms'], 'Miss')
    name = name.replace('Mme', 'Mrs')
    name = pd.get_dummies(name, drop_first=False)
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
    ticket = pd.get_dummies(ticket, drop_first=False)
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



train = pd.read_csv("./datasets/train.csv")
y_train = train.Survived
train.drop('Survived', axis=1, inplace=True)
test = pd.read_csv("./datasets/test.csv")
passenger_id = test.PassengerId
full = train.append( test , ignore_index = True )
full.drop('PassengerId', axis=1, inplace=True)


sex = sex_extraction(full)
embarked = embarked_extraction(full)
pclass = pclass_extraction(full)
age = age_extraction(full)
fare = fare_extraction(full)
name = name_extraction(full)
ticket = ticket_extraction(full)
cabin = cabin_extraction(full)
siblings, parents, size, isAlone = family_extraction(full)

full = pd.concat([sex,embarked,pclass,age,fare,name,ticket,cabin,siblings,parents,size,isAlone], axis=1)

# full.columns.age = 'text'
# print(full.head(2))

# sns.heatmap(full.isnull(),yticklabels=False, cbar=False, cmap='YlGnBu')
# plt.show()

def preprocessed_data():
    x_train = full[0:891]
    X_test = full[891:]
    
    return x_train, y_train, X_test

preprocessed_data()