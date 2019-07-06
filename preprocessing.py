import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def normalizer(x):
    m = x.shape[0]
    x = x - np.mean(x)
    x = (x * m) / (np.sum(x ** 2))
    return x


train = pd.read_csv("./datasets/train.csv")


########### Pclass ################
# sns.countplot(x='Survived',hue='Pclass',data=train)
# plt.show()
# Pclass 2 is obviously insignificant
# Pclass 1 should be tested


######### Embarked ############# 
# sns.countplot(x='Survived',hue='Embarked',data=train)
# plt.show()
# print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# should be tested


######### SEX ############# 
# sns.countplot(x='Survived',hue='Sex',data=train)
# plt.show()
# print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# obviously important



######### Name ############# 
# name = train.Name
# name = name.map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
# name = name.replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
# # name = name.replace(['Mlle', 'Ms'], 'Miss')
# # name = name.replace('Mme', 'Mrs')
# name = name.replace(['Mme', 'Mrs', 'Mlle', 'Ms', 'Miss'], 'Female')
# name = pd.get_dummies(name, drop_first=False)
# print(name.head)
# name.drop('Miss', axis=1, inplace=True)
# tmp = pd.concat([train['Survived'],name], axis=1)
# sns.countplot(x='Survived',hue='Name',data=tmp)
# plt.show()
# print(tmp[['Name', 'Survived']].groupby(['Name'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# Master is not important



########## Ticket ############
# ticket = train.Ticket
def ticket_handler(ticket):
        ticket = ticket.replace( '.' , '' )
        ticket = ticket.replace( '/' , '' )
        ticket = ticket.split()
        ticket = map( lambda t : t.strip() , ticket )
        ticket = list(filter( lambda t : not t.isdigit() , ticket ))
        if len( ticket ) > 0:
            return (ticket[0])
        else: 
            return 'X'
# ticket = ticket.map( ticket_handler )
# ticket = pd.get_dummies(ticket, drop_first=False)
# print(ticket.head(5))
# tmp = pd.concat([train['Survived'],ticket], axis=1)
# print(tmp.head(3))
# sns.countplot(x='Survived',hue='Ticket',data=tmp)
# plt.show()
# print(tmp[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(tmp.head(10))




############# Cabin #######################
# cabin = train.Cabin
# cabin = cabin.fillna( 'Without Cabin' )
# cabin = cabin.map( lambda c : c[0] )
# tmp = pd.concat([train['Survived'],cabin], axis=1)
# sns.countplot(x='Survived',hue='Cabin',data=tmp)
# plt.show()
# print(tmp[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# use only E,B,D,W if they have significant impact



############# Family ####################
# siblings = train.SibSp
# parents = train.Parch
# size = siblings + parents
# train['isAlone'] = size.map( lambda s : 1 if s == 1 else 0 )
# tmp = pd.concat([train['Survived'],siblings], axis=1)
# sns.countplot(x='Survived',hue='SibSp',data=tmp)
# plt.show()
# print(tmp[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# tmp = pd.concat([train['Survived'],train['isAlone']], axis=1)
# print(tmp.head())
# sns.countplot(x='Survived',hue='isAlone',data=tmp)
# plt.show()
# print(tmp[['isAlone', 'Survived']].groupby(['isAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False))




z = pd.DataFrame()
z['E'] = train['Embarked']
z['S'] = train['Sex']
z = pd.get_dummies(z, drop_first=False)
# print(z.head())
z['new'] = z['E_C']*z['S_male'] + z['E_S']*z['S_female'] + z['E_Q']*z['S_female']
z['Survived'] = train['Survived']
# sns.countplot(x='Survived',hue='new',data=z)
# plt.show() 
# print(z[['new', 'Survived']].groupby(['new'], as_index=False).mean().sort_values(by='Survived', ascending=False))


z = pd.DataFrame()
z['A'] = train['Age']
z['S'] = train['Sex']
z['S'] = z['S'].replace('male', 1)
z['S'] = z['S'].replace('female', -1)
z['N'] = z['S'] * z['A']
z['N'] = z['N'].mask(z['N'].between(0,11.2), 0)
z['N'] = z['N'].mask(z['N'].between(-38,-30), 0)
z['N'] = z['N'].mask(z['N']!=0, 1)
z['Survived'] = train['Survived']
# print(z.head())
# sns.countplot(x='Survived',hue='N',data=z)
# plt.show() 
# print(z[['N', 'Survived']].groupby(['N'], as_index=False).mean().sort_values(by='Survived', ascending=False))



### Numercial features ###
def correlating_numerical_features(data, feature):
    g = sns.FacetGrid(data, col='Survived')
    g.map(plt.hist, feature, bins=20)
    plt.show()
# correlating_numerical_features(train, 'Age')

def correlating_numerical_and_ordinal_features(data, feature_A, feature_B):
    grid = sns.FacetGrid(data, col='Survived', row=feature_A, height=2.2, aspect=1.6)
    grid.map(plt.hist, feature_B, alpha=.5, bins=20)
    grid.add_legend()
    plt.show()
# correlating_numerical_and_ordinal_features(train, 'Sex', 'Embarked')

def correlating_categorical_features(data, feature_A, feature_B, feature_C):
    grid = sns.FacetGrid(data, row=feature_A, height=2.2, aspect=1.6)
    grid.map(sns.pointplot, feature_B, 'Survived', feature_C, palette='deep')
    grid.add_legend()
    plt.show()
# correlating_categorical_features(train, 'Embarked', 'Pclass', 'Sex')
# correlating_categorical_features(train, 'Embarked', 'Pclass', 'Sex')

def correlating_categorical_and_numerical_features(data, feature_A, feature_B, feature_C):
    grid = sns.FacetGrid(data, row=feature_A, col='Survived', height=2.2, aspect=1.6)
    grid.map(sns.barplot, feature_B, feature_C, alpha=.5, ci=None)
    grid.add_legend()
    plt.show()
# correlating_categorical_and_numerical_features(train, 'Embarked', 'Sex', 'Fare')
# correlating_categorical_and_numerical_features(train, 'Embarked', 'Sex', 'Age')


def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    plt.show()
# plot_distribution( train , var = 'Age' , target = 'Survived' , row = 'Sex' )


########### Age #############
# train['AgeBand'] = pd.qcut(train['Age'], 20)
# print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))


############ Fare ##############
# train['FareBand'] = pd.qcut(train['Fare'], 4)
# print(train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))






train.drop('Survived', axis=1, inplace=True)
test = pd.read_csv("./datasets/test.csv")
data = train.append(test , ignore_index = True)
data = data.sort_values('Name')



# data.drop('PassengerId', axis=1, inplace=True)
# sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
# plt.show()

# print(data.describe())
# print(data.describe(include=['O']))
# print(data.info())

"""
1 nan in Fare
Embarked has two nans
Cabin and Age have many nans
"""


passenger_id = data['PassengerId']
sex = pd.get_dummies(data['Sex'], drop_first=True)

embarked = pd.get_dummies(data['Embarked'], drop_first=False, prefix='Embark') #fills nan with 0 0 0
embarked.drop('Embark_C', axis=1, inplace=True)
# embarked.drop('Embark_Q', axis=1, inplace=True)

# pclass = data['Pclass']
pclass = pd.get_dummies(data['Pclass'], drop_first=False, prefix='class')
pclass.drop('class_2', axis=1, inplace=True)
pclass.drop('class_1', axis=1, inplace=True)



# sns.boxplot(x='Pclass', y='Age', data=data)
# plt.show() 
#### 39 for pclass=1 & 29 pclass=2  & 24 pclass=3
def age_handler(cols):
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

    
age = data[['Age', 'Pclass']].apply(age_handler, axis=1)

bins = (0, 20, 28, 38, 80)   #LR coefficient = -0.29
group_names = [0, 1, 2, 3]
age = pd.cut(age, bins, labels=group_names)
# data['age'] = age  ## Solving naming issue



fare = data['Fare'].fillna( data.Fare.mean() )
# print(data.Fare.describe())
# plt.hist(fare, bins=100)
# plt.show()
bins = (-1, 12, 31, 1000)   #LR Coefficient = -0.16
group_names = [0, 1, 2]
fare = pd.cut(fare, bins, labels=group_names)



name = data['Name']
name = name.map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
name = name.replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
name = name.replace(['Mme', 'Mrs', 'Mlle', 'Ms', 'Miss'], 'Female')
name = pd.get_dummies(name, drop_first=False)
name.drop('Master', axis=1, inplace=True)





ticket = data.Ticket
def ticket_handler(ticket):
        ticket = ticket.replace( '.' , '' )
        ticket = ticket.replace( '/' , '' )
        ticket = ticket.split()
        ticket = map( lambda t : t.strip() , ticket )
        ticket = list(filter( lambda t : not t.isdigit() , ticket ))
        if len( ticket ) > 0:
            return (ticket[0])
        else: 
            return 'XX'
ticket = ticket.map( ticket_handler )
ticket = ticket.replace(['SWPP', 'SC'], 'AAA')
ticket = ticket.replace('FCC', 'BBB')
ticket = ticket.replace(['SCAH', 'PP', 'PC'], 'CCC')
ticket = ticket.replace(['CA', 'WEP'], 'DDD')
ticket = ticket.replace('LINE', 'EEE')
ticket = ticket.replace(['SOC', 'SOTONOQ'], 'FFF')
ticket = ticket.replace(['WC', 'A5'], 'GGG')
ticket = ticket.replace(['AS', 'CASOTON', 'SP', 'SOTONO2', 'SCA4', 'SOPP', 'SOP', 'FC', 'Fa', 'SCOW', 'A4'], 'HHH')
ticket = pd.get_dummies(ticket, drop_first=False)
ticket.drop(['PPP', 'STONO', 'STONO2', 'SCParis', 'SCPARIS', 'XX'], axis=1, inplace=True)
ticket.drop(['A', 'AQ3', 'AQ4', 'C', 'LP', 'SCA3', 'STONOQ'], axis=1, inplace=True)






cabin = data.Cabin
cabin = cabin.fillna( 'Without Cabin' )
cabin = cabin.map( lambda c : c[0] )
cabin = cabin.replace(['A','C','F','G','T'], 'drop')
cabin = pd.get_dummies(cabin, drop_first=False)
cabin.drop('drop', axis=1, inplace=True)






siblings = data.SibSp
parents = data.Parch
size = siblings + parents
isAlone = size.map( lambda s : 1 if s == 1 else 0 )
siblings = pd.get_dummies(siblings, drop_first=False)
# siblings.drop([1,2,3,4,5,8], axis=1, inplace=True)


family = pd.DataFrame()
family['Name'] = data['Name']
family['Name'] = family['Name'].map( lambda nam: nam.split( ',' )[0])
family['distinction'] = 0
last = ''
last_index = 0

for index, row in family.iterrows():
    if row['Name']==last :
        family.loc[index, 'distinction'] = last_index
    else:
        last_index += 1
        last = row['Name']
        family.loc[index, 'distinction'] = last_index

family['distinction'] = normalizer(family['distinction'])


z = pd.DataFrame()
z['E'] = data['Embarked']
z['S'] = data['Sex']
z = pd.get_dummies(z, drop_first=False)
z['new'] = z['E_C']*z['S_male'] + z['E_S']*z['S_female'] + z['E_Q']*z['S_female']
my_feature = pd.DataFrame()
my_feature['feature'] = z['new']
my_feature = pd.get_dummies(my_feature['feature'], drop_first=False)


z = pd.DataFrame()
z['S'] = data['Sex']
z['S'] = z['S'].replace('male', 1)
z['S'] = z['S'].replace('female', -1)
z['N'] = z['S'] * train['Age']
z['N'] = z['N'].mask(z['N'].between(0,11.2), 0)
z['N'] = z['N'].mask(z['N'].between(-38,-30), 0)
z['N'] = z['N'].mask(z['N']!=0, 'Y')
my_feature2 = z['N']
my_feature2 = my_feature2.replace(0, 'X')
my_feature2 = my_feature2.replace(1, 'Y')
my_feature2 = pd.get_dummies(my_feature2, drop_first=False)
my_feature2.drop('Y', axis=1, inplace=True)
# print(my_feature2.head())



processed_data = pd.concat([sex,pclass,fare,name,ticket,cabin,my_feature], axis=1)
processed_data['age'] = age
processed_data['isAlone'] = isAlone
# processed_data['distinction'] = family['distinction']
processed_data['id'] = passenger_id

processed_data = processed_data.sort_values('id')
processed_data.drop('id', axis=1, inplace=True)



def after_preprocessing():
    return processed_data