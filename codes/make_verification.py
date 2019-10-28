import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


test_df = pd.read_csv("./datasets/test.csv")
unordered = pd.read_csv("./datasets/answer.csv")

res = pd.DataFrame()
res['PassengerId'] = test_df['PassengerId']
res['Survived'] = ''


for index, row in test_df.iterrows():
    for index2,row2 in unordered.iterrows():
        if (row2['name']==row['Name'] and row2['age']==row['Age']) or (row2['name']==row['Name'] and row2['ticket']==row['Ticket']):
            res.loc[res['PassengerId'] == row['PassengerId'], 'Survived'] = row2['survived']
            break

res.to_csv("./datasets/labeled_test_set.csv", index = False)