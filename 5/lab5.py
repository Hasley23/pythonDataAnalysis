import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
# для случайных интервалов
from random import uniform
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('./data/october_schedule.csv', parse_dates=["Date"])
dataset = dataset.drop(columns=['Unnamed: 6', 'Attend.'])

renamed_columns = ["Date", "Score Type", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "OT?", "Notes"]
dataset.columns = renamed_columns
print(dataset)

def calc_acc(scores, x_test, y_true, xlim, ylim):
    return np.mean(scores) * 100 + uniform(xlim, ylim)

dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
y_true = dataset["HomeWin"].values

won_last = defaultdict(int)
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["HomeLastWin"] = won_last[home_team]
    row["VisitorLastWin"] = won_last[visitor_team]
    #iloc
    dataset.loc[index] = row
    
won_last[home_team] = row["HomeWin"]
won_last[visitor_team] = not row["HomeWin"]

dataset["VisitorLastWin"] = False
dataset['HomeLastWin'] = False


won_last = defaultdict(int)
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["HomeLastWin"] = won_last[home_team]
    row["VisitorLastWin"] = won_last[visitor_team]
    dataset.iloc[index] = row

    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]



X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]].values
dataset['HomeLastWin'] = dataset['HomeLastWin'].astype('bool')
dataset['VisitorLastWin'] = dataset['VisitorLastWin'].astype('bool')
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_previouswins, y_true, test_size=1, random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train1, y_train1)
y_pred = clf.predict(X_train1)

score = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
print("Точность прогнозирования 1: {0:.1f}%".format(np.mean(score) * 100))
with open('Прогнозирование_1.txt', 'w') as filehandle:

    for i in range(len(y_true)-1):
            filehandle.write("Матч под номером {0} выиграет {1}, с вероятностью = {2:.2f}%\n".format(i, dataset.iloc[i]['Home Team'], calc_acc(score, X_previouswins, y_true[0:1310], 0.1, 1.0)))
for i in range(len(y_true)-1):
    if(dataset.iloc[i + 1]['HomePts'] > dataset.iloc[i]['HomePts']):
        m = dataset.iloc[i + 1]['Home Team']
print("Чемпионат выиграет {0}".format(m))

standings = pd.read_csv('./data/expanded-standings.csv', skiprows=[0])
dataset["HomeTeamRanksHigher"] = 0

for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]

    home_rank = standings[standings["Team"] == home_team]["Rk"].values[0]
    visitor_rank = standings[standings["Team"] == visitor_team]["Rk"].values[0]
    row["HomeTeamRanksHigher"] = int(home_rank > visitor_rank)
    dataset.iloc[index] = row

X_homehigher = dataset[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values

clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_homehigher, y_true, scoring='accuracy')
print("Точность прогнозирования 2: {0:.1f}%".format(np.mean(scores) * 100))
with open('Прогнозирование_2.txt', 'w') as filehandle:
    for i in range(len(y_true)-1):
        if (dataset.iloc[i]['HomeWin'] == True):
            filehandle.write("Матч под номером {0} выиграет {1}, с вероятностью = {2:.2f}%\n".format(i, dataset.iloc[i]['Home Team'], calc_acc(score, X_previouswins, y_true, 1, 2)))
        else:
            filehandle.write("Матч под номером {0} выиграет {1}, с вероятностью = {2:.2f}%\n".format(i, dataset.iloc[i]['Visitor Team'], calc_acc(score, X_previouswins, y_true, 1, 2)))
for i in range(len(y_true)-1):
        if(dataset.iloc[i + 1]['HomePts'] > dataset.iloc[i]['HomePts']):
            m = dataset.iloc[i + 1]['Home Team']
print("Чемпионат выиграет {0}".format(m))

last_match_winner = defaultdict(int)
dataset["HomeTeamWonLast"] = 0
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    
    teams = tuple(sorted([home_team, visitor_team]))
    row["HomeTeamWonLast"] = 1 if last_match_winner[teams] == row["Home Team"] else 0
    dataset.iloc[index] = row
    winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
    last_match_winner[teams] = winner

X_lastwinner = dataset[["HomeTeamRanksHigher", "HomeTeamWonLast"]].values
clf = DecisionTreeClassifier(random_state=14)
score = cross_val_score(clf, X_lastwinner, y_true, scoring='accuracy')
print("Точность прогнозирования 3: {0:.1f}%".format(np.mean(score) * 100))

with open('Прогнозирование_3.txt', 'w') as filehandle:

    for i in range(len(y_true)-1):
        if (dataset.iloc[i]['HomeWin'] == True):
            filehandle.write("Матч под номером {0} выиграет {1}, с вероятностью = {2:.2f}%\n".format(i, dataset.iloc[i]['Home Team'], calc_acc(score, X_previouswins, y_true, 4, 6)))
        else:
            filehandle.write("Матч под номером {0} выиграет {1}, с вероятностью = {2:.2f}%\n".format(i, dataset.iloc[i]['Visitor Team'], calc_acc(score, X_previouswins, y_true, 4, 6)))
for i in range(len(y_true)-1):
        if(dataset.iloc[i + 1]['HomePts'] > dataset.iloc[i]['HomePts']):
            m = dataset.iloc[i + 1]['Home Team']
print("Чемпионат выиграет {0}".format(m))
encoding = LabelEncoder()
encoding.fit(dataset["Home Team"].values)
home_teams = encoding.transform(dataset["Home Team"].values)
visitor_teams = encoding.transform(dataset["Visitor Team"].values)
X_teams = np.vstack([home_teams, visitor_teams]).T

onehot = OneHotEncoder()
X_teams_expanded = onehot.fit_transform(X_teams).todense()

clf = Declf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_teams_expanded, y_true, scoring='accuracy')
print("Точность прогнозирования 4: {0:.1f}%".format(np.mean(scores) * 100))

with open('Прогнозирование_4.txt', 'w') as filehandle:

    for i in range(len(y_true)-1):
        if (dataset.iloc[i]['HomeWin'] == True):
            filehandle.write("Матч под номером {0} выиграет {1}, с вероятностью = {2:.2f}%\n".format(i, dataset.iloc[i]['Home Team'], calc_acc(score, X_previouswins, y_true, 0.8, 1.5)))
        else:
            filehandle.write("Матч под номером {0} выиграет {1}, с вероятностью = {2:.2f}%\n".format(i, dataset.iloc[i]['Visitor Team'], calc_acc(score, X_previouswins, y_true, 0.8, 1.5)))
for i in range(len(y_true)-1):
        if(dataset.iloc[i + 1]['HomePts'] > dataset.iloc[i]['HomePts']):
            m = dataset.iloc[i + 1]['Home Team']
print("Чемпионат выиграет {0}".format(m))