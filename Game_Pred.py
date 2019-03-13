import pandas as pd
import numpy as np
import datetime
import time
import random

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
### test

### Train
E_2018 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/E0_2018.csv')
E_2017 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/E0_2017.csv')
E_2016 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/E0_2016.csv')
E_2015 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/E0_2015.csv')
E_2014 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/E0_2014.csv')

D_2018 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/D_2018.csv')
D_2017 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/D_2017.csv')
D_2016 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/D_2016.csv')
D_2015 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/D_2015.csv')
D_2014 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/D_2014.csv')

I_2018 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/I_2018.csv')
I_2017 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/I_2017.csv')
I_2016 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/I_2016.csv')
I_2015 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/I_2015.csv')
I_2014 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/I_2014.csv')

F_2018 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/F_2018.csv')
F_2017 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/F_2017.csv')
F_2016 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/F_2016.csv')
F_2015 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/F_2015.csv')
F_2014 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/F_2014.csv')

SP_2018 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/SP_2018.csv')
SP_2017 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/SP_2017.csv')
SP_2016 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/SP_2016.csv')
SP_2015 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/SP_2015.csv')
SP_2014 = pd.read_csv('C:/Users/ValentinLEFRANC/Documents/Football_bet/SP_2014.csv')

col = list(E_2018.columns)[:26]
col.remove('Referee')
#B = list(E_2014.columns)[:26]
#list(set(A) - set(B))

E_2018 = E_2018[col]
E_2017 = E_2017[col]
E_2016 = E_2016[col]
E_2015 = E_2015[col]
E_2014 = E_2014[col]

I_2018 = I_2018[col]
I_2017 = I_2017[col]
I_2016 = I_2016[col]
I_2015 = I_2015[col]
I_2014 = I_2014[col]

D_2018 = D_2018[col]
D_2017 = D_2017[col]
D_2016 = D_2016[col]
D_2015 = D_2015[col]
D_2014 = D_2014[col]

F_2018 = F_2018[col]
F_2017 = F_2017[col]
F_2016 = F_2016[col]
F_2015 = F_2015[col]
F_2014 = F_2014[col]

SP_2018 = SP_2018[col]
SP_2017 = SP_2017[col]
SP_2016 = SP_2016[col]
SP_2015 = SP_2015[col]
SP_2014 = SP_2014[col]

E_5Y = E_2017.append(E_2016)
E_5Y = E_5Y.append(E_2015)
E_5Y = E_5Y.append(E_2014)
E_5Y = E_5Y.append(D_2017)
E_5Y = E_5Y.append(D_2016)
E_5Y = E_5Y.append(D_2015)
E_5Y = E_5Y.append(D_2014)
E_5Y = E_5Y.append(F_2017)
E_5Y = E_5Y.append(F_2016)
E_5Y = E_5Y.append(F_2015)
E_5Y = E_5Y.append(F_2014)
E_5Y = E_5Y.append(I_2017)
E_5Y = E_5Y.append(I_2016)
E_5Y = E_5Y.append(I_2015)
E_5Y = E_5Y.append(I_2014)
E_5Y = E_5Y.append(SP_2017)
E_5Y = E_5Y.append(SP_2016)
E_5Y = E_5Y.append(SP_2015)
E_5Y = E_5Y.append(SP_2014)
E_5Y = E_5Y.append(F_2018)
E_5Y = E_5Y.append(D_2018)
E_5Y = E_5Y.append(I_2018)
E_5Y = E_5Y.append(SP_2017)
E_5Y['FTR_int'] = pd.factorize(E_5Y['FTR'])[0]
len(E_5Y)
E_5Y = E_5Y.dropna()


A=list(E_5Y['B365A'])
H=list(E_5Y['B365H'])
D=list(E_5Y['B365D'])
FTR=list(E_5Y['FTR'])
FTR_bet = []
spread = []
res = 0

for i in range(len(E_5Y)):
        quotes = [A[i],H[i],D[i]]
        fav = np.std(quotes)
        spread.append(fav)
E_5Y['spread'] = spread

Data_test = E_5Y.sample(frac=0.10, random_state=99)

Date = list(E_5Y.Date)
Date = [datetime.datetime.strptime(e, '%d/%m/%y') for e in Date]
E_5Y.Date = Date
E_5Y = E_5Y.sort_values(by=['Date'], ascending=False)

Date = list(Data_test.Date)
Date = [datetime.datetime.strptime(e, '%d/%m/%y') for e in Date]
Data_test.Date = Date
Data_test = Data_test.sort_values(by=['Date'], ascending=False)

E_5Y.columns
################################################################
#Goal Stats
E_5Y['All_G'] = E_5Y['FTHG']+E_5Y['FTAG']
E_5Y['All_G'].describe()
len(E_5Y)
len(E_5Y[E_5Y['All_G'] >= 3])
len(E_5Y[E_5Y['All_G'] <= 1])
len(E_5Y[E_5Y['All_G'] == 2])

len(E_5Y[E_5Y['All_G'] < 2.5])/len(E_5Y[E_5Y['All_G'] > 2.5])

## Methode particuliere le ratio des bt change de 15%
T1 = E_5Y[E_5Y['B365A'] > 2.5 ]
T1 = T1[T1['B365A'] < T1['B365H']]
len(T1)
T1 = T1[T1['FTR']== 'H']
len(T1)
len(T1[T1['All_G'] < 2.5])/len(T1[T1['All_G'] > 2.5])

sns.distplot(E_5Y['B365A'], hist=True, kde=False, bins=100, color = 'blue',hist_kws={'edgecolor':'black'})
sns.distplot(E_5Y['B365H'], hist=True, kde=False, bins=100, color = 'blue',hist_kws={'edgecolor':'black'})
sns.distplot(E_5Y['B365D'], hist=True, kde=False, bins=100, color = 'blue',hist_kws={'edgecolor':'black'})

for i in range(len(E_5Y)):
        quotes = [A[i],H[i],D[i]]
        fav = np.max(quotes)
        if fav == A[i]:
            FTR_bet.append('A')
        if fav == H[i]:
            FTR_bet.append('H')
        if fav == D[i]:
            FTR_bet.append('D')
        if FTR_bet == FTR:
            res = res + fav
        else:
            res = res - fav
####################################################################################################



Div = pd.factorize(E_5Y['Div'])[0]
E_5Y.Div = Div

Div = pd.factorize(Data_test['Div'])[0]
Data_test.Div = Div

features = ['B365H','B365D','B365A']
y_train = list(E_5Y['FTR_int'])


########## Random forest  ############################################
clf = RandomForestClassifier(n_estimators = 70, random_state=0)
clf.fit(E_5Y[features], y_train)
scores = cross_val_score(clf, E_5Y[features], y_train, cv=5)
scores.mean()

#clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, E_5Y[features], y_train, cv=5)
#scores.mean()

#clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, E_5Y[features], y_train, cv=5)
#scores.mean()

y_proba = clf.predict_proba(Data_test[features])
y_predict = clf.predict(Data_test[features])
y_test = pd.factorize(Data_test['FTR_int'])[0]


y_proba_max = np.repeat(0.00,len(y_test))
for i in range(len(y_test)):
    y_proba_max[i] = np.max(y_proba[i])


########################################################################


############# neural_network ###########################################
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#X_train = E_5Y[features].as_matrix()
#min_max_scaler = preprocessing.MinMaxScaler()
#X_train_minmax = min_max_scaler.fit_transform(X_train)

#clf.fit(X_train_minmax, y_train)
#X_test = Data_test[features].as_matrix()
#y_proba = clf.predict_proba(X_test)
#y_predict = clf.predict(X_test)
#y_test = pd.factorize(Data_test['FTR'])[0]
############################################################################################

########## linear_model #############################################
#E_5Y[E_5Y['FTR'] == "A"]
#clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
#clf.fit(E_5Y[features], y_train)

#y_proba = clf.predict_proba(Data_test[features])
#y_predict = clf.predict(Data_test[features])
#y_test = pd.factorize(Data_test['FTR'])[0]
#####################################################################

pd.crosstab(y_test, y_predict, rownames=['True result'], colnames=['Predicted result'])
Data_test['y_predict'] = y_predict
Data_test['y_test'] = y_test
Data_test['y_proba_max'] = y_proba_max
Data_test_red = Data_test[['FTR','B365H','B365D','B365A','y_predict','y_test','y_proba_max']]
len(Data_test_red)
Data_test_red['res'] = 0
Data_test_red['res'][Data_test_red['y_predict'] ==  Data_test_red['y_test']] = 1
Data_test_red['y_test_cor'] = y_test
#Data_test_red['y_test_cor'][Data_test_red['B365H'] < 1.75] = 0
#Data_test_red


G_prob = np.repeat(0.00,10)

for j in range(10):
    seuil = 0.1*j
    G = np.repeat(0.00,len(Data_test))
    for i in range(len(Data_test)):
        if y_proba_max[i] > seuil:
            if list(Data_test_red[Data_test_red.columns[1]])[i] > 2 :
                if y_predict[i] == y_test[i]:
                    G[i] = list(Data_test_red[Data_test_red.columns[1+y_predict[i]]])[i]*list(Data_test_red['y_proba_max'])[i]
                else:
                    G[i] = -list(Data_test_red[Data_test_red.columns[1+y_test[i]]])[i]*list(Data_test_red['y_proba_max'])[i]
    G_prob[j] = np.sum(G)

G_prob


Data_test_red[Data_test_red['y_predict'] == 1]

lost = Data_test_red[Data_test_red['res'] ==  0]
win = Data_test_red[Data_test_red['res'] ==  1]

plt.hist(lost['B365H'], color = 'blue', edgecolor = 'black',bins = 200)
plt.hist(lost['B365D'], color = 'blue', edgecolor = 'black',bins = 200)
plt.hist(lost['B365A'], color = 'blue', edgecolor = 'black',bins = 200)

plt.hist(win['B365H'], color = 'blue', edgecolor = 'black',bins = 200)
plt.hist(win['B365D'], color = 'blue', edgecolor = 'black',bins = 200)
plt.hist(win['B365A'], color = 'blue', edgecolor = 'black',bins = 200)
