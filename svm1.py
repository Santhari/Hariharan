import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('MAIN RAEDINGS.csv')
print(df)
df = df.drop(["ID"],axis=1)
df.rename(columns={'acc X':'acc Y'}, inplace=True)
df['fall or not'].value_counts()
df["fall or not"]=df["fall or not"].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
df["Heart rate"]=df["Heart rate"].map({0:3,1:1,2:2,3:3})
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True, cmap='rainbow',linewidth=0.5, fmt='.2f')
df['fall or not'].value_counts()
plt.title('Postural Imbalnce or not - data imbalance check')
ax1 = sns.countplot(x= 'fall or not', data = df)
ax1.set_xticklabels(['No imbalance','Imbalance detected'])
plt.figure(figsize=(5,5))
plt.show()
plt.title('Heart rate')
ax2 = sns.countplot(x= 'Heart rate', hue = 'fall or not', data = df)
ax2.set_xticklabels(['Heart rate vs fall'],rotation = 90)
plt.show()
plt.title('Heart rate Distribution \n Default(Cyan) vs. No Default(Blue)')
agedist0 = df[df['fall or not']==0]['Heart rate']
agedist1 = df[df['fall or not']==1]['Heart rate']
sns.distplot(agedist0, bins = 100, color = 'blue')
sns.distplot(agedist1, bins = 100, color = 'cyan')
plt.show()
plt.title('Temperature Distribution \n Default(Mediumorchid) vs. No Default(Hotpink)')
cadist0 = df[df['fall or not']==0]['Temp']
cadist1 = df[df['fall or not']==1]['Temp']
sns.distplot(cadist0, bins = 100, color = 'hotpink')
sns.distplot(cadist1, bins = 100, color = 'mediumorchid')
plt.xlabel('Temperature')
plt.show()
q1=df.quantile(.25)
q3=df.quantile(.75)
iqr=q3-q1
print(iqr)
df = df[~((df < (q1 - 1.5 * iqr)) |(df > (q3 + 1.5 * iqr))).any(axis=1)]
print(df.shape)

df.hist(figsize = (15,15), ec = 'black')


from sklearn.datasets import make_classification
x=df.drop('fall or not',axis=1).values
y=df['fall or not'].values
x, y = make_classification(n_samples = 1000, n_features = 50, n_informative = 10, n_redundant = 40)
x = pd.DataFrame(x)
y = pd.Series(y)

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,classification_report
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
scalar = StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
svc = SVC(kernel='sigmoid')
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
svm1_score=accuracy_score(y_test, y_pred)
print(svm1_score)
model2 = SVC(kernel='sigmoid')
model2.fit(x_train, y_train)
predictions1 = model2.predict(x_test)
print(classification_report(y_test, y_pred))



