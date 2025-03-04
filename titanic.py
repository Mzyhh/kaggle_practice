import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# загрузим датасет о пассижирах титаника
df = pd.read_csv("https://raw.githubusercontent.com/Adelaaas/Data_science_basic_22-23_2/main/%D0%92%D0%B5%D1%81%D0%B5%D0%BD%D0%BD%D0%B8%D0%B9%20%D1%81%D0%B5%D0%BC%D0%B5%D1%81%D1%82%D1%80/class_work_2_log_reg/train.csv")

df.drop(['Ticket', 'Cabin', 'PassengerId', 'Name'], axis=1, inplace=True)

sex_encoder = LabelEncoder()
df['Sex'] = sex_encoder.fit_transform(df['Sex'])

df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

mean_class1 = df.groupby("Pclass")['Age'].mean().round().loc[1]
mean_class2 = df.groupby("Pclass")['Age'].mean().round().loc[2]
mean_class3 = df.groupby("Pclass")['Age'].mean().round().loc[3]
df.loc[df["Pclass"]==1, 'Age'] = df.loc[df["Pclass"]==1, 'Age'].fillna(mean_class1)
df.loc[df["Pclass"]==2, 'Age'] = df.loc[df["Pclass"]==2, 'Age'].fillna(mean_class2)
df.loc[df["Pclass"]==3, 'Age'] = df.loc[df["Pclass"]==3, 'Age'].fillna(mean_class3)

X = df.drop('Survived', axis=1)
Y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("f1 score: ", f1_score(y_test, y_pred))