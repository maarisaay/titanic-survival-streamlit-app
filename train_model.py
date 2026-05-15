import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression

train = pd.read_csv('data/titanic.csv')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

df = train[features + ['Survived']].copy()

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

with open('models/titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Model zapisany")