import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#load the train data
train = pd.read_csv('..\\data\\train.csv')

# clean the data
# Fill the Age value with median age value if its NULL or Not present. 
train["Age"] = train["Age"].fillna(train["Age"].median())
# Remove or delete the Embarked value if its not available.
train = train.dropna(subset=["Embarked"])
# drop unwanted columns.
train.drop(columns=["Cabin","Name","Ticket","PassengerId"], inplace=True)
# map male to 0 and female to 1. Just converting it to INT values.
train["Sex"] = train["Sex"].map({"male":0,"female":1})
# Emabarked is changed to 0,1,2 based on its value.
train["Embarked"] = train["Embarked"].map({"S":0,"C":1,"Q":2})

X = train.drop(columns=["Survived"])
y = train["Survived"]

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)


for depth in [3, 4, 5, 6, 7, 8]:
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"max_depth={depth}, Accuracy: {accuracy}")