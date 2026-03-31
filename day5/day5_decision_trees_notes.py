import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the train data

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

X_train,  X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Decision tree classifier
dt_model= DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Precision:", precision_score(y_test, dt_pred))
print("Decision Tree Recall:", recall_score(y_test, dt_pred))
print("Decision Tree F1 Score:", f1_score(y_test, dt_pred))

dr_model = RandomForestClassifier(max_depth=5, random_state=42)
dr_model.fit(X_train,y_train)
dr_predict = dr_model.predict(X_test)
dr_accuracy = accuracy_score(y_test, dr_predict)
print("Random Forest Accuracy:", dr_accuracy)
print("Random Forest Precision:", precision_score(y_test, dr_predict))
print("Random Forest Recall:", recall_score(y_test, dr_predict))
print("Random Forest F1 Score:", f1_score(y_test, dr_predict))

