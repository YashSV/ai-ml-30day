import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Pulling the CSV data into the dataframe. 
train = pd.read_csv('..\\data\\train.csv')

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

# Removing Survived from X and keeping it in y.
X= train.drop(columns=["Survived"])
y = train["Survived"]

# Calling the train_test-split function to split data into training and testing sets. We kept 80/20.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Printing train and test shapes. 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# I am using Logisticregression model to train the data. I gave 1000 iterations as I have close to 890 records.
model =LogisticRegression(max_iter=1000)
# model fit with X and Y data.
model.fit(X_train,y_train)

# magic is happening here, predicting the X_test data and storing it in y_pred field.
y_pred = model.predict(X_test)

print("Predictions:", y_pred[:10])

# from scikit.learn used the accuracy score of the model. we use y-test and y_pred fields for this.
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

Recall = recall_score(y_test, y_pred)
print("Recall:", Recall)    

f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)