Day 2: Built My First ML Model

I created my first machine learning model using the Titanic dataset. 

First, I cleaned the data:
- Removed unwanted columns (Cabin, Name, Ticket, PassengerId)
- Filled missing Age values with the median
- Mapped categorical variables (Sex, Embarked) to integers

Then I prepared the data for modeling:
- Separated features (X) from the target (Survived in y)
- Split data into 80% training and 20% testing using train_test_split

I trained a Logistic Regression model on the training data and made predictions on the test set.

Finally, I evaluated the model using metrics:
- Accuracy: 78.65% (correct predictions)
- Precision: 70.67% (of predicted survivors, how many actually survived)
- Recall: 76.81% (of actual survivors, how many I caught)
- F1 Score: 0.736 (balanced metric)

Key Learning: The model is decent but not perfect. I want to improve it by trying different algorithms and feature engineering.