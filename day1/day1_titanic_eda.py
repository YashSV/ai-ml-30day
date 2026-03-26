import pandas as pd
train = pd.read_csv('..\\data\\train.csv')
print('Missing values:')
print(train.isnull().sum())
print('\nSurvival rate:', train['Survived'].mean())
print('\nAge distribution:')
print(train['Age'].describe())
print('\nFare distribution:')
print(train['Fare'].describe())
