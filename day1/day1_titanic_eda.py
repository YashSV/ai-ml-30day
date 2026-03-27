import pandas as pd
train = pd.read_csv('..\\data\\train.csv')

print("how many rows and columns in the train dataset?")
print(train.shape)
print(train.info())


print('Missing values:')
print(train.isnull().sum())

print('\nSurvival rate:', train['Survived'].mean())

print('\nAge distribution:')
print(train['Age'].describe())

print('\nFare distribution:')
print(train['Fare'].describe())
