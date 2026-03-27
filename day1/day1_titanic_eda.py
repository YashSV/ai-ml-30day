import pandas as pd
train = pd.read_csv('..\\data\\train.csv')

print("How many rows and columns in the train dataset?")
print(train.shape)
print(train.info())

print('Fare:')
print(train['Fare'].describe())

print('Missing values:')
print(train.isnull().sum())

print('\nSurvival rate:', train['Survived'].mean())

print('\nAge:')
print(train['Age'].describe())



# Survival by gender
print('\n survival gender', train.groupby('Sex')['Survived'].mean())
# Survival by class
print('\n survival class', train.groupby('Pclass')['Survived'].mean())
# Age distribution by survival status
print('\n Age distribution by survival status')
print(train.groupby('Survived')['Age'].describe())
# Fare distribution by class
print('\n Fare distribution by class')
print(train.groupby('Pclass')['Fare'].describe())
# Correlation between variables
print('\n Correlation between variables')
# print(train.corr())