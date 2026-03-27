import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../data/train.csv')

# Plot 1: Survival by Sex (bar chart)
fig, ax = plt.subplots()
train.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=ax)
ax.set_title('Survival Rate by Gender')
ax.set_ylabel('Survival Rate')
plt.savefig('../day1/plots/survival_by_sex.png')
plt.close()

# Plot 2: Age distribution (histogram)
fig, ax = plt.subplots()
train['Age'].hist(bins=30, ax=ax)
ax.set_title('Age Distribution')
ax.set_xlabel('Age')
plt.savefig('../day1/plots/age_distribution.png')
plt.close()

# Plot 3: Fare by Class (boxplot)
fig, ax = plt.subplots()
sns.boxplot(x='Pclass', y='Fare', data=train, ax=ax)
ax.set_title('Fare Distribution by Passenger Class')
plt.savefig('../day1/plots/fare_by_class.png')
plt.close()

# Plot 4: Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(train.corr(numeric_only=True), annot=True, ax=ax)
ax.set_title('Correlation Heatmap')
plt.savefig('../day1/plots/correlation_heatmap.png')
plt.close()

print("All plots saved.")