import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_df = pd.read_csv(url)


print(titanic_df.head())

print(f"Dimensions of the dataset: {titanic_df.shape}")


print(titanic_df.describe())


print(titanic_df.isnull().sum())

titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)


plt.figure(figsize=(8, 6))
sns.histplot(titanic_df['Age'], bins=20, kde=True, color='blue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', hue='Survived', data=titanic_df)
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

