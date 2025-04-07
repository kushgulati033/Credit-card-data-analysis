import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = pd.read_csv('german_credit_data.csv')

# data description
print(df.head())
print(df.describe())
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Handle missing values
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

# Visualize distributions
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='Age', hue='Sex', multiple='stack')
plt.title('Age Distribution by Sex')
plt.show()

# Credit amount distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Credit amount', kde=True)
plt.title('Credit Amount Distribution')
plt.show()

# Correlation between numerical variables
corr = df[['Age', 'Job', 'Credit amount', 'Duration']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Credit amount vs Duration with Purpose
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Credit amount', y='Duration', hue='Purpose')
plt.title('Credit Amount vs Duration by Purpose')
plt.show()

# Check average credit amount by sex and job
avg_credit = df.groupby(['Sex', 'Job'])['Credit amount'].mean().reset_index()
print(avg_credit)