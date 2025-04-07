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