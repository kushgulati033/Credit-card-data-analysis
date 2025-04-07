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

# Create age groups
bins = [18, 25, 35, 50, 75]
labels = ['Young', 'Adult', 'Middle-aged', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Create credit amount categories
bins = [0, 1000, 5000, 10000, 20000]
labels = ['Low', 'Medium', 'High', 'Very High']
df['Credit_Category'] = pd.cut(df['Credit amount'], bins=bins, labels=labels)

plt.figure(figsize=(14, 10))

# Credit amount by sex
plt.subplot(2, 2, 1)
sns.boxplot(x='Sex', y='Credit amount', data=df)
plt.title('Credit Amount by Sex')

# Credit amount by age groups
plt.subplot(2, 2, 2)
bins = [18, 25, 35, 45, 60, 76]
df['Age_Group'] = pd.cut(df['Age'], bins=bins)
sns.boxplot(x='Age_Group', y='Credit amount', data=df)
plt.title('Credit Amount by Age Group')
plt.xticks(rotation=45)

# Credit amount by job
plt.subplot(2, 2, 3)
sns.boxplot(x='Job', y='Credit amount', data=df)
plt.title('Credit Amount by Job Category')

# Credit amount by housing
plt.subplot(2, 2, 4)
sns.boxplot(x='Housing', y='Credit amount', data=df)
plt.title('Credit Amount by Housing Type')

plt.tight_layout()
plt.show()

# Statistical analysis
print("Average Credit Amount by Demographics:")
demo_groups = ['Sex', 'Job', 'Housing']
for group in demo_groups:
    print(f"\nBy {group}:")
    print(df.groupby(group)['Credit amount'].agg(['mean', 'median', 'count']))

# Age correlation
age_corr = df['Age'].corr(df['Credit amount'])
print(f"\nCorrelation between Age and Credit Amount: {age_corr:.3f}")

# Create a profile of high-credit customers
high_credit = df[df['Credit amount'] > df['Credit amount'].quantile(0.75)]
print("\nProfile of High-Credit Customers:")
for col in ['Sex', 'Job', 'Housing', 'Age_Group']:
    print(f"\n{col} distribution in high-credit group:")
    print(high_credit[col].value_counts(normalize=True).round(3) * 100, "%")