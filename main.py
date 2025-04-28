# Importing libraries
import pandas as pd
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
plt.show()

plt.figure(figsize=(12, 6))

# Scatter plot with regression line
plt.subplot(1, 2, 1)
sns.regplot(x='Duration', y='Credit amount', data=df)
plt.title('Credit Amount vs Duration')

# Calculate correlation
duration_corr = df['Duration'].corr(df['Credit amount'])
plt.xlabel(f'Duration (months) - Correlation: {duration_corr:.3f}')

# Grouped bar chart of average credit by duration
plt.subplot(1, 2, 2)
bins = [0, 12, 24, 36, 48, 72]
df['Duration_Group'] = pd.cut(df['Duration'], bins=bins)
avg_by_duration = df.groupby('Duration_Group')['Credit amount'].mean().reset_index()
sns.barplot(x='Duration_Group', y='Credit amount', data=avg_by_duration)
plt.title('Average Credit Amount by Duration Group')
plt.xticks(rotation=45)

plt.show()

# Joint analysis of saving and checking
print("\nJoint Analysis of Saving and Checking Accounts:")
pivot = pd.pivot_table(
    df,
    values='Credit amount',
    index='Saving accounts',
    columns='Checking account',
    aggfunc='mean'
)
print(pivot)

# Visualize the joint effect
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.0f')
plt.title('Average Credit Amount by Account Types')
plt.show()

plt.figure(figsize=(12, 6))

# Sort purposes by average credit amount
purpose_avg = df.groupby('Purpose')['Credit amount'].mean().sort_values(ascending=False)

# Bar chart of credit amount by purpose
plt.subplot(1, 2, 1)
sns.barplot(x=purpose_avg.values, y=purpose_avg.index)
plt.title('Average Credit Amount by Purpose')
plt.xlabel('Average Credit Amount')

# Box plot for distribution
plt.subplot(1, 2, 2)
purpose_order = purpose_avg.index.tolist()
sns.boxplot(y='Purpose', x='Credit amount', data=df, order=purpose_order)
plt.title('Credit Amount Distribution by Purpose')

plt.show()

# Statistical analysis
print("Credit Amount Statistics by Purpose:")
print(df.groupby('Purpose')['Credit amount'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False))

# Purpose by demographic analysis
print("\nTop Purpose by Sex:")
print(pd.crosstab(df['Sex'], df['Purpose'], normalize='index').round(3) * 100, "%")

# Create risk indicators
df['High_Credit'] = df['Credit amount'] > df['Credit amount'].quantile(0.75)
df['Long_Duration'] = df['Duration'] > df['Duration'].quantile(0.75)
df['Risk_Score'] = 0

# Risk factors:
# 1. High credit amount + long duration
df.loc[(df['High_Credit'] & df['Long_Duration']), 'Risk_Score'] += 2

# 2. No checking account with high credit
df.loc[(df['Checking account'].isnull() & df['High_Credit']), 'Risk_Score'] += 1

# 3. Young age with high credit
df.loc[(df['Age'] < 25 & df['High_Credit']), 'Risk_Score'] += 1

# 4. Unemployed (Job=0) with credit
df.loc[(df['Job'] == 0), 'Risk_Score'] += 1

plt.figure(figsize=(14, 10))

# Risk score distribution
plt.subplot(2, 2, 1)
sns.countplot(x='Risk_Score', data=df)
plt.title('Risk Score Distribution')

# Risk score by purpose
plt.subplot(2, 2, 2)
sns.boxplot(x='Purpose', y='Risk_Score', data=df)
plt.xticks(rotation=90)
plt.title('Risk Score by Purpose')

# Credit-to-age ratio (another risk indicator)
df['Credit_Age_Ratio'] = df['Credit amount'] / df['Age']
plt.subplot(2, 2, 3)
sns.histplot(df['Credit_Age_Ratio'], kde=True)
plt.title('Credit Amount to Age Ratio')

# Credit-duration ratio
df['Credit_Duration_Ratio'] = df['Credit amount'] / df['Duration']
plt.subplot(2, 2, 4)
sns.histplot(df['Credit_Duration_Ratio'], kde=True)
plt.title('Credit Amount to Duration Ratio')

plt.show()

