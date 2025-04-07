# Credit Data Analysis Project

## Overview
This project provides a comprehensive analysis of credit card data, examining key relationships between demographics, loan duration, account types, loan purposes, and risk factors. The analysis helps identify patterns in credit behavior and risk profiles.

## Dataset
The dataset contains information about 1,000 credit accounts with the following features:
- Age: Customer's age (19-75 years)
- Sex: Customer's gender (male/female)
- Job: Job category (0-3)
- Housing: Housing status (own/rent/etc.)
- Saving accounts: Saving account status
- Checking account: Checking account status
- Credit amount: Loan amount (250-18,424)
- Duration: Loan duration in months (4-72)
- Purpose: Purpose of the loan (car, education, etc.)

## Analysis Components

### 1. Demographics Analysis
Examines how demographic factors like age, gender, job category, and housing type affect credit amounts. Identifies which demographic groups tend to take higher loans and profiles high-credit customers.

### 2. Duration and Credit Amount Relationship
Analyzes the correlation between loan duration and credit amount. Includes regression analysis and visualization of how credit amounts vary across different duration groups.

### 3. Saving/Checking Accounts Analysis
Investigates how different banking relationships (saving and checking accounts) influence credit behavior. Examines both individual and combined effects of account types on credit amounts.

### 4. Purpose Analysis
Identifies which loan purposes drive higher credit amounts and examines how purposes vary across demographic groups. Includes statistical significance testing using ANOVA.

### 5. Risk Factor Analysis
Combines multiple variables to create risk scores and indicators. Uses machine learning to predict high-credit risk and identifies the most important features contributing to risk assessment.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn

## Installation
```
pip install pandas numpy matplotlib seaborn 
```

## Usage
1. Clone this repository
2. Ensure your credit data CSV file is in the project directory
3. Run the analysis:
```python
# Import the main analysis script
from credit_analysis import *

# Load your data
df = pd.read_csv('your_credit_data.csv')

# Run all analyses
analyze_demographics()
analyze_duration_credit()
analyze_accounts()
analyze_purposes()
analyze_risk_factors()
```

## Code Structure
- `main.py`: Main script containing all analysis functions
- `credit_analysis.py`: Module with analysis functions
- `visualization.py`: Helper functions for creating plots

## Key Findings
- Middle-aged customers and those with higher job categories tend to take larger loans
- Strong positive correlation between loan duration and credit amount
- Customers with both robust saving and checking accounts receive higher credit amounts
- Business, education, and car loans typically show higher average credit amounts
- Risk factors combining high credit amounts, long durations, and limited banking relationships identify potentially problematic loans

## Future Work
- Incorporate payment history and default data for more comprehensive risk analysis
- Develop a predictive model for loan approval decision support
- Segment customers for targeted marketing strategies
- Compare findings with industry benchmarks

## Acknowledgements
This analysis was conducted using Python's data science ecosystem, particularly pandas for data manipulation and scikit-learn for machine learning components.

## Contact
For questions or feedback about this analysis, please contact [Your Name] at [your.email@example.com].
