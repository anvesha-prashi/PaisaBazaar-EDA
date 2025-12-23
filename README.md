# PaisaBazaar-EDA
[Streamlit App Link](https://paisabazaar-capstone.streamlit.app/)
## 📝 <u> **Summary** </u>
This EDA is for PaisaBazaar company to help them enhance their credit assessment process and reduce risks of potential fraud and default. 

By identifying deviations from the typical behaviour, we can eliminate the anomalies and have the dataset cleaned for better predictions and to improve future fraud detection models.

## 📖 Project Overview
This project focuses on analyzing customer financial data to understand credit risk patterns 
and explore factors influencing credit scores. The goal is to identify potential indicators 
of high-risk customers using exploratory data analysis (EDA) and statistical insights.

## 📊 Dataset Description
The dataset contains anonymized customer-level financial and credit-related information, 
including demographic details, credit history, loan behavior, payment patterns, and credit scores.

Key characteristics:
- Multiple records per customer with unique customer ID
- Mix of numerical and categorical features
- No abnormal values in numerical columns from important computations like mean, min, max, std etc
- Minimal Feature Engineering required as dataset already contains rich domain-specific features
- Target variable: Credit Score

## 🎯 Objective
The primary objectives of this project are:
- To aggregate customer-level data from unique customer IDs
- To explore relationships between financial variables and credit scores
- To identify meaningful patterns and risk indicators
- To prepare the data for potential predictive modeling

## 📈 Exploratory Data Analysis (EDA)
EDA was conducted to understand the distribution and behavior of key numerical and categorical variables.

- Univariate analysis using box plots, KDE plots, and count plots
- Bivariate analysis across credit score categories
- Multivariate analysis using correlation heatmaps and pair plots

Extreme values were retained, as they often represent high-risk customers rather than data errors.

## 🔍 Key Insights
Poor credit score customers consistently shift toward:
  - Higher outstanding debt and interest rates.
  - Higher number of payment delays as well as number of delayed days from payment
  - Lower credit history
  - More credit inquiries.

Strong Categorical indicators include Credit-Mix and Minimum Amount Payments.
  - Customers who have bad credit-mix almost certainly have poor credit scores.
  - Customers who have only pay minimum amount also tend to have low credit scores.
    
Several variables exhibit overlapping distributions, highlighting the complexity of credit behavior

