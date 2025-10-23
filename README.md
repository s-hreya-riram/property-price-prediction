# 🏠 Singapore Property Price Prediction

A comprehensive machine learning project to predict residential property prices in Singapore using advanced feature engineering and multiple regression algorithms.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation & Setup](#-installation--setup)
- [Project Structure](#-project-structure)

## 🎯 Project Overview

This project develops a machine learning pipeline to predict Singapore residential property transaction prices using historical sales data. The project incorporates comprehensive preprocessing of property-specific attributes, feature engineering techniques and training multiple models such as linear regression (baseline), Random Forests, Gradient Boosting, Decision Trees and KNN against the training data to determine the most suitable model for predictions on an unknown dataset.

### Objectives
- Build accurate predictive models for Singapore property prices
- Implement robust data preprocessing and feature engineering
- Handle high-cardinality categorical variables effectively
- Analyze key factors influencing property valuations

## 📊 Dataset

The dataset contains **7,500+ property transactions** with comprehensive property details:

### Data Sources
- **Training Data**: `data/train.csv` (6,000+ transactions with prices)
- **Test Data**: `data/test.csv` (1,500+ transactions for prediction)

### Key Attributes
| Feature | Description | Type |
|---------|-------------|------|
| **Project Name** | Property development name | Categorical (2,242 unique) |
| **Transacted Price ($)** | Sale price in SGD | Target Variable |
| **Area (SQFT)** | Property size in square feet | Numerical |
| **Sale Date** | Transaction date | Date |
| **Street Name** | Property location | Categorical (829 unique) |
| **Property Type** | Apartment/Condominium | Categorical |
| **Tenure** | Leasehold/Freehold details | Categorical |
| **Postal District** | Singapore postal district (1-28) | Categorical |
| **Market Segment** | Core/Rest/Outside Central Region | Categorical |
| **Floor Level** | Property floor range | Categorical |

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of target variable (log-transformation for skewness)
- Correlation analysis between features and prices
- Categorical variable cardinality assessment
- Missing value pattern analysis

### 2. Feature Engineering Pipeline
```python
# Target Encoding for High-Cardinality Variables
train_target_means = train_data.groupby('Project Name')['Price'].mean()
df['Project_Name_Encoded'] = df['Project Name'].map(train_target_means)
# Handle unseen categories with global mean fallback
```

### 3. Model Selection & Training
- **Linear Models**: Linear Regression
- **Tree-Based Models**: Random Forest, Gradient Boosting, Decision Trees
- **Instance-Based**: K-Nearest Neighbors

### 4. Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **R² Score**

### 5. Feature Importance Analysis using SHAP
Identified the most important features in the model

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab

### Dependencies
```bash
# Clone the repository
git clone https://github.com/s-hreya-riram/property-price-prediction.git
cd property-price-prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt
```

## 📁 Project Structure

```
property-price-prediction/
├── README.md
├── Assignment-1.ipynb         # Main analysis notebook
├── Assignment-1.html          # HTML equivalent (export) of the analysis notebook
├── Assignment.pdf             # Project description
├── Analysis.pdf               # PDF with the written answers to the questions of interest
├── Analysis.TeX               # TeX file used to generate the PDF
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset for predictions
│   └── final_predictions.csv  # Model predictions output with just the predictions
│   └── final_predictions_with_other_features.csv        # Model predictions output with the predictions alongside the other test features
├── requirements.txt           # Exhaustive list of project requirements
├── .gitignore                 # Git ignore rules
```
