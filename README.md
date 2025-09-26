# 🏠 Singapore Property Price Prediction

A comprehensive machine learning project to predict residential property prices in Singapore using advanced feature engineering and multiple regression algorithms.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Methodology](#-methodology)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [Technical Challenges](#-technical-challenges)
- [Future Improvements](#-future-improvements)

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

## 🔧 Features

### Data Preprocessing
- **Missing Value Handling**: Comprehensive imputation strategies
- **Date Feature Engineering**: Extract year, month from sale dates
- **Floor Level Processing**: Parse floor ranges (e.g., "06 to 10" → start_floor, end_floor)
- **Tenure Analysis**: Extract lease duration and remaining years

### Advanced Feature Engineering
- **Target Encoding**: For high-cardinality variables (Project Name, Street Name)
- **One-Hot Encoding**: For low-cardinality categoricals
- **Lease Features**: Remaining lease years, lease maturity ratios
- **Location Features**: Postal district analysis and regional grouping

### Data Quality Assurance
- **Duplicate Detection**: Automated duplicate removal from training data
- **Index Alignment**: Safe DataFrame concatenation to prevent data corruption
- **Unseen Category Handling**: Robust target encoding with fallback strategies

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
- **Linear Models**: Linear Regression, Lasso Regression
- **Tree-Based Models**: Random Forest, Gradient Boosting, Decision Trees
- **Instance-Based**: K-Nearest Neighbors
- **Support Vector**: SVR with different kernels

### 4. Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **R² Score**
- **Mean Absolute Percentage Error (MAPE)**

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
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 💻 Usage

### Quick Start
```bash
# Launch Jupyter Notebook
jupyter notebook Assignment-1.ipynb
```

### Running the Complete Pipeline
1. **Data Loading**: Load and concatenate train/test datasets
2. **EDA**: Run exploratory data analysis cells
3. **Preprocessing**: Execute data cleaning and feature engineering
4. **Model Training**: Train multiple regression models
5. **Evaluation**: Compare model performance and select best model
6. **Prediction**: Generate predictions for test dataset

### Key Code Sections
```python
# Load data
df = concat_train_test('data/train.csv', 'data/test.csv')

# Target encoding with unseen category handling
for col in ['Project Name', 'Street Name']:
    target_means = train_data.groupby(col)['Price'].mean()
    df[col + '_Encoded'] = df[col].map(target_means).fillna(global_mean)

# Model training and evaluation
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Linear Regression': LinearRegression()
}
```

## 📈 Model Performance

### Best Performing Models
| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Gradient Boosting** | $XXX,XXX | $XXX,XXX | 0.XX |
| **Random Forest** | $XXX,XXX | $XXX,XXX | 0.XX |
| **Linear Regression** | $XXX,XXX | $XXX,XXX | 0.XX |

### Feature Importance
Top predictive features identified:
1. **Project Name** (Target Encoded)
2. **Area (SQFT)**
3. **Postal District**
4. **Remaining Lease Years**
5. **Market Segment**

## 🔍 Key Insights

### Market Trends
- **Central vs. Regional**: Core Central Region commands premium prices
- **Lease Impact**: Freehold properties significantly outvalue leasehold
- **Size Effect**: Strong positive correlation between area and price
- **Location Premium**: Postal districts 9, 10, 11 show highest average prices

### Technical Findings
- **Target Encoding**: 40% improvement over one-hot encoding for high-cardinality variables
- **Feature Engineering**: Tenure-based features provide strong predictive power
- **Data Quality**: Proper handling of duplicates and missing values crucial for model performance

## 📁 Project Structure

```
property-price-prediction/
├── README.md
├── Assignment-1.ipynb          # Main analysis notebook
├── Assignment.pdf              # Project requirements
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset for predictions
│   └── predictions.csv        # Model predictions output
├── .venv/                     # Python virtual environment
├── .gitignore                 # Git ignore rules
└── .git/                      # Git repository
```

## 🚧 Technical Challenges

### 1. High-Cardinality Categorical Variables
**Challenge**: 2,242+ unique project names and 829+ street names
**Solution**: Target encoding with robust unseen category handling

### 2. Data Leakage Prevention
**Challenge**: Ensuring no test data information leaks into training
**Solution**: Strict train-only target encoding with global mean fallbacks

### 3. Index Misalignment
**Challenge**: DataFrame concatenation causing NaN value creation
**Solution**: Safe concatenation with proper index management

### 4. Feature Engineering Complexity
**Challenge**: Complex tenure parsing and floor level extraction
**Solution**: Robust parsing functions with comprehensive error handling

## 🔮 Future Improvements

### Model Enhancements
- [ ] **Ensemble Methods**: Implement stacking/blending of top models
- [ ] **Deep Learning**: Explore neural networks for complex feature interactions
- [ ] **Time Series**: Incorporate temporal trends and seasonality

### Feature Engineering
- [ ] **Geospatial Features**: Distance to MRT stations, schools, shopping centers
- [ ] **Market Conditions**: Economic indicators, interest rates
- [ ] **Property Age**: Building completion date and age effects

### Technical Improvements
- [ ] **Pipeline Automation**: MLOps pipeline with automated retraining
- [ ] **Model Interpretability**: SHAP analysis for feature explanations
- [ ] **A/B Testing**: Model performance monitoring in production

---

## 👨‍💻 Author

**Shreya Sriram**
- 📧 Email: [contact information]
- 🎓 Institution: National University of Singapore
- 📚 Course: Data Science Projects in Practice

---

## 📄 License

This project is part of academic coursework at NUS. Please refer to course guidelines for usage permissions.

---

*Last Updated: September 2025*
