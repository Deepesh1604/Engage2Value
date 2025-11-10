# Engage2Value: From Clicks to Conversions

Predict purchase value from multi-session digital behavior using ML.

## Project Overview

This project predicts a customer's purchase value based on their multi-session behavior across digital touchpoints. The dataset captures anonymized user interactions such as browser types, traffic sources, device details, and geographical indicators. By modeling these patterns, we estimate the purchase potential of each user to optimize marketing and engagement strategies.

## Dataset

- **Training Data**: 116,023 samples with 52 features
- **Test Data**: Unlabeled samples for prediction
- **Target Variable**: `purchaseValue` (continuous regression target)
- **Features**: Mix of categorical and numerical features including user behavior, device info, and session data

## Technical Implementation

### 1. Data Preprocessing & Analysis
- **Exploratory Data Analysis**: Comprehensive analysis of feature distributions, correlations, and target variable characteristics
- **Memory Optimization**: Efficient data type conversions to reduce memory usage
- **Missing Value Handling**: Strategic imputation based on feature types
- **Feature Engineering**: Advanced feature creation including interaction terms and transformations

### 2. Machine Learning Pipeline
- **Feature Selection**: Multi-stage approach using statistical tests and Random Forest importance
- **Preprocessing**: RobustScaler for handling outliers and feature normalization
- **Cross-Validation**: 5-fold stratified cross-validation for robust model evaluation

### 3. Two-Stage Ensemble Model
**Stage 1 - Base Models:**
- **LightGBM**: Gradient boosting with optimized hyperparameters
- **XGBoost**: Extreme gradient boosting for robust predictions
- **CatBoost**: Categorical feature handling with advanced boosting

**Stage 2 - Meta-Learning:**
- **Meta-Model**: LightGBM trained on Stage 1 predictions
- **Stacking Strategy**: Combines base model predictions for improved performance

### 4. Model Performance
- **Evaluation Metrics**: R² Score, RMSE, MAE
- **Validation Strategy**: Holdout validation + cross-validation
- **Feature Count**: ~600 engineered features after selection

## Key Features

🔍 **Comprehensive EDA** - Detailed analysis of data patterns and relationships  
⚙️ **Advanced Feature Engineering** - Automated feature creation and selection  
🤖 **Two-Stage Ensemble** - Sophisticated stacking approach for optimal performance  
📊 **Robust Validation** - Multiple validation strategies to ensure generalization  
💾 **Memory Efficient** - Optimized data handling for large datasets  

## Files Structure

```
├── train_data.csv              # Training dataset
├── test_data.csv               # Test dataset for predictions
├── submission.csv              # Final predictions
├── 22f2001290-notebook-*.ipynb # Main analysis notebook
└── catboost_info/              # CatBoost model artifacts
```

## Usage

1. **Data Loading**: Load training and test datasets
2. **Preprocessing**: Run comprehensive feature engineering pipeline
3. **Model Training**: Train two-stage ensemble model
4. **Prediction**: Generate predictions for test set
5. **Submission**: Export results in required format

## Applications

This predictive model enables businesses to:
- **Optimize Marketing Strategies** - Focus resources on high-value potential customers
- **Improve Customer Engagement** - Personalize interactions based on purchase likelihood
- **Budget Allocation** - Allocate marketing spend more effectively
- **Risk Assessment** - Identify and prioritize valuable customer segments

## Technologies Used

- **Python** - Core programming language
- **Pandas/NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning framework
- **LightGBM/XGBoost/CatBoost** - Gradient boosting algorithms
- **Matplotlib/Seaborn** - Data visualization
- **Scipy** - Statistical analysis

---

*This project demonstrates advanced machine learning techniques for customer value prediction using ensemble methods and comprehensive feature engineering.*