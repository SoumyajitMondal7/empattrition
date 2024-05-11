import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def handle_missing_values(data, method='drop'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    - data (DataFrame): Input dataset.
    - method (str): Method for handling missing values. Options: 'drop', 'impute', etc.
    
    Returns:
    - data (DataFrame): Dataset after handling missing values.
    """
    if method == 'drop':
        return data.dropna()
    elif method == 'impute':
        # Example: impute missing values using mean, median, etc.
        return data.fillna(data.mean())
    else:
        raise ValueError(f"Invalid method for handling missing values: {method}")

def preprocess_data(data):
    """Preprocess the dataset."""
    # Encode target variable 'Attrition' to numeric values
    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

    # Separate features and target variable
    X = data.drop(columns=['Attrition'])
    y = data['Attrition']

    # Handle missing values
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    preprocessed_data = preprocessor.fit_transform(X)

    return preprocessed_data, y
