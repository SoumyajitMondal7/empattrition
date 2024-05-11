import logging
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def load_data(file_path):
    """Load the dataset from the specified file path."""
    if not os.path.exists(file_path):
        logging.error(f"File not found at path: {file_path}")
        raise FileNotFoundError(f"File not found at path: {file_path}")

    logging.info("Loading dataset...")
    data = pd.read_csv(file_path)
    logging.info("Dataset loaded successfully.")
    return data

def preprocess_data(data):
    """Preprocess the dataset."""
    logging.info("Preprocessing data...")

    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    preprocessed_data = preprocessor.fit_transform(data)

    logging.info("Data preprocessing completed.")
    return preprocessed_data

def explore_data(data):
    """Explore the dataset."""
    logging.info("Exploring data...")

    # Drop non-numeric columns for correlation analysis
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    
    # Correlation analysis
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()  # Explicitly display the figure

    # Distribution plots
    for col in numeric_data.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_data[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()  # Explicitly display the figure

    # Feature importance
    X = data.drop(columns=['Attrition'])
    y = data['Attrition']
    model = RandomForestClassifier()
    model.fit(X, y)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()  # Explicitly display the figure

    logging.info("Data exploration completed.")

def main(file_path):
    """Main function for data analysis and preprocessing."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting application...")
    logging.info("Performing data analysis and preprocessing...")

    # Call load_data and preprocess_data functions
    data = load_data(file_path)
    preprocessed_data = preprocess_data(data)

    # Convert preprocessed data back to DataFrame if needed
    preprocessed_data = pd.DataFrame(preprocessed_data)

    # Save preprocessed data
    preprocessed_data.to_csv('preprocessed_data.csv', index=False)
    logging.info("Preprocessed data saved to preprocessed_data.csv")

    # Explore the data
    explore_data(data)

    logging.info("Data analysis and preprocessing completed.")

if __name__ == "__main__":
    # Replace "your_dataset.csv" with the actual path to your dataset
    main("C:\\Users\\Soumyajit\\Downloads\\IBM_HR_Attrition.csv")
