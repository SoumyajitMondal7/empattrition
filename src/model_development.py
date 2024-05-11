import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

def load_data(file_path):
    """Load the dataset from the specified file path."""
    data = pd.read_csv(file_path)
    return data

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

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the machine learning model."""
    logging.info("Training the model...")
    # Train model
    models = [
        ('RandomForest', RandomForestClassifier()),
        ('LogisticRegression', LogisticRegression()),
        ('SVM', SVC())
    ]

    voting_clf = VotingClassifier(estimators=models)
    stacking_clf = StackingClassifier(estimators=models, final_estimator=LogisticRegression())

    # Fit models
    voting_clf.fit(X_train, y_train)
    stacking_clf.fit(X_train, y_train)

    logging.info("Model training completed.")
    return voting_clf, stacking_clf

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    logging.info("Evaluating the model...")
    # Evaluate model
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logging.info(f"Model evaluation completed. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    return accuracy, precision, recall, f1

def optimize_model(X_train, y_train):
    """Optimize model hyperparameters."""
    logging.info("Optimizing model hyperparameters...")
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    logging.info(f"Best hyperparameters: {best_params}")

    # Feature selection
    feature_selector = SelectFromModel(RandomForestClassifier(**best_params))
    feature_selector.fit(X_train, y_train)
    selected_features = feature_selector.get_support(indices=True)

    logging.info("Model optimization completed.")
    return best_params, selected_features

def main():
    """Main function for model development."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data
    data = load_data("C:\\Users\\Soumyajit\\Downloads\\IBM_HR_Attrition.csv")

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model_voting, model_stacking = train_model(X_train, y_train)

    # Evaluate the models
    evaluate_model(model_voting, X_test, y_test)
    evaluate_model(model_stacking, X_test, y_test)

    # Optimize the model
    best_params, selected_features = optimize_model(X_train, y_train)

    logging.info("Model development completed.")

if __name__ == "__main__":
    main()
