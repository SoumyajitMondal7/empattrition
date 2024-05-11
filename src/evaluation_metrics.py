import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_performance(model, X_test, y_test):
    """
    Evaluate the performance of the trained model on the test data.
    
    Parameters:
    - model: Trained model to be evaluated.
    - X_test: Features of the test data.
    - y_test: Target variable of the test data.
    
    Returns:
    - metrics_dict: Dictionary containing evaluation metrics (accuracy, precision, recall, F1 score, ROC-AUC score, PR-AUC score).
    """
    try:
        # Predictions
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_pred)

        # Log evaluation metrics
        logging.info("Evaluation Metrics:")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"ROC-AUC Score: {roc_auc}")
        logging.info(f"PR-AUC Score: {pr_auc}")

        # Return evaluation metrics as a dictionary
        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc_score": roc_auc,
            "pr_auc_score": pr_auc
        }
        return metrics_dict
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {str(e)}")
        return None

def load_test_data(test_data_file, target_column):
    """Load the test dataset from the specified file path."""
    try:
        test_data = pd.read_csv(test_data_file)
        if target_column in test_data.columns:
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            return X_test, y_test
        else:
            logging.error(f"Target variable '{target_column}' not found in the test data.")
            return None, None
    except Exception as e:
        logging.error(f"An error occurred while loading test data: {str(e)}")
        return None, None

def load_trained_model():
    # Load your actual trained model here
    # For example, let's assume you have a saved model file named 'trained_model.pkl'
    from joblib import load
    model = load('trained_model.pkl')
    return model

def main():
    """Main function for evaluating model performance."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Specify the path to your test data file and the name of the target column
    test_data_file = "C:\\Users\\Soumyajit\\Downloads\\test_data.csv"
    target_column = "target"

    # Load test data
    X_test, y_test = load_test_data(test_data_file, target_column)
    if X_test is None or y_test is None:
        logging.error("Failed to load test data.")
        return

    # Load trained model
    trained_model = load_trained_model()

    # Evaluate model performance
    evaluate_performance(trained_model, X_test, y_test)

if __name__ == "__main__":
    main()
