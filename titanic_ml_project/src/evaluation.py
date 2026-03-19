from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.utils import setup_logger

logger = setup_logger(__name__)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model using accuracy, confusion matrix, and classification report.
    Prints the results cleanly and returns the metrics.
    """
    logger.info(f"Evaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    print("-" * 30 + "\n")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
