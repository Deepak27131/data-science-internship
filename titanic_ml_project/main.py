import os
from src.utils import setup_logger, print_section
from src.data_preprocessing import load_data, clean_data, encode_features, get_train_test_split
from src.model_training import train_logistic_regression, train_decision_tree
from src.evaluation import evaluate_model

logger = setup_logger(__name__)

def main():
    print_section("Titanic ML Pipeline Started")
    
    # Paths setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'titanic.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}. Please download titanic.csv into the data/ folder.")
        return
        
    print_section("1. Data Loading & Preprocessing")
    df = load_data(data_path)
    logger.info(f"Initial data shape: {df.shape}")
    
    df_cleaned = clean_data(df)
    logger.info(f"Cleaned data shape: {df_cleaned.shape}")
    
    df_encoded = encode_features(df_cleaned)
    logger.info(f"Encoded data shape: {df_encoded.shape}")
    
    print_section("2. Train-Test Split")
    X_train, X_test, y_train, y_test = get_train_test_split(df_encoded, target_col='Survived', test_size=0.2)
    logger.info(f"Training set size: {X_train.shape[0]} samples")
    logger.info(f"Testing set size: {X_test.shape[0]} samples")
    
    print_section("3. Model Training & Evaluation")
    
    # Logistic Regression (Mandatory)
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression")
    
    # Decision Tree (Optional extension)
    dt_model = train_decision_tree(X_train, y_train, max_depth=5)
    evaluate_model(dt_model, X_test, y_test, model_name="Decision Tree Classifier")
    
    print_section("Pipeline Execution Complete")

if __name__ == "__main__":
    main()
