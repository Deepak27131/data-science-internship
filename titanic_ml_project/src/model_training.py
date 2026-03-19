from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.utils import setup_logger

logger = setup_logger(__name__)

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth=5):
    """
    Train a Decision Tree classifier.
    """
    logger.info(f"Training Decision Tree model with max_depth={max_depth}...")
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model
