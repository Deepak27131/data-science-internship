import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils import setup_logger

logger = setup_logger(__name__)

def load_data(filepath):
    """
    Load the Titanic dataset using pandas.
    """
    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Handle missing values and drop irrelevant columns.
    - Fills 'Age' with median.
    - Fills 'Embarked' with mode.
    - Drops 'Cabin' due to excessive missing values.
    - Drops identifying columns like 'PassengerId', 'Name', 'Ticket'.
    """
    logger.info("Cleaning data (handling missing values and dropping columns)...")
    
    # 1. Drop Cabin as it has too many missing values (>75%)
    if 'Cabin' in df.columns:
        df = df.drop('Cabin', axis=1)
    
    # 2. Fill missing Age with median
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # 3. Fill missing Embarked with the most frequent value (mode)
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
    # 4. Fill missing Fare (just in case)
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
    # 5. Drop irrelevant columns that don't help prediction
    cols_to_drop = ['PassengerId', 'Name', 'Ticket']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return df

def encode_features(df):
    """
    Encode categorical variables ('Sex', 'Embarked').
    - 'Sex' is label encoded.
    - 'Embarked' is one-hot encoded.
    """
    logger.info("Encoding categorical variables...")
    
    # Label encoding for Sex
    if 'Sex' in df.columns:
        le_sex = LabelEncoder()
        df['Sex'] = le_sex.fit_transform(df['Sex']) # 1 usually for male, 0 for female
    
    # One-hot encoding for Embarked (creates dummy variables)
    if 'Embarked' in df.columns:
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    return df

def get_train_test_split(df, target_col='Survived', test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe!")
        
    logger.info(f"Splitting data with test_size={test_size}")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
