
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def prepare_data_for_ml(features_df):
    """
    Prepare feature DataFrame for ML training.

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features and labels

    Returns:
    --------
    X : np.array
        Feature matrix
    y : np.array
        Labels
    feature_names : list
        Names of features
    """
    # Get feature columns (exclude metadata columns)
    meta_cols = ['subject_id', 'epoch', 'label']
    feature_cols = [col for col in features_df.columns if col not in meta_cols]

    X = features_df[feature_cols].values
    y = features_df['label'].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, feature_cols, le

def train_classical_models(X, y, cv_folds=5):
    """
    Train and evaluate classical ML models using cross-validation.

    Parameters:
    -----------
    X : np.array
        Feature matrix
    y : np.array
        Labels
    cv_folds : int
        Number of cross-validation folds

    Returns:
    --------
    results : dict
        Dictionary with model performance metrics
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define models
    models = {
        # 'SVM (RBF)': SVC(kernel='rbf', C=1.0, random_state=42),
        # 'SVM (Linear)': SVC(kernel='linear', C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        results[name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }
        print(f"  Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

    return results, scaler

print("eeg_linear.py recreated successfully!")
