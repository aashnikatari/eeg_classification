
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def prepare_data_for_lstm(X, y):
    """
    Reshape data for LSTM input (samples, time_steps, features).
    For feature-based approach, we treat each epoch as a sequence.

    Parameters:
    -----------
    X : np.array
        Feature matrix (samples, features)
    y : np.array
        Labels

    Returns:
    --------
    X_lstm : np.array
        Reshaped data for LSTM
    y_lstm : np.array
        Corresponding labels
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Reshape each sample to (1, n_features) to represent a single time step
    X_lstm = X.reshape((n_samples, 1, n_features))

    return X_lstm, y

def build_lstm_model(input_shape, n_classes):
    """
    Build a simple LSTM model for EEG classification.

    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (time_steps, features)
    n_classes : int
        Number of output classes

    Returns:
    --------
    model : keras.Model
        Compiled LSTM model
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
    ])

    loss = 'sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return model

def train_lstm_model(X, y, test_size=0.2, epochs=50, batch_size=32):
    """
    Train LSTM model with early stopping.

    Parameters:
    -----------
    X : np.array
        Feature matrix
    y : np.array
        Labels
    test_size : float
        Proportion of data for testing
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size for training

    Returns:
    --------
    model : keras.Model
        Trained model
    history : History
        Training history
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Prepare for LSTM
    X_lstm, y_lstm = prepare_data_for_lstm(X_scaled, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y_lstm, test_size=test_size, random_state=42, stratify=y_lstm
    )

    # Build model
    n_classes = len(np.unique(y))
    model = build_lstm_model(X_train.shape[1:], n_classes)

    print(model.summary())

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    return model, history, scaler, (X_test, y_test)

    def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'training_history.png'), dpi=150)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix.png'), dpi=150)
    plt.show()

def compare_models(results):
    """
    Compare performance of different models.
    """
    models = list(results.keys())
    accuracies = [results[m]['mean_accuracy'] for m in models]
    stds = [results[m]['std_accuracy'] for m in models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison - OCD vs Anxiety Classification')
    plt.ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'model_comparison.png'), dpi=150)
    plt.show()

def save_results(features_df, results, output_path):
    """
    Save processed features and results to files.
    """
    # Save features
    features_df.to_csv(os.path.join(output_path, 'extracted_features.csv'), index=False)
    print(f"Features saved to {output_path}/extracted_features.csv")

    # Save results summary
    results_summary = pd.DataFrame([
        {'Model': name, 'Mean_Accuracy': r['mean_accuracy'], 'Std_Accuracy': r['std_accuracy']}
        for name, r in results.items()
    ])
    results_summary.to_csv(os.path.join(output_path, 'results_summary.csv'), index=False)
    print(f"Results saved to {output_path}/results_summary.csv")