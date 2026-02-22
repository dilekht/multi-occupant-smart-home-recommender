#!/usr/bin/env python3
"""
Baseline Model Implementations
==============================
Implements all baseline models for the KAIS revision:
1. GLM (Logistic Regression) - Our main approach
2. XGBoost - Strong tabular baseline (Chen & Guestrin, KDD 2016)
3. LightGBM - Efficient gradient boosting (Ke et al., NeurIPS 2017)
4. Random Forest - Non-boosting ensemble
5. LSTM - Standard deep learning sequential baseline
6. Transformer - State-of-the-art architecture

All models return a dictionary with:
- accuracy
- f1_weighted
- f1_macro
- precision
- recall
- predictions
- model_size_mb (approximate)
"""

import numpy as np
import pickle
import sys
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute standard classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'predictions': y_pred
    }


# =============================================================================
# 1. GLM (LOGISTIC REGRESSION) - Our Main Approach
# =============================================================================
def train_glm_baseline(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Train Generalized Linear Model (Multinomial Logistic Regression).
    
    This is our main approach as described in the paper.
    """
    from sklearn.linear_model import LogisticRegression
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    # Note: multi_class parameter removed in sklearn 1.7+ (auto-determined)
    # Note: n_jobs parameter deprecated in sklearn 1.8+
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        C=1.0,  # L2 regularization
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    
    # Estimate model size
    model_bytes = len(pickle.dumps(model))
    metrics['model_size_mb'] = model_bytes / (1024 * 1024)
    metrics['model'] = model
    
    return metrics


# =============================================================================
# 2. XGBOOST - Strong Tabular Baseline
# =============================================================================
def train_xgboost_baseline(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Train XGBoost classifier.
    
    Reference: Chen & Guestrin (2016) - XGBoost: A Scalable Tree Boosting System
    Published at KDD (one of the venues mentioned by the editor).
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost not installed. Install with: pip install xgboost")
        return {'accuracy': None, 'error': 'XGBoost not installed'}
    
    # Encode labels - fit only on training labels to ensure consecutive 0-indexed
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    
    # For test, only evaluate on classes that were in training
    # (This is fair - model can only predict what it was trained on)
    known_classes = set(le.classes_)
    test_mask = np.array([y in known_classes for y in y_test])
    
    X_test_filtered = X_test[test_mask]
    y_test_filtered = y_test[test_mask]
    y_test_enc = le.transform(y_test_filtered)
    
    # Report how many test samples were filtered
    n_filtered = len(y_test) - len(y_test_filtered)
    if n_filtered > 0:
        pct_filtered = n_filtered / len(y_test) * 100
        # Only warn if significant
        if pct_filtered > 1:
            print(f"[Note: {n_filtered} test samples ({pct_filtered:.1f}%) have unseen classes]", end=" ")
    
    # Get number of classes
    num_classes = len(le.classes_)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=num_classes,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train_enc)
    
    # Predict
    y_pred_enc = model.predict(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_enc)
    
    # Compute metrics on filtered test set
    metrics = compute_metrics(y_test_filtered, y_pred)
    
    # Model size
    model_bytes = len(pickle.dumps(model))
    metrics['model_size_mb'] = model_bytes / (1024 * 1024)
    
    return metrics


# =============================================================================
# 3. LIGHTGBM - Efficient Gradient Boosting
# =============================================================================
def train_lightgbm_baseline(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Train LightGBM classifier.
    
    Reference: Ke et al. (2017) - LightGBM: A Highly Efficient Gradient Boosting 
    Decision Tree. Published at NeurIPS.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("LightGBM not installed. Install with: pip install lightgbm")
        return {'accuracy': None, 'error': 'LightGBM not installed'}
    
    # Encode labels (fit on ALL labels to handle unseen in test)
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_test])
    le.fit(all_labels)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Scale features (important for LightGBM performance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get number of classes
    num_classes = len(le.classes_)
    
    # Train model with tuned hyperparameters
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        force_col_wise=True  # Avoid OpenMP warnings
    )
    model.fit(X_train_scaled, y_train_enc)
    
    # Predict
    y_pred_enc = model.predict(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_enc)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    
    # Model size
    model_bytes = len(pickle.dumps(model))
    metrics['model_size_mb'] = model_bytes / (1024 * 1024)
    
    return metrics


# =============================================================================
# 4. RANDOM FOREST - Non-Boosting Ensemble
# =============================================================================
def train_random_forest_baseline(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Train Random Forest classifier.
    
    Reference: Breiman (2001) - Random Forests. Machine Learning.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    
    # Model size
    model_bytes = len(pickle.dumps(model))
    metrics['model_size_mb'] = model_bytes / (1024 * 1024)
    
    return metrics


# =============================================================================
# 5. LSTM - Deep Learning Sequential Baseline
# =============================================================================
def train_lstm_baseline(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    sequence_length: int = 10,
    epochs: int = 10,
    batch_size: int = 256
) -> Dict[str, Any]:
    """
    Train LSTM classifier.
    
    Standard deep learning baseline for sequential activity recognition.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("TensorFlow not installed. Install with: pip install tensorflow")
        return {'accuracy': None, 'error': 'TensorFlow not installed'}
    
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    num_classes = len(le.classes_)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create sequences
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_cat, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_cat, sequence_length)
    
    # Build model
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(sequence_length, X_train.shape[1])),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predict
    y_pred_proba = model.predict(X_test_seq, verbose=0)
    y_pred_enc = np.argmax(y_pred_proba, axis=1)
    y_pred = le.inverse_transform(y_pred_enc)
    
    # Ground truth for test (accounting for sequence offset)
    y_test_aligned = y_test[sequence_length:]
    
    # Compute metrics
    metrics = compute_metrics(y_test_aligned, y_pred)
    
    # Model size (approximate)
    metrics['model_size_mb'] = model.count_params() * 4 / (1024 * 1024)  # 4 bytes per float32
    
    return metrics


# =============================================================================
# 6. TRANSFORMER - State-of-the-Art Architecture
# =============================================================================
def train_transformer_baseline(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    sequence_length: int = 10,
    epochs: int = 10,
    batch_size: int = 256
) -> Dict[str, Any]:
    """
    Train Transformer-based classifier.
    
    Reference: Huang & Zhang (2023) - Transformer-based HAR in Smart Homes (ACM CACML)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, LayerNormalization,
            MultiHeadAttention, GlobalAveragePooling1D
        )
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("TensorFlow not installed. Install with: pip install tensorflow")
        return {'accuracy': None, 'error': 'TensorFlow not installed'}
    
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    num_classes = len(le.classes_)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create sequences
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_cat, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_cat, sequence_length)
    
    # Build Transformer model
    def build_transformer(seq_length, n_features, n_classes, d_model=64, num_heads=4, ff_dim=128):
        inputs = Input(shape=(seq_length, n_features))
        
        # Project to d_model dimensions
        x = Dense(d_model)(inputs)
        
        # Transformer block
        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads
        )(x, x)
        attn_output = Dropout(0.1)(attn_output)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ffn = Dense(ff_dim, activation='relu')(x)
        ffn = Dense(d_model)(ffn)
        ffn = Dropout(0.1)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global pooling and classification
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(n_classes, activation='softmax')(x)
        
        return Model(inputs, outputs)
    
    model = build_transformer(
        seq_length=sequence_length,
        n_features=X_train.shape[1],
        n_classes=num_classes
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predict
    y_pred_proba = model.predict(X_test_seq, verbose=0)
    y_pred_enc = np.argmax(y_pred_proba, axis=1)
    y_pred = le.inverse_transform(y_pred_enc)
    
    # Ground truth for test (accounting for sequence offset)
    y_test_aligned = y_test[sequence_length:]
    
    # Compute metrics
    metrics = compute_metrics(y_test_aligned, y_pred)
    
    # Model size
    metrics['model_size_mb'] = model.count_params() * 4 / (1024 * 1024)
    
    return metrics


# =============================================================================
# HELPER: Run All Baselines
# =============================================================================
def run_all_baselines(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    include_deep_learning: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run all baseline models and return results.
    """
    results = {}
    
    # Traditional ML baselines
    print("  Training GLM...", end=" ")
    results['GLM'] = train_glm_baseline(X_train, y_train, X_test, y_test)
    print(f"Acc: {results['GLM']['accuracy']:.4f}")
    
    print("  Training XGBoost...", end=" ")
    results['XGBoost'] = train_xgboost_baseline(X_train, y_train, X_test, y_test)
    if results['XGBoost']['accuracy']:
        print(f"Acc: {results['XGBoost']['accuracy']:.4f}")
    else:
        print("SKIPPED")
    
    print("  Training LightGBM...", end=" ")
    results['LightGBM'] = train_lightgbm_baseline(X_train, y_train, X_test, y_test)
    if results['LightGBM']['accuracy']:
        print(f"Acc: {results['LightGBM']['accuracy']:.4f}")
    else:
        print("SKIPPED")
    
    print("  Training Random Forest...", end=" ")
    results['RandomForest'] = train_random_forest_baseline(X_train, y_train, X_test, y_test)
    print(f"Acc: {results['RandomForest']['accuracy']:.4f}")
    
    # Deep learning baselines
    if include_deep_learning:
        print("  Training LSTM...", end=" ")
        results['LSTM'] = train_lstm_baseline(X_train, y_train, X_test, y_test)
        if results['LSTM']['accuracy']:
            print(f"Acc: {results['LSTM']['accuracy']:.4f}")
        else:
            print("SKIPPED")
        
        print("  Training Transformer...", end=" ")
        results['Transformer'] = train_transformer_baseline(X_train, y_train, X_test, y_test)
        if results['Transformer']['accuracy']:
            print(f"Acc: {results['Transformer']['accuracy']:.4f}")
        else:
            print("SKIPPED")
    
    return results


if __name__ == '__main__':
    # Test with synthetic data
    from preprocessing import SyntheticArasGenerator
    from feature_engineering import MultiResidentFeatureEngineer
    
    print("Testing baseline implementations with synthetic data...")
    
    # Generate data
    generator = SyntheticArasGenerator(seed=42)
    data = generator.generate(num_days=3)
    
    # Generate features
    engineer = MultiResidentFeatureEngineer()
    X, y_r1, y_r2, y_conflict = engineer.generate_features(data)
    
    # Simple train/test split
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_r1[:split_idx], y_r1[split_idx:]
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Run baselines
    results = run_all_baselines(X_train, y_train, X_test, y_test, include_deep_learning=False)
    
    print("\nResults Summary:")
    for name, metrics in results.items():
        if metrics.get('accuracy'):
            print(f"  {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_weighted']:.4f}")
