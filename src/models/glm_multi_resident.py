"""
Extended GLM Model for Multi-Resident Smart Home Recommendations
================================================================

This module extends the Dilekh et al. (2024) GLM approach to handle
multi-resident scenarios with conflict detection and resolution.

Original Paper: "Dynamic Context-Aware Recommender System for Home Automation
                Through Synergistic Unsupervised and Supervised Learning"
DOI: https://doi.org/10.18267/j.aip.228

Extension: Multi-resident activity prediction with conflict awareness

Key Components:
1. Multi-output GLM for concurrent activity prediction
2. Conflict prediction model
3. FP-Growth pattern integration
4. Household-adaptive recommendations

Requirements:
    pip install scikit-learn pandas numpy scipy joblib

Author: Research Extension Project
Date: January 2026
Version: 1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import warnings
import pickle
from datetime import datetime

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    multilabel_confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# For multi-output
from sklearn.multioutput import MultiOutputClassifier

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

ACTIVITIES = {
    1: "Other", 2: "Going Out", 3: "Preparing Breakfast", 4: "Having Breakfast",
    5: "Preparing Lunch", 6: "Having Lunch", 7: "Preparing Dinner", 8: "Having Dinner",
    9: "Washing Dishes", 10: "Having Snack", 11: "Sleeping", 12: "Watching TV",
    13: "Studying", 14: "Having Shower", 15: "Toileting", 16: "Napping",
    17: "Using Internet", 18: "Reading Book", 19: "Laundry", 20: "Shaving",
    21: "Brushing Teeth", 22: "Talking on Phone", 23: "Listening to Music",
    24: "Cleaning", 25: "Having Conversation", 26: "Having Guest", 27: "Changing Clothes"
}

ACTIVITY_CATEGORIES = {
    "rest": [11, 16],
    "entertainment": [12, 17, 18, 23],
    "meal_prep": [3, 5, 7],
    "meal_consumption": [4, 6, 8, 10],
    "hygiene": [14, 15, 20, 21, 27],
    "work": [13],
    "household": [9, 19, 24],
    "social": [22, 25, 26],
    "away": [2],
    "other": [1]
}


@dataclass
class ModelConfig:
    """Configuration for GLM models."""
    # Model parameters
    max_iter: int = 1000
    C: float = 1.0  # Regularization strength
    solver: str = 'lbfgs'
    
    # Training parameters
    test_size: float = 0.25
    val_size: float = 0.25  # From training set
    random_state: int = 42
    cv_folds: int = 5
    
    # Feature parameters
    use_sensor_features: bool = True
    use_temporal_features: bool = True
    use_lag_features: bool = True
    lag_window: int = 5  # Number of previous timesteps
    use_pattern_features: bool = True
    
    # Multi-resident specific
    use_cross_resident_features: bool = True
    use_conflict_features: bool = True


@dataclass
class ModelResults:
    """Container for model evaluation results."""
    model_name: str = ""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    confusion_matrix: np.ndarray = None
    classification_report: str = ""
    feature_importance: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class MultiResidentFeatureEngineer:
    """
    Feature engineering for multi-resident smart home data.
    
    Extends single-occupant features with:
    - Cross-resident activity features
    - Synchronization indicators
    - Conflict probability features
    - Pattern-based features from FP-Growth
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.sensor_columns: List[str] = []
        self.feature_columns: List[str] = []
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'MultiResidentFeatureEngineer':
        """Fit feature transformers on training data."""
        # Identify sensor columns
        self.sensor_columns = [c for c in df.columns if c.startswith('S') and '_' in c]
        
        # Fit scalers for numerical features
        numerical_cols = ['Hour', 'MinuteOfDay', 'ActiveSensorCount']
        numerical_cols = [c for c in numerical_cols if c in df.columns]
        
        if numerical_cols:
            self.scalers['numerical'] = StandardScaler()
            self.scalers['numerical'].fit(df[numerical_cols])
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data with multi-resident features."""
        if not self._fitted:
            raise ValueError("FeatureEngineer not fitted. Call fit() first.")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Sensor features
        if self.config.use_sensor_features:
            for col in self.sensor_columns:
                if col in df.columns:
                    features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 2. Temporal features
        if self.config.use_temporal_features:
            features = self._add_temporal_features(df, features)
        
        # 3. Lag features (previous activities)
        if self.config.use_lag_features:
            features = self._add_lag_features(df, features)
        
        # 4. Cross-resident features
        if self.config.use_cross_resident_features:
            features = self._add_cross_resident_features(df, features)
        
        # 5. Conflict features
        if self.config.use_conflict_features:
            features = self._add_conflict_features(df, features)
        
        # 6. Pattern-based features
        if self.config.use_pattern_features:
            features = self._add_pattern_features(df, features)
        
        # Fill any remaining NaN values with 0
        features = features.fillna(0)
        
        # Store feature columns
        self.feature_columns = list(features.columns)
        
        return features
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def _add_temporal_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add temporal context features."""
        # Hour (cyclical encoding)
        if 'Hour' in df.columns:
            hour = pd.to_numeric(df['Hour'], errors='coerce').fillna(12)
            features['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Time of day one-hot
        if 'TimeOfDay' in df.columns:
            tod_dummies = pd.get_dummies(df['TimeOfDay'].fillna('Unknown'), prefix='TOD')
            features = pd.concat([features, tod_dummies], axis=1)
        
        # Weekend
        if 'IsWeekend' in df.columns:
            features['IsWeekend'] = df['IsWeekend']
        
        # Day of week (cyclical)
        if 'DayOfWeek' in df.columns:
            features['DOW_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            features['DOW_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        return features
    
    def _add_lag_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add lagged activity features."""
        # Previous activities for both residents
        for lag in range(1, self.config.lag_window + 1):
            if 'Activity_R1' in df.columns:
                features[f'R1_Act_Lag_{lag}'] = df['Activity_R1'].shift(lag).fillna(0).astype(int)
            if 'Activity_R2' in df.columns:
                features[f'R2_Act_Lag_{lag}'] = df['Activity_R2'].shift(lag).fillna(0).astype(int)
        
        # Activity change indicators
        if 'Activity_R1' in df.columns:
            features['R1_Activity_Changed'] = (df['Activity_R1'] != df['Activity_R1'].shift(1)).astype(int)
        if 'Activity_R2' in df.columns:
            features['R2_Activity_Changed'] = (df['Activity_R2'] != df['Activity_R2'].shift(1)).astype(int)
        
        return features
    
    def _add_cross_resident_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add features capturing inter-resident relationships."""
        # Synchronization status
        if 'IsSynchronized' in df.columns:
            features['IsSynchronized'] = df['IsSynchronized']
        elif 'Activity_R1' in df.columns and 'Activity_R2' in df.columns:
            features['IsSynchronized'] = (df['Activity_R1'] == df['Activity_R2']).astype(int)
        
        # Activity category match
        if 'Category_R1' in df.columns and 'Category_R2' in df.columns:
            features['SameCategory'] = (df['Category_R1'] == df['Category_R2']).astype(int)
        
        # Both home indicator
        if 'Activity_R1' in df.columns and 'Activity_R2' in df.columns:
            features['BothHome'] = ((df['Activity_R1'] != 2) & (df['Activity_R2'] != 2)).astype(int)
            features['BothAway'] = ((df['Activity_R1'] == 2) & (df['Activity_R2'] == 2)).astype(int)
            features['OneAway'] = ((df['Activity_R1'] == 2) ^ (df['Activity_R2'] == 2)).astype(int)
        
        # Activity pair encoding (top 20 most frequent)
        if 'ActivityPair' in df.columns:
            top_pairs = df['ActivityPair'].value_counts().head(20).index.tolist()
            for pair in top_pairs:
                features[f'Pair_{pair}'] = (df['ActivityPair'] == pair).astype(int)
        
        return features
    
    def _add_conflict_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add conflict-related features."""
        # Direct conflict indicators
        if 'HasConflict' in df.columns:
            features['HasConflict'] = df['HasConflict']
        
        # Conflict type encoding
        if 'ConflictType' in df.columns:
            conflict_dummies = pd.get_dummies(df['ConflictType'].fillna('none'), prefix='Conflict')
            features = pd.concat([features, conflict_dummies], axis=1)
        
        # Potential conflict indicators based on activity combinations
        if 'Activity_R1' in df.columns and 'Activity_R2' in df.columns:
            # TV conflict potential
            features['TVConflictRisk'] = (
                ((df['Activity_R1'] == 12) & (df['Activity_R2'].isin([11, 13, 16, 18]))) |
                ((df['Activity_R2'] == 12) & (df['Activity_R1'].isin([11, 13, 16, 18])))
            ).astype(int)
            
            # Noise conflict potential
            features['NoiseConflictRisk'] = (
                ((df['Activity_R1'] == 23) & (df['Activity_R2'].isin([11, 13, 16, 18]))) |
                ((df['Activity_R2'] == 23) & (df['Activity_R1'].isin([11, 13, 16, 18])))
            ).astype(int)
        
        return features
    
    def _add_pattern_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add features derived from FP-Growth patterns."""
        # Zone-based features
        if 'PrimaryZone' in df.columns:
            zone_dummies = pd.get_dummies(df['PrimaryZone'], prefix='Zone')
            features = pd.concat([features, zone_dummies], axis=1)
        
        # Activity category features
        if 'Category_R1' in df.columns:
            cat_dummies = pd.get_dummies(df['Category_R1'], prefix='R1_Cat')
            features = pd.concat([features, cat_dummies], axis=1)
        
        if 'Category_R2' in df.columns:
            cat_dummies = pd.get_dummies(df['Category_R2'], prefix='R2_Cat')
            features = pd.concat([features, cat_dummies], axis=1)
        
        # Sensor activation count
        if self.sensor_columns:
            features['ActiveSensorCount'] = df[self.sensor_columns].sum(axis=1)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_columns


# =============================================================================
# GLM MODELS
# =============================================================================

class MultiResidentGLM:
    """
    Extended GLM for multi-resident activity prediction.
    
    Implements three prediction tasks:
    1. Resident 1 activity prediction
    2. Resident 2 activity prediction
    3. Conflict prediction
    
    Based on Dilekh et al. (2024) with multi-resident extensions.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.feature_engineer = MultiResidentFeatureEngineer(config)
        
        # Models
        self.model_r1: Optional[LogisticRegression] = None
        self.model_r2: Optional[LogisticRegression] = None
        self.model_conflict: Optional[LogisticRegression] = None
        self.model_joint: Optional[MultiOutputClassifier] = None
        
        # Label encoders
        self.le_r1 = LabelEncoder()
        self.le_r2 = LabelEncoder()
        
        # Results
        self.results: Dict[str, ModelResults] = {}
        
        # State
        self._fitted = False
        
    def prepare_data(self, df: pd.DataFrame
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Generate features
        X = self.feature_engineer.fit_transform(df)
        
        # Prepare labels
        y = pd.DataFrame({
            'Activity_R1': df['Activity_R1'].values,
            'Activity_R2': df['Activity_R2'].values,
            'HasConflict': df['HasConflict'].values if 'HasConflict' in df.columns else 0
        })
        
        # Remove rows with NaN (from lag features)
        valid_idx = ~X.isnull().any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y['Activity_R1']  # Stratify by R1 activity
        )
        
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> 'MultiResidentGLM':
        """
        Fit all GLM models.
        
        Args:
            X_train: Training features
            y_train: Training labels (Activity_R1, Activity_R2, HasConflict)
        """
        print("\n" + "=" * 60)
        print("TRAINING MULTI-RESIDENT GLM MODELS")
        print("=" * 60)
        
        # 1. Train Resident 1 Activity Model
        print("\n[1/4] Training Resident 1 Activity Model...")
        self.model_r1 = LogisticRegression(
            max_iter=self.config.max_iter,
            C=self.config.C,
            solver=self.config.solver,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        y_r1 = self.le_r1.fit_transform(y_train['Activity_R1'])
        self.model_r1.fit(X_train, y_r1)
        print(f"   Classes: {len(self.le_r1.classes_)}")
        
        # 2. Train Resident 2 Activity Model
        print("\n[2/4] Training Resident 2 Activity Model...")
        self.model_r2 = LogisticRegression(
            max_iter=self.config.max_iter,
            C=self.config.C,
            solver=self.config.solver,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        y_r2 = self.le_r2.fit_transform(y_train['Activity_R2'])
        self.model_r2.fit(X_train, y_r2)
        print(f"   Classes: {len(self.le_r2.classes_)}")
        
        # 3. Train Conflict Model
        print("\n[3/4] Training Conflict Prediction Model...")
        self.model_conflict = LogisticRegression(
            max_iter=self.config.max_iter,
            C=self.config.C,
            solver='lbfgs',
            random_state=self.config.random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1
        )
        y_conflict = y_train['HasConflict'].values
        if len(np.unique(y_conflict)) > 1:
            self.model_conflict.fit(X_train, y_conflict)
            print(f"   Conflict rate: {y_conflict.mean()*100:.2f}%")
        else:
            print("   WARNING: No conflicts in training data, skipping conflict model")
            self.model_conflict = None
        
        # 4. Train Joint Multi-Output Model
        print("\n[4/4] Training Joint Multi-Output Model...")
        base_model = LogisticRegression(
            max_iter=self.config.max_iter,
            C=self.config.C,
            solver=self.config.solver,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        self.model_joint = MultiOutputClassifier(base_model, n_jobs=-1)
        y_joint = np.column_stack([y_r1, y_r2])
        self.model_joint.fit(X_train, y_joint)
        print("   Multi-output model trained")
        
        self._fitted = True
        print("\n✓ All models trained successfully")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict activities and conflicts.
        
        Returns:
            Dictionary with predictions for R1, R2, and conflict
        """
        if not self._fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        predictions = {}
        
        # Resident 1 prediction
        pred_r1_encoded = self.model_r1.predict(X)
        predictions['Activity_R1'] = self.le_r1.inverse_transform(pred_r1_encoded)
        predictions['Activity_R1_Proba'] = self.model_r1.predict_proba(X)
        
        # Resident 2 prediction
        pred_r2_encoded = self.model_r2.predict(X)
        predictions['Activity_R2'] = self.le_r2.inverse_transform(pred_r2_encoded)
        predictions['Activity_R2_Proba'] = self.model_r2.predict_proba(X)
        
        # Conflict prediction
        if self.model_conflict is not None:
            predictions['HasConflict'] = self.model_conflict.predict(X)
            predictions['Conflict_Proba'] = self.model_conflict.predict_proba(X)[:, 1]
        else:
            predictions['HasConflict'] = np.zeros(len(X))
            predictions['Conflict_Proba'] = np.zeros(len(X))
        
        # Joint prediction
        pred_joint = self.model_joint.predict(X)
        predictions['Joint_R1'] = self.le_r1.inverse_transform(pred_joint[:, 0])
        predictions['Joint_R2'] = self.le_r2.inverse_transform(pred_joint[:, 1])
        
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, ModelResults]:
        """
        Evaluate all models on test data.
        
        Returns:
            Dictionary of ModelResults for each model
        """
        print("\n" + "=" * 60)
        print("EVALUATING MULTI-RESIDENT GLM MODELS")
        print("=" * 60)
        
        predictions = self.predict(X_test)
        
        # 1. Evaluate Resident 1 Model
        print("\n[1] Resident 1 Activity Prediction")
        print("-" * 40)
        self.results['R1_Activity'] = self._evaluate_single_model(
            y_test['Activity_R1'].values,
            predictions['Activity_R1'],
            "Resident 1 Activity"
        )
        
        # 2. Evaluate Resident 2 Model
        print("\n[2] Resident 2 Activity Prediction")
        print("-" * 40)
        self.results['R2_Activity'] = self._evaluate_single_model(
            y_test['Activity_R2'].values,
            predictions['Activity_R2'],
            "Resident 2 Activity"
        )
        
        # 3. Evaluate Conflict Model
        print("\n[3] Conflict Prediction")
        print("-" * 40)
        if self.model_conflict is not None and y_test['HasConflict'].sum() > 0:
            self.results['Conflict'] = self._evaluate_binary_model(
                y_test['HasConflict'].values,
                predictions['HasConflict'],
                predictions['Conflict_Proba'],
                "Conflict"
            )
        else:
            print("   No conflicts in test data or model not trained")
            self.results['Conflict'] = ModelResults(model_name="Conflict", accuracy=0)
        
        # 4. Evaluate Joint Model
        print("\n[4] Joint Multi-Resident Prediction")
        print("-" * 40)
        self.results['Joint'] = self._evaluate_joint_model(
            y_test[['Activity_R1', 'Activity_R2']].values,
            np.column_stack([predictions['Joint_R1'], predictions['Joint_R2']]),
            "Joint"
        )
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _evaluate_single_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                name: str) -> ModelResults:
        """Evaluate a single multi-class model."""
        results = ModelResults(model_name=name)
        
        results.accuracy = accuracy_score(y_true, y_pred)
        results.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results.f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"   Accuracy:  {results.accuracy:.4f} ({results.accuracy*100:.2f}%)")
        print(f"   Precision: {results.precision:.4f}")
        print(f"   Recall:    {results.recall:.4f}")
        print(f"   F1-Score:  {results.f1:.4f}")
        
        results.classification_report = classification_report(y_true, y_pred, zero_division=0)
        
        return results
    
    def _evaluate_binary_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_proba: np.ndarray, name: str) -> ModelResults:
        """Evaluate a binary classification model."""
        results = ModelResults(model_name=name)
        
        results.accuracy = accuracy_score(y_true, y_pred)
        results.precision = precision_score(y_true, y_pred, zero_division=0)
        results.recall = recall_score(y_true, y_pred, zero_division=0)
        results.f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"   Accuracy:  {results.accuracy:.4f}")
        print(f"   Precision: {results.precision:.4f}")
        print(f"   Recall:    {results.recall:.4f}")
        print(f"   F1-Score:  {results.f1:.4f}")
        
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
                print(f"   ROC-AUC:   {roc_auc:.4f}")
            except:
                pass
        
        return results
    
    def _evaluate_joint_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                               name: str) -> ModelResults:
        """Evaluate joint multi-output model."""
        results = ModelResults(model_name=name)
        
        # Exact match accuracy (both predictions correct)
        exact_match = np.all(y_true == y_pred, axis=1).mean()
        
        # Individual accuracies
        acc_r1 = accuracy_score(y_true[:, 0], y_pred[:, 0])
        acc_r2 = accuracy_score(y_true[:, 1], y_pred[:, 1])
        
        results.accuracy = exact_match
        results.precision = (acc_r1 + acc_r2) / 2  # Average individual accuracy
        
        print(f"   Exact Match Accuracy: {exact_match:.4f} ({exact_match*100:.2f}%)")
        print(f"   R1 Individual Acc:    {acc_r1:.4f}")
        print(f"   R2 Individual Acc:    {acc_r2:.4f}")
        
        return results
    
    def _print_summary(self) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\n{'Model':<25} {'Accuracy':>10} {'F1-Score':>10}")
        print("-" * 47)
        
        for name, res in self.results.items():
            print(f"{res.model_name:<25} {res.accuracy:>10.4f} {res.f1:>10.4f}")
        
        # Calculate combined metric
        if 'R1_Activity' in self.results and 'R2_Activity' in self.results:
            avg_acc = (self.results['R1_Activity'].accuracy + 
                      self.results['R2_Activity'].accuracy) / 2
            print("-" * 47)
            print(f"{'Average (R1+R2)':<25} {avg_acc:>10.4f}")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Perform cross-validation on all models.
        
        Returns:
            Dictionary with CV scores for each model
        """
        print("\n" + "=" * 60)
        print(f"CROSS-VALIDATION ({self.config.cv_folds}-fold)")
        print("=" * 60)
        
        cv_results = {}
        kfold = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        # R1 Activity
        print("\n[1] Resident 1 Activity Model...")
        model_r1 = LogisticRegression(
            max_iter=self.config.max_iter,
            C=self.config.C,
            solver=self.config.solver,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        y_r1 = self.le_r1.fit_transform(y['Activity_R1'])
        scores_r1 = cross_val_score(model_r1, X, y_r1, cv=kfold, scoring='accuracy', n_jobs=-1)
        cv_results['R1_Activity'] = scores_r1.tolist()
        print(f"   Mean: {scores_r1.mean():.4f} (+/- {scores_r1.std()*2:.4f})")
        
        # R2 Activity
        print("\n[2] Resident 2 Activity Model...")
        model_r2 = LogisticRegression(
            max_iter=self.config.max_iter,
            C=self.config.C,
            solver=self.config.solver,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        y_r2 = self.le_r2.fit_transform(y['Activity_R2'])
        scores_r2 = cross_val_score(model_r2, X, y_r2, cv=kfold, scoring='accuracy', n_jobs=-1)
        cv_results['R2_Activity'] = scores_r2.tolist()
        print(f"   Mean: {scores_r2.mean():.4f} (+/- {scores_r2.std()*2:.4f})")
        
        # Conflict (if applicable)
        if y['HasConflict'].sum() > 0:
            print("\n[3] Conflict Model...")
            model_conflict = LogisticRegression(
                max_iter=self.config.max_iter,
                C=self.config.C,
                random_state=self.config.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            scores_conflict = cross_val_score(
                model_conflict, X, y['HasConflict'], 
                cv=kfold, scoring='f1', n_jobs=-1
            )
            cv_results['Conflict'] = scores_conflict.tolist()
            print(f"   Mean F1: {scores_conflict.mean():.4f} (+/- {scores_conflict.std()*2:.4f})")
        
        return cv_results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from GLM coefficients."""
        if not self._fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        feature_names = self.feature_engineer.get_feature_names()
        
        importance_data = []
        
        # R1 model importance (average absolute coefficient across classes)
        if self.model_r1 is not None:
            r1_importance = np.abs(self.model_r1.coef_).mean(axis=0)
            for i, name in enumerate(feature_names):
                importance_data.append({
                    'feature': name,
                    'model': 'R1_Activity',
                    'importance': r1_importance[i]
                })
        
        # R2 model importance
        if self.model_r2 is not None:
            r2_importance = np.abs(self.model_r2.coef_).mean(axis=0)
            for i, name in enumerate(feature_names):
                importance_data.append({
                    'feature': name,
                    'model': 'R2_Activity',
                    'importance': r2_importance[i]
                })
        
        # Conflict model importance
        if self.model_conflict is not None:
            conflict_importance = np.abs(self.model_conflict.coef_[0])
            for i, name in enumerate(feature_names):
                importance_data.append({
                    'feature': name,
                    'model': 'Conflict',
                    'importance': conflict_importance[i]
                })
        
        df = pd.DataFrame(importance_data)
        
        # Pivot and rank
        pivot_df = df.pivot(index='feature', columns='model', values='importance')
        pivot_df['Average'] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values('Average', ascending=False)
        
        return pivot_df
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            'config': self.config,
            'model_r1': self.model_r1,
            'model_r2': self.model_r2,
            'model_conflict': self.model_conflict,
            'model_joint': self.model_joint,
            'le_r1': self.le_r1,
            'le_r2': self.le_r2,
            'feature_engineer': self.feature_engineer,
            'results': self.results,
            'fitted': self._fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MultiResidentGLM':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_data['config'])
        instance.model_r1 = model_data['model_r1']
        instance.model_r2 = model_data['model_r2']
        instance.model_conflict = model_data['model_conflict']
        instance.model_joint = model_data['model_joint']
        instance.le_r1 = model_data['le_r1']
        instance.le_r2 = model_data['le_r2']
        instance.feature_engineer = model_data['feature_engineer']
        instance.results = model_data['results']
        instance._fitted = model_data['fitted']
        
        print(f"Model loaded from {filepath}")
        return instance


# =============================================================================
# COMPARISON WITH SINGLE-OCCUPANT BASELINE
# =============================================================================

class SingleOccupantBaseline:
    """
    Single-occupant GLM baseline for comparison.
    
    Implements the original Dilekh et al. (2024) approach without
    multi-resident features.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model: Optional[LogisticRegression] = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.results: Optional[ModelResults] = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare single-occupant features (no cross-resident features)."""
        features = pd.DataFrame(index=df.index)
        
        # Sensor features
        sensor_cols = [c for c in df.columns if c.startswith('S') and '_' in c]
        for col in sensor_cols:
            features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Basic temporal features
        if 'Hour' in df.columns:
            hour = pd.to_numeric(df['Hour'], errors='coerce').fillna(12)
            features['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        if 'TimeOfDay' in df.columns:
            tod_dummies = pd.get_dummies(df['TimeOfDay'].fillna('Unknown'), prefix='TOD')
            features = pd.concat([features, tod_dummies], axis=1)
        
        if 'IsWeekend' in df.columns:
            features['IsWeekend'] = pd.to_numeric(df['IsWeekend'], errors='coerce').fillna(0)
        
        # Fill any remaining NaN with 0
        features = features.fillna(0)
        
        return features
    
    def fit_predict_evaluate(self, df: pd.DataFrame, target_col: str = 'Activity_R1'
                             ) -> ModelResults:
        """Train and evaluate single-occupant model."""
        print(f"\n{'='*60}")
        print(f"SINGLE-OCCUPANT BASELINE ({target_col})")
        print("="*60)
        
        # Prepare data
        X = self.prepare_features(df)
        y = pd.to_numeric(df[target_col], errors='coerce').fillna(1).astype(int).values
        
        # Remove any rows with NaN in features
        valid_idx = ~X.isnull().any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"   Valid samples: {len(X):,}")
        
        if len(X) == 0:
            print("   ERROR: No valid samples after preprocessing")
            return ModelResults(model_name=f"Single-Occupant ({target_col})", accuracy=0)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")
        
        # Encode
        y_train_enc = self.le.fit_transform(y_train)
        y_test_enc = self.le.transform(y_test)
        
        # Train
        print("\nTraining single-occupant GLM...")
        self.model = LogisticRegression(
            max_iter=self.config.max_iter,
            C=self.config.C,
            solver=self.config.solver,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train_enc)
        
        # Predict
        y_pred_enc = self.model.predict(X_test)
        y_pred = self.le.inverse_transform(y_pred_enc)
        
        # Evaluate
        self.results = ModelResults(model_name=f"Single-Occupant ({target_col})")
        self.results.accuracy = accuracy_score(y_test, y_pred)
        self.results.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        self.results.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        self.results.f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nResults:")
        print(f"   Accuracy:  {self.results.accuracy:.4f} ({self.results.accuracy*100:.2f}%)")
        print(f"   Precision: {self.results.precision:.4f}")
        print(f"   Recall:    {self.results.recall:.4f}")
        print(f"   F1-Score:  {self.results.f1:.4f}")
        
        return self.results


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(processed_data_path: str,
                   output_dir: str,
                   house_name: str = "combined") -> Dict:
    """
    Run complete experiment comparing single vs multi-occupant models.
    
    Args:
        processed_data_path: Path to processed_data.csv
        output_dir: Directory for outputs
        house_name: Name for output files
        
    Returns:
        Dictionary with all results
    """
    print("=" * 70)
    print("MULTI-RESIDENT GLM EXPERIMENT")
    print("=" * 70)
    
    # Load data with low_memory=False to avoid mixed type warnings
    print(f"\nLoading data from {processed_data_path}...")
    df = pd.read_csv(processed_data_path, low_memory=False)
    print(f"Loaded {len(df):,} records")
    
    # Handle mixed type columns (ConflictType, ConflictSeverity)
    if 'ConflictType' in df.columns:
        df['ConflictType'] = df['ConflictType'].fillna('none').astype(str)
    if 'ConflictSeverity' in df.columns:
        df['ConflictSeverity'] = df['ConflictSeverity'].fillna('none').astype(str)
    
    # Ensure HasConflict exists and is numeric
    if 'HasConflict' not in df.columns:
        df['HasConflict'] = 0
    df['HasConflict'] = pd.to_numeric(df['HasConflict'], errors='coerce').fillna(0).astype(int)
    
    # Ensure activity columns are numeric
    df['Activity_R1'] = pd.to_numeric(df['Activity_R1'], errors='coerce').fillna(1).astype(int)
    df['Activity_R2'] = pd.to_numeric(df['Activity_R2'], errors='coerce').fillna(1).astype(int)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'house': house_name,
        'total_samples': len(df),
        'models': {}
    }
    
    # =========================================================================
    # 1. Single-Occupant Baselines
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: SINGLE-OCCUPANT BASELINES")
    print("=" * 70)
    
    baseline_r1 = SingleOccupantBaseline()
    results['models']['baseline_r1'] = baseline_r1.fit_predict_evaluate(df, 'Activity_R1')
    
    baseline_r2 = SingleOccupantBaseline()
    results['models']['baseline_r2'] = baseline_r2.fit_predict_evaluate(df, 'Activity_R2')
    
    # =========================================================================
    # 2. Multi-Resident Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: MULTI-RESIDENT MODEL")
    print("=" * 70)
    
    config = ModelConfig(
        use_cross_resident_features=True,
        use_conflict_features=True,
        use_lag_features=True,
        lag_window=5
    )
    
    multi_model = MultiResidentGLM(config)
    X_train, X_test, y_train, y_test = multi_model.prepare_data(df)
    multi_model.fit(X_train, y_train)
    multi_results = multi_model.evaluate(X_test, y_test)
    
    results['models']['multi_r1'] = multi_results['R1_Activity']
    results['models']['multi_r2'] = multi_results['R2_Activity']
    results['models']['multi_conflict'] = multi_results.get('Conflict')
    results['models']['multi_joint'] = multi_results['Joint']
    
    # =========================================================================
    # 3. Feature Importance
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    importance_df = multi_model.get_feature_importance()
    print("\nTop 15 Features:")
    print(importance_df.head(15).to_string())
    
    importance_df.to_csv(output_path / f"{house_name}_feature_importance.csv")
    
    # =========================================================================
    # 4. Cross-Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: CROSS-VALIDATION")
    print("=" * 70)
    
    cv_results = multi_model.cross_validate(X_train, y_train)
    results['cv_results'] = cv_results
    
    # =========================================================================
    # 5. Comparison Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: SINGLE VS MULTI-OCCUPANT")
    print("=" * 70)
    
    print(f"\n{'Model':<35} {'Accuracy':>10} {'Improvement':>12}")
    print("-" * 60)
    
    # R1 comparison
    baseline_acc_r1 = results['models']['baseline_r1'].accuracy
    multi_acc_r1 = results['models']['multi_r1'].accuracy
    improvement_r1 = (multi_acc_r1 - baseline_acc_r1) / baseline_acc_r1 * 100
    
    print(f"{'Baseline R1 (Single-Occupant)':<35} {baseline_acc_r1:>10.4f}")
    print(f"{'Multi-Resident R1':<35} {multi_acc_r1:>10.4f} {improvement_r1:>+11.2f}%")
    
    # R2 comparison
    baseline_acc_r2 = results['models']['baseline_r2'].accuracy
    multi_acc_r2 = results['models']['multi_r2'].accuracy
    improvement_r2 = (multi_acc_r2 - baseline_acc_r2) / baseline_acc_r2 * 100
    
    print(f"{'Baseline R2 (Single-Occupant)':<35} {baseline_acc_r2:>10.4f}")
    print(f"{'Multi-Resident R2':<35} {multi_acc_r2:>10.4f} {improvement_r2:>+11.2f}%")
    
    # Average
    avg_baseline = (baseline_acc_r1 + baseline_acc_r2) / 2
    avg_multi = (multi_acc_r1 + multi_acc_r2) / 2
    avg_improvement = (avg_multi - avg_baseline) / avg_baseline * 100
    
    print("-" * 60)
    print(f"{'Average Baseline':<35} {avg_baseline:>10.4f}")
    print(f"{'Average Multi-Resident':<35} {avg_multi:>10.4f} {avg_improvement:>+11.2f}%")
    
    results['comparison'] = {
        'baseline_avg': avg_baseline,
        'multi_avg': avg_multi,
        'improvement_percent': avg_improvement
    }
    
    # =========================================================================
    # 6. Save Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save model
    multi_model.save(str(output_path / f"{house_name}_multi_resident_glm.pkl"))
    
    # Save results summary
    results_summary = {
        'house': house_name,
        'total_samples': len(df),
        'baseline_r1_accuracy': baseline_acc_r1,
        'baseline_r2_accuracy': baseline_acc_r2,
        'multi_r1_accuracy': multi_acc_r1,
        'multi_r2_accuracy': multi_acc_r2,
        'improvement_r1_percent': improvement_r1,
        'improvement_r2_percent': improvement_r2,
        'average_improvement_percent': avg_improvement,
        'conflict_f1': results['models']['multi_conflict'].f1 if results['models']['multi_conflict'] else 0,
        'joint_exact_match': results['models']['multi_joint'].accuracy,
        'cv_r1_mean': np.mean(cv_results.get('R1_Activity', [0])),
        'cv_r2_mean': np.mean(cv_results.get('R2_Activity', [0]))
    }
    
    with open(output_path / f"{house_name}_results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("MULTI-RESIDENT GLM MODEL")
    print("=" * 70)
    print("\nUsage:")
    print("  from glm_multi_resident import run_experiment")
    print()
    print("  results = run_experiment(")
    print("      processed_data_path='path/to/processed_data.csv',")
    print("      output_dir='output/',")
    print("      house_name='house_a'")
    print("  )")
    print()
    print("Or use the MultiResidentGLM class directly:")
    print()
    print("  from glm_multi_resident import MultiResidentGLM, ModelConfig")
    print()
    print("  config = ModelConfig(use_lag_features=True, lag_window=5)")
    print("  model = MultiResidentGLM(config)")
    print("  X_train, X_test, y_train, y_test = model.prepare_data(df)")
    print("  model.fit(X_train, y_train)")
    print("  results = model.evaluate(X_test, y_test)")


if __name__ == "__main__":
    main()
