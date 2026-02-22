#!/usr/bin/env python3
"""
Multi-Resident Feature Engineering
===================================
Generates feature sets for multi-occupant smart home activity prediction.
Supports ablation studies by allowing selective feature group inclusion.

Feature Groups:
1. Temporal features (cyclical time encoding)
2. Cross-resident features (synchronization, same category, etc.)
3. Lag features (previous activities)
4. Conflict risk features
5. Spatial features (room zones from sensors)
6. FP-Growth pattern features
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MultiResidentFeatureEngineer:
    """
    Generate features for multi-occupant activity prediction with ablation support.
    """
    
    # Activity categories
    ACTIVITY_CATEGORIES = {
        'Rest': [11, 16],
        'Entertainment': [12, 17, 18, 23],
        'Work': [13],
        'Hygiene': [14, 15, 20, 21],
        'Eating': [3, 4, 5, 6, 7, 8, 10],
        'Household': [9, 19, 24, 27],
        'Social': [22, 25],
        'Other': [1, 2, 26]
    }
    
    # Quiet activities (for conflict detection)
    QUIET_ACTIVITIES = {11, 13, 16, 18, 22}  # Sleep, Study, Nap, Read, Phone
    
    # Noisy activities
    NOISY_ACTIVITIES = {12, 23}  # TV, Music
    
    # Sensor to room mapping (approximate)
    SENSOR_ROOMS = {
        'PhoneJack_A': 'living_room', 'PhoneJack_B': 'living_room',
        'PhoneJack_C': 'bedroom', 'PhoneJack_D': 'bedroom',
        'PhoneJack_E': 'kitchen', 'PhoneJack_F': 'bathroom',
        'IrProx_A': 'entrance', 'IrProx_B': 'living_room',
        'IrProx_C': 'bedroom', 'IrProx_D': 'kitchen',
        'Contact_A': 'entrance', 'Contact_B': 'bedroom',
        'Contact_C': 'bathroom', 'Contact_D': 'kitchen',
        'Contact_E': 'living_room',
        'ForceSnsr_A': 'living_room', 'ForceSnsr_B': 'bedroom',
        'ForceSnsr_C': 'kitchen', 'ForceSnsr_D': 'bathroom',
        'ForceSnsr_E': 'other'
    }
    
    def __init__(self, lag_window: int = 5):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        lag_window : int
            Number of previous timesteps to include as lag features
        """
        self.lag_window = lag_window
        self.scaler = StandardScaler()
        self.category_map = self._build_category_map()
        
    def _build_category_map(self) -> Dict[int, str]:
        """Build activity to category mapping."""
        cat_map = {}
        for cat, activities in self.ACTIVITY_CATEGORIES.items():
            for act in activities:
                cat_map[act] = cat
        return cat_map
    
    def generate_features(
        self,
        data: pd.DataFrame,
        include_temporal: bool = True,
        include_cross_resident: bool = True,
        include_lag: bool = True,
        include_conflict_risk: bool = True,
        include_spatial: bool = True,
        include_fpgrowth: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate feature matrix with selective feature groups.
        
        Parameters:
        -----------
        data : pd.DataFrame
            ARAS data with sensor columns, Activity_R1, Activity_R2
        include_* : bool
            Whether to include each feature group
            
        Returns:
        --------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y_r1 : np.ndarray
            Resident 1 activity labels
        y_r2 : np.ndarray
            Resident 2 activity labels
        y_conflict : np.ndarray
            Binary conflict labels
        """
        features = []
        feature_names = []
        
        # Base sensor features (always included)
        sensor_cols = [c for c in data.columns if c.startswith(('PhoneJack', 'IrProx', 'Contact', 'ForceSnsr'))]
        sensor_features = data[sensor_cols].values
        features.append(sensor_features)
        feature_names.extend(sensor_cols)
        
        # 1. TEMPORAL FEATURES
        if include_temporal:
            temporal_feats, temporal_names = self._generate_temporal_features(data)
            features.append(temporal_feats)
            feature_names.extend(temporal_names)
        
        # 2. CROSS-RESIDENT FEATURES
        if include_cross_resident:
            cross_feats, cross_names = self._generate_cross_resident_features(data)
            features.append(cross_feats)
            feature_names.extend(cross_names)
        
        # 3. LAG FEATURES
        if include_lag:
            lag_feats, lag_names = self._generate_lag_features(data)
            features.append(lag_feats)
            feature_names.extend(lag_names)
        
        # 4. CONFLICT RISK FEATURES
        if include_conflict_risk:
            conflict_feats, conflict_names = self._generate_conflict_features(data)
            features.append(conflict_feats)
            feature_names.extend(conflict_names)
        
        # 5. SPATIAL FEATURES
        if include_spatial:
            spatial_feats, spatial_names = self._generate_spatial_features(data)
            features.append(spatial_feats)
            feature_names.extend(spatial_names)
        
        # 6. FP-GROWTH PATTERN FEATURES
        if include_fpgrowth:
            fpgrowth_feats, fpgrowth_names = self._generate_fpgrowth_features(data)
            features.append(fpgrowth_feats)
            feature_names.extend(fpgrowth_names)
        
        # Combine all features
        X = np.hstack(features)
        
        # Handle NaN values from lag features
        X = np.nan_to_num(X, nan=0.0)
        
        # Get labels
        y_r1 = data['Activity_R1'].values
        y_r2 = data['Activity_R2'].values
        
        # Compute conflict labels
        y_conflict = self._compute_conflicts(data).values
        
        # Store feature names for interpretation
        self.feature_names_ = feature_names
        
        return X, y_r1, y_r2, y_conflict
    
    def _generate_temporal_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Generate temporal features using cyclical encoding.
        
        Features:
        - Hour sin/cos encoding
        - Day of week encoding
        - Time of day category (night/morning/afternoon/evening)
        - Weekend indicator
        """
        features = []
        names = []
        
        # Get hour from Second_of_Day
        hour = data['Second_of_Day'] // 3600
        
        # Cyclical hour encoding (Equation 1 in paper)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        features.extend([hour_sin.values.reshape(-1, 1), hour_cos.values.reshape(-1, 1)])
        names.extend(['Hour_sin', 'Hour_cos'])
        
        # Time of day categories
        tod_night = ((hour >= 0) & (hour < 6)).astype(int)
        tod_morning = ((hour >= 6) & (hour < 12)).astype(int)
        tod_afternoon = ((hour >= 12) & (hour < 18)).astype(int)
        tod_evening = ((hour >= 18) & (hour < 24)).astype(int)
        
        features.extend([
            tod_night.values.reshape(-1, 1),
            tod_morning.values.reshape(-1, 1),
            tod_afternoon.values.reshape(-1, 1),
            tod_evening.values.reshape(-1, 1)
        ])
        names.extend(['TOD_Night', 'TOD_Morning', 'TOD_Afternoon', 'TOD_Evening'])
        
        # Day of week (assuming Day column represents consecutive days)
        day_of_week = (data['Day'] - 1) % 7
        for dow in range(7):
            dow_feat = (day_of_week == dow).astype(int)
            features.append(dow_feat.values.reshape(-1, 1))
            names.append(f'DOW_{dow}')
        
        # Weekend indicator
        is_weekend = ((day_of_week == 5) | (day_of_week == 6)).astype(int)
        features.append(is_weekend.values.reshape(-1, 1))
        names.append('Is_Weekend')
        
        return np.hstack(features), names
    
    def _generate_cross_resident_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Generate cross-resident features (Equations 2-4 in paper).
        
        Features:
        - IsSynchronized: Both residents doing same activity
        - SameCategory: Both in same activity category
        - BothHome: Neither is "Going Out"
        - OneAway: Exactly one resident is away
        - Activity category for each resident
        """
        features = []
        names = []
        
        r1 = data['Activity_R1']
        r2 = data['Activity_R2']
        
        # IsSynchronized (Equation 2)
        is_sync = (r1 == r2).astype(int)
        features.append(is_sync.values.reshape(-1, 1))
        names.append('IsSynchronized')
        
        # SameCategory (Equation 3)
        r1_cat = r1.map(self.category_map)
        r2_cat = r2.map(self.category_map)
        same_cat = (r1_cat == r2_cat).astype(int)
        features.append(same_cat.values.reshape(-1, 1))
        names.append('SameCategory')
        
        # BothHome (Equation 4) - activity 2 is "Going Out"
        both_home = ((r1 != 2) & (r2 != 2)).astype(int)
        features.append(both_home.values.reshape(-1, 1))
        names.append('BothHome')
        
        # OneAway
        one_away = (((r1 == 2) & (r2 != 2)) | ((r1 != 2) & (r2 == 2))).astype(int)
        features.append(one_away.values.reshape(-1, 1))
        names.append('OneAway')
        
        # Activity categories as one-hot
        for cat in self.ACTIVITY_CATEGORIES.keys():
            r1_in_cat = (r1_cat == cat).astype(int)
            r2_in_cat = (r2_cat == cat).astype(int)
            features.extend([
                r1_in_cat.values.reshape(-1, 1),
                r2_in_cat.values.reshape(-1, 1)
            ])
            names.extend([f'R1_Cat_{cat}', f'R2_Cat_{cat}'])
        
        return np.hstack(features), names
    
    def _generate_lag_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Generate lag features (Equation 5 in paper).
        
        WARNING: These features can cause data leakage since activities
        persist for long periods at 1Hz sampling. The ablation study
        will show their impact.
        """
        features = []
        names = []
        
        r1 = data['Activity_R1']
        r2 = data['Activity_R2']
        
        # Create lag features for both residents
        for lag in range(1, self.lag_window + 1):
            # R1 previous activity
            r1_lag = r1.shift(lag)
            features.append(r1_lag.values.reshape(-1, 1))
            names.append(f'R1_Lag_{lag}')
            
            # R2 previous activity
            r2_lag = r2.shift(lag)
            features.append(r2_lag.values.reshape(-1, 1))
            names.append(f'R2_Lag_{lag}')
        
        return np.hstack(features), names
    
    def _generate_conflict_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Generate conflict risk features (Equation 6 in paper).
        """
        features = []
        names = []
        
        r1 = data['Activity_R1']
        r2 = data['Activity_R2']
        
        # TV Conflict Risk: One watching TV while other needs quiet
        tv_risk = (
            ((r1 == 12) & r2.isin(self.QUIET_ACTIVITIES)) |
            ((r2 == 12) & r1.isin(self.QUIET_ACTIVITIES))
        ).astype(int)
        features.append(tv_risk.values.reshape(-1, 1))
        names.append('TVConflictRisk')
        
        # Music Conflict Risk
        music_risk = (
            ((r1 == 23) & r2.isin(self.QUIET_ACTIVITIES)) |
            ((r2 == 23) & r1.isin(self.QUIET_ACTIVITIES))
        ).astype(int)
        features.append(music_risk.values.reshape(-1, 1))
        names.append('MusicConflictRisk')
        
        # Bathroom Resource Risk
        bathroom_risk = (
            (r1.isin([14, 15])) & (r2.isin([14, 15]))
        ).astype(int)
        features.append(bathroom_risk.values.reshape(-1, 1))
        names.append('BathroomConflictRisk')
        
        # General conflict indicator
        has_conflict = self._compute_conflicts(data).astype(int)
        features.append(has_conflict.values.reshape(-1, 1))
        names.append('HasConflict')
        
        # No conflict indicator (useful for some models)
        no_conflict = (~has_conflict.astype(bool)).astype(int)
        features.append(no_conflict.values.reshape(-1, 1))
        names.append('Conflict_none')
        
        return np.hstack(features), names
    
    def _generate_spatial_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Generate spatial features based on sensor activations.
        """
        features = []
        names = []
        
        # Get sensor columns
        sensor_cols = [c for c in data.columns if c.startswith(('PhoneJack', 'IrProx', 'Contact', 'ForceSnsr'))]
        
        # Room activity indicators based on sensor groupings
        rooms = ['living_room', 'bedroom', 'kitchen', 'bathroom', 'entrance', 'other']
        
        for room in rooms:
            room_sensors = [s for s, r in self.SENSOR_ROOMS.items() if r == room and s in sensor_cols]
            if room_sensors:
                room_activity = data[room_sensors].max(axis=1)
                features.append(room_activity.values.reshape(-1, 1))
                names.append(f'Zone_{room}')
        
        # Total sensor activations
        total_active = data[sensor_cols].sum(axis=1)
        features.append(total_active.values.reshape(-1, 1))
        names.append('Total_Sensors_Active')
        
        return np.hstack(features), names
    
    def _generate_fpgrowth_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Generate FP-Growth derived pattern features.
        
        Note: Full FP-Growth implementation would require mlxtend.
        This generates simplified pattern-based features.
        """
        features = []
        names = []
        
        r1 = data['Activity_R1']
        r2 = data['Activity_R2']
        
        # Common activity pairs (simplified pattern features)
        common_pairs = [
            (11, 11),  # Both sleeping
            (12, 12),  # Both watching TV
            (6, 6),    # Both having lunch
            (8, 8),    # Both having dinner
            (17, 17),  # Both using internet
        ]
        
        for a1, a2 in common_pairs:
            pair_active = ((r1 == a1) & (r2 == a2)).astype(int)
            features.append(pair_active.values.reshape(-1, 1))
            names.append(f'Pattern_R1_{a1}_R2_{a2}')
        
        # High-confidence patterns (meal times)
        hour = data['Second_of_Day'] // 3600
        
        # Breakfast pattern (6-9 AM)
        breakfast_time = ((hour >= 6) & (hour < 9)).astype(int)
        features.append(breakfast_time.values.reshape(-1, 1))
        names.append('Pattern_Breakfast_Time')
        
        # Lunch pattern (12-14)
        lunch_time = ((hour >= 12) & (hour < 14)).astype(int)
        features.append(lunch_time.values.reshape(-1, 1))
        names.append('Pattern_Lunch_Time')
        
        # Dinner pattern (18-21)
        dinner_time = ((hour >= 18) & (hour < 21)).astype(int)
        features.append(dinner_time.values.reshape(-1, 1))
        names.append('Pattern_Dinner_Time')
        
        # Sleep pattern (night hours)
        sleep_time = ((hour >= 22) | (hour < 6)).astype(int)
        features.append(sleep_time.values.reshape(-1, 1))
        names.append('Pattern_Sleep_Time')
        
        return np.hstack(features), names
    
    def _compute_conflicts(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute binary conflict labels.
        """
        r1 = data['Activity_R1']
        r2 = data['Activity_R2']
        
        # Noise conflicts
        noise_conflict = (
            (r1.isin(self.NOISY_ACTIVITIES) & r2.isin(self.QUIET_ACTIVITIES)) |
            (r2.isin(self.NOISY_ACTIVITIES) & r1.isin(self.QUIET_ACTIVITIES))
        )
        
        # Resource conflicts (bathroom)
        resource_conflict = (
            (r1.isin([14, 15]) & r2.isin([14, 15]))
        )
        
        return (noise_conflict | resource_conflict)
    
    def get_feature_importance_analysis(self, model, X: np.ndarray) -> pd.DataFrame:
        """
        Analyze feature importance from a trained model.
        
        Parameters:
        -----------
        model : sklearn model with coef_ or feature_importances_
        X : feature matrix used for training
        
        Returns:
        --------
        DataFrame with feature names and importance scores
        """
        if hasattr(model, 'coef_'):
            # For linear models (GLM)
            importances = np.abs(model.coef_).mean(axis=0)
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
        else:
            raise ValueError("Model must have coef_ or feature_importances_")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names_,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df


if __name__ == '__main__':
    # Test with synthetic data
    from preprocessing import SyntheticArasGenerator
    
    print("Testing feature engineering...")
    generator = SyntheticArasGenerator(seed=42)
    data = generator.generate(num_days=5)
    
    engineer = MultiResidentFeatureEngineer()
    
    # Test full feature set
    X_full, y_r1, y_r2, y_conflict = engineer.generate_features(data)
    print(f"Full features: {X_full.shape}")
    
    # Test without lag features
    X_no_lag, _, _, _ = engineer.generate_features(data, include_lag=False)
    print(f"Without lag: {X_no_lag.shape}")
    
    # Test sensors only
    X_sensors, _, _, _ = engineer.generate_features(
        data,
        include_temporal=False,
        include_cross_resident=False,
        include_lag=False,
        include_conflict_risk=False,
        include_spatial=False,
        include_fpgrowth=False
    )
    print(f"Sensors only: {X_sensors.shape}")
    
    print(f"Conflict rate: {y_conflict.mean():.2%}")
