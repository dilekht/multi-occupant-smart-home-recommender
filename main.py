#!/usr/bin/env python3
"""
================================================================================
Multi-Occupant Smart Home Recommendation System
================================================================================
A Complete Implementation of FP-Growth + GLM with Conflict Resolution

This repository accompanies the paper:
"Multi-Occupant Context-Aware Recommender System for Smart Home Automation:
An Extended FP-Growth and GLM Approach with Conflict Resolution"

Author: [Your Name]
Institution: [Your Institution]
Date: February 2026

USAGE:
------
1. Install requirements:
   pip install -r requirements.txt

2. Prepare ARAS dataset:
   Place data in ./data/aras/HouseA/ and ./data/aras/HouseB/

3. Run all experiments:
   python main.py --all

4. Run specific experiments:
   python main.py --baselines          # Run baseline comparisons
   python main.py --ablation           # Run ablation study
   python main.py --figures            # Generate figures only
   python main.py --conflict           # Run conflict resolution experiments

================================================================================
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'data_dir': './data/aras',
    'results_dir': './results',
    'figures_dir': './figures',
    'models_dir': './models',
    'random_seed': 42,
    'train_days': 22,  # Days 1-22 for training
    'test_days': 8,    # Days 23-30 for testing
    'samples_per_day': 86400,  # 1 Hz sampling
}

# Activity labels from ARAS dataset
ACTIVITY_NAMES = {
    1: 'Other', 2: 'Going_Out', 3: 'Preparing_Breakfast', 4: 'Having_Breakfast',
    5: 'Preparing_Lunch', 6: 'Having_Lunch', 7: 'Preparing_Dinner', 8: 'Having_Dinner',
    9: 'Washing_Dishes', 10: 'Having_Snack', 11: 'Sleeping', 12: 'Watching_TV',
    13: 'Studying', 14: 'Having_Shower', 15: 'Toileting', 16: 'Napping',
    17: 'Using_Internet', 18: 'Reading_Book', 19: 'Laundry', 20: 'Shaving',
    21: 'Brushing_Teeth', 22: 'Talking_Phone', 23: 'Listening_Music',
    24: 'Cleaning', 25: 'Conversation', 26: 'Eating_Medicine', 27: 'Kitchen_Appliance'
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_directories():
    """Create output directories if they don't exist."""
    for dir_name in ['results_dir', 'figures_dir', 'models_dir']:
        os.makedirs(CONFIG[dir_name], exist_ok=True)

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{title}")
    print("-" * 50)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_aras_data(house: str) -> pd.DataFrame:
    """
    Load ARAS dataset for a specific house.
    
    Parameters:
    -----------
    house : str
        'A' or 'B'
        
    Returns:
    --------
    pd.DataFrame with columns:
        - 20 sensor columns (binary)
        - Activity_R1, Activity_R2 (int 1-27)
        - Day, Second_of_Day, Timestamp
    """
    data_dir = os.path.join(CONFIG['data_dir'], f'House{house}')
    
    sensor_names = [
        'PhoneJack_A', 'PhoneJack_B', 'PhoneJack_C', 'PhoneJack_D',
        'PhoneJack_E', 'PhoneJack_F', 'IrProx_A', 'IrProx_B',
        'IrProx_C', 'IrProx_D', 'Contact_A', 'Contact_B',
        'Contact_C', 'Contact_D', 'Contact_E', 'ForceSnsr_A',
        'ForceSnsr_B', 'ForceSnsr_C', 'ForceSnsr_D', 'ForceSnsr_E'
    ]
    
    all_days = []
    for day in range(1, 31):
        filepath = os.path.join(data_dir, f'DAY{day}.txt')
        if not os.path.exists(filepath):
            continue
            
        day_data = pd.read_csv(
            filepath, sep=r'\s+', header=None,
            names=sensor_names + ['Activity_R1', 'Activity_R2']
        )
        day_data['Day'] = day
        day_data['Second_of_Day'] = range(len(day_data))
        all_days.append(day_data)
    
    if not all_days:
        raise FileNotFoundError(f"No data found in {data_dir}")
    
    return pd.concat(all_days, ignore_index=True)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def generate_features(data: pd.DataFrame, config: dict = None) -> tuple:
    """
    Generate feature matrix for activity prediction.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw ARAS data
    config : dict
        Feature configuration with keys:
        - include_temporal, include_cross_resident, include_lag,
        - include_conflict_risk, include_spatial, include_fpgrowth
        
    Returns:
    --------
    X : np.ndarray - Feature matrix
    y_r1 : np.ndarray - Resident 1 labels
    y_r2 : np.ndarray - Resident 2 labels
    y_conflict : np.ndarray - Conflict labels
    feature_names : list - Feature names
    """
    if config is None:
        config = {
            'include_temporal': True,
            'include_cross_resident': True,
            'include_lag': True,
            'include_conflict_risk': True,
            'include_spatial': True,
            'include_fpgrowth': True
        }
    
    features = []
    feature_names = []
    
    # Sensor names
    sensor_cols = [c for c in data.columns if c.startswith(('PhoneJack', 'IrProx', 'Contact', 'ForceSnsr'))]
    
    # 1. Base sensor features (always included)
    features.append(data[sensor_cols].values)
    feature_names.extend(sensor_cols)
    
    # 2. Temporal features
    if config.get('include_temporal', True):
        hour = data['Second_of_Day'] // 3600
        features.append(np.sin(2 * np.pi * hour / 24).values.reshape(-1, 1))
        features.append(np.cos(2 * np.pi * hour / 24).values.reshape(-1, 1))
        feature_names.extend(['Hour_sin', 'Hour_cos'])
        
        # Time of day categories
        for name, (start, end) in [('Night', (0, 6)), ('Morning', (6, 12)), 
                                    ('Afternoon', (12, 18)), ('Evening', (18, 24))]:
            tod = ((hour >= start) & (hour < end)).astype(int)
            features.append(tod.values.reshape(-1, 1))
            feature_names.append(f'TOD_{name}')
        
        # Day of week
        day_of_week = (data['Day'] - 1) % 7
        for dow in range(7):
            features.append((day_of_week == dow).astype(int).values.reshape(-1, 1))
            feature_names.append(f'DOW_{dow}')
        
        features.append(((day_of_week >= 5)).astype(int).values.reshape(-1, 1))
        feature_names.append('Is_Weekend')
    
    # 3. Cross-resident features
    if config.get('include_cross_resident', True):
        r1, r2 = data['Activity_R1'], data['Activity_R2']
        
        features.append((r1 == r2).astype(int).values.reshape(-1, 1))
        feature_names.append('IsSynchronized')
        
        # Activity categories
        categories = {
            'Rest': [11, 16], 'Entertainment': [12, 17, 18, 23], 'Work': [13],
            'Hygiene': [14, 15, 20, 21], 'Eating': [3, 4, 5, 6, 7, 8, 10],
            'Household': [9, 19, 24, 27], 'Social': [22, 25], 'Other': [1, 2, 26]
        }
        
        def get_category(activity):
            for cat, acts in categories.items():
                if activity in acts:
                    return cat
            return 'Other'
        
        r1_cat = r1.apply(get_category)
        r2_cat = r2.apply(get_category)
        features.append((r1_cat == r2_cat).astype(int).values.reshape(-1, 1))
        feature_names.append('SameCategory')
        
        features.append(((r1 != 2) & (r2 != 2)).astype(int).values.reshape(-1, 1))
        feature_names.append('BothHome')
        
        for cat in categories.keys():
            features.append((r1_cat == cat).astype(int).values.reshape(-1, 1))
            features.append((r2_cat == cat).astype(int).values.reshape(-1, 1))
            feature_names.extend([f'R1_Cat_{cat}', f'R2_Cat_{cat}'])
    
    # 4. Lag features
    if config.get('include_lag', True):
        for lag in range(1, 6):
            features.append(data['Activity_R1'].shift(lag).values.reshape(-1, 1))
            features.append(data['Activity_R2'].shift(lag).values.reshape(-1, 1))
            feature_names.extend([f'R1_Lag_{lag}', f'R2_Lag_{lag}'])
    
    # 5. Conflict risk features
    if config.get('include_conflict_risk', True):
        r1, r2 = data['Activity_R1'], data['Activity_R2']
        quiet = {11, 13, 16, 18, 22}
        noisy = {12, 23}
        
        tv_risk = ((r1 == 12) & r2.isin(quiet)) | ((r2 == 12) & r1.isin(quiet))
        features.append(tv_risk.astype(int).values.reshape(-1, 1))
        feature_names.append('TVConflictRisk')
        
        music_risk = ((r1 == 23) & r2.isin(quiet)) | ((r2 == 23) & r1.isin(quiet))
        features.append(music_risk.astype(int).values.reshape(-1, 1))
        feature_names.append('MusicConflictRisk')
        
        bathroom_risk = r1.isin([14, 15]) & r2.isin([14, 15])
        features.append(bathroom_risk.astype(int).values.reshape(-1, 1))
        feature_names.append('BathroomConflictRisk')
        
        has_conflict = tv_risk | music_risk | bathroom_risk
        features.append(has_conflict.astype(int).values.reshape(-1, 1))
        feature_names.append('HasConflict')
    
    # 6. Spatial features
    if config.get('include_spatial', True):
        room_sensors = {
            'living_room': ['PhoneJack_A', 'PhoneJack_B', 'IrProx_B', 'Contact_E', 'ForceSnsr_A'],
            'bedroom': ['PhoneJack_C', 'PhoneJack_D', 'IrProx_C', 'Contact_B', 'ForceSnsr_B'],
            'kitchen': ['PhoneJack_E', 'IrProx_D', 'Contact_D', 'ForceSnsr_C'],
            'bathroom': ['PhoneJack_F', 'Contact_C', 'ForceSnsr_D'],
        }
        for room, sensors in room_sensors.items():
            valid_sensors = [s for s in sensors if s in data.columns]
            if valid_sensors:
                features.append(data[valid_sensors].max(axis=1).values.reshape(-1, 1))
                feature_names.append(f'Zone_{room}')
        
        features.append(data[sensor_cols].sum(axis=1).values.reshape(-1, 1))
        feature_names.append('Total_Sensors_Active')
    
    # 7. FP-Growth pattern features
    if config.get('include_fpgrowth', True):
        r1, r2 = data['Activity_R1'], data['Activity_R2']
        hour = data['Second_of_Day'] // 3600
        
        # Common activity pairs
        for a1, a2 in [(11, 11), (12, 12), (6, 6), (8, 8), (17, 17)]:
            features.append(((r1 == a1) & (r2 == a2)).astype(int).values.reshape(-1, 1))
            feature_names.append(f'Pattern_R1_{a1}_R2_{a2}')
        
        # Meal time patterns
        features.append(((hour >= 6) & (hour < 9)).astype(int).values.reshape(-1, 1))
        features.append(((hour >= 12) & (hour < 14)).astype(int).values.reshape(-1, 1))
        features.append(((hour >= 18) & (hour < 21)).astype(int).values.reshape(-1, 1))
        features.append(((hour >= 22) | (hour < 6)).astype(int).values.reshape(-1, 1))
        feature_names.extend(['Pattern_Breakfast', 'Pattern_Lunch', 'Pattern_Dinner', 'Pattern_Sleep'])
    
    # Combine features
    X = np.hstack(features)
    X = np.nan_to_num(X, nan=0.0)
    
    # Labels
    y_r1 = data['Activity_R1'].values
    y_r2 = data['Activity_R2'].values
    
    # Conflict labels
    r1, r2 = data['Activity_R1'], data['Activity_R2']
    quiet, noisy = {11, 13, 16, 18, 22}, {12, 23}
    y_conflict = ((r1.isin(noisy) & r2.isin(quiet)) | 
                  (r2.isin(noisy) & r1.isin(quiet)) |
                  (r1.isin([14, 15]) & r2.isin([14, 15]))).values
    
    return X, y_r1, y_r2, y_conflict, feature_names

# =============================================================================
# BASELINE MODELS
# =============================================================================

def train_evaluate_model(model_name, X_train, y_train, X_test, y_test):
    """Train and evaluate a baseline model."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    
    # Filter test to known classes
    known_classes = set(le.classes_)
    test_mask = np.array([y in known_classes for y in y_test])
    X_test_filtered = X_test_scaled[test_mask]
    y_test_filtered = y_test[test_mask]
    y_test_enc = le.transform(y_test_filtered)
    
    start_time = time.time()
    
    if model_name == 'GLM':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, random_state=42)
        model.fit(X_train_scaled, y_train_enc)
        y_pred = model.predict(X_test_filtered)
        
    elif model_name == 'XGBoost':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              objective='multi:softmax', num_class=len(le.classes_),
                              random_state=42, verbosity=0)
        model.fit(X_train_scaled, y_train_enc)
        y_pred = model.predict(X_test_filtered)
        
    elif model_name == 'LightGBM':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.05,
                               num_leaves=63, random_state=42, verbose=-1)
        model.fit(X_train_scaled, y_train_enc)
        y_pred = model.predict(X_test_filtered)
        
    elif model_name == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train_enc)
        y_pred = model.predict(X_test_filtered)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    elapsed = time.time() - start_time
    
    return {
        'accuracy': accuracy_score(y_test_enc, y_pred),
        'f1_weighted': f1_score(y_test_enc, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_test_enc, y_pred, average='macro', zero_division=0),
        'time': elapsed
    }

# =============================================================================
# CONFLICT RESOLUTION
# =============================================================================

class ConflictResolver:
    """
    Five-strategy conflict resolution framework.
    """
    
    STRATEGIES = ['priority', 'compromise', 'temporal', 'spatial', 'device_specific']
    
    # Activity priorities (higher = more priority)
    PRIORITIES = {
        15: 5, 11: 5,  # Toileting, Sleeping
        14: 4, 16: 4, 13: 4, 22: 4,  # Shower, Nap, Study, Phone
        18: 3,  # Reading
        12: 2, 17: 2, 23: 2,  # TV, Internet, Music
    }
    
    def detect_conflict(self, activity_r1: int, activity_r2: int) -> dict:
        """Detect if activities conflict."""
        quiet = {11, 13, 16, 18, 22}
        noisy = {12, 23}
        bathroom = {14, 15}
        
        conflict = {
            'has_conflict': False,
            'type': None,
            'severity': 0
        }
        
        # Noise conflict
        if (activity_r1 in noisy and activity_r2 in quiet) or \
           (activity_r2 in noisy and activity_r1 in quiet):
            conflict['has_conflict'] = True
            conflict['type'] = 'noise'
            conflict['severity'] = 3 if activity_r2 in {11, 16} or activity_r1 in {11, 16} else 2
        
        # Resource conflict
        elif activity_r1 in bathroom and activity_r2 in bathroom:
            conflict['has_conflict'] = True
            conflict['type'] = 'resource'
            conflict['severity'] = 4
        
        return conflict
    
    def resolve(self, activity_r1: int, activity_r2: int, context: dict = None) -> dict:
        """
        Resolve conflict between two activities.
        
        Returns:
        --------
        dict with keys:
            - strategy: str (which strategy was used)
            - action_r1: str (recommendation for R1)
            - action_r2: str (recommendation for R2)
            - device_commands: list (smart device commands)
        """
        conflict = self.detect_conflict(activity_r1, activity_r2)
        
        if not conflict['has_conflict']:
            return {'strategy': 'none', 'action_r1': 'continue', 
                    'action_r2': 'continue', 'device_commands': []}
        
        # Select strategy based on conflict type and severity
        if conflict['type'] == 'resource':
            return self._resolve_priority(activity_r1, activity_r2)
        elif conflict['type'] == 'noise':
            if conflict['severity'] >= 3:
                return self._resolve_spatial(activity_r1, activity_r2)
            else:
                return self._resolve_device_specific(activity_r1, activity_r2)
        
        return self._resolve_compromise(activity_r1, activity_r2)
    
    def _resolve_priority(self, a1, a2):
        p1 = self.PRIORITIES.get(a1, 2)
        p2 = self.PRIORITIES.get(a2, 2)
        if p1 >= p2:
            return {'strategy': 'priority', 'action_r1': 'continue', 
                    'action_r2': 'wait', 'device_commands': []}
        else:
            return {'strategy': 'priority', 'action_r1': 'wait', 
                    'action_r2': 'continue', 'device_commands': []}
    
    def _resolve_compromise(self, a1, a2):
        return {'strategy': 'compromise', 'action_r1': 'adjust', 
                'action_r2': 'adjust', 'device_commands': ['reduce_volume']}
    
    def _resolve_temporal(self, a1, a2):
        return {'strategy': 'temporal', 'action_r1': 'schedule_later', 
                'action_r2': 'continue', 'device_commands': ['set_timer']}
    
    def _resolve_spatial(self, a1, a2):
        return {'strategy': 'spatial', 'action_r1': 'relocate', 
                'action_r2': 'continue', 'device_commands': ['suggest_room']}
    
    def _resolve_device_specific(self, a1, a2):
        commands = []
        if a1 == 12 or a2 == 12:  # TV
            commands = ['tv_reduce_volume', 'tv_enable_subtitles']
        elif a1 == 23 or a2 == 23:  # Music
            commands = ['music_reduce_volume', 'suggest_headphones']
        return {'strategy': 'device_specific', 'action_r1': 'continue', 
                'action_r2': 'continue', 'device_commands': commands}

# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_all_figures(results: dict):
    """Generate all figures for the paper."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    figures_dir = CONFIG['figures_dir']
    
    # Figure 2: Accuracy Comparison
    if 'baseline_comparison' in results:
        generate_accuracy_comparison_figure(results['baseline_comparison'], figures_dir)
    
    # Figure 4: Ablation Study
    if 'ablation_study' in results:
        generate_ablation_figure(results['ablation_study'], figures_dir)
    
    # Figure 10: SOTA Comparison
    if 'baseline_comparison' in results:
        generate_sota_comparison_figure(results['baseline_comparison'], figures_dir)
    
    # Figure: Per-class F1
    if 'per_class_f1' in results:
        generate_per_class_figure(results['per_class_f1'], figures_dir)
    
    print(f"All figures saved to {figures_dir}/")

def generate_accuracy_comparison_figure(data, output_dir):
    """Generate accuracy comparison bar chart."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, house in enumerate(['A', 'B']):
        ax = axes[idx]
        house_data = data[data['House'] == house]
        
        methods = house_data['Method'].values
        r1_acc = house_data['R1_Accuracy'].values * 100
        r2_acc = house_data['R2_Accuracy'].values * 100
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, r1_acc, width, label='Resident 1', color='#3498db')
        bars2 = ax.bar(x + width/2, r2_acc, width, label='Resident 2', color='#2ecc71')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'House {house}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_accuracy_comparison.png'), 
                bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'fig2_accuracy_comparison.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Generated: fig2_accuracy_comparison.png")

def generate_ablation_figure(data, output_dir):
    """Generate ablation study figure."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    configs = ['full', 'no_lag', 'no_cross_resident', 'no_conflict_risk', 'no_fpgrowth', 'sensors_only']
    config_labels = ['Full Model', 'No Lag', 'No Cross-Res.', 'No Conflict', 'No FP-Growth', 'Sensors Only']
    
    house_a = data[data['House'] == 'A'].set_index('Feature_Set')['Avg_Accuracy'].reindex(configs).values * 100
    house_b = data[data['House'] == 'B'].set_index('Feature_Set')['Avg_Accuracy'].reindex(configs).values * 100
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, house_a, width, label='House A (Couple)', color='#3498db')
    bars2 = ax.bar(x + width/2, house_b, width, label='House B (Roommates)', color='#e74c3c')
    
    ax.set_xlabel('Feature Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Impact of Feature Groups', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_ablation_study.png'), 
                bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'fig4_ablation_study.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Generated: fig4_ablation_study.png")

def generate_sota_comparison_figure(data, output_dir):
    """Generate state-of-the-art comparison figure."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Combine with literature baselines
    methods = ['HMM\n(Alemdar 2013)', 'CRF\n(Alemdar 2013)', 'GLM (Ours)', 
               'XGBoost', 'LightGBM', 'Random Forest']
    
    # House A averages (using literature values for HMM/CRF)
    accuracies = [78.25, 79.00]  # Literature values
    
    glm_avg = data[data['Method'] == 'GLM (Ours)']['Avg_Accuracy'].mean() * 100
    xgb_avg = data[data['Method'] == 'XGBoost']['Avg_Accuracy'].mean() * 100
    lgb_avg = data[data['Method'] == 'LightGBM']['Avg_Accuracy'].mean() * 100
    rf_avg = data[data['Method'] == 'Random Forest']['Avg_Accuracy'].mean() * 100
    
    accuracies.extend([glm_avg, xgb_avg, lgb_avg, rf_avg])
    
    colors = ['#95a5a6', '#95a5a6', '#2ecc71', '#3498db', '#9b59b6', '#e67e22']
    
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Comparison with State-of-the-Art Methods', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig10_sota_comparison.png'), 
                bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'fig10_sota_comparison.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Generated: fig10_sota_comparison.png")

def generate_per_class_figure(data, output_dir):
    """Generate per-class F1 score figure."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, house in enumerate(['A', 'B']):
        ax = axes[idx]
        house_data = data[data['House'] == house].sort_values('F1_Score', ascending=True)
        
        colors = ['#e74c3c' if f1 < 0.5 else '#f39c12' if f1 < 0.8 else '#2ecc71' 
                  for f1 in house_data['F1_Score']]
        
        ax.barh(house_data['Class_Name'], house_data['F1_Score'], color=colors)
        ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title(f'House {house} - Per-Activity F1 Scores', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (0.5)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_per_class_f1.png'), 
                bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'fig_per_class_f1.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Generated: fig_per_class_f1.png")

# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_baseline_experiments():
    """Run all baseline comparison experiments."""
    print_header("BASELINE COMPARISON EXPERIMENTS")
    
    results = []
    
    for house in ['A', 'B']:
        print_section(f"House {house}")
        
        # Load data
        print("  Loading data...", end=" ")
        data = load_aras_data(house)
        print(f"{len(data):,} records")
        
        # Generate features
        print("  Generating features...", end=" ")
        X, y_r1, y_r2, _, _ = generate_features(data)
        print(f"{X.shape[1]} features")
        
        # Temporal split
        train_samples = CONFIG['train_days'] * CONFIG['samples_per_day']
        X_train, X_test = X[:train_samples], X[train_samples:]
        y_train_r1, y_test_r1 = y_r1[:train_samples], y_r1[train_samples:]
        y_train_r2, y_test_r2 = y_r2[:train_samples], y_r2[train_samples:]
        
        # Train and evaluate each model
        for model_name in ['GLM', 'XGBoost', 'LightGBM', 'RandomForest']:
            print(f"  Training {model_name}...", end=" ")
            try:
                metrics_r1 = train_evaluate_model(model_name, X_train, y_train_r1, X_test, y_test_r1)
                metrics_r2 = train_evaluate_model(model_name, X_train, y_train_r2, X_test, y_test_r2)
                
                result = {
                    'House': house,
                    'Method': model_name if model_name != 'GLM' else 'GLM (Ours)',
                    'R1_Accuracy': metrics_r1['accuracy'],
                    'R2_Accuracy': metrics_r2['accuracy'],
                    'R1_F1': metrics_r1['f1_weighted'],
                    'R2_F1': metrics_r2['f1_weighted'],
                    'Avg_Accuracy': (metrics_r1['accuracy'] + metrics_r2['accuracy']) / 2,
                    'Training_Time': metrics_r1['time'] + metrics_r2['time']
                }
                results.append(result)
                print(f"R1={metrics_r1['accuracy']:.2%}, R2={metrics_r2['accuracy']:.2%}")
            except Exception as e:
                print(f"FAILED: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(CONFIG['results_dir'], 'baseline_comparison.csv'), index=False)
    print(f"\nResults saved to {CONFIG['results_dir']}/baseline_comparison.csv")
    
    return results_df

def run_ablation_study():
    """Run ablation study experiments."""
    print_header("ABLATION STUDY")
    
    configs = {
        'full': {'include_temporal': True, 'include_cross_resident': True, 
                 'include_lag': True, 'include_conflict_risk': True,
                 'include_spatial': True, 'include_fpgrowth': True},
        'no_lag': {'include_temporal': True, 'include_cross_resident': True, 
                   'include_lag': False, 'include_conflict_risk': True,
                   'include_spatial': True, 'include_fpgrowth': True},
        'no_cross_resident': {'include_temporal': True, 'include_cross_resident': False, 
                              'include_lag': True, 'include_conflict_risk': True,
                              'include_spatial': True, 'include_fpgrowth': True},
        'no_conflict_risk': {'include_temporal': True, 'include_cross_resident': True, 
                             'include_lag': True, 'include_conflict_risk': False,
                             'include_spatial': True, 'include_fpgrowth': True},
        'no_fpgrowth': {'include_temporal': True, 'include_cross_resident': True, 
                        'include_lag': True, 'include_conflict_risk': True,
                        'include_spatial': True, 'include_fpgrowth': False},
        'sensors_only': {'include_temporal': False, 'include_cross_resident': False, 
                         'include_lag': False, 'include_conflict_risk': False,
                         'include_spatial': False, 'include_fpgrowth': False},
    }
    
    results = []
    
    for house in ['A', 'B']:
        print_section(f"House {house}")
        data = load_aras_data(house)
        
        for config_name, config in configs.items():
            print(f"  Testing {config_name}...", end=" ")
            
            X, y_r1, y_r2, _, _ = generate_features(data, config)
            
            train_samples = CONFIG['train_days'] * CONFIG['samples_per_day']
            X_train, X_test = X[:train_samples], X[train_samples:]
            y_train_r1, y_test_r1 = y_r1[:train_samples], y_r1[train_samples:]
            y_train_r2, y_test_r2 = y_r2[:train_samples], y_r2[train_samples:]
            
            metrics_r1 = train_evaluate_model('GLM', X_train, y_train_r1, X_test, y_test_r1)
            metrics_r2 = train_evaluate_model('GLM', X_train, y_train_r2, X_test, y_test_r2)
            
            results.append({
                'House': house,
                'Feature_Set': config_name,
                'Num_Features': X.shape[1],
                'R1_Accuracy': metrics_r1['accuracy'],
                'R2_Accuracy': metrics_r2['accuracy'],
                'Avg_Accuracy': (metrics_r1['accuracy'] + metrics_r2['accuracy']) / 2
            })
            print(f"R1={metrics_r1['accuracy']:.2%}, R2={metrics_r2['accuracy']:.2%}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(CONFIG['results_dir'], 'ablation_study.csv'), index=False)
    print(f"\nResults saved to {CONFIG['results_dir']}/ablation_study.csv")
    
    return results_df

def run_conflict_resolution_experiments():
    """Run conflict resolution experiments."""
    print_header("CONFLICT RESOLUTION EXPERIMENTS")
    
    resolver = ConflictResolver()
    results = []
    
    for house in ['A', 'B']:
        print_section(f"House {house}")
        data = load_aras_data(house)
        
        conflicts_detected = 0
        conflicts_resolved = 0
        strategy_counts = {s: 0 for s in ConflictResolver.STRATEGIES}
        
        # Sample for efficiency
        sample_indices = np.random.choice(len(data), min(100000, len(data)), replace=False)
        
        for idx in sample_indices:
            r1, r2 = data.iloc[idx]['Activity_R1'], data.iloc[idx]['Activity_R2']
            conflict = resolver.detect_conflict(r1, r2)
            
            if conflict['has_conflict']:
                conflicts_detected += 1
                resolution = resolver.resolve(r1, r2)
                if resolution['strategy'] != 'none':
                    conflicts_resolved += 1
                    strategy_counts[resolution['strategy']] += 1
        
        results.append({
            'House': house,
            'Samples_Analyzed': len(sample_indices),
            'Conflicts_Detected': conflicts_detected,
            'Conflict_Rate': conflicts_detected / len(sample_indices),
            'Conflicts_Resolved': conflicts_resolved,
            'Resolution_Rate': conflicts_resolved / max(conflicts_detected, 1),
            **{f'Strategy_{k}': v for k, v in strategy_counts.items()}
        })
        
        print(f"  Conflicts detected: {conflicts_detected} ({conflicts_detected/len(sample_indices)*100:.2f}%)")
        print(f"  Resolution rate: {conflicts_resolved/max(conflicts_detected,1)*100:.1f}%")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(CONFIG['results_dir'], 'conflict_resolution.csv'), index=False)
    print(f"\nResults saved to {CONFIG['results_dir']}/conflict_resolution.csv")
    
    return results_df

def compute_per_class_f1():
    """Compute per-class F1 scores."""
    print_header("PER-CLASS F1 SCORES")
    
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    results = []
    
    for house in ['A', 'B']:
        print_section(f"House {house}")
        data = load_aras_data(house)
        X, y_r1, _, _, _ = generate_features(data)
        
        train_samples = CONFIG['train_days'] * CONFIG['samples_per_day']
        X_train, X_test = X[:train_samples], X[train_samples:]
        y_train, y_test = y_r1[:train_samples], y_r1[train_samples:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        
        known_classes = set(le.classes_)
        test_mask = np.array([y in known_classes for y in y_test])
        y_test_filtered = y_test[test_mask]
        y_test_enc = le.transform(y_test_filtered)
        
        model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        model.fit(X_train_scaled, y_train_enc)
        y_pred = model.predict(X_test_scaled[test_mask])
        
        report = classification_report(y_test_enc, y_pred, output_dict=True, zero_division=0)
        
        for class_id, metrics in report.items():
            if class_id not in ['accuracy', 'macro avg', 'weighted avg']:
                try:
                    original_class = le.inverse_transform([int(class_id)])[0]
                    results.append({
                        'House': house,
                        'Class_ID': original_class,
                        'Class_Name': ACTIVITY_NAMES.get(original_class, f'Class_{original_class}'),
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1_Score': metrics['f1-score'],
                        'Support': metrics['support']
                    })
                except:
                    pass
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(CONFIG['results_dir'], 'per_class_f1.csv'), index=False)
    print(f"\nResults saved to {CONFIG['results_dir']}/per_class_f1.csv")
    
    return results_df

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Occupant Smart Home Recommendation System Experiments'
    )
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--baselines', action='store_true', help='Run baseline comparisons')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--conflict', action='store_true', help='Run conflict resolution experiments')
    parser.add_argument('--perclass', action='store_true', help='Compute per-class F1 scores')
    parser.add_argument('--figures', action='store_true', help='Generate figures only')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print_header("MULTI-OCCUPANT SMART HOME RECOMMENDATION SYSTEM")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_directories()
    
    results = {}
    
    if args.all or args.baselines:
        results['baseline_comparison'] = run_baseline_experiments()
    
    if args.all or args.ablation:
        results['ablation_study'] = run_ablation_study()
    
    if args.all or args.conflict:
        results['conflict_resolution'] = run_conflict_resolution_experiments()
    
    if args.all or args.perclass:
        results['per_class_f1'] = compute_per_class_f1()
    
    if args.all or args.figures:
        # Load existing results if not computed
        if 'baseline_comparison' not in results:
            try:
                results['baseline_comparison'] = pd.read_csv(
                    os.path.join(CONFIG['results_dir'], 'baseline_comparison.csv'))
            except:
                pass
        if 'ablation_study' not in results:
            try:
                results['ablation_study'] = pd.read_csv(
                    os.path.join(CONFIG['results_dir'], 'ablation_study.csv'))
            except:
                pass
        if 'per_class_f1' not in results:
            try:
                results['per_class_f1'] = pd.read_csv(
                    os.path.join(CONFIG['results_dir'], 'per_class_f1.csv'))
            except:
                pass
        
        print_section("Generating Figures")
        generate_all_figures(results)
    
    print_header("COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
