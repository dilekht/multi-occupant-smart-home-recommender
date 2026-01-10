"""
ARAS Dataset Preprocessor v2.0
==============================
Multi-Occupant Context-Aware Recommender System Research

This module processes the ARAS dataset for extending the Dilekh et al. (2024) 
methodology from single-occupant to multi-occupant smart home scenarios.

Paper: "Dynamic Context-Aware Recommender System for Home Automation Through 
        Synergistic Unsupervised and Supervised Learning Algorithms"
DOI: https://doi.org/10.18267/j.aip.228

Dataset: ARAS - Activity Recognition with Ambient Sensing
Paper: Alemdar et al. (2013) - "ARAS Human Activity Datasets in Multiple Homes 
        with Multiple Residents"

Author: Research Extension Project
Date: January 2026
Version: 2.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import warnings

# =============================================================================
# OFFICIAL ARAS CONFIGURATION (from README files)
# =============================================================================

ACTIVITIES = {
    1: "Other",
    2: "Going Out", 
    3: "Preparing Breakfast",
    4: "Having Breakfast",
    5: "Preparing Lunch",
    6: "Having Lunch",
    7: "Preparing Dinner",
    8: "Having Dinner",
    9: "Washing Dishes",
    10: "Having Snack",
    11: "Sleeping",
    12: "Watching TV",
    13: "Studying",
    14: "Having Shower",
    15: "Toileting",
    16: "Napping",
    17: "Using Internet",
    18: "Reading Book",
    19: "Laundry",
    20: "Shaving",
    21: "Brushing Teeth",
    22: "Talking on Phone",
    23: "Listening to Music",
    24: "Cleaning",
    25: "Having Conversation",
    26: "Having Guest",
    27: "Changing Clothes"
}

# House A: 50m², married couple (both age 34), 2071 activity instances
HOUSE_A_SENSORS = {
    0: {"id": "Ph1", "type": "Photocell", "location": "Wardrobe"},
    1: {"id": "Ph2", "type": "Photocell", "location": "Convertible Couch (Bed R2)"},
    2: {"id": "Ir1", "type": "IR", "location": "TV receiver"},
    3: {"id": "Fo1", "type": "Force", "location": "Couch"},
    4: {"id": "Fo2", "type": "Force", "location": "Couch"},
    5: {"id": "Di3", "type": "Distance", "location": "Chair"},
    6: {"id": "Di4", "type": "Distance", "location": "Chair"},
    7: {"id": "Ph3", "type": "Photocell", "location": "Fridge"},
    8: {"id": "Ph4", "type": "Photocell", "location": "Kitchen Drawer"},
    9: {"id": "Ph5", "type": "Photocell", "location": "Wardrobe"},
    10: {"id": "Ph6", "type": "Photocell", "location": "Bathroom Cabinet"},
    11: {"id": "Co1", "type": "Contact", "location": "House Door"},
    12: {"id": "Co2", "type": "Contact", "location": "Bathroom Door"},
    13: {"id": "Co3", "type": "Contact", "location": "Shower Cabinet Door"},
    14: {"id": "So1", "type": "Sonar", "location": "Hall"},
    15: {"id": "So2", "type": "Sonar", "location": "Kitchen"},
    16: {"id": "Di1", "type": "Distance", "location": "Tap"},
    17: {"id": "Di2", "type": "Distance", "location": "Water Closet"},
    18: {"id": "Te1", "type": "Temperature", "location": "Kitchen"},
    19: {"id": "Fo3", "type": "Force", "location": "Bed"}
}

# House B: 90m², 2 males (both age 25), 1021 activity instances  
HOUSE_B_SENSORS = {
    0: {"id": "co1", "type": "Contact", "location": "Kitchen cupboard"},
    1: {"id": "co2", "type": "Contact", "location": "Kitchen cupboard"},
    2: {"id": "co3", "type": "Contact", "location": "House Door"},
    3: {"id": "co4", "type": "Contact", "location": "Wardrobe Door"},
    4: {"id": "co5", "type": "Contact", "location": "Wardrobe Door"},
    5: {"id": "co6", "type": "Contact", "location": "Shower Cabinet Door"},
    6: {"id": "di2", "type": "Distance", "location": "Tap"},
    7: {"id": "fo1", "type": "Force", "location": "Chair"},
    8: {"id": "fo2", "type": "Force", "location": "Chair"},
    9: {"id": "fo3", "type": "Force", "location": "Chair"},
    10: {"id": "ph1", "type": "Photocell", "location": "Fridge"},
    11: {"id": "ph2", "type": "Photocell", "location": "Kitchen Drawer"},
    12: {"id": "pr1", "type": "Pressure", "location": "Couch"},
    13: {"id": "pr2", "type": "Pressure", "location": "Couch"},
    14: {"id": "pr3", "type": "Pressure", "location": "Bed"},
    15: {"id": "pr4", "type": "Pressure", "location": "Bed"},
    16: {"id": "pr5", "type": "Pressure", "location": "Armchair"},
    17: {"id": "so1", "type": "Sonar", "location": "Bathroom Door"},
    18: {"id": "so2", "type": "Sonar", "location": "Kitchen"},
    19: {"id": "so3", "type": "Sonar", "location": "Closet"}
}

# Activity categories for conflict analysis and recommendation
ACTIVITY_CATEGORIES = {
    "rest": [11, 16],  # Sleeping, Napping
    "entertainment": [12, 17, 18, 23],  # TV, Internet, Reading, Music
    "meal_prep": [3, 5, 7],  # Preparing meals
    "meal_consumption": [4, 6, 8, 10],  # Having meals/snacks
    "hygiene": [14, 15, 20, 21, 27],  # Shower, Toilet, Shave, Brush, Change
    "work": [13],  # Studying
    "household": [9, 19, 24],  # Dishes, Laundry, Cleaning
    "social": [22, 25, 26],  # Phone, Conversation, Guest
    "away": [2],  # Going out
    "other": [1]  # Other
}

# Potential conflict pairs for multi-occupant scenarios
# (activity1, activity2, conflict_type, severity)
CONFLICT_DEFINITIONS = [
    (11, 12, "noise", "high"),      # Sleeping vs TV
    (11, 23, "noise", "high"),      # Sleeping vs Music
    (16, 12, "noise", "medium"),    # Napping vs TV
    (16, 23, "noise", "medium"),    # Napping vs Music
    (13, 12, "distraction", "medium"),  # Studying vs TV
    (13, 23, "distraction", "low"),     # Studying vs Music (may be ok)
    (18, 12, "distraction", "medium"),  # Reading vs TV
    (18, 23, "distraction", "low"),     # Reading vs Music
    (14, 15, "resource", "high"),   # Shower vs Toilet (same bathroom)
]

# Location zones for spatial context
LOCATION_ZONES = {
    "bedroom": ["Bed", "Wardrobe", "Convertible Couch"],
    "living_room": ["Couch", "TV", "Armchair"],
    "kitchen": ["Fridge", "Kitchen", "cupboard", "Drawer"],
    "bathroom": ["Bathroom", "Shower", "Water Closet", "Tap"],
    "entrance": ["House Door", "Hall"],
    "other": ["Chair", "Closet"]
}


@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline."""
    window_size: int = 60  # Aggregation window in seconds
    min_activity_duration: int = 10  # Minimum activity duration to consider
    include_temporal: bool = True
    include_spatial: bool = True
    include_conflict_detection: bool = True
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    test_ratio: float = 0.25


@dataclass 
class DataStats:
    """Container for dataset statistics."""
    total_seconds: int = 0
    total_days: int = 0
    r1_activity_counts: Dict[int, int] = field(default_factory=dict)
    r2_activity_counts: Dict[int, int] = field(default_factory=dict)
    concurrent_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    conflict_counts: Dict[str, int] = field(default_factory=dict)
    sensor_activations: Dict[str, int] = field(default_factory=dict)
    sync_seconds: int = 0
    async_seconds: int = 0


class ARASPreprocessor:
    """
    Production-ready preprocessor for ARAS dataset.
    
    Designed for multi-occupant context-aware recommender system research,
    extending the Dilekh et al. (2024) FP-Growth + GLM methodology.
    """
    
    def __init__(self, house: str = "B", config: Optional[ProcessingConfig] = None):
        """
        Initialize preprocessor.
        
        Args:
            house: "A" or "B"
            config: Processing configuration (uses defaults if None)
        """
        self.house = house.upper()
        if self.house not in ["A", "B"]:
            raise ValueError("House must be 'A' or 'B'")
            
        self.sensor_config = HOUSE_A_SENSORS if self.house == "A" else HOUSE_B_SENSORS
        self.config = config or ProcessingConfig()
        self.stats = DataStats()
        
        # Generate column names
        self.sensor_columns = [
            f"S{i+1}_{self.sensor_config[i]['id']}" 
            for i in range(20)
        ]
        self.all_columns = self.sensor_columns + ["Activity_R1", "Activity_R2"]
        
        # Storage
        self._raw_data: List[pd.DataFrame] = []
        self._processed_data: Optional[pd.DataFrame] = None
        
    def load_day(self, filepath: Union[str, Path], day_num: int) -> pd.DataFrame:
        """
        Load a single day's data file.
        
        Args:
            filepath: Path to DAY_X.csv file
            day_num: Day number (1-30)
            
        Returns:
            DataFrame with proper column names and day identifier
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load data - try space-separated first, then comma
        try:
            df = pd.read_csv(filepath, sep=r'\s+', header=None)
            # If we got only 1 column, it might be comma-separated
            if len(df.columns) == 1:
                df = pd.read_csv(filepath, sep=',', header=None)
        except Exception:
            df = pd.read_csv(filepath, header=None)
        
        # Validate structure
        if len(df) != 86400:
            warnings.warn(f"Expected 86400 rows, got {len(df)} in {filepath}")
        if len(df.columns) != 22:
            raise ValueError(f"Expected 22 columns, got {len(df.columns)} in {filepath}")
        
        # Assign column names
        df.columns = self.all_columns
        
        # Add metadata
        df["Day"] = day_num
        df["House"] = self.house
        df["Second"] = range(len(df))
        
        self._raw_data.append(df)
        self.stats.total_days += 1
        self.stats.total_seconds += len(df)
        
        return df
    
    def load_all_days(self, data_dir: Union[str, Path], 
                       pattern: Optional[str] = None,
                       days: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load multiple days from a directory.
        
        Args:
            data_dir: Directory containing day files
            pattern: Filename pattern with {} for day number (auto-detected if None)
            days: Specific days to load (default: 1-30)
            
        Returns:
            Combined DataFrame
        """
        data_dir = Path(data_dir)
        days = days or list(range(1, 31))
        
        # Auto-detect file pattern if not specified
        if pattern is None:
            pattern = self._detect_file_pattern(data_dir)
            print(f"Auto-detected pattern: {pattern}")
        
        for day in days:
            filepath = data_dir / pattern.format(day)
            if filepath.exists():
                self.load_day(filepath, day)
                print(f"✓ Loaded Day {day}")
            else:
                print(f"✗ Day {day} not found: {filepath}")
        
        if self._raw_data:
            combined = pd.concat(self._raw_data, ignore_index=True)
            print(f"\nTotal: {len(combined):,} records from {len(self._raw_data)} days")
            return combined
        else:
            raise ValueError("No data files were loaded. Check directory path and file names.")
    
    def _detect_file_pattern(self, data_dir: Path) -> str:
        """Auto-detect the file naming pattern in the directory."""
        import re
        
        # Common patterns to try
        patterns = [
            "DAY_{}.csv",
            "DAY_{}.txt", 
            "day_{}.csv",
            "day_{}.txt",
            "Day{}.csv",
            "Day{}.txt",
            "day{}.csv",
            "day{}.txt",
            "{}.csv",
            "{}.txt",
        ]
        
        # List files in directory
        try:
            files = list(data_dir.iterdir())
        except Exception as e:
            print(f"Error reading directory: {e}")
            return "DAY_{}.csv"
        
        # Show what files exist
        print(f"Files found in {data_dir}:")
        for f in sorted(files)[:10]:  # Show first 10
            print(f"  - {f.name}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
        
        # Try to match patterns
        for pattern in patterns:
            test_file = data_dir / pattern.format(1)
            if test_file.exists():
                return pattern
        
        # Try regex to find day files
        for f in files:
            # Match patterns like "DAY_1", "day_1", "Day1", etc.
            match = re.match(r'(DAY_?|day_?|Day_?)(\d+)\.(csv|txt)', f.name, re.IGNORECASE)
            if match:
                prefix = match.group(1)
                ext = match.group(3)
                detected = f"{prefix}{{}}.{ext}"
                print(f"Detected pattern from file '{f.name}': {detected}")
                return detected
        
        # Default fallback
        print("Could not auto-detect pattern, using default: DAY_{}.csv")
        return "DAY_{}.csv"
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal context features."""
        df = df.copy()
        
        # Time decomposition
        df["Hour"] = df["Second"] // 3600
        df["Minute"] = (df["Second"] % 3600) // 60
        df["MinuteOfDay"] = df["Second"] // 60
        
        # Time of day categories
        df["TimeOfDay"] = pd.cut(
            df["Hour"],
            bins=[-1, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"]
        )
        
        # Weekend indicator (assuming Day 1 = Monday)
        df["IsWeekend"] = ((df["Day"] - 1) % 7 >= 5).astype(int)
        
        # Day of week
        df["DayOfWeek"] = (df["Day"] - 1) % 7
        
        return df
    
    def add_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add activity-related features for both residents."""
        df = df.copy()
        
        # Activity names
        df["Activity_R1_Name"] = df["Activity_R1"].map(ACTIVITIES)
        df["Activity_R2_Name"] = df["Activity_R2"].map(ACTIVITIES)
        
        # Activity categories
        def get_category(act_id):
            for cat, acts in ACTIVITY_CATEGORIES.items():
                if act_id in acts:
                    return cat
            return "other"
        
        df["Category_R1"] = df["Activity_R1"].apply(get_category)
        df["Category_R2"] = df["Activity_R2"].apply(get_category)
        
        # Synchronization flag
        df["IsSynchronized"] = (df["Activity_R1"] == df["Activity_R2"]).astype(int)
        
        # Combined activity pattern (for pattern mining)
        df["ActivityPair"] = df["Activity_R1"].astype(str) + "_" + df["Activity_R2"].astype(str)
        
        return df
    
    def add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive spatial context from sensor activations."""
        df = df.copy()
        
        def get_zone(row):
            """Determine primary zone from active sensors."""
            active_locations = []
            for i, col in enumerate(self.sensor_columns):
                if row[col] == 1:
                    active_locations.append(self.sensor_config[i]["location"])
            
            if not active_locations:
                return "unknown"
            
            # Check each zone
            for zone, keywords in LOCATION_ZONES.items():
                for loc in active_locations:
                    if any(kw.lower() in loc.lower() for kw in keywords):
                        return zone
            return "other"
        
        df["PrimaryZone"] = df.apply(get_zone, axis=1)
        
        # Count active sensors
        df["ActiveSensorCount"] = df[self.sensor_columns].sum(axis=1)
        
        return df
    
    def detect_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect potential preference conflicts between residents."""
        df = df.copy()
        
        # Initialize conflict columns
        df["HasConflict"] = 0
        df["ConflictType"] = None
        df["ConflictSeverity"] = None
        
        for a1, a2, conf_type, severity in CONFLICT_DEFINITIONS:
            # Check both directions
            mask = (
                ((df["Activity_R1"] == a1) & (df["Activity_R2"] == a2)) |
                ((df["Activity_R1"] == a2) & (df["Activity_R2"] == a1))
            )
            df.loc[mask, "HasConflict"] = 1
            df.loc[mask, "ConflictType"] = conf_type
            df.loc[mask, "ConflictSeverity"] = severity
        
        return df
    
    def process(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.
        
        Args:
            df: DataFrame to process (uses combined raw data if None)
            
        Returns:
            Fully processed DataFrame
        """
        if df is None:
            if not self._raw_data:
                raise ValueError("No data loaded. Call load_day() first.")
            df = pd.concat(self._raw_data, ignore_index=True)
        
        print("Processing pipeline:")
        
        # 1. Temporal features
        if self.config.include_temporal:
            df = self.add_temporal_features(df)
            print("  ✓ Temporal features added")
        
        # 2. Activity features
        df = self.add_activity_features(df)
        print("  ✓ Activity features added")
        
        # 3. Spatial features
        if self.config.include_spatial:
            df = self.add_spatial_features(df)
            print("  ✓ Spatial features added")
        
        # 4. Conflict detection
        if self.config.include_conflict_detection:
            df = self.detect_conflicts(df)
            print("  ✓ Conflict detection complete")
        
        # Update statistics
        self._update_stats(df)
        
        self._processed_data = df
        return df
    
    def _update_stats(self, df: pd.DataFrame) -> None:
        """Update dataset statistics."""
        # Activity counts
        self.stats.r1_activity_counts = df["Activity_R1"].value_counts().to_dict()
        self.stats.r2_activity_counts = df["Activity_R2"].value_counts().to_dict()
        
        # Concurrent activities
        concurrent = df.groupby(["Activity_R1", "Activity_R2"]).size()
        self.stats.concurrent_counts = {
            (int(a1), int(a2)): int(count) 
            for (a1, a2), count in concurrent.items()
        }
        
        # Synchronization
        self.stats.sync_seconds = int(df["IsSynchronized"].sum())
        self.stats.async_seconds = int(len(df) - self.stats.sync_seconds)
        
        # Conflicts
        if "HasConflict" in df.columns:
            conflict_df = df[df["HasConflict"] == 1]
            if len(conflict_df) > 0:
                self.stats.conflict_counts = conflict_df["ConflictType"].value_counts().to_dict()
        
        # Sensor activations
        for col in self.sensor_columns:
            self.stats.sensor_activations[col] = int(df[col].sum())
    
    def generate_transactions(self, df: Optional[pd.DataFrame] = None,
                               window_size: Optional[int] = None) -> pd.DataFrame:
        """
        Generate transaction data for FP-Growth algorithm.
        
        Extends Dilekh methodology for multi-resident patterns.
        
        Args:
            df: Processed DataFrame
            window_size: Aggregation window in seconds
            
        Returns:
            Transaction DataFrame for FP-Growth
        """
        if df is None:
            df = self._processed_data
        if df is None:
            raise ValueError("No processed data available")
        
        window_size = window_size or self.config.window_size
        transactions = []
        
        for start in range(0, len(df), window_size):
            window = df.iloc[start:start + window_size]
            if len(window) == 0:
                continue
            
            items = set()
            
            # 1. Active sensors (majority voting)
            for col in self.sensor_columns:
                if window[col].mean() > 0.5:
                    items.add(f"SENSOR:{col}")
            
            # 2. Temporal context
            hour = int(window["Hour"].mode().iloc[0])
            tod = window["TimeOfDay"].mode().iloc[0]
            items.add(f"HOUR:{hour}")
            items.add(f"TOD:{tod}")
            
            if "IsWeekend" in window.columns:
                is_weekend = int(window["IsWeekend"].mode().iloc[0])
                items.add(f"WEEKEND:{is_weekend}")
            
            # 3. Resident activities
            act_r1 = int(window["Activity_R1"].mode().iloc[0])
            act_r2 = int(window["Activity_R2"].mode().iloc[0])
            items.add(f"R1:{ACTIVITIES.get(act_r1, 'Unknown')}")
            items.add(f"R2:{ACTIVITIES.get(act_r2, 'Unknown')}")
            
            # 4. Activity categories
            cat_r1 = window["Category_R1"].mode().iloc[0]
            cat_r2 = window["Category_R2"].mode().iloc[0]
            items.add(f"R1_CAT:{cat_r1}")
            items.add(f"R2_CAT:{cat_r2}")
            
            # 5. Synchronization
            is_sync = int(window["IsSynchronized"].mode().iloc[0])
            items.add(f"SYNC:{is_sync}")
            
            # 6. Spatial context
            if "PrimaryZone" in window.columns:
                zone = window["PrimaryZone"].mode().iloc[0]
                items.add(f"ZONE:{zone}")
            
            # 7. Conflict status
            if "HasConflict" in window.columns and window["HasConflict"].any():
                items.add("CONFLICT:Yes")
                conf_type = window.loc[window["HasConflict"] == 1, "ConflictType"].mode()
                if len(conf_type) > 0:
                    items.add(f"CONFLICT_TYPE:{conf_type.iloc[0]}")
            
            transactions.append({
                "window_id": start // window_size,
                "start_second": start,
                "end_second": start + window_size,
                "day": int(window["Day"].iloc[0]),
                "items": list(items),
                "itemset": frozenset(items),
                "activity_r1": act_r1,
                "activity_r2": act_r2,
                "is_synchronized": is_sync,
                "has_conflict": 1 if "CONFLICT:Yes" in items else 0
            })
        
        return pd.DataFrame(transactions)
    
    def prepare_ml_data(self, df: Optional[pd.DataFrame] = None
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Prepare data for machine learning models.
        
        Returns:
            Tuple of (features, labels, column_info)
        """
        if df is None:
            df = self._processed_data
        if df is None:
            raise ValueError("No processed data available")
        
        # Feature columns
        feature_cols = self.sensor_columns.copy()
        
        if self.config.include_temporal:
            feature_cols.extend(["Hour", "MinuteOfDay", "IsWeekend", "DayOfWeek"])
        
        features = df[feature_cols].copy()
        
        # One-hot encode categorical features
        if "TimeOfDay" in df.columns:
            tod_dummies = pd.get_dummies(df["TimeOfDay"], prefix="TOD")
            features = pd.concat([features, tod_dummies], axis=1)
        
        if "PrimaryZone" in df.columns:
            zone_dummies = pd.get_dummies(df["PrimaryZone"], prefix="ZONE")
            features = pd.concat([features, zone_dummies], axis=1)
        
        # Labels
        label_cols = ["Activity_R1", "Activity_R2"]
        if "HasConflict" in df.columns:
            label_cols.append("HasConflict")
        
        labels = df[label_cols].copy()
        
        # Column info for reference
        col_info = {
            "sensor_cols": self.sensor_columns,
            "temporal_cols": ["Hour", "MinuteOfDay", "IsWeekend", "DayOfWeek"],
            "feature_cols": list(features.columns),
            "label_cols": label_cols,
            "n_features": len(features.columns),
            "n_samples": len(features)
        }
        
        return features, labels, col_info
    
    def split_data(self, features: pd.DataFrame, labels: pd.DataFrame,
                   ) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame],
                              Tuple[pd.DataFrame, pd.DataFrame],
                              Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into train/validation/test sets.
        
        Uses temporal split (not random) to preserve time-series nature.
        Default: 50% train, 25% validation, 25% test
        """
        n = len(features)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train = (features.iloc[:train_end], labels.iloc[:train_end])
        val = (features.iloc[train_end:val_end], labels.iloc[train_end:val_end])
        test = (features.iloc[val_end:], labels.iloc[val_end:])
        
        print(f"Data split: Train={len(train[0]):,}, Val={len(val[0]):,}, Test={len(test[0]):,}")
        
        return train, val, test
    
    def export_transactions_fpgrowth(self, transactions: pd.DataFrame,
                                      output_path: Union[str, Path]) -> None:
        """Export transactions for FP-Growth (one itemset per line)."""
        with open(output_path, 'w') as f:
            for _, row in transactions.iterrows():
                f.write(" ".join(row["items"]) + "\n")
        print(f"Exported {len(transactions)} transactions to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics as dictionary."""
        return {
            "house": self.house,
            "total_seconds": self.stats.total_seconds,
            "total_days": self.stats.total_days,
            "total_hours": round(self.stats.total_seconds / 3600, 1),
            "r1_activities": {
                ACTIVITIES.get(k, k): v 
                for k, v in self.stats.r1_activity_counts.items()
            },
            "r2_activities": {
                ACTIVITIES.get(k, k): v 
                for k, v in self.stats.r2_activity_counts.items()
            },
            "synchronization": {
                "synchronized_seconds": self.stats.sync_seconds,
                "async_seconds": self.stats.async_seconds,
                "sync_percentage": round(
                    self.stats.sync_seconds / max(1, self.stats.total_seconds) * 100, 1
                )
            },
            "conflicts": self.stats.conflict_counts,
            "sensor_activations": self.stats.sensor_activations
        }
    
    def save_processed(self, output_dir: Union[str, Path], prefix: str = "") -> Dict[str, Path]:
        """
        Save all processed outputs.
        
        Returns:
            Dictionary of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{prefix}_" if prefix else ""
        outputs = {}
        
        # 1. Processed data
        if self._processed_data is not None:
            path = output_dir / f"{prefix}processed_data.csv"
            self._processed_data.to_csv(path, index=False)
            outputs["processed_data"] = path
        
        # 2. Transactions
        if self._processed_data is not None:
            transactions = self.generate_transactions()
            path = output_dir / f"{prefix}transactions.csv"
            transactions.to_csv(path, index=False)
            outputs["transactions"] = path
            
            # FP-Growth format
            path_fpg = output_dir / f"{prefix}fpgrowth_input.txt"
            self.export_transactions_fpgrowth(transactions, path_fpg)
            outputs["fpgrowth_input"] = path_fpg
        
        # 3. ML-ready data
        if self._processed_data is not None:
            features, labels, col_info = self.prepare_ml_data()
            
            path = output_dir / f"{prefix}features.csv"
            features.to_csv(path, index=False)
            outputs["features"] = path
            
            path = output_dir / f"{prefix}labels.csv"
            labels.to_csv(path, index=False)
            outputs["labels"] = path
            
            path = output_dir / f"{prefix}column_info.json"
            with open(path, 'w') as f:
                json.dump(col_info, f, indent=2)
            outputs["column_info"] = path
        
        # 4. Statistics
        path = output_dir / f"{prefix}statistics.json"
        with open(path, 'w') as f:
            json.dump(self.get_statistics(), f, indent=2, default=str)
        outputs["statistics"] = path
        
        print(f"\nSaved {len(outputs)} files to {output_dir}")
        return outputs


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run preprocessing on available data."""
    
    print("=" * 70)
    print("ARAS Multi-Occupant Preprocessor v2.0")
    print("=" * 70)
    
    # Check for data
    data_file = Path("/mnt/user-data/uploads/DAY_1.csv")
    if not data_file.exists():
        print("No data file found in uploads.")
        return
    
    # Initialize preprocessor for House B
    preprocessor = ARASPreprocessor(house="B")
    
    # Load data
    print("\n1. Loading data...")
    preprocessor.load_day(data_file, day_num=1)
    
    # Process
    print("\n2. Running preprocessing pipeline...")
    df = preprocessor.process()
    
    # Generate outputs
    print("\n3. Generating outputs...")
    output_dir = Path("/home/claude/aras_preprocessing/output_v2")
    outputs = preprocessor.save_processed(output_dir, prefix="house_b_day1")
    
    # Summary
    print("\n4. Summary Statistics:")
    stats = preprocessor.get_statistics()
    print(f"   Total time: {stats['total_hours']} hours")
    print(f"   Synchronized: {stats['synchronization']['sync_percentage']}%")
    print(f"   Conflicts detected: {sum(stats['conflicts'].values()) if stats['conflicts'] else 0}")
    
    print("\n" + "=" * 70)
    print("Preprocessing complete!")
    print("=" * 70)
    
    return preprocessor, df


if __name__ == "__main__":
    preprocessor, df = main()
