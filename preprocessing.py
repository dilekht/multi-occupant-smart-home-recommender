#!/usr/bin/env python3
"""
ARAS Dataset Preprocessor
=========================
Handles loading and initial preprocessing of the ARAS dataset.

ARAS Dataset Structure:
- 30 days of data per house
- 20 binary sensors + 2 activity labels per row
- 86,400 rows per day (1 Hz sampling)
- Total: 2,592,000 rows per house
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


class ArasPreprocessor:
    """
    Preprocessor for the ARAS (Activity Recognition with Ambient Sensing) dataset.
    """
    
    # ARAS sensor names (20 sensors per house)
    SENSOR_NAMES = [
        'PhoneJack_A', 'PhoneJack_B', 'PhoneJack_C', 'PhoneJack_D',
        'PhoneJack_E', 'PhoneJack_F', 'IrProx_A', 'IrProx_B',
        'IrProx_C', 'IrProx_D', 'Contact_A', 'Contact_B',
        'Contact_C', 'Contact_D', 'Contact_E', 'ForceSnsr_A',
        'ForceSnsr_B', 'ForceSnsr_C', 'ForceSnsr_D', 'ForceSnsr_E'
    ]
    
    # Activity labels (27 activities)
    ACTIVITY_NAMES = {
        1: 'Other',
        2: 'Going_Out',
        3: 'Preparing_Breakfast',
        4: 'Having_Breakfast',
        5: 'Preparing_Lunch',
        6: 'Having_Lunch',
        7: 'Preparing_Dinner',
        8: 'Having_Dinner',
        9: 'Washing_Dishes',
        10: 'Having_Snack',
        11: 'Sleeping',
        12: 'Watching_TV',
        13: 'Studying',
        14: 'Having_Shower',
        15: 'Toileting',
        16: 'Napping',
        17: 'Using_Internet',
        18: 'Reading_Book',
        19: 'Laundry',
        20: 'Shaving',
        21: 'Brushing_Teeth',
        22: 'Talking_Phone',
        23: 'Listening_Music',
        24: 'Cleaning',
        25: 'Conversation',
        26: 'Eating_Medicine',
        27: 'Using_Kitchen_Appliance'
    }
    
    # Activity categories for grouping
    ACTIVITY_CATEGORIES = {
        'Rest': [11, 16],  # Sleeping, Napping
        'Entertainment': [12, 17, 18, 23],  # TV, Internet, Reading, Music
        'Work': [13],  # Studying
        'Hygiene': [14, 15, 20, 21],  # Shower, Toilet, Shaving, Brushing
        'Eating': [3, 4, 5, 6, 7, 8, 10],  # All meal-related
        'Household': [9, 19, 24, 27],  # Dishes, Laundry, Cleaning, Kitchen
        'Social': [22, 25],  # Phone, Conversation
        'Other': [1, 2, 26]  # Other, Going Out, Medicine
    }
    
    # Activity priorities for conflict resolution (higher = more priority)
    ACTIVITY_PRIORITIES = {
        15: 5,  # Toileting - highest priority
        11: 5,  # Sleeping
        14: 4,  # Shower
        16: 4,  # Napping
        13: 4,  # Studying
        22: 4,  # Phone
        18: 3,  # Reading
        12: 2,  # TV
        17: 2,  # Internet
        23: 2,  # Music
    }
    
    # Noise tolerance levels (1=low tolerance, 5=high tolerance)
    NOISE_TOLERANCE = {
        11: 1, 16: 1,  # Sleeping, Napping - very low tolerance
        13: 2, 18: 2, 22: 2,  # Studying, Reading, Phone - low tolerance
        17: 3,  # Internet - medium tolerance
        12: 5, 23: 5, 14: 5, 15: 5  # TV, Music, Shower, Toilet - high tolerance
    }
    
    def __init__(self):
        self.house_data = {}
        
    def load_house_data(self, house: str, data_dir: str = './data/aras') -> pd.DataFrame:
        """
        Load all 30 days of data for a house.
        
        Parameters:
        -----------
        house : str
            'A' or 'B'
        data_dir : str
            Path to the house data directory containing DAY1.txt through DAY30.txt
            
        Returns:
        --------
        pd.DataFrame
            Combined data for all 30 days with columns:
            - Timestamp (datetime)
            - 20 sensor columns (binary)
            - Activity_R1 (int 1-27)
            - Activity_R2 (int 1-27)
            - Day (int 1-30)
            - Second_of_Day (int 0-86399)
        """
        all_days = []
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Please download the ARAS dataset and place it in the correct location.\n"
                f"Expected structure:\n"
                f"  {data_dir}/DAY1.txt\n"
                f"  {data_dir}/DAY2.txt\n"
                f"  ...\n"
                f"  {data_dir}/DAY30.txt"
            )
        
        for day in range(1, 31):
            filename = f"DAY{day}.txt"
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"  Warning: {filepath} not found, skipping...")
                continue
            
            # Load day file (space-separated, no header)
            # Columns: 20 sensors + Activity_R1 + Activity_R2
            day_data = pd.read_csv(
                filepath, 
                sep=r'\s+', 
                header=None,
                names=self.SENSOR_NAMES + ['Activity_R1', 'Activity_R2']
            )
            
            # Add metadata columns
            day_data['Day'] = day
            day_data['Second_of_Day'] = range(len(day_data))
            
            # Create timestamp (assuming data starts at midnight)
            base_date = datetime(2026, 1, 1) + timedelta(days=day-1)
            day_data['Timestamp'] = [
                base_date + timedelta(seconds=s) 
                for s in range(len(day_data))
            ]
            
            all_days.append(day_data)
        
        if not all_days:
            raise ValueError(f"No data files found in {data_dir}")
        
        # Combine all days
        combined = pd.concat(all_days, ignore_index=True)
        
        # Store and return
        self.house_data[house] = combined
        return combined
    
    def get_activity_name(self, activity_id: int) -> str:
        """Get human-readable activity name."""
        return self.ACTIVITY_NAMES.get(activity_id, f'Unknown_{activity_id}')
    
    def get_activity_category(self, activity_id: int) -> str:
        """Get category for an activity."""
        for category, activities in self.ACTIVITY_CATEGORIES.items():
            if activity_id in activities:
                return category
        return 'Other'
    
    def get_activity_priority(self, activity_id: int) -> int:
        """Get priority level (1-5) for an activity."""
        return self.ACTIVITY_PRIORITIES.get(activity_id, 2)
    
    def get_noise_tolerance(self, activity_id: int) -> int:
        """Get noise tolerance level (1-5) for an activity."""
        return self.NOISE_TOLERANCE.get(activity_id, 3)
    
    def compute_statistics(self, data: pd.DataFrame) -> dict:
        """
        Compute summary statistics for loaded data.
        
        Returns:
        --------
        dict with keys:
            - total_samples
            - days
            - r1_activity_distribution
            - r2_activity_distribution
            - sensor_activation_rates
            - conflict_rate
        """
        stats = {
            'total_samples': len(data),
            'days': data['Day'].nunique(),
            'r1_activities': data['Activity_R1'].nunique(),
            'r2_activities': data['Activity_R2'].nunique(),
        }
        
        # Activity distributions
        stats['r1_distribution'] = data['Activity_R1'].value_counts().to_dict()
        stats['r2_distribution'] = data['Activity_R2'].value_counts().to_dict()
        
        # Sensor activation rates
        sensor_cols = self.SENSOR_NAMES
        stats['sensor_rates'] = {
            col: data[col].mean() for col in sensor_cols
        }
        
        # Conflict detection
        conflicts = self._detect_conflicts(data)
        stats['conflict_rate'] = conflicts.mean()
        stats['total_conflicts'] = conflicts.sum()
        
        return stats
    
    def _detect_conflicts(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect potential conflicts between resident activities.
        
        Conflict types:
        - Noise: One resident doing noisy activity while other needs quiet
        - Resource: Both residents need same shared resource (e.g., bathroom)
        """
        r1 = data['Activity_R1']
        r2 = data['Activity_R2']
        
        # Define conflict pairs
        # (activity_a, activity_b) = conflict if a is noisy and b needs quiet
        noise_conflicts = [
            (12, 11), (12, 16), (12, 13), (12, 18), (12, 22),  # TV vs quiet activities
            (23, 11), (23, 16), (23, 13), (23, 18),  # Music vs quiet activities
            (11, 12), (16, 12), (13, 12), (18, 12), (22, 12),  # Reverse
            (11, 23), (16, 23), (13, 23), (18, 23),
        ]
        
        # Resource conflicts (bathroom)
        resource_conflicts = [
            (14, 15), (15, 14),  # Shower vs Toilet
            (14, 14), (15, 15),  # Both trying same activity
        ]
        
        all_conflicts = noise_conflicts + resource_conflicts
        
        # Create boolean mask for conflicts
        conflict_mask = pd.Series(False, index=data.index)
        for a, b in all_conflicts:
            conflict_mask |= ((r1 == a) & (r2 == b)) | ((r1 == b) & (r2 == a))
        
        return conflict_mask


# Synthetic data generator for testing when ARAS is unavailable
class SyntheticArasGenerator:
    """
    Generate synthetic ARAS-like data for testing.
    Useful when actual ARAS data is not available.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.preprocessor = ArasPreprocessor()
        
    def generate(self, num_days: int = 30) -> pd.DataFrame:
        """
        Generate synthetic data with realistic patterns.
        """
        samples_per_day = 86400
        total_samples = num_days * samples_per_day
        
        # Generate sensor data (20 sensors, binary)
        sensors = self.rng.binomial(1, 0.1, (total_samples, 20))
        
        # Generate activities with temporal patterns
        activities_r1 = self._generate_activities(total_samples)
        activities_r2 = self._generate_activities(total_samples)
        
        # Create DataFrame
        df = pd.DataFrame(
            sensors, 
            columns=self.preprocessor.SENSOR_NAMES
        )
        df['Activity_R1'] = activities_r1
        df['Activity_R2'] = activities_r2
        df['Day'] = np.repeat(range(1, num_days + 1), samples_per_day)
        df['Second_of_Day'] = np.tile(range(samples_per_day), num_days)
        
        # Add timestamp
        base_date = datetime(2026, 1, 1)
        df['Timestamp'] = [
            base_date + timedelta(seconds=i) 
            for i in range(total_samples)
        ]
        
        return df
    
    def _generate_activities(self, n: int) -> np.ndarray:
        """Generate activity sequence with realistic patterns."""
        activities = np.zeros(n, dtype=int)
        
        # Define activity durations (in seconds)
        durations = {
            1: 60, 2: 3600, 3: 900, 4: 1200, 5: 900, 6: 1800,
            7: 1200, 8: 2400, 9: 600, 10: 300, 11: 28800, 12: 3600,
            13: 7200, 14: 900, 15: 300, 16: 3600, 17: 3600, 18: 1800,
            19: 1800, 20: 300, 21: 180, 22: 600, 23: 1800, 24: 1800,
            25: 1800, 26: 60, 27: 600
        }
        
        i = 0
        while i < n:
            # Pick activity based on time of day
            hour = (i % 86400) // 3600
            
            if 0 <= hour < 6:  # Night - mostly sleeping
                activity = self.rng.choice([11, 16, 1], p=[0.9, 0.05, 0.05])
            elif 6 <= hour < 9:  # Morning routine
                activity = self.rng.choice([11, 3, 4, 14, 21], p=[0.3, 0.2, 0.2, 0.15, 0.15])
            elif 9 <= hour < 12:  # Morning
                activity = self.rng.choice([13, 17, 12, 2], p=[0.3, 0.3, 0.2, 0.2])
            elif 12 <= hour < 14:  # Lunch
                activity = self.rng.choice([5, 6, 9], p=[0.3, 0.5, 0.2])
            elif 14 <= hour < 18:  # Afternoon
                activity = self.rng.choice([13, 17, 12, 18], p=[0.3, 0.3, 0.2, 0.2])
            elif 18 <= hour < 21:  # Evening
                activity = self.rng.choice([7, 8, 12, 25], p=[0.2, 0.3, 0.3, 0.2])
            else:  # Night
                activity = self.rng.choice([12, 17, 11], p=[0.3, 0.3, 0.4])
            
            duration = durations.get(activity, 600)
            # Add some randomness to duration
            duration = max(60, int(duration * self.rng.uniform(0.5, 1.5)))
            
            end_i = min(i + duration, n)
            activities[i:end_i] = activity
            i = end_i
        
        return activities


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing with synthetic data...")
    generator = SyntheticArasGenerator(seed=42)
    synthetic_data = generator.generate(num_days=5)
    
    preprocessor = ArasPreprocessor()
    stats = preprocessor.compute_statistics(synthetic_data)
    
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Days: {stats['days']}")
    print(f"R1 activities: {stats['r1_activities']}")
    print(f"R2 activities: {stats['r2_activities']}")
    print(f"Conflict rate: {stats['conflict_rate']:.2%}")
