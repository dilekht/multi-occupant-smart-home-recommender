"""
Integrated Smart Home Recommendation System
============================================

This script integrates:
1. ARAS Preprocessor (data preparation)
2. FP-Growth Patterns (pattern mining)
3. Multi-Resident GLM (activity prediction)
4. Conflict Resolution Module (recommendation adjustment)

Complete pipeline for multi-occupant smart home recommendations.

Usage:
    python run_recommendation_system.py

Author: Research Extension Project
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from conflict_resolution_module import (
    SmartHomeRecommendationEngine,
    ConflictResolver,
    ConflictType,
    ConflictSeverity,
    ResolutionStrategy,
    ACTIVITY_PROFILES
)

# Try to import GLM model
try:
    import sys
    sys.path.append(r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project")
    from glm_multi_resident import MultiResidentGLM, ModelConfig
    GLM_AVAILABLE = True
except ImportError:
    GLM_AVAILABLE = False
    print("Note: GLM model not imported. Using standalone conflict resolution.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - Update these to your actual paths
HOUSE_A_MODEL = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\a_output\glm_results\house_a_multi_resident_glm.pkl"
HOUSE_B_MODEL = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\b_output\glm_results\house_b_multi_resident_glm.pkl"
HOUSE_A_DATA = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\a_output\processed_data.csv"
HOUSE_B_DATA = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\b_output\processed_data.csv"
OUTPUT_DIR = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\recommendation_results"


# =============================================================================
# INTEGRATED RECOMMENDATION SYSTEM
# =============================================================================

class IntegratedRecommendationSystem:
    """
    Complete integrated recommendation system.
    
    Combines activity prediction with conflict resolution
    for multi-occupant smart home automation.
    """
    
    def __init__(self, house: str = "A"):
        """
        Initialize the integrated system.
        
        Args:
            house: "A" or "B" to select household model
        """
        self.house = house
        self.glm_model = None
        self.recommendation_engine = SmartHomeRecommendationEngine()
        self.is_loaded = False
        
        # Statistics
        self.prediction_count = 0
        self.conflict_count = 0
        self.resolution_count = 0
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the trained GLM model.
        
        Args:
            model_path: Path to model file, or None to use default
            
        Returns:
            True if loaded successfully
        """
        if model_path is None:
            model_path = HOUSE_A_MODEL if self.house == "A" else HOUSE_B_MODEL
        
        try:
            if GLM_AVAILABLE:
                self.glm_model = MultiResidentGLM.load(model_path)
                self.recommendation_engine.set_model(self.glm_model)
                self.is_loaded = True
                print(f"✓ Loaded GLM model from {model_path}")
                return True
            else:
                print("GLM module not available. Running in standalone mode.")
                self.is_loaded = True
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running in standalone conflict resolution mode.")
            self.is_loaded = True
            return True
    
    def predict_activities(self, features: pd.DataFrame) -> Dict:
        """
        Predict activities for both residents.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Dictionary with predictions
        """
        if self.glm_model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        predictions = self.glm_model.predict(features)
        self.prediction_count += len(features)
        
        return predictions
    
    def get_recommendations(self, 
                           activity_r1: int, 
                           activity_r2: int,
                           context: Optional[Dict] = None) -> Dict:
        """
        Get recommendations for given activities.
        
        Args:
            activity_r1: Activity ID for resident 1
            activity_r2: Activity ID for resident 2
            context: Optional context dictionary
            
        Returns:
            Complete recommendation dictionary
        """
        result = self.recommendation_engine.recommend_from_activities(
            activity_r1, activity_r2
        )
        
        if result['has_conflict']:
            self.conflict_count += 1
            self.resolution_count += 1
        
        # Add context-aware adjustments
        if context:
            result = self._apply_context(result, context)
        
        return result
    
    def _apply_context(self, result: Dict, context: Dict) -> Dict:
        """Apply contextual adjustments to recommendations."""
        hour = context.get('hour', datetime.now().hour)
        is_weekend = context.get('is_weekend', datetime.now().weekday() >= 5)
        
        # Night mode adjustments (10 PM - 6 AM)
        if 22 <= hour or hour < 6:
            result['context_adjustments'] = ['night_mode']
            
            # Lower all volume recommendations further at night
            for rec in result.get('recommendations', []):
                if rec.get('action') == 'set_volume':
                    rec['value'] = max(5, rec['value'] - 10)
                    rec['reason'] += " (night mode: further reduced)"
        
        # Weekend adjustments
        if is_weekend:
            result['context_adjustments'] = result.get('context_adjustments', []) + ['weekend_mode']
            # More lenient on weekends for entertainment activities
        
        return result
    
    def process_realtime(self, 
                         sensor_data: Dict,
                         current_r1_activity: Optional[int] = None,
                         current_r2_activity: Optional[int] = None) -> Dict:
        """
        Process real-time sensor data and generate recommendations.
        
        Args:
            sensor_data: Dictionary of sensor readings
            current_r1_activity: Known activity for R1 (optional)
            current_r2_activity: Known activity for R2 (optional)
            
        Returns:
            Real-time recommendation
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'sensor_summary': {
                'active_sensors': sum(1 for v in sensor_data.values() if v == 1),
                'total_sensors': len(sensor_data)
            },
            'predictions': None,
            'recommendations': None
        }
        
        # If activities provided, use them directly
        if current_r1_activity is not None and current_r2_activity is not None:
            recommendations = self.get_recommendations(
                current_r1_activity, 
                current_r2_activity,
                context={'hour': datetime.now().hour}
            )
            result['predictions'] = {
                'activity_r1': current_r1_activity,
                'activity_r2': current_r2_activity,
                'source': 'provided'
            }
            result['recommendations'] = recommendations
        
        # Otherwise, would use GLM to predict (if model loaded)
        elif self.glm_model is not None:
            # Convert sensor data to feature format
            # This would need proper feature engineering
            result['predictions'] = {'source': 'model', 'status': 'requires_feature_engineering'}
        
        return result
    
    def batch_process(self, 
                      data: pd.DataFrame,
                      activity_r1_col: str = 'Activity_R1',
                      activity_r2_col: str = 'Activity_R2',
                      sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Process batch data and generate recommendations.
        
        Args:
            data: DataFrame with activity columns
            activity_r1_col: Column name for R1 activity
            activity_r2_col: Column name for R2 activity
            sample_size: Number of samples to process (None for all)
            
        Returns:
            DataFrame with recommendations
        """
        if sample_size:
            data = data.sample(n=min(sample_size, len(data)), random_state=42)
        
        print(f"Processing {len(data):,} records...")
        
        results = []
        conflicts_found = 0
        
        for idx, row in data.iterrows():
            activity_r1 = int(row[activity_r1_col])
            activity_r2 = int(row[activity_r2_col])
            
            rec = self.get_recommendations(activity_r1, activity_r2)
            
            results.append({
                'index': idx,
                'activity_r1': activity_r1,
                'activity_r1_name': rec['activity_r1_name'],
                'activity_r2': activity_r2,
                'activity_r2_name': rec['activity_r2_name'],
                'has_conflict': rec['has_conflict'],
                'conflict_type': rec['conflict']['type'] if rec['conflict'] else None,
                'conflict_severity': rec['conflict']['severity'] if rec['conflict'] else None,
                'resolution_strategy': rec['resolution']['strategy'] if rec['resolution'] else None,
                'resolution_confidence': rec['resolution']['confidence'] if rec['resolution'] else None,
                'num_recommendations': len(rec.get('recommendations', [])),
                'primary_action': rec['recommendations'][0]['action'] if rec.get('recommendations') else None,
                'primary_device': rec['recommendations'][0]['device'] if rec.get('recommendations') else None
            })
            
            if rec['has_conflict']:
                conflicts_found += 1
        
        print(f"✓ Processed {len(data):,} records")
        print(f"  Conflicts found: {conflicts_found:,} ({conflicts_found/len(data)*100:.2f}%)")
        
        return pd.DataFrame(results)
    
    def get_statistics(self) -> Dict:
        """Get system statistics."""
        stats = self.recommendation_engine.get_summary()
        stats.update({
            'house': self.house,
            'model_loaded': self.glm_model is not None,
            'total_predictions': self.prediction_count,
            'total_conflicts': self.conflict_count,
            'total_resolutions': self.resolution_count
        })
        return stats
    
    def export_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export all results and statistics."""
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        # Export statistics
        stats_path = output_path / f"house_{self.house.lower()}_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.get_statistics(), f, indent=2, default=str)
        outputs['statistics'] = str(stats_path)
        
        # Export recommendation history
        history_path = output_path / f"house_{self.house.lower()}_recommendations.json"
        self.recommendation_engine.export_recommendations(str(history_path))
        outputs['recommendations'] = str(history_path)
        
        print(f"\n✓ Results exported to {output_path}")
        return outputs


# =============================================================================
# DEMONSTRATION SCENARIOS
# =============================================================================

def run_demo_scenarios():
    """Run demonstration scenarios."""
    print("=" * 70)
    print("INTEGRATED RECOMMENDATION SYSTEM DEMO")
    print("=" * 70)
    
    # Initialize system
    system = IntegratedRecommendationSystem(house="A")
    system.load_model()  # Will work in standalone mode if model not found
    
    # Demo scenarios representing typical household situations
    scenarios = [
        # Evening conflicts
        {
            "name": "Evening TV vs Study",
            "r1": 13,  # Studying
            "r2": 12,  # Watching TV
            "context": {"hour": 20, "is_weekend": False},
            "description": "R1 needs to study for work, R2 wants to relax with TV"
        },
        {
            "name": "Night Sleep vs Music",
            "r1": 11,  # Sleeping
            "r2": 23,  # Listening to Music
            "context": {"hour": 23, "is_weekend": True},
            "description": "R1 went to bed early, R2 still listening to music"
        },
        {
            "name": "Morning Bathroom Rush",
            "r1": 14,  # Shower
            "r2": 15,  # Toileting
            "context": {"hour": 7, "is_weekend": False},
            "description": "Both need bathroom before work"
        },
        # Non-conflict scenarios
        {
            "name": "Quiet Evening",
            "r1": 18,  # Reading
            "r2": 17,  # Using Internet
            "context": {"hour": 21, "is_weekend": False},
            "description": "Both doing quiet activities - no conflict"
        },
        {
            "name": "Both Sleeping",
            "r1": 11,  # Sleeping
            "r2": 11,  # Sleeping
            "context": {"hour": 2, "is_weekend": False},
            "description": "Both asleep - no conflict"
        },
        # Edge cases
        {
            "name": "Phone Call vs TV",
            "r1": 22,  # Phone call
            "r2": 12,  # TV
            "context": {"hour": 19, "is_weekend": True},
            "description": "R1 gets important call while R2 watching TV"
        },
        {
            "name": "Nap vs Music",
            "r1": 16,  # Napping
            "r2": 23,  # Music
            "context": {"hour": 15, "is_weekend": True},
            "description": "R1 taking afternoon nap, R2 playing music"
        },
    ]
    
    print("\n" + "=" * 70)
    print("SCENARIO ANALYSIS")
    print("=" * 70)
    
    for scenario in scenarios:
        print(f"\n{'─' * 70}")
        print(f"📋 SCENARIO: {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"{'─' * 70}")
        
        result = system.get_recommendations(
            scenario['r1'], 
            scenario['r2'],
            context=scenario['context']
        )
        
        print(f"\n   👤 Resident 1: {result['activity_r1_name']}")
        print(f"   👤 Resident 2: {result['activity_r2_name']}")
        print(f"   🕐 Time: {scenario['context']['hour']}:00 ({'Weekend' if scenario['context']['is_weekend'] else 'Weekday'})")
        
        if result['has_conflict']:
            print(f"\n   ⚠️  CONFLICT DETECTED")
            print(f"   ├─ Type: {result['conflict']['type'].upper()}")
            print(f"   ├─ Severity: {result['conflict']['severity']}")
            print(f"   └─ Description: {result['conflict']['description']}")
            
            print(f"\n   ✅ RESOLUTION ({result['resolution']['strategy'].upper()})")
            print(f"   ├─ Confidence: {result['resolution']['confidence']:.0%}")
            print(f"   └─ {result['resolution']['description']}")
            
            if result.get('recommendations'):
                print(f"\n   📱 SMART HOME ACTIONS:")
                for i, rec in enumerate(result['recommendations'][:5], 1):
                    device = rec['device'].upper()
                    action = rec['action']
                    value = rec['value']
                    print(f"   {i}. [{device}] {action}")
                    print(f"      Value: {value}")
                    print(f"      Reason: {rec['reason']}")
            
            if result['resolution'].get('alternatives'):
                print(f"\n   🔄 ALTERNATIVE STRATEGIES:")
                for alt in result['resolution']['alternatives'][:2]:
                    print(f"   • {alt['strategy']}: {alt['description'][:60]}...")
        else:
            print(f"\n   ✓ No conflict - activities are compatible")
        
        # Context adjustments
        if result.get('context_adjustments'):
            print(f"\n   ⚙️  Context Adjustments: {', '.join(result['context_adjustments'])}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    
    stats = system.get_statistics()
    print(f"\nScenarios tested: {len(scenarios)}")
    print(f"Conflicts detected: {stats.get('recommendations_with_conflicts', 0)}")
    print(f"Conflict rate: {stats.get('conflict_rate', 0):.1%}")
    
    if stats.get('resolution_stats'):
        print(f"\nResolution strategies used:")
        for strategy, count in stats['resolution_stats'].get('by_strategy', {}).items():
            print(f"  • {strategy}: {count}")
        print(f"Average confidence: {stats['resolution_stats'].get('avg_confidence', 0):.1%}")
    
    return system


def run_batch_analysis(house: str = "A", sample_size: int = 10000):
    """Run batch analysis on actual dataset."""
    print("=" * 70)
    print(f"BATCH ANALYSIS - HOUSE {house}")
    print("=" * 70)
    
    # Initialize system
    system = IntegratedRecommendationSystem(house=house)
    system.load_model()
    
    # Load data
    data_path = HOUSE_A_DATA if house == "A" else HOUSE_B_DATA
    
    try:
        print(f"\nLoading data from {data_path}...")
        data = pd.read_csv(data_path, low_memory=False)
        print(f"Loaded {len(data):,} records")
        
        # Process batch
        results = system.batch_process(data, sample_size=sample_size)
        
        # Analyze results
        print("\n" + "-" * 70)
        print("BATCH ANALYSIS RESULTS")
        print("-" * 70)
        
        total = len(results)
        conflicts = results['has_conflict'].sum()
        
        print(f"\nTotal processed: {total:,}")
        print(f"Conflicts found: {conflicts:,} ({conflicts/total*100:.2f}%)")
        
        if conflicts > 0:
            print(f"\nConflict Types:")
            for ctype, count in results[results['has_conflict']]['conflict_type'].value_counts().items():
                print(f"  • {ctype}: {count:,} ({count/conflicts*100:.1f}%)")
            
            print(f"\nConflict Severities:")
            for sev, count in results[results['has_conflict']]['conflict_severity'].value_counts().items():
                print(f"  • {sev}: {count:,} ({count/conflicts*100:.1f}%)")
            
            print(f"\nResolution Strategies:")
            for strat, count in results[results['has_conflict']]['resolution_strategy'].value_counts().items():
                print(f"  • {strat}: {count:,} ({count/conflicts*100:.1f}%)")
            
            print(f"\nTop Device Actions:")
            for action, count in results[results['has_conflict']]['primary_action'].value_counts().head(5).items():
                print(f"  • {action}: {count:,}")
        
        # Save results
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"house_{house.lower()}_batch_results.csv"
        results.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to {results_path}")
        
        # Export system stats
        system.export_results(str(output_dir))
        
        return results
        
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Running demo scenarios instead...")
        return run_demo_scenarios()


def interactive_mode():
    """Run interactive recommendation mode."""
    print("=" * 70)
    print("INTERACTIVE RECOMMENDATION MODE")
    print("=" * 70)
    
    system = IntegratedRecommendationSystem()
    system.load_model()
    
    print("\nAvailable Activities:")
    print("-" * 40)
    for aid, profile in sorted(ACTIVITY_PROFILES.items()):
        print(f"  {aid:2d}: {profile['name']}")
    
    print("\nEnter activity IDs for each resident (or 'q' to quit)")
    print("-" * 40)
    
    while True:
        try:
            r1_input = input("\nResident 1 Activity ID: ").strip()
            if r1_input.lower() == 'q':
                break
            
            r2_input = input("Resident 2 Activity ID: ").strip()
            if r2_input.lower() == 'q':
                break
            
            r1 = int(r1_input)
            r2 = int(r2_input)
            
            if r1 not in ACTIVITY_PROFILES or r2 not in ACTIVITY_PROFILES:
                print("Invalid activity ID. Please try again.")
                continue
            
            result = system.get_recommendations(r1, r2)
            
            print(f"\n{'─' * 50}")
            print(f"R1: {result['activity_r1_name']}")
            print(f"R2: {result['activity_r2_name']}")
            
            if result['has_conflict']:
                print(f"\n⚠️  CONFLICT: {result['conflict']['type']} ({result['conflict']['severity']})")
                print(f"Resolution: {result['resolution']['strategy']} ({result['resolution']['confidence']:.0%})")
                print(f"\nRecommendations:")
                for rec in result['recommendations'][:3]:
                    print(f"  • [{rec['device']}] {rec['action']}: {rec['value']}")
            else:
                print("\n✓ No conflict detected")
            
        except ValueError:
            print("Please enter valid numeric IDs.")
        except KeyboardInterrupt:
            break
    
    print("\n✓ Session ended")
    stats = system.get_statistics()
    print(f"Total queries: {stats.get('total_recommendations', 0)}")
    print(f"Conflicts resolved: {stats.get('total_resolutions', 0)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    import sys
    
    print("=" * 70)
    print("MULTI-RESIDENT SMART HOME RECOMMENDATION SYSTEM")
    print("=" * 70)
    print("\nOptions:")
    print("  1. Run demo scenarios")
    print("  2. Batch analysis (House A)")
    print("  3. Batch analysis (House B)")
    print("  4. Interactive mode")
    print("  5. Quick demo (default)")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = "5"
    
    if choice == "1":
        run_demo_scenarios()
    elif choice == "2":
        run_batch_analysis("A")
    elif choice == "3":
        run_batch_analysis("B")
    elif choice == "4":
        interactive_mode()
    else:
        run_demo_scenarios()


if __name__ == "__main__":
    main()
