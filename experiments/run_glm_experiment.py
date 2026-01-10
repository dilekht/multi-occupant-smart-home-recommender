"""
GLM Experiment Runner for ARAS Multi-Resident Dataset
======================================================

Run this script to train and evaluate the Extended GLM models.

Usage:
    python run_glm_experiment.py

Make sure to update the paths below to match your directory structure.

Requirements:
    pip install scikit-learn pandas numpy scipy joblib
"""

from glm_multi_resident import (
    MultiResidentGLM,
    SingleOccupantBaseline,
    ModelConfig,
    run_experiment
)
from pathlib import Path
import pandas as pd
import json

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Input: Processed data files from preprocessing step
HOUSE_A_PROCESSED = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\a_output\processed_data.csv"
HOUSE_B_PROCESSED = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\b_output\processed_data.csv"

# Output directories
HOUSE_A_OUTPUT = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\a_output\glm_results"
HOUSE_B_OUTPUT = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\b_output\glm_results"
COMBINED_OUTPUT = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\combined_glm_results"


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def run_house_a():
    """Run experiment on House A data."""
    print("\n" + "=" * 70)
    print("HOUSE A EXPERIMENT (Married Couple)")
    print("=" * 70)
    
    results = run_experiment(
        processed_data_path=HOUSE_A_PROCESSED,
        output_dir=HOUSE_A_OUTPUT,
        house_name="house_a"
    )
    return results


def run_house_b():
    """Run experiment on House B data."""
    print("\n" + "=" * 70)
    print("HOUSE B EXPERIMENT (Male Roommates)")
    print("=" * 70)
    
    results = run_experiment(
        processed_data_path=HOUSE_B_PROCESSED,
        output_dir=HOUSE_B_OUTPUT,
        house_name="house_b"
    )
    return results


def run_combined():
    """Run experiment on combined House A + B data."""
    print("\n" + "=" * 70)
    print("COMBINED EXPERIMENT (House A + B)")
    print("=" * 70)
    
    # Load and combine data with low_memory=False
    print("\nLoading and combining data...")
    df_a = pd.read_csv(HOUSE_A_PROCESSED, low_memory=False)
    df_b = pd.read_csv(HOUSE_B_PROCESSED, low_memory=False)
    
    # Add house identifier
    df_a['SourceHouse'] = 'A'
    df_b['SourceHouse'] = 'B'
    
    # Combine
    df_combined = pd.concat([df_a, df_b], ignore_index=True)
    print(f"Combined dataset: {len(df_combined):,} records")
    
    # Save combined data temporarily
    combined_path = Path(COMBINED_OUTPUT)
    combined_path.mkdir(parents=True, exist_ok=True)
    combined_data_path = combined_path / "combined_processed_data.csv"
    df_combined.to_csv(combined_data_path, index=False)
    
    # Run experiment
    results = run_experiment(
        processed_data_path=str(combined_data_path),
        output_dir=COMBINED_OUTPUT,
        house_name="combined"
    )
    return results


def run_all_experiments():
    """Run all experiments and generate comparison report."""
    print("=" * 70)
    print("COMPLETE GLM EXPERIMENT SUITE")
    print("=" * 70)
    
    all_results = {}
    
    # Run individual house experiments
    try:
        all_results['house_a'] = run_house_a()
    except Exception as e:
        print(f"Error in House A experiment: {e}")
        
    try:
        all_results['house_b'] = run_house_b()
    except Exception as e:
        print(f"Error in House B experiment: {e}")
    
    # Run combined experiment
    try:
        all_results['combined'] = run_combined()
    except Exception as e:
        print(f"Error in Combined experiment: {e}")
    
    # Generate comparison report
    generate_comparison_report(all_results)
    
    return all_results


def generate_comparison_report(all_results: dict):
    """Generate a comprehensive comparison report."""
    print("\n" + "=" * 70)
    print("FINAL COMPARISON REPORT")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("ACCURACY COMPARISON")
    print("-" * 70)
    
    print(f"\n{'Dataset':<15} {'Baseline R1':>12} {'Multi R1':>10} {'Δ R1':>8} "
          f"{'Baseline R2':>12} {'Multi R2':>10} {'Δ R2':>8}")
    print("-" * 85)
    
    for name, results in all_results.items():
        if results and 'comparison' in results:
            comp = results['comparison']
            models = results['models']
            
            b_r1 = models['baseline_r1'].accuracy
            m_r1 = models['multi_r1'].accuracy
            d_r1 = (m_r1 - b_r1) * 100
            
            b_r2 = models['baseline_r2'].accuracy
            m_r2 = models['multi_r2'].accuracy
            d_r2 = (m_r2 - b_r2) * 100
            
            print(f"{name:<15} {b_r1:>12.4f} {m_r1:>10.4f} {d_r1:>+7.2f}% "
                  f"{b_r2:>12.4f} {m_r2:>10.4f} {d_r2:>+7.2f}%")
    
    print("\n" + "-" * 70)
    print("CONFLICT PREDICTION")
    print("-" * 70)
    
    print(f"\n{'Dataset':<15} {'Conflict F1':>12} {'Joint Exact Match':>18}")
    print("-" * 50)
    
    for name, results in all_results.items():
        if results and 'models' in results:
            models = results['models']
            conflict_f1 = models['multi_conflict'].f1 if models.get('multi_conflict') else 0
            joint_acc = models['multi_joint'].accuracy if models.get('multi_joint') else 0
            print(f"{name:<15} {conflict_f1:>12.4f} {joint_acc:>18.4f}")
    
    print("\n" + "-" * 70)
    print("KEY INSIGHTS")
    print("-" * 70)
    
    # Analyze results
    if 'house_a' in all_results and 'house_b' in all_results:
        a_improvement = all_results['house_a']['comparison']['improvement_percent']
        b_improvement = all_results['house_b']['comparison']['improvement_percent']
        
        print(f"\n1. House A (Couple) improvement: {a_improvement:+.2f}%")
        print(f"2. House B (Roommates) improvement: {b_improvement:+.2f}%")
        
        if a_improvement > b_improvement:
            print("\n   → Multi-resident features MORE beneficial for couples")
            print("   → Cross-resident features capture shared lifestyle patterns")
        else:
            print("\n   → Multi-resident features MORE beneficial for roommates")
            print("   → Cross-resident features help predict independent behaviors")
        
        # Conflict analysis
        a_conflict = all_results['house_a']['models'].get('multi_conflict')
        b_conflict = all_results['house_b']['models'].get('multi_conflict')
        
        if a_conflict and a_conflict.f1 > 0:
            print(f"\n3. Conflict prediction F1 (House A): {a_conflict.f1:.4f}")
        if b_conflict and b_conflict.f1 > 0:
            print(f"   Conflict prediction F1 (House B): {b_conflict.f1:.4f}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


# =============================================================================
# CUSTOM EXPERIMENT CONFIGURATION
# =============================================================================

def run_custom_experiment(
    data_path: str,
    output_dir: str,
    name: str,
    use_lag_features: bool = True,
    lag_window: int = 5,
    use_conflict_features: bool = True,
    test_size: float = 0.25
):
    """
    Run experiment with custom configuration.
    
    Args:
        data_path: Path to processed_data.csv
        output_dir: Output directory
        name: Experiment name
        use_lag_features: Whether to use lag features
        lag_window: Number of lag timesteps
        use_conflict_features: Whether to use conflict features
        test_size: Test set proportion
    """
    config = ModelConfig(
        use_lag_features=use_lag_features,
        lag_window=lag_window,
        use_conflict_features=use_conflict_features,
        test_size=test_size
    )
    
    print(f"\nCustom Experiment: {name}")
    print(f"  Lag features: {use_lag_features} (window={lag_window})")
    print(f"  Conflict features: {use_conflict_features}")
    print(f"  Test size: {test_size}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Create model
    model = MultiResidentGLM(config)
    X_train, X_test, y_train, y_test = model.prepare_data(df)
    model.fit(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path / f"{name}_model.pkl"))
    
    return model, results


# =============================================================================
# ABLATION STUDY
# =============================================================================

def run_ablation_study(data_path: str, output_dir: str):
    """
    Run ablation study to analyze impact of each feature group.
    
    Tests:
    1. Base features only (sensors + temporal)
    2. + Lag features
    3. + Cross-resident features
    4. + Conflict features
    5. All features (full model)
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    
    df = pd.read_csv(data_path, low_memory=False)
    
    # Handle data types
    if 'ConflictType' in df.columns:
        df['ConflictType'] = df['ConflictType'].fillna('none').astype(str)
    if 'ConflictSeverity' in df.columns:
        df['ConflictSeverity'] = df['ConflictSeverity'].fillna('none').astype(str)
    if 'HasConflict' not in df.columns:
        df['HasConflict'] = 0
    df['HasConflict'] = pd.to_numeric(df['HasConflict'], errors='coerce').fillna(0).astype(int)
    df['Activity_R1'] = pd.to_numeric(df['Activity_R1'], errors='coerce').fillna(1).astype(int)
    df['Activity_R2'] = pd.to_numeric(df['Activity_R2'], errors='coerce').fillna(1).astype(int)
    
    configurations = [
        ("Base (Sensors+Temporal)", {
            'use_lag_features': False,
            'use_cross_resident_features': False,
            'use_conflict_features': False
        }),
        ("+ Lag Features", {
            'use_lag_features': True,
            'lag_window': 5,
            'use_cross_resident_features': False,
            'use_conflict_features': False
        }),
        ("+ Cross-Resident", {
            'use_lag_features': True,
            'lag_window': 5,
            'use_cross_resident_features': True,
            'use_conflict_features': False
        }),
        ("Full Model", {
            'use_lag_features': True,
            'lag_window': 5,
            'use_cross_resident_features': True,
            'use_conflict_features': True
        }),
    ]
    
    ablation_results = []
    
    for name, params in configurations:
        print(f"\n--- {name} ---")
        
        config = ModelConfig(**params)
        model = MultiResidentGLM(config)
        X_train, X_test, y_train, y_test = model.prepare_data(df)
        model.fit(X_train, y_train)
        results = model.evaluate(X_test, y_test)
        
        ablation_results.append({
            'configuration': name,
            'r1_accuracy': results['R1_Activity'].accuracy,
            'r2_accuracy': results['R2_Activity'].accuracy,
            'avg_accuracy': (results['R1_Activity'].accuracy + results['R2_Activity'].accuracy) / 2
        })
    
    # Print summary
    print("\n" + "-" * 70)
    print("ABLATION STUDY RESULTS")
    print("-" * 70)
    
    print(f"\n{'Configuration':<25} {'R1 Acc':>10} {'R2 Acc':>10} {'Avg Acc':>10}")
    print("-" * 57)
    
    base_acc = ablation_results[0]['avg_accuracy']
    for result in ablation_results:
        delta = (result['avg_accuracy'] - base_acc) / base_acc * 100 if base_acc > 0 else 0
        print(f"{result['configuration']:<25} {result['r1_accuracy']:>10.4f} "
              f"{result['r2_accuracy']:>10.4f} {result['avg_accuracy']:>10.4f} ({delta:+.1f}%)")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "ablation_results.json", 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    return ablation_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run all experiments
    results = run_all_experiments()
    
    # Optionally run ablation study on combined data
    # run_ablation_study(
    #     data_path=str(Path(COMBINED_OUTPUT) / "combined_processed_data.csv"),
    #     output_dir=COMBINED_OUTPUT
    # )
