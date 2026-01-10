"""
FP-Growth Runner Script for ARAS Dataset
=========================================

Run this script to mine patterns from your preprocessed ARAS data.

Usage:
    python run_fpgrowth.py

Make sure to update the paths below to match your directory structure.

Requirements:
    pip install mlxtend pandas numpy
"""

from fpgrowth_multi_resident import (
    MultiResidentPatternMiner, 
    FPGrowthConfig,
    mine_patterns,
    combine_house_patterns
)
from pathlib import Path

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Input files (update these to your actual paths)
HOUSE_A_TRANSACTIONS = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\a_output\fpgrowth_input.txt"
HOUSE_B_TRANSACTIONS = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\b_output\fpgrowth_input.txt"

# Output directories
HOUSE_A_OUTPUT = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\a_output\patterns"
HOUSE_B_OUTPUT = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\b_output\patterns"
COMBINED_OUTPUT = r"D:\My files\Post-doc\Pr\Papers\AI & IoT\Project\datasets\combined_patterns"

# Mining parameters
MIN_SUPPORT = 0.01      # 1% minimum support
MIN_CONFIDENCE = 0.5    # 50% minimum confidence
MIN_LIFT = 1.0          # Positive correlation only


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_house_a():
    """Mine patterns from House A only."""
    print("\n" + "=" * 70)
    print("MINING HOUSE A PATTERNS")
    print("=" * 70)
    
    results = mine_patterns(
        transaction_file=HOUSE_A_TRANSACTIONS,
        output_dir=HOUSE_A_OUTPUT,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        prefix="house_a"
    )
    return results


def run_house_b():
    """Mine patterns from House B only."""
    print("\n" + "=" * 70)
    print("MINING HOUSE B PATTERNS")
    print("=" * 70)
    
    results = mine_patterns(
        transaction_file=HOUSE_B_TRANSACTIONS,
        output_dir=HOUSE_B_OUTPUT,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        prefix="house_b"
    )
    return results


def run_combined():
    """Mine patterns from combined House A + B data."""
    print("\n" + "=" * 70)
    print("MINING COMBINED PATTERNS (House A + B)")
    print("=" * 70)
    
    results = combine_house_patterns(
        house_a_file=HOUSE_A_TRANSACTIONS,
        house_b_file=HOUSE_B_TRANSACTIONS,
        output_dir=COMBINED_OUTPUT,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE
    )
    return results


def run_all():
    """Run complete analysis on all data."""
    print("=" * 70)
    print("COMPLETE FP-GROWTH ANALYSIS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Min Support: {MIN_SUPPORT}")
    print(f"  Min Confidence: {MIN_CONFIDENCE}")
    print(f"  Min Lift: {MIN_LIFT}")
    
    results = {}
    
    # House A
    try:
        results['house_a'] = run_house_a()
    except Exception as e:
        print(f"Error processing House A: {e}")
    
    # House B
    try:
        results['house_b'] = run_house_b()
    except Exception as e:
        print(f"Error processing House B: {e}")
    
    # Combined
    try:
        results['combined'] = run_combined()
    except Exception as e:
        print(f"Error processing combined: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - FINAL SUMMARY")
    print("=" * 70)
    
    for name, res in results.items():
        if res:
            print(f"\n{name.upper()}:")
            print(f"  Frequent itemsets: {res.summary.get('total_frequent_itemsets', 0):,}")
            print(f"  Association rules: {res.summary.get('total_association_rules', 0):,}")
            print(f"  Activity patterns: {res.summary.get('activity_patterns', 0):,}")
            print(f"  Conflict patterns: {res.summary.get('conflict_patterns', 0):,}")
    
    return results


def explore_patterns(results):
    """Interactive pattern exploration."""
    print("\n" + "=" * 70)
    print("PATTERN EXPLORATION")
    print("=" * 70)
    
    # Top activity patterns
    print("\n📊 TOP 5 ACTIVITY PATTERNS:")
    print("-" * 50)
    for i, p in enumerate(results.activity_patterns[:5], 1):
        print(f"\n  {i}. IF {' AND '.join(p['antecedent'])}")
        print(f"     THEN {' AND '.join(p['consequent'])}")
        print(f"     Confidence: {p['confidence']*100:.1f}%, Lift: {p['lift']:.2f}")
    
    # Top conflict patterns
    print("\n\n⚠️ TOP 5 CONFLICT PATTERNS:")
    print("-" * 50)
    for i, p in enumerate(results.conflict_patterns[:5], 1):
        print(f"\n  {i}. IF {' AND '.join(p['antecedent'])}")
        print(f"     THEN {' AND '.join(p['consequent'])}")
        print(f"     Confidence: {p['confidence']*100:.1f}%, Lift: {p['lift']:.2f}")
    
    # Top temporal patterns
    print("\n\n⏰ TOP 5 TEMPORAL PATTERNS:")
    print("-" * 50)
    for i, p in enumerate(results.temporal_patterns[:5], 1):
        print(f"\n  {i}. IF {' AND '.join(p['antecedent'])}")
        print(f"     THEN {' AND '.join(p['consequent'])}")
        print(f"     Confidence: {p['confidence']*100:.1f}%, Lift: {p['lift']:.2f}")
    
    # Multi-resident specific
    print("\n\n👥 MULTI-RESIDENT INTERACTION PATTERNS:")
    print("-" * 50)
    
    # Find patterns where both R1 and R2 appear
    multi_patterns = [p for p in results.activity_patterns 
                     if p['r1_activities'] and p['r2_activities']]
    
    for i, p in enumerate(multi_patterns[:5], 1):
        print(f"\n  {i}. R1: {p['r1_activities']}, R2: {p['r2_activities']}")
        print(f"     Rule: {' + '.join(p['antecedent'])} => {' + '.join(p['consequent'])}")
        print(f"     Lift: {p['lift']:.2f}")


if __name__ == "__main__":
    # Run complete analysis
    results = run_all()
    
    # Explore combined patterns
    if 'combined' in results and results['combined']:
        explore_patterns(results['combined'])
