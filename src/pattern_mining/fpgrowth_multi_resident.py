"""
FP-Growth Pattern Mining for Multi-Resident Smart Home
=======================================================

This module discovers frequent patterns in multi-resident smart home data
using the FP-Growth algorithm, extending Dilekh et al. (2024) methodology.

Key pattern types discovered:
1. Multi-resident activity associations
2. Sensor-activity correlations
3. Temporal-activity patterns
4. Conflict-related patterns
5. Synchronization patterns

Requirements:
    pip install mlxtend pandas numpy

Author: Research Extension Project
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import json
import warnings
from dataclasses import dataclass, field
import time

# Try to import mlxtend, provide installation instructions if not available
try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("=" * 60)
    print("WARNING: mlxtend not installed!")
    print("Install with: pip install mlxtend")
    print("=" * 60)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FPGrowthConfig:
    """Configuration for FP-Growth mining."""
    min_support: float = 0.01  # Minimum support threshold (1%)
    min_confidence: float = 0.5  # Minimum confidence for rules (50%)
    min_lift: float = 1.0  # Minimum lift (>1 means positive correlation)
    max_itemset_length: Optional[int] = None  # Max items in itemset
    
    # Pattern filtering
    focus_on_activities: bool = True  # Prioritize activity patterns
    focus_on_conflicts: bool = True  # Prioritize conflict patterns
    include_sensors: bool = True  # Include sensor patterns
    include_temporal: bool = True  # Include temporal patterns


@dataclass
class PatternResults:
    """Container for mining results."""
    frequent_itemsets: pd.DataFrame = None
    association_rules: pd.DataFrame = None
    activity_patterns: List[Dict] = field(default_factory=list)
    conflict_patterns: List[Dict] = field(default_factory=list)
    temporal_patterns: List[Dict] = field(default_factory=list)
    sensor_patterns: List[Dict] = field(default_factory=list)
    sync_patterns: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


# =============================================================================
# FP-GROWTH MINER CLASS
# =============================================================================

class MultiResidentPatternMiner:
    """
    FP-Growth pattern miner for multi-resident smart home data.
    
    Discovers frequent patterns and association rules specifically
    designed for multi-occupant context-aware recommendations.
    """
    
    def __init__(self, config: Optional[FPGrowthConfig] = None):
        """
        Initialize the pattern miner.
        
        Args:
            config: Mining configuration
        """
        self.config = config or FPGrowthConfig()
        self.transactions: List[List[str]] = []
        self.results = PatternResults()
        self._encoded_df: Optional[pd.DataFrame] = None
        
    def load_transactions_from_file(self, filepath: str) -> int:
        """
        Load transactions from FP-Growth input file.
        
        Args:
            filepath: Path to fpgrowth_input.txt
            
        Returns:
            Number of transactions loaded
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            for line in f:
                items = line.strip().split()
                if items:
                    self.transactions.append(items)
        
        print(f"Loaded {len(self.transactions):,} transactions from {filepath.name}")
        return len(self.transactions)
    
    def load_transactions_from_df(self, df: pd.DataFrame, 
                                   items_column: str = "items") -> int:
        """
        Load transactions from DataFrame.
        
        Args:
            df: DataFrame with transaction data
            items_column: Column containing item lists
            
        Returns:
            Number of transactions loaded
        """
        for _, row in df.iterrows():
            items = row[items_column]
            if isinstance(items, str):
                items = eval(items)  # Convert string repr to list
            self.transactions.append(list(items))
        
        print(f"Loaded {len(self.transactions):,} transactions")
        return len(self.transactions)
    
    def _encode_transactions(self) -> pd.DataFrame:
        """Encode transactions into binary matrix for mlxtend."""
        if not MLXTEND_AVAILABLE:
            raise ImportError("mlxtend is required. Install with: pip install mlxtend")
        
        print("Encoding transactions...")
        te = TransactionEncoder()
        te_array = te.fit_transform(self.transactions)
        df = pd.DataFrame(te_array, columns=te.columns_)
        
        print(f"Encoded matrix: {df.shape[0]:,} transactions × {df.shape[1]:,} items")
        self._encoded_df = df
        return df
    
    def mine_frequent_itemsets(self) -> pd.DataFrame:
        """
        Run FP-Growth to find frequent itemsets.
        
        Returns:
            DataFrame with frequent itemsets and support
        """
        if not MLXTEND_AVAILABLE:
            raise ImportError("mlxtend is required. Install with: pip install mlxtend")
        
        if self._encoded_df is None:
            self._encode_transactions()
        
        print(f"\nRunning FP-Growth (min_support={self.config.min_support})...")
        start_time = time.time()
        
        # Run FP-Growth
        frequent_itemsets = fpgrowth(
            self._encoded_df,
            min_support=self.config.min_support,
            use_colnames=True,
            max_len=self.config.max_itemset_length
        )
        
        elapsed = time.time() - start_time
        print(f"Found {len(frequent_itemsets):,} frequent itemsets in {elapsed:.2f}s")
        
        # Add itemset length
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
        
        # Sort by support
        frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
        
        self.results.frequent_itemsets = frequent_itemsets
        return frequent_itemsets
    
    def generate_association_rules(self) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.
        
        Returns:
            DataFrame with association rules
        """
        if self.results.frequent_itemsets is None:
            self.mine_frequent_itemsets()
        
        print(f"\nGenerating association rules (min_confidence={self.config.min_confidence})...")
        
        # Generate rules
        rules = association_rules(
            self.results.frequent_itemsets,
            metric="confidence",
            min_threshold=self.config.min_confidence
        )
        
        # Filter by lift
        rules = rules[rules['lift'] >= self.config.min_lift]
        
        # Sort by lift (most interesting rules first)
        rules = rules.sort_values('lift', ascending=False)
        
        print(f"Generated {len(rules):,} association rules")
        
        self.results.association_rules = rules
        return rules
    
    def extract_activity_patterns(self) -> List[Dict]:
        """Extract patterns involving resident activities."""
        if self.results.association_rules is None:
            self.generate_association_rules()
        
        patterns = []
        rules = self.results.association_rules
        
        for _, rule in rules.iterrows():
            antecedent = set(rule['antecedents'])
            consequent = set(rule['consequents'])
            all_items = antecedent | consequent
            
            # Check if rule involves activities
            r1_acts = [i for i in all_items if i.startswith('R1:')]
            r2_acts = [i for i in all_items if i.startswith('R2:')]
            
            if r1_acts or r2_acts:
                pattern = {
                    'antecedent': list(antecedent),
                    'consequent': list(consequent),
                    'support': round(rule['support'], 4),
                    'confidence': round(rule['confidence'], 4),
                    'lift': round(rule['lift'], 4),
                    'r1_activities': r1_acts,
                    'r2_activities': r2_acts,
                    'type': 'activity_pattern'
                }
                patterns.append(pattern)
        
        # Sort by lift
        patterns.sort(key=lambda x: x['lift'], reverse=True)
        
        self.results.activity_patterns = patterns
        print(f"Extracted {len(patterns)} activity patterns")
        return patterns
    
    def extract_conflict_patterns(self) -> List[Dict]:
        """Extract patterns related to conflicts."""
        if self.results.association_rules is None:
            self.generate_association_rules()
        
        patterns = []
        rules = self.results.association_rules
        
        for _, rule in rules.iterrows():
            antecedent = set(rule['antecedents'])
            consequent = set(rule['consequents'])
            all_items = antecedent | consequent
            
            # Check if rule involves conflicts
            conflict_items = [i for i in all_items if 'CONFLICT' in i]
            
            if conflict_items:
                # Get context (what leads to or accompanies conflicts)
                context_items = [i for i in all_items if 'CONFLICT' not in i]
                
                pattern = {
                    'antecedent': list(antecedent),
                    'consequent': list(consequent),
                    'support': round(rule['support'], 4),
                    'confidence': round(rule['confidence'], 4),
                    'lift': round(rule['lift'], 4),
                    'conflict_indicators': conflict_items,
                    'context': context_items,
                    'type': 'conflict_pattern'
                }
                patterns.append(pattern)
        
        patterns.sort(key=lambda x: x['lift'], reverse=True)
        
        self.results.conflict_patterns = patterns
        print(f"Extracted {len(patterns)} conflict patterns")
        return patterns
    
    def extract_temporal_patterns(self) -> List[Dict]:
        """Extract patterns involving time context."""
        if self.results.association_rules is None:
            self.generate_association_rules()
        
        patterns = []
        rules = self.results.association_rules
        
        for _, rule in rules.iterrows():
            antecedent = set(rule['antecedents'])
            consequent = set(rule['consequents'])
            all_items = antecedent | consequent
            
            # Check if rule involves temporal items
            temporal_items = [i for i in all_items 
                           if i.startswith(('HOUR:', 'TOD:', 'WEEKEND:'))]
            
            if temporal_items:
                activity_items = [i for i in all_items 
                                if i.startswith(('R1:', 'R2:'))]
                
                if activity_items:  # Must also involve activities
                    pattern = {
                        'antecedent': list(antecedent),
                        'consequent': list(consequent),
                        'support': round(rule['support'], 4),
                        'confidence': round(rule['confidence'], 4),
                        'lift': round(rule['lift'], 4),
                        'temporal_context': temporal_items,
                        'activities': activity_items,
                        'type': 'temporal_pattern'
                    }
                    patterns.append(pattern)
        
        patterns.sort(key=lambda x: x['lift'], reverse=True)
        
        self.results.temporal_patterns = patterns
        print(f"Extracted {len(patterns)} temporal patterns")
        return patterns
    
    def extract_sync_patterns(self) -> List[Dict]:
        """Extract patterns related to activity synchronization."""
        if self.results.association_rules is None:
            self.generate_association_rules()
        
        patterns = []
        rules = self.results.association_rules
        
        for _, rule in rules.iterrows():
            antecedent = set(rule['antecedents'])
            consequent = set(rule['consequents'])
            all_items = antecedent | consequent
            
            # Check if rule involves sync status
            sync_items = [i for i in all_items if i.startswith('SYNC:')]
            
            if sync_items:
                pattern = {
                    'antecedent': list(antecedent),
                    'consequent': list(consequent),
                    'support': round(rule['support'], 4),
                    'confidence': round(rule['confidence'], 4),
                    'lift': round(rule['lift'], 4),
                    'sync_status': sync_items,
                    'context': [i for i in all_items if not i.startswith('SYNC:')],
                    'type': 'sync_pattern'
                }
                patterns.append(pattern)
        
        patterns.sort(key=lambda x: x['lift'], reverse=True)
        
        self.results.sync_patterns = patterns
        print(f"Extracted {len(patterns)} synchronization patterns")
        return patterns
    
    def extract_sensor_activity_patterns(self) -> List[Dict]:
        """Extract patterns linking sensors to activities."""
        if self.results.association_rules is None:
            self.generate_association_rules()
        
        patterns = []
        rules = self.results.association_rules
        
        for _, rule in rules.iterrows():
            antecedent = set(rule['antecedents'])
            consequent = set(rule['consequents'])
            all_items = antecedent | consequent
            
            # Check if rule involves sensors and activities
            sensor_items = [i for i in all_items if i.startswith('SENSOR:')]
            activity_items = [i for i in all_items if i.startswith(('R1:', 'R2:'))]
            
            if sensor_items and activity_items:
                pattern = {
                    'antecedent': list(antecedent),
                    'consequent': list(consequent),
                    'support': round(rule['support'], 4),
                    'confidence': round(rule['confidence'], 4),
                    'lift': round(rule['lift'], 4),
                    'sensors': sensor_items,
                    'activities': activity_items,
                    'type': 'sensor_activity_pattern'
                }
                patterns.append(pattern)
        
        patterns.sort(key=lambda x: x['lift'], reverse=True)
        
        self.results.sensor_patterns = patterns
        print(f"Extracted {len(patterns)} sensor-activity patterns")
        return patterns
    
    def run_full_analysis(self) -> PatternResults:
        """
        Run complete pattern mining analysis.
        
        Returns:
            PatternResults with all discovered patterns
        """
        print("=" * 70)
        print("MULTI-RESIDENT PATTERN MINING")
        print("=" * 70)
        
        # Mine frequent itemsets
        self.mine_frequent_itemsets()
        
        # Generate association rules
        self.generate_association_rules()
        
        # Extract specific pattern types
        print("\nExtracting pattern categories...")
        self.extract_activity_patterns()
        self.extract_conflict_patterns()
        self.extract_temporal_patterns()
        self.extract_sync_patterns()
        self.extract_sensor_activity_patterns()
        
        # Generate summary
        self._generate_summary()
        
        print("\n" + "=" * 70)
        print("MINING COMPLETE")
        print("=" * 70)
        
        return self.results
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        summary = {
            'total_transactions': len(self.transactions),
            'total_frequent_itemsets': len(self.results.frequent_itemsets) if self.results.frequent_itemsets is not None else 0,
            'total_association_rules': len(self.results.association_rules) if self.results.association_rules is not None else 0,
            'activity_patterns': len(self.results.activity_patterns),
            'conflict_patterns': len(self.results.conflict_patterns),
            'temporal_patterns': len(self.results.temporal_patterns),
            'sync_patterns': len(self.results.sync_patterns),
            'sensor_patterns': len(self.results.sensor_patterns),
            'config': {
                'min_support': self.config.min_support,
                'min_confidence': self.config.min_confidence,
                'min_lift': self.config.min_lift
            }
        }
        
        # Top patterns summary
        if self.results.association_rules is not None and len(self.results.association_rules) > 0:
            rules = self.results.association_rules
            summary['avg_support'] = round(rules['support'].mean(), 4)
            summary['avg_confidence'] = round(rules['confidence'].mean(), 4)
            summary['avg_lift'] = round(rules['lift'].mean(), 4)
            summary['max_lift'] = round(rules['lift'].max(), 4)
        
        self.results.summary = summary
        return summary
    
    def get_top_patterns(self, n: int = 20, pattern_type: str = 'all') -> List[Dict]:
        """
        Get top N patterns by lift.
        
        Args:
            n: Number of patterns to return
            pattern_type: 'all', 'activity', 'conflict', 'temporal', 'sync', 'sensor'
        """
        if pattern_type == 'activity':
            patterns = self.results.activity_patterns
        elif pattern_type == 'conflict':
            patterns = self.results.conflict_patterns
        elif pattern_type == 'temporal':
            patterns = self.results.temporal_patterns
        elif pattern_type == 'sync':
            patterns = self.results.sync_patterns
        elif pattern_type == 'sensor':
            patterns = self.results.sensor_patterns
        else:
            # Combine all patterns
            patterns = (
                self.results.activity_patterns +
                self.results.conflict_patterns +
                self.results.temporal_patterns +
                self.results.sync_patterns +
                self.results.sensor_patterns
            )
        
        # Sort by lift and return top N
        patterns.sort(key=lambda x: x['lift'], reverse=True)
        return patterns[:n]
    
    def get_multi_resident_rules(self) -> pd.DataFrame:
        """Get rules that specifically involve both residents."""
        if self.results.association_rules is None:
            return pd.DataFrame()
        
        rules = self.results.association_rules.copy()
        
        def involves_both_residents(row):
            all_items = set(row['antecedents']) | set(row['consequents'])
            has_r1 = any(i.startswith('R1:') for i in all_items)
            has_r2 = any(i.startswith('R2:') for i in all_items)
            return has_r1 and has_r2
        
        multi_resident_rules = rules[rules.apply(involves_both_residents, axis=1)]
        return multi_resident_rules.sort_values('lift', ascending=False)
    
    def save_results(self, output_dir: str, prefix: str = "") -> Dict[str, Path]:
        """
        Save all results to files.
        
        Args:
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            Dictionary of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{prefix}_" if prefix else ""
        
        outputs = {}
        
        # 1. Frequent itemsets
        if self.results.frequent_itemsets is not None:
            path = output_dir / f"{prefix}frequent_itemsets.csv"
            # Convert frozensets to strings for CSV
            df = self.results.frequent_itemsets.copy()
            df['itemsets'] = df['itemsets'].apply(lambda x: ', '.join(sorted(x)))
            df.to_csv(path, index=False)
            outputs['frequent_itemsets'] = path
        
        # 2. Association rules
        if self.results.association_rules is not None:
            path = output_dir / f"{prefix}association_rules.csv"
            df = self.results.association_rules.copy()
            df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(sorted(x)))
            df['consequents'] = df['consequents'].apply(lambda x: ', '.join(sorted(x)))
            df.to_csv(path, index=False)
            outputs['association_rules'] = path
        
        # 3. Pattern categories (JSON)
        patterns_data = {
            'activity_patterns': self.results.activity_patterns[:100],  # Top 100
            'conflict_patterns': self.results.conflict_patterns[:100],
            'temporal_patterns': self.results.temporal_patterns[:100],
            'sync_patterns': self.results.sync_patterns[:100],
            'sensor_patterns': self.results.sensor_patterns[:100],
        }
        path = output_dir / f"{prefix}pattern_categories.json"
        with open(path, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        outputs['pattern_categories'] = path
        
        # 4. Summary
        path = output_dir / f"{prefix}mining_summary.json"
        with open(path, 'w') as f:
            json.dump(self.results.summary, f, indent=2)
        outputs['summary'] = path
        
        # 5. Multi-resident specific rules
        multi_rules = self.get_multi_resident_rules()
        if len(multi_rules) > 0:
            path = output_dir / f"{prefix}multi_resident_rules.csv"
            df = multi_rules.copy()
            df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(sorted(x)))
            df['consequents'] = df['consequents'].apply(lambda x: ', '.join(sorted(x)))
            df.to_csv(path, index=False)
            outputs['multi_resident_rules'] = path
        
        # 6. Human-readable report
        report = self._generate_report()
        path = output_dir / f"{prefix}pattern_report.txt"
        with open(path, 'w') as f:
            f.write(report)
        outputs['report'] = path
        
        print(f"\nSaved {len(outputs)} files to {output_dir}")
        return outputs
    
    def _generate_report(self) -> str:
        """Generate human-readable pattern report."""
        lines = []
        lines.append("=" * 70)
        lines.append("MULTI-RESIDENT PATTERN MINING REPORT")
        lines.append("=" * 70)
        
        # Summary
        lines.append("\n## SUMMARY")
        lines.append("-" * 40)
        for key, value in self.results.summary.items():
            if key != 'config':
                lines.append(f"  {key}: {value}")
        
        # Top activity patterns
        lines.append("\n## TOP 10 ACTIVITY PATTERNS (by lift)")
        lines.append("-" * 40)
        for i, p in enumerate(self.results.activity_patterns[:10], 1):
            lines.append(f"\n  {i}. {' + '.join(p['antecedent'])} => {' + '.join(p['consequent'])}")
            lines.append(f"     Support: {p['support']:.3f}, Confidence: {p['confidence']:.3f}, Lift: {p['lift']:.3f}")
        
        # Top conflict patterns
        lines.append("\n\n## TOP 10 CONFLICT PATTERNS (by lift)")
        lines.append("-" * 40)
        for i, p in enumerate(self.results.conflict_patterns[:10], 1):
            lines.append(f"\n  {i}. {' + '.join(p['antecedent'])} => {' + '.join(p['consequent'])}")
            lines.append(f"     Support: {p['support']:.3f}, Confidence: {p['confidence']:.3f}, Lift: {p['lift']:.3f}")
        
        # Top temporal patterns
        lines.append("\n\n## TOP 10 TEMPORAL PATTERNS (by lift)")
        lines.append("-" * 40)
        for i, p in enumerate(self.results.temporal_patterns[:10], 1):
            lines.append(f"\n  {i}. {' + '.join(p['antecedent'])} => {' + '.join(p['consequent'])}")
            lines.append(f"     Support: {p['support']:.3f}, Confidence: {p['confidence']:.3f}, Lift: {p['lift']:.3f}")
        
        # Multi-resident insights
        lines.append("\n\n## MULTI-RESIDENT INSIGHTS")
        lines.append("-" * 40)
        
        multi_rules = self.get_multi_resident_rules()
        if len(multi_rules) > 0:
            lines.append(f"  Rules involving both residents: {len(multi_rules)}")
            lines.append("\n  Top 5 multi-resident rules:")
            for i, (_, rule) in enumerate(multi_rules.head(5).iterrows(), 1):
                ant = ', '.join(sorted(rule['antecedents']))
                cons = ', '.join(sorted(rule['consequents']))
                lines.append(f"    {i}. {ant} => {cons}")
                lines.append(f"       Lift: {rule['lift']:.3f}")
        
        lines.append("\n" + "=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)
        
        return '\n'.join(lines)
    
    def print_summary(self) -> None:
        """Print summary to console."""
        print("\n" + "=" * 70)
        print("PATTERN MINING SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 Overview:")
        print(f"   Transactions analyzed: {self.results.summary.get('total_transactions', 0):,}")
        print(f"   Frequent itemsets: {self.results.summary.get('total_frequent_itemsets', 0):,}")
        print(f"   Association rules: {self.results.summary.get('total_association_rules', 0):,}")
        
        print(f"\n📁 Pattern Categories:")
        print(f"   Activity patterns: {self.results.summary.get('activity_patterns', 0):,}")
        print(f"   Conflict patterns: {self.results.summary.get('conflict_patterns', 0):,}")
        print(f"   Temporal patterns: {self.results.summary.get('temporal_patterns', 0):,}")
        print(f"   Sync patterns: {self.results.summary.get('sync_patterns', 0):,}")
        print(f"   Sensor patterns: {self.results.summary.get('sensor_patterns', 0):,}")
        
        if 'avg_lift' in self.results.summary:
            print(f"\n📈 Rule Quality:")
            print(f"   Avg support: {self.results.summary['avg_support']:.4f}")
            print(f"   Avg confidence: {self.results.summary['avg_confidence']:.4f}")
            print(f"   Avg lift: {self.results.summary['avg_lift']:.4f}")
            print(f"   Max lift: {self.results.summary['max_lift']:.4f}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def mine_patterns(transaction_file: str,
                  output_dir: str,
                  min_support: float = 0.01,
                  min_confidence: float = 0.5,
                  prefix: str = "") -> PatternResults:
    """
    Convenience function to run complete pattern mining.
    
    Args:
        transaction_file: Path to fpgrowth_input.txt
        output_dir: Output directory for results
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        prefix: Output filename prefix
        
    Returns:
        PatternResults object
    """
    config = FPGrowthConfig(
        min_support=min_support,
        min_confidence=min_confidence
    )
    
    miner = MultiResidentPatternMiner(config)
    miner.load_transactions_from_file(transaction_file)
    results = miner.run_full_analysis()
    miner.save_results(output_dir, prefix)
    miner.print_summary()
    
    return results


def combine_house_patterns(house_a_file: str, 
                           house_b_file: str,
                           output_dir: str,
                           min_support: float = 0.01,
                           min_confidence: float = 0.5) -> PatternResults:
    """
    Mine patterns from combined House A and B data.
    
    Args:
        house_a_file: Path to House A fpgrowth_input.txt
        house_b_file: Path to House B fpgrowth_input.txt
        output_dir: Output directory
        min_support: Minimum support
        min_confidence: Minimum confidence
        
    Returns:
        Combined PatternResults
    """
    config = FPGrowthConfig(
        min_support=min_support,
        min_confidence=min_confidence
    )
    
    miner = MultiResidentPatternMiner(config)
    
    # Load both houses
    print("Loading House A transactions...")
    miner.load_transactions_from_file(house_a_file)
    
    print("Loading House B transactions...")
    miner.load_transactions_from_file(house_b_file)
    
    print(f"\nTotal combined transactions: {len(miner.transactions):,}")
    
    # Run analysis
    results = miner.run_full_analysis()
    miner.save_results(output_dir, "combined")
    miner.print_summary()
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Demo execution."""
    print("=" * 70)
    print("FP-GROWTH MULTI-RESIDENT PATTERN MINER")
    print("=" * 70)
    
    print("\nUsage:")
    print("  from fpgrowth_multi_resident import mine_patterns, combine_house_patterns")
    print()
    print("  # Mine single house")
    print("  results = mine_patterns(")
    print("      'path/to/fpgrowth_input.txt',")
    print("      'output_dir/',")
    print("      min_support=0.01,")
    print("      min_confidence=0.5")
    print("  )")
    print()
    print("  # Mine combined houses")
    print("  results = combine_house_patterns(")
    print("      'house_a/fpgrowth_input.txt',")
    print("      'house_b/fpgrowth_input.txt',")
    print("      'combined_output/'")
    print("  )")
    print()
    print("  # Access results")
    print("  print(results.summary)")
    print("  top_patterns = results.activity_patterns[:10]")
    

if __name__ == "__main__":
    main()
