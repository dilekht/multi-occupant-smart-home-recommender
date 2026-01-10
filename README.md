# Multi-Occupant Smart Home Recommender System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Expert%20Systems%20with%20Applications-green.svg)](https://doi.org/XXXX)

A comprehensive framework for multi-occupant context-aware smart home recommendations, extending the FP-Growth and GLM methodology with conflict detection and resolution capabilities.

## 📋 Overview

This repository contains the implementation of our paper:

> **Multi-Occupant Context-Aware Recommender System for Smart Home Automation: An Extended FP-Growth and GLM Approach with Conflict Resolution**
> 
> Expert Systems with Applications, 2026

### Key Features

- ✅ **Multi-resident activity prediction** with 99.87% accuracy
- ✅ **Perfect conflict detection** (F1 = 1.0)
- ✅ **Five resolution strategies**: Priority, Compromise, Temporal, Spatial, Device-specific
- ✅ **Cross-resident feature engineering** for capturing household dynamics
- ✅ **FP-Growth pattern mining** for multi-resident behavioral patterns
- ✅ **Actionable device recommendations** for smart home automation

### Results Summary

| Metric | House A (Couple) | House B (Roommates) |
|--------|------------------|---------------------|
| Baseline Accuracy | 73.84% | 93.09% |
| Multi-Resident Accuracy | **99.86%** | **99.96%** |
| Improvement | **+35.25%** | **+7.38%** |
| Conflict Prediction F1 | 1.0000 | 0.9975 |
| Joint Exact Match | 99.70% | 99.91% |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SMART HOME RECOMMENDATION SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐                                                   │
│  │  ARAS Dataset    │                                                   │
│  │  (5.18M records) │                                                   │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐  │
│  │  Preprocessing   │────▶│   FP-Growth      │────▶│  Pattern        │  │
│  │  • Sensors       │     │   Pattern Mining │     │  Database       │  │
│  │  • Temporal      │     │   (86,400 trans) │     │  (475K+ rules)  │  │
│  │  • Cross-resident│     └──────────────────┘     └─────────────────┘  │
│  │  • Lag features  │                                       │           │
│  └────────┬─────────┘                                       │           │
│           │                                                  │           │
│           ▼                                                  ▼           │
│  ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐  │
│  │  Extended GLM    │────▶│  Conflict        │────▶│  Resolution     │  │
│  │  • R1 Activity   │     │  Detector        │     │  Strategies     │  │
│  │  • R2 Activity   │     │  • Noise         │     │  • Priority     │  │
│  │  • Conflict Prob │     │  • Distraction   │     │  • Compromise   │  │
│  │  • Joint Model   │     │  • Resource      │     │  • Temporal     │  │
│  └──────────────────┘     └──────────────────┘     │  • Spatial      │  │
│                                                     │  • Device       │  │
│                                                     └────────┬────────┘  │
│                                                              │           │
│                                                              ▼           │
│                                                     ┌─────────────────┐  │
│                                                     │  Smart Device   │  │
│                                                     │  Recommendations│  │
│                                                     │  • TV volume    │  │
│                                                     │  • Headphones   │  │
│                                                     │  • Lighting     │  │
│                                                     │  • Notifications│  │
│                                                     └─────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
multi-occupant-smart-home-recommender/
│
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package installation
├── .gitignore                          # Git ignore rules
│
├── data/
│   ├── README.md                       # Dataset information & download links
│   └── sample/                         # Sample data for testing
│       ├── house_b_day1.csv            # 1-day sample
│       └── expected_output.csv         # Expected preprocessing output
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── aras_preprocessor.py        # Data preprocessing pipeline
│   │
│   ├── pattern_mining/
│   │   ├── __init__.py
│   │   └── fpgrowth_multi_resident.py  # FP-Growth extension
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── glm_multi_resident.py       # Extended GLM models
│   │
│   └── conflict_resolution/
│       ├── __init__.py
│       └── conflict_resolver.py        # Conflict detection & resolution
│
├── experiments/
│   ├── run_preprocessing.py            # Preprocess ARAS dataset
│   ├── run_fpgrowth.py                 # Run pattern mining
│   ├── run_glm_experiment.py           # Train and evaluate GLM
│   ├── run_recommendation_system.py    # Full pipeline demo
│   └── run_ablation_study.py           # Feature ablation analysis
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA and visualization
│   ├── 02_pattern_analysis.ipynb       # FP-Growth results analysis
│   ├── 03_model_training.ipynb         # GLM training walkthrough
│   └── 04_conflict_resolution_demo.ipynb # Interactive demo
│
├── results/
│   ├── figures/                        # Paper figures
│   │   ├── architecture.pdf
│   │   ├── accuracy_comparison.pdf
│   │   ├── feature_importance.pdf
│   │   └── conflict_resolution_flow.pdf
│   │
│   ├── tables/                         # Result tables (CSV)
│   │   ├── main_results.csv
│   │   ├── pattern_mining_results.csv
│   │   └── feature_importance.csv
│   │
│   └── reports/                        # Generated reports
│       ├── house_a_pattern_report.txt
│       ├── house_b_pattern_report.txt
│       └── experimental_results.md
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_pattern_mining.py
│   ├── test_glm_model.py
│   └── test_conflict_resolution.py
│
├── docs/
│   ├── API.md                          # API documentation
│   ├── METHODOLOGY.md                  # Detailed methodology
│   ├── INSTALLATION.md                 # Installation guide
│   └── CONTRIBUTING.md                 # Contribution guidelines
│
└── paper/
    ├── main.tex                        # LaTeX source
    ├── references.bib                  # Bibliography
    └── figures/                        # Paper figures (source)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dielkht/multi-occupant-smart-home-recommender.git
cd multi-occupant-smart-home-recommender

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Download Dataset

The ARAS dataset is available from:
- **Original**: [Bogazici University](http://aras.cmpe.boun.edu.tr/)
- **Mirror**: [GitHub Repository](https://github.com/ronsm/ARAS-SKMulti-Model-Generator)

Place the data files in the `data/` directory.

### Run Demo

```bash
# Quick demo with conflict resolution scenarios
python experiments/run_recommendation_system.py

# Full experiment pipeline
python experiments/run_glm_experiment.py
```

## 📖 Usage Examples

### Basic Usage

```python
from src.conflict_resolution import SmartHomeRecommendationEngine

# Initialize engine
engine = SmartHomeRecommendationEngine()

# Get recommendations for activities
result = engine.recommend_from_activities(
    activity_r1=11,  # Sleeping
    activity_r2=12   # Watching TV
)

# Check conflict
if result['has_conflict']:
    print(f"Conflict: {result['conflict']['type']}")
    print(f"Resolution: {result['resolution']['strategy']}")
    
    for rec in result['recommendations']:
        print(f"  [{rec['device']}] {rec['action']}: {rec['value']}")
```

### Full Pipeline

```python
from src.preprocessing import ARASPreprocessor
from src.pattern_mining import MultiResidentPatternMiner
from src.models import MultiResidentGLM

# 1. Preprocess data
preprocessor = ARASPreprocessor(house="A")
preprocessor.load_all_days("/path/to/data/")
df = preprocessor.process()

# 2. Mine patterns
miner = MultiResidentPatternMiner()
miner.load_transactions_from_file("fpgrowth_input.txt")
patterns = miner.run_full_analysis()

# 3. Train GLM
model = MultiResidentGLM()
X_train, X_test, y_train, y_test = model.prepare_data(df)
model.fit(X_train, y_train)
results = model.evaluate(X_test, y_test)

print(f"R1 Accuracy: {results['R1_Activity'].accuracy:.4f}")
print(f"R2 Accuracy: {results['R2_Activity'].accuracy:.4f}")
```

### Conflict Resolution

```python
from src.conflict_resolution import ConflictResolver

resolver = ConflictResolver()

# Detect and resolve
conflict, resolution = resolver.detect_and_resolve(
    activity_r1=13,  # Studying
    activity_r2=12   # Watching TV
)

if conflict:
    print(f"Conflict Type: {conflict.conflict_type.value}")
    print(f"Severity: {conflict.severity.name}")
    print(f"Resolution Strategy: {resolution.strategy.value}")
    print(f"Confidence: {resolution.confidence:.0%}")
    
    for rec in resolution.recommendations:
        print(f"  {rec.device_type.value}: {rec.action} = {rec.value}")
```

## 📊 Reproducing Results

To reproduce the paper results:

```bash
# 1. Preprocess both houses
python experiments/run_preprocessing.py --house A --input /path/to/house_a
python experiments/run_preprocessing.py --house B --input /path/to/house_b

# 2. Run FP-Growth pattern mining
python experiments/run_fpgrowth.py

# 3. Train and evaluate GLM models
python experiments/run_glm_experiment.py

# 4. Generate paper figures
python experiments/generate_figures.py
```

Expected output:
```
House A (Couple):     99.86% accuracy (+35.25% improvement)
House B (Roommates):  99.96% accuracy (+7.38% improvement)
Conflict F1:          1.0000 (House A), 0.9975 (House B)
```

## 📈 Results Visualization

### Accuracy Comparison
![Accuracy Comparison](results/figures/accuracy_comparison.pdf)

### Feature Importance
![Feature Importance](results/figures/feature_importance.pdf)

## 🔧 Configuration

Key configuration options in `config.yaml`:

```yaml
preprocessing:
  lag_window: 5
  transaction_window: 60  # seconds

pattern_mining:
  min_support: 0.01
  min_confidence: 0.5
  min_lift: 1.0

model:
  max_iter: 1000
  C: 1.0
  test_size: 0.25
  cv_folds: 5

conflict_resolution:
  strategies:
    - device
    - compromise
    - priority
    - spatial
    - temporal
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{dilekh2026multioccupant,
  title={Multi-Occupant Context-Aware Recommender System for Smart Home 
         Automation: An Extended FP-Growth and GLM Approach with 
         Conflict Resolution},
  author={Author, First and Author, Second},
  journal={},
  volume={XXX},
  pages={XXX--XXX},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.eswa.2026.XXXXX}
}
```

Also cite the original works:

```bibtex
@article{dilekh2024dynamic,
  title={Dynamic Context-Aware Recommender System for Home Automation},
  author={Dilekh, Tarek and Ouhbi, Brahim},
  journal={Acta Informatica Pragensia},
  year={2024},
  doi={10.18267/j.aip.228}
}

@inproceedings{alemdar2013aras,
  title={ARAS human activity datasets in multiple homes with multiple residents},
  author={Alemdar, Hande and Ertan, Halil and Incel, Ozlem Durmaz and Ersoy, Cem},
  booktitle={PervasiveHealth},
  year={2013}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

- **Author**: [Your Name](mailto:tahar.dilekh@univ-batna2.dz)
- **Issues**: [GitHub Issues](https://github.com/dilekht/multi-occupant-smart-home-recommender/issues)

## 🙏 Acknowledgments

- ARAS dataset creators at Boğaziçi University
- Dilekh & Ouhbi for the original FP-Growth + GLM methodology
- scikit-learn and mlxtend development teams

---

**⭐ Star this repository if you find it useful!**
