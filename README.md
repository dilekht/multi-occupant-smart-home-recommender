# Multi-Occupant Smart Home Recommender System

A comprehensive multi-resident context-aware recommendation system for smart home automation, featuring comparative machine learning approaches and conflict resolution mechanisms.

## Overview

This project implements a smart home recommendation system designed for **multi-occupant households**. Unlike traditional single-occupant systems, it handles the complex dynamics of multiple residents with potentially conflicting preferences.

### Key Features

- **Multi-Resident Activity Prediction**: Predicts activities for multiple residents simultaneously
- **Cross-Resident Feature Engineering**: Novel features capturing synchronization patterns and inter-resident dynamics
- **Comparative ML Evaluation**: GLM, XGBoost, LightGBM, and Random Forest implementations
- **Conflict Detection**: Rule-based detection using activity compatibility matrix
- **Five-Strategy Resolution**: Priority-based, Compromise, Temporal, Spatial, and Device-Specific solutions
- **FP-Growth Pattern Mining**: Discovers 475K+ multi-resident behavioral patterns

## Performance

| Method | House A | House B | Average |
|--------|---------|---------|---------|
| GLM | 96.32% | 95.04% | 95.68% |
| XGBoost | 99.51% | 99.43% | 99.47% |
| LightGBM | 99.69% | 99.58% | 99.63% |
| Random Forest | 99.94% | 99.18% | 99.56% |

## Installation

```bash
# Clone the repository
git clone https://github.com/dilekht/multi-occupant-smart-home-recommender.git
cd multi-occupant-smart-home-recommender

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the **ARAS (Activity Recognition with Ambient Sensing)** dataset:
- **House A**: Married couple, 30 days, 2.59M samples
- **House B**: Two male roommates, 30 days, 2.59M samples
- **Sensors**: 20 binary sensors per house (motion, contact, pressure, proximity)
- **Activities**: 27 activity classes

Download the dataset from: http://aras.cmpe.boun.edu.tr/

Place the data files in:
```
data/aras/HouseA/
data/aras/HouseB/
```

## Usage

### Run Complete Pipeline

```bash
python main.py
```

This will:
1. Load and preprocess the ARAS dataset
2. Engineer 85 features (temporal, cross-resident, lag, conflict risk)
3. Mine patterns using FP-Growth
4. Train and evaluate all ML models
5. Perform conflict detection and resolution
6. Generate results and figures

### Generate Figures Only

```bash
python generate_figures.py
```

### Run Individual Components

```python
from preprocessing import load_aras_data, preprocess_data
from feature_engineering import engineer_features
from baselines import train_glm, train_xgboost, train_lightgbm, train_random_forest

# Load data
data = load_aras_data('data/aras/HouseA/')

# Preprocess
processed = preprocess_data(data)

# Engineer features
features = engineer_features(processed)

# Train models
glm_model = train_glm(features)
xgb_model = train_xgboost(features)
```

## Project Structure

```
├── main.py                 # Main pipeline script
├── preprocessing.py        # Data loading and preprocessing
├── feature_engineering.py  # Feature extraction (85 features)
├── baselines.py           # ML model implementations
├── generate_figures.py    # Visualization generation
├── requirements.txt       # Python dependencies
├── data/
│   └── aras/
│       ├── HouseA/        # House A data files
│       └── HouseB/        # House B data files
├── figures/               # Generated visualizations
├── models/               # Saved trained models
└── results/              # Experiment outputs
```

## Features Engineered

| Category | Features | Count |
|----------|----------|-------|
| Temporal | Hour (sin/cos), day of week, weekend, time of day | 12 |
| Sensor | Raw binary sensor values | 20 |
| Cross-Resident | IsSynchronized, SameCategory, BothHome, OneAway | 8 |
| Lag | Previous 5 activities per resident | 10 |
| Conflict Risk | TVConflictRisk, BathroomConflictRisk | 5 |
| FP-Growth Patterns | Mined association rules | 30 |
| **Total** | | **85** |

## Conflict Resolution Strategies

1. **Priority-Based**: Higher-priority activities take precedence
2. **Compromise**: Both residents adjust preferences
3. **Temporal**: Postpone lower-priority activity
4. **Spatial**: Suggest relocation to separate spaces
5. **Device-Specific**: Adjust smart device settings (volume, subtitles, headphones)

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.2.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0
- mlxtend >= 0.22.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dilekh2024multioccupant,
  author = {Dilekh, Tahar and Abdelhadi, Adel and Mokeddem, Ayoub},
  title = {Multi-Occupant Smart Home Recommender System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/dilekht/multi-occupant-smart-home-recommender}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Tahar Dilekh** - *Lead Developer*
- **Adel Abdelhadi** - *Contributor*
- **Ayoub Mokeddem** - *Contributor*

## Acknowledgments

- ARAS dataset creators for providing the multi-resident activity data
- The scikit-learn, XGBoost, and LightGBM teams for excellent ML libraries
