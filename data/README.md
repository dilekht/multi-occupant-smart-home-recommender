# Dataset Information

## ARAS Dataset

This project uses the **ARAS (Activity Recognition with Ambient Sensing)** dataset.

### Dataset Description

- **Source**: Boğaziçi University, Turkey
- **Reference**: Alemdar et al., "ARAS human activity datasets in multiple homes with multiple residents" (PervasiveHealth 2013)
- **DOI**: 10.4108/PERVASIVEHEALTH.2013.252120

### Dataset Characteristics

| Property | House A | House B |
|----------|---------|---------|
| Residents | Married couple (age 34) | Male roommates (age 25) |
| House Size | 50 m² | 90 m² |
| Duration | 30 days | 30 days |
| Sensors | 20 binary sensors | 20 binary sensors |
| Activities | 27 classes | 27 classes |
| Sampling Rate | 1 Hz | 1 Hz |
| Records | 2,592,000 | 2,592,000 |

### Sensor Types

The 20 sensors include:
- Motion detectors
- Contact sensors (doors, cabinets)
- Pressure sensors (chairs, sofa, bed)
- Proximity sensors
- Temperature sensors

### Activity Classes

| ID | Activity | ID | Activity |
|----|----------|----| ---------|
| 1 | Other | 15 | Toileting |
| 2 | Going Out | 16 | Napping |
| 3 | Preparing Breakfast | 17 | Using Internet |
| 4 | Having Breakfast | 18 | Reading Book |
| 5 | Preparing Lunch | 19 | Laundry |
| 6 | Having Lunch | 20 | Shaving |
| 7 | Preparing Dinner | 21 | Brushing Teeth |
| 8 | Having Dinner | 22 | Talking on Phone |
| 9 | Washing Dishes | 23 | Listening to Music |
| 10 | Having Snack | 24 | Cleaning |
| 11 | Sleeping | 25 | Having Conversation |
| 12 | Watching TV | 26 | Having Guest |
| 13 | Studying | 27 | Changing Clothes |
| 14 | Having Shower | | |

## Download Instructions

### Option 1: GitHub Mirror (Recommended)

```bash
git clone https://github.com/ronsm/ARAS-SKMulti-Model-Generator.git
cp -r ARAS-SKMulti-Model-Generator/ARAS/* data/
```

### Option 2: Direct Download

Visit: https://github.com/ronsm/ARAS-SKMulti-Model-Generator

### Option 3: Original Source

Visit: http://aras.cmpe.boun.edu.tr/ (may be unavailable)

## Data Format

Each daily file contains 86,400 rows (1 per second) with 22 columns:

```
S1_co1, S2_co2, ..., S20_so3, Activity_R1, Activity_R2
```

- Columns 1-20: Binary sensor readings (0/1)
- Column 21: Activity label for Resident 1 (1-27)
- Column 22: Activity label for Resident 2 (1-27)

## Directory Structure

After downloading, organize as:

```
data/
├── house_a/
│   ├── DAY_1.csv
│   ├── DAY_2.csv
│   └── ... (30 files)
│
├── house_b/
│   ├── DAY_1.csv
│   ├── DAY_2.csv
│   └── ... (30 files)
│
└── sample/
    └── house_b_day1.csv  # Included for testing
```

## Sample Data

A sample file (`sample/house_b_day1.csv`) is included in this repository for testing without downloading the full dataset.

## License

The ARAS dataset is provided by Boğaziçi University for research purposes. Please cite the original paper if you use this dataset.

## Citation

```bibtex
@inproceedings{alemdar2013aras,
  title={ARAS human activity datasets in multiple homes with multiple residents},
  author={Alemdar, Hande and Ertan, Halil and Incel, Ozlem Durmaz and Ersoy, Cem},
  booktitle={2013 7th International Conference on Pervasive Computing Technologies 
             for Healthcare and Workshops},
  pages={232--235},
  year={2013},
  organization={IEEE},
  doi={10.4108/PERVASIVEHEALTH.2013.252120}
}
```
