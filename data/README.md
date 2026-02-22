# ARAS Dataset

Download the ARAS dataset from: http://aras.cmpe.boun.edu.tr/

## Directory Structure

Place the data files as follows:

```
data/aras/
├── HouseA/
│   ├── DAY1.txt
│   ├── DAY2.txt
│   └── ... (DAY1-DAY30)
└── HouseB/
    ├── DAY1.txt
    ├── DAY2.txt
    └── ... (DAY1-DAY30)
```

## Data Format

Each DAY file contains tab-separated values:
- Columns 1-20: Binary sensor readings (0/1)
- Column 21: Activity label for Resident 1
- Column 22: Activity label for Resident 2

## Activity Classes

| ID | Activity |
|----|----------|
| 1 | Other |
| 2 | Going Out |
| 11 | Sleeping |
| 12 | Watching TV |
| 13 | Studying |
| 14 | Having Shower |
| 15 | Toileting |
| 16 | Napping |
| ... | ... |

See the ARAS documentation for the complete list of 27 activity classes.
