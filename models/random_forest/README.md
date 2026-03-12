# Random Forest Model

## Overview
This folder contains the Random Forest implementation for the group project.

## Performance Metrics
- **Fraud Detection**: 96.32% accuracy, 96.30% F1-score
- **Late Delivery**: 91.96% accuracy, 89.79% F1-score

## Confusion Matrices

### Fraud Detection:
[[14979 1329]
[ 0 19796]]

### Late Delivery:
[[ 8318 0 0 0]
[ 0 19796 0 0]
[ 352 960 0 239]
[ 0 1353 0 5086]]

## Files
- `run_random_forest_fixed.py` - Main script for training and evaluation

## Usage
```bash
# Install dependencies
pip install numpy pandas scikit-learn

# Run the model
python3 run_random_forest_fixed.py
nano .gitignore
# Dataset (large file - DO NOT COMMIT)
DataCoSupplyChainDataset.csv

# Virtual environment
venv/
.ipynb_checkpoints/
__pycache__/
*.pyc

# OS files
.DS_Store
Thumbs.db

