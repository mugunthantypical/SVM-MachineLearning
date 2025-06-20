# SVMHeart(Final).ipynb

## Overview

**SVMHeart(Final).ipynb** is a Jupyter Notebook that demonstrates the process of building and evaluating Support Vector Machine (SVM) models to predict heart disease using the UCI Heart Disease dataset. The notebook covers the full data science workflow, including data cleaning, exploratory data analysis (EDA), feature selection, model training, and evaluation.

## Features

- **Data Cleaning:** Handles duplicate and irregular data, checks for missing values, and prepares the dataset for modeling.
- **Exploratory Data Analysis:** Provides descriptive statistics and insights into the dataset.
- **Feature Selection:** Experiments with different subsets of features (all features, top 10, top 8, top 5) to evaluate their impact on model performance.
- **Model Training:** Trains SVM classifiers using various kernels (`linear`, `poly`, `rbf`, `sigmoid`) and different train-test splits (60/40, 70/30, 80/20).
- **Evaluation:** Reports accuracy, recall, and precision for both training and testing sets. Visualizes results using confusion matrices and accuracy plots.
- **Comparison:** Compares the performance of different kernels and feature sets to identify the optimal configuration for heart disease prediction.

## Experiments

The notebook runs 12 experiments, varying the feature set and train-test split. For each experiment, it:
- Tests multiple SVM kernels.
- Plots training and testing accuracy.
- Selects the best kernel based on performance.
- Reports detailed metrics and confusion matrices.

## Results

- **Best Performance:** Achieved in Experiment 3 (60% training, 40% test, top 8 features) with:
  - Accuracy: 0.85
  - Recall: 0.85
  - Precision: 0.85

- **Worst Performance:** Observed in Experiment 12 (80% training, 20% test, top 5 features) with:
  - Accuracy: 0.80
  - Recall: 0.80
  - Precision: 0.80

## Usage

1. Place `heart_disease.csv` and `heart.csv` in the same directory as the notebook.
2. Open `SVMHeart(Final).ipynb` in Jupyter Notebook or VS Code.
3. Run the notebook cells sequentially to reproduce the experiments and results.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

Install dependencies with:
```sh
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Dataset
The notebook uses the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

## License
This project is for educational and research purposes.