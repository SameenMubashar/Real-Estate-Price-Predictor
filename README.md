## Real Estate Price Predictor — Model Comparison & Visual Diagnostics

A clean, reproducible pipeline for training and comparing multiple regression models on the Ames Housing dataset. It handles robust preprocessing (ordinal encodings with explicit order, one-hot encoding, numeric imputation), applies a log-transform to target for skew handling, and reports R²/MAE/RMSE across models with Actual vs. Predicted and Residual plots.

# Table of Contents
	•	Overview
	•	Key Features
	•	Data & Assumptions
	•	Environment & Requirements
	•	Quick Start
	•	What the Script Does
	•	Preprocessing Details
	•	Models Compared
	•	Outputs
	•	Results Interpretation
	•	License
	•	Acknowledgments

# Overview
	1.	Loads train.csv (Ames Housing-style).
	2.	Visualizes SalePrice distribution and its log1p transform.
	3.	Builds a ColumnTransformer-based preprocessing pipeline:
	   •	Ordinal features with explicit category order.
	   •	Nominal features with OneHotEncoder (handle_unknown="ignore").
	   •	Numeric features with median imputation.
	4.	Produces a fully encoded feature matrix and correlation ranking vs. target.
	5.	Selects a curated set of top features for a compact model.
	6.	Trains and evaluates multiple models with a uniform train/test split.
	7.	Generates Actual vs. Predicted plots for each model (and optionally residual plots).

# Key Features
	•	Robust missing-value handling across numeric, nominal, and ordinal features.
	•	Ordinal encodings that respect domain-specific quality/condition hierarchies.
	•	Consistent, comparable metrics across 5 models (R², MAE, RMSE).
	•	Clear diagnostics via plots saved to disk.
	•	Modular blocks you can reuse (preprocessor, evaluation, plotting).

# Data & Assumptions
	•	Input file: train.csv (expects Ames Housing-like columns).
	•	Target: SalePrice.
	•	The script applies SalePrice = log1p(SalePrice) to reduce skew.

# Environment & Requirements

Python: 3.9+

Core libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn

Install:

pip install -r requirements.txt
# or
pip install pandas numpy matplotlib seaborn scikit-learn

Note on scikit-learn:
	•	If you use OneHotEncoder(sparse_output=False), you need scikit-learn ≥ 1.2.
	•	On older versions, replace with sparse=False.

# Quick Start
	1.	Place train.csv in the project root (same folder as the script).
	2.	Run the script (Jupyter or plain Python):

python your_script_name.py

or open it in Jupyter Notebook / VS Code and run all cells.

# Preprocessing Details
	•	Ordinal columns (e.g., ExterQual, KitchenQual, HeatingQC, BsmtQual, etc.)
Encoded with OrdinalEncoder using explicit category orderings such as:

'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex']
'BsmtQual' : ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
'GarageFinish': ['NA', 'Unf', 'RFn', 'Fin']
... (see script for full mapping)

Missing values are imputed as the string "NA" so they map consistently.

	•	Nominal columns
Imputed to "Missing", then one-hot encoded with handle_unknown="ignore" for robustness.
	•	Numeric columns
Imputed with median.
	•	Target transform
df['SalePrice'] = np.log1p(df['SalePrice'])

⸻

# Models Compared
	•	Linear Regression
	•	Ridge Regression (alpha=10)
	•	Lasso Regression (alpha=0.001, max_iter=5000)
	•	Random Forest Regressor (n_estimators=300, max_depth=15)
	•	Gradient Boosting Regressor (n_estimators=500, learning_rate=0.05, max_depth=4)


## Outputs

# Console
	•	Top correlations with SalePrice (log-scaled).
	•	Sorted model leaderboard:

     Model           R²      MAE     RMSE
    Random Forest 0.8589  0.1078   0.1623
    Gradient Boosting 0.8585  0.1094   0.1625
    Lasso Regression 0.8559  0.1157   0.1640
    Ridge Regression 0.8553  0.1161   0.1643
    Linear Regression 0.8547  0.1164   0.1647

Plots (saved to files)
	•	One Actual vs. Predicted scatter plot per model, e.g.:

linear_regression_actual_vs_pred.png
ridge_regression_actual_vs_pred.png
lasso_regression_actual_vs_pred.png
random_forest_actual_vs_pred.png
gradient_boosting_actual_vs_pred.png


# Results Interpretation
	•	The dataset is largely linear (strong performance from Linear/Ridge/Lasso).
	•	Tree ensembles (RF/GB) provide a small but consistent lift, capturing mild nonlinearities and interactions.
	•	The log-transform stabilizes variance and improves fit quality; keep it for training and evaluation consistency.

## License

This project is provided under the MIT License. Feel free to use and adapt.


## Acknowledgments
	•	Ames Housing Dataset: Dean De Cock
	•	scikit-learn for preprocessing, modeling, and metrics
	•	Matplotlib/Seaborn for visualizations
