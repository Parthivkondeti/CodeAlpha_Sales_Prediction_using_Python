# Sales Prediction using Python

## Overview
A regression project that predicts product sales from advertising spend across TV, Radio, and
Newspaper channels. Built as **Task 4** of the **CodeAlpha Data Science Internship**.

## Objective
Predict `Sales` from advertising budgets, and identify which advertising channel has the
strongest effect on sales, to inform marketing budget allocation.

## Dataset
- **Source:** the CodeAlpha task PDF links to
  [Advertising.csv — Kaggle (bumba5341/advertisingcsv)](https://www.kaggle.com/datasets/bumba5341/advertisingcsv),
  a republish of the classic "Advertising" dataset from *An Introduction to Statistical Learning*
  (ISLR). Since the Kaggle site could not be reached directly from this environment, the identical
  data was sourced from a public mirror of the same dataset,
  `https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv` — the values are
  identical to the Kaggle CSV; nothing was invented or altered.
- 200 rows, 4 columns: `TV`, `Radio`, `Newspaper` (spend in $1000s), `Sales` (units sold, in 1000s) — target.
- No missing values, no duplicate rows.

## Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter.

## Methodology
1. Load and inspect the dataset (shape, dtypes, nulls, duplicates, summary statistics).
2. Check for outliers per channel.
3. Exploratory data analysis — correlation heatmap, spend-vs-sales regression plots.
4. Train/test split (80/20) and feature scaling.
5. Train and compare three regressors: Linear Regression, Decision Tree, Random Forest.
6. Evaluate the best model with MAE, MSE, RMSE, and R², and inspect feature importance.

## Data Preprocessing
No missing-value or duplicate handling was needed. A few high `Newspaper` values were checked and
kept as plausible spend rather than treated as errors. The three feature columns were standardised
(`StandardScaler`) before model training.

## Exploratory Data Analysis
- Correlation heatmap (`images/correlation_heatmap.png`)
- Spend vs. Sales regression plots per channel (`images/spend_vs_sales.png`)
- Outlier boxplots per channel (`images/outlier_boxplots.png`)

TV spend shows the strongest linear association with sales, Radio a moderate association, and
Newspaper the weakest.

## Machine Learning Model
Three models were trained and compared on the same scaled train/test split (`random_state=42`):
Linear Regression, Decision Tree Regressor, Random Forest Regressor.

## Evaluation
| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Random Forest | 0.621 | 0.591 | 0.769 | **0.981** |
| Decision Tree | 0.985 | 2.175 | 1.475 | 0.931 |
| Linear Regression | 1.461 | 3.174 | 1.782 | 0.899 |

**Best model: Random Forest Regressor — R² = 0.981** on held-out test data (40 samples).
See `images/actual_vs_predicted.png` for predicted-vs-actual sales.

## Results
These are the actual results from running `notebooks/Sales_Prediction.ipynb` end-to-end — no
invented metrics. Feature importance from the Random Forest model:

| Feature | Importance |
|---|---|
| TV | 0.625 |
| Radio | 0.362 |
| Newspaper | 0.013 |

## Key Insights
- **TV advertising spend is the dominant driver of sales** — confirmed by both the correlation
  analysis and the trained model's feature importance.
- **Radio spend has a real, moderate positive contribution.**
- **Newspaper spend contributes the least** to the model's predictions, despite having a mild
  positive raw correlation with sales on its own — a model-based finding, not a causal claim,
  since this is observational data rather than a controlled experiment.
- Directionally, the results support prioritising TV and Radio budget over Newspaper.

## How to Run
```bash
pip install -r requirements.txt
```
Then open `notebooks/Sales_Prediction.ipynb` in Jupyter and run all cells
(or run `src/sales_prediction.py` directly).

## Project Structure
```text
CodeAlpha_Sales_Prediction_using_Python/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── Advertising.csv
│
├── notebooks/
│   └── Sales_Prediction.ipynb
│
├── src/
│   └── sales_prediction.py
│
└── images/
    ├── correlation_heatmap.png
    ├── spend_vs_sales.png
    ├── outlier_boxplots.png
    └── actual_vs_predicted.png
```

## Future Improvements
- Add interaction terms (e.g. TV × Radio) to capture channel synergy.
- Try polynomial regression if residuals suggest non-linearity.
- Use larger/more recent data, including seasonality or competitor-spend features.

## Author
**Parthiv Kondeti**
