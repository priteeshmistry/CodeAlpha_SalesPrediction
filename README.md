# 📊 CodeAlpha Task 4: Sales Prediction using Python

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0%2B-red.svg)](https://xgboost.ai/)
[![Status](https://img.shields.io/badge/Task-Completed-success.svg)]()

> **Internship Program:** CodeAlpha Data Science Internship  
> **Task Name:** Task 4 - Sales Prediction using Python  
> **Author:** Priteesh Mistry  
> **GitHub Repository:** `CodeAlpha_SalesPrediction`

---

## 📌 Project Overview
Sales forecasting based on promotional advertising spend is a core discipline in **Marketing Mix Modeling (MMM)**. Businesses must understand which advertising platforms drive consumer conversions, how media channels interact with each other, and how to optimally distribute limited budgets to maximize product sales volume.

This project implements an end-to-end econometric regression pipeline analyzing advertising budgets across **TV**, **Radio**, and **Newspaper** to forecast **Sales units**, uncovering channel synergies and simulating optimal budget allocations.

---

## 📊 Dataset Description
- **Records:** 200 marketing campaign observations
- **Target Variable:** `Sales` (in thousands of product units)
- **Predictor Variables:**
  - `TV`: Television advertising spend (in thousands of dollars, `$1,000s`)
  - `Radio`: Radio advertising spend (in thousands of dollars, `$1,000s`)
  - `Newspaper`: Newspaper advertising spend (in thousands of dollars, `$1,000s`)

### Summary Statistics

| Variable | Min | Mean | Median | Max | Std Dev |
|---|---|---|---|---|---|
| **TV ($1k)** | 0.70 | 147.04 | 149.75 | 296.40 | 85.85 |
| **Radio ($1k)** | 0.00 | 23.26 | 22.90 | 49.60 | 14.85 |
| **Newspaper ($1k)** | 0.30 | 30.55 | 25.75 | 114.00 | 21.78 |
| **Sales (1k units)** | 1.60 | 14.02 | 12.90 | 27.00 | 5.22 |

---

## 🔍 Exploratory Data Analysis & Strategic Insights

1. **TV as the Primary Driver:**  
   `TV` spend demonstrates the highest single-channel Pearson correlation with sales ($r = 0.782$).
2. **Radio as a High-Efficiency Multiplier:**  
   `Radio` spend has a strong positive correlation ($r = 0.576$) and yields the highest direct marginal return per thousand dollars spent.
3. **The Newspaper Myth (Confounding Variable):**  
   While `Newspaper` appears weakly correlated ($r = 0.228$), in multiple regression its coefficient is statistically insignificant ($p > 0.05$). Its apparent relationship is largely driven by correlation with Radio spend ($r = 0.354$).
4. **Cross-Channel Synergy (Interaction Effect):**  
   Simultaneous advertising on TV and Radio creates a powerful compounding effect. Introducing the interaction term $\text{TV} \times \text{Radio}$ leaps linear model $R^2$ from **0.8994** to **0.9742**!

---

## 🏆 Model Benchmarking & Performance Comparison

| Model | 5-Fold CV Mean R² | CV Std R² | CV Mean MAE | Test R² Score | Test MAE (1k units) | Test RMSE (1k units) |
|---|---|---|---|---|---|---|
| **Gradient Boosting Regressor** | **0.9671** | **±0.0134** | **0.6122** | **0.9831** | **0.6181** | **0.7295** |
| **Random Forest Regressor** | 0.9663 | ±0.0132 | 0.6382 | 0.9813 | 0.6210 | 0.7689 |
| **XGBoost Regressor** | 0.9651 | ±0.0123 | 0.6655 | 0.9810 | 0.6540 | 0.7737 |
| **Linear Regression (with Synergy)** | 0.9570 | ±0.0283 | 0.7009 | 0.9742 | 0.6718 | 0.9025 |
| **Linear Regression (Base OLS)** | 0.8703 | ±0.0718 | 1.2371 | 0.8994 | 1.4608 | 1.7816 |
| **Ridge Regression** | 0.8705 | ±0.0703 | 1.2371 | 0.8988 | 1.4643 | 1.7872 |
| **Lasso Regression** | 0.8726 | ±0.0672 | 1.2280 | 0.8983 | 1.4613 | 1.7913 |

### 🎯 Best Model: Gradient Boosting Regressor
- **Coefficient of Determination ($R^2$):** **0.9831** (Explains 98.31% of sales variance on test set)
- **Mean Absolute Error (MAE):** **0.618k units** (618 units average error)
- **Root Mean Squared Error (RMSE):** **0.729k units**

---

## 💡 Business Marketing Strategies & Budget Optimization

1. **Eliminate or Substantially Reduce Newspaper Spend:**  
   Given its negligible contribution to sales, newspaper budgets should be reallocated towards digital radio or TV campaigns.
2. **Optimal Multi-Channel Budget Allocation:**  
   Our budget simulation algorithm tested spending curves for a fixed budget of **$200,000**:
   - **Recommended Split:** **65% to 70% TV** and **30% to 35% Radio**.
   - This split triggers the maximum cross-channel synergy multiplier, generating an expected **21,500+ product sales units**.

---

## 📈 Visualizations
All charts are rendered at 300 DPI in [`plots/`](./plots/):
- `01_feature_distributions.png`: Ad spend and sales density distributions
- `02_scatter_sales_vs_channels.png`: OLS regression lines for TV, Radio, Newspaper vs Sales
- `03_correlation_heatmap.png`: Pearson correlation matrix
- `04_synergy_interaction_plot.png`: TV and Radio tier interaction matrix
- `05_model_performance_comparison.png`: Comparative R² and RMSE bar charts
- `06_actual_vs_predicted.png`: Actual vs Predicted sales scatter plot
- `07_residual_analysis.png`: Residual error distribution and homoscedasticity plot
- `08_budget_allocation_simulation.png`: Optimal marketing allocation curve

---

## 🚀 How to Run

```bash
cd CodeAlpha_Task4_Sales_Prediction

# Install dependencies
pip install numpy pandas scikit-learn xgboost matplotlib seaborn joblib

# Run forecasting pipeline
python src/sales_prediction.py

# Launch interactive notebook
jupyter notebook Sales_Prediction.ipynb
```

---

## 📂 Project Structure
```
CodeAlpha_Task4_Sales_Prediction/
│
├── data/
│   └── Advertising.csv
├── models/
│   └── sales_prediction_model.joblib      # Serialized best model pipeline
├── plots/
│   ├── 01_feature_distributions.png
│   ├── 02_scatter_sales_vs_channels.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_synergy_interaction_plot.png
│   ├── 05_model_performance_comparison.png
│   ├── 06_actual_vs_predicted.png
│   ├── 07_residual_analysis.png
│   └── 08_budget_allocation_simulation.png
├── src/
│   └── sales_prediction.py
├── Sales_Prediction.ipynb
├── sales_model_comparison.csv
└── README.md
```

---

## 🎥 LinkedIn Presentation Video Script (Word-for-Word Guide)
*(For completing Step 4 of the CodeAlpha Intern Instructions)*

> **Introduction (0:00 - 0:20):**  
> *"Hello everyone! I'm Priteesh Mistry, a Data Science Intern at CodeAlpha. Today, I'm thrilled to share my work on Task 4: Sales Prediction using Python."*
>
> **Problem & Dataset (0:20 - 0:50):**  
> *"In marketing mix modeling, the goal is to predict product sales based on advertising investments across TV, Radio, and Newspaper channels. Our dataset contains 200 campaign instances with budget expenditures and corresponding sales outcomes."*
>
> **Analysis & Modeling (0:50 - 1:30):**  
> *"Through exploratory data analysis, we observed that TV has the strongest direct correlation at 0.78, while Radio acts as a powerful multiplier. Interestingly, Newspaper spend showed an insignificant marginal ROI. Furthermore, we demonstrated a pronounced synergy effect between TV and Radio advertising. Benchmarking across Linear, Regularized, Random Forest, and Gradient Boosting models, our Gradient Boosting Regressor achieved an outstanding R² score of 0.9831 with an RMSE of only 0.73 units."*
>
> **Marketing Recommendations & Closing (1:30 - 2:00):**  
> *"We concluded by simulating budget allocations, proving that shifting budget from print newspaper into a 70/30 TV and Radio mix maximizes sales. The complete reproducible code and interactive notebook are available on GitHub. Thank you CodeAlpha for this fantastic learning journey!"*
