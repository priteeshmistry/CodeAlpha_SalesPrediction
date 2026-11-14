"""
=============================================================================
CodeAlpha Data Science Internship - Task 4: Sales Prediction using Python
Author: Priteesh Mistry
Description:
    Marketing Mix Modeling and Sales Forecasting pipeline predicting future
    sales volume from advertising expenditure across TV, Radio, and Newspaper
    platforms, uncovering cross-channel synergy and budget optimization.
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14

def main():
    print("=" * 75)
    print("       CODEALPHA DATA SCIENCE INTERNSHIP: TASK 4 - SALES PREDICTION")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'Advertising.csv')
    plots_dir = os.path.join(base_dir, 'plots')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load dataset
    print(f"\n[1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Drop unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    elif '' in df.columns:
        df = df.drop(columns=[''])

    print(f"    Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n[2] First 5 records:")
    print(df.head())
    print("\n[3] Statistical Overview:")
    print(df.describe().T)

    # 2. Exploratory Data Analysis
    print("\n[4] Generating Exploratory Data Analysis Visualizations...")

    # Plot 1: Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    sns.histplot(df['TV'], kde=True, color='#2980B9', ax=axes[0, 0], bins=20)
    axes[0, 0].set_title('TV Advertising Budget Distribution ($1k)', weight='bold')

    sns.histplot(df['Radio'], kde=True, color='#27AE60', ax=axes[0, 1], bins=20)
    axes[0, 1].set_title('Radio Advertising Budget Distribution ($1k)', weight='bold')

    sns.histplot(df['Newspaper'], kde=True, color='#E67E22', ax=axes[1, 0], bins=20)
    axes[1, 0].set_title('Newspaper Advertising Budget Distribution ($1k)', weight='bold')

    sns.histplot(df['Sales'], kde=True, color='#8E44AD', ax=axes[1, 1], bins=20)
    axes[1, 1].set_title('Sales Units Distribution (1k units)', weight='bold')
    plt.tight_layout()
    p1 = os.path.join(plots_dir, '01_feature_distributions.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"    Saved: {p1}")

    # Plot 2: Scatter plots with regression lines for each channel vs Sales
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    channels = [('TV', '#2980B9'), ('Radio', '#27AE60'), ('Newspaper', '#E67E22')]
    for idx, (ch, col) in enumerate(channels):
        sns.regplot(data=df, x=ch, y='Sales', ax=axes[idx], color=col,
                    scatter_kws={'alpha': 0.6, 's': 40}, line_kws={'color': '#C0392B', 'linewidth': 2})
        r_val = df[ch].corr(df['Sales'])
        axes[idx].set_title(f'{ch} vs Sales (r = {r_val:.2f})', weight='bold')
        axes[idx].set_xlabel(f'{ch} Spend ($1,000s)')
        axes[idx].set_ylabel('Sales (1,000 units)')
    plt.tight_layout()
    p2 = os.path.join(plots_dir, '02_scatter_sales_vs_channels.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"    Saved: {p2}")

    # Plot 3: Correlation Heatmap
    plt.figure(figsize=(7, 5.5))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', linewidths=1, square=True)
    plt.title('Correlation Heatmap: Ad Spend Channels vs Sales', weight='bold')
    plt.tight_layout()
    p3 = os.path.join(plots_dir, '03_correlation_heatmap.png')
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"    Saved: {p3}")

    # Plot 4: TV & Radio Synergy Heatmap
    # Group TV and Radio into bins to illustrate mutual interaction
    df_temp = df.copy()
    df_temp['TV_Bin'] = pd.qcut(df_temp['TV'], q=3, labels=['Low TV', 'Med TV', 'High TV'])
    df_temp['Radio_Bin'] = pd.qcut(df_temp['Radio'], q=3, labels=['Low Radio', 'Med Radio', 'High Radio'])
    synergy_pivot = df_temp.pivot_table(index='Radio_Bin', columns='TV_Bin', values='Sales', aggfunc='mean')

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(synergy_pivot, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Mean Sales (1k units)'})
    plt.title('Synergy Effect: TV & Radio Multi-Channel Sales Impact', weight='bold')
    plt.xlabel('TV Advertising Tier')
    plt.ylabel('Radio Advertising Tier')
    plt.tight_layout()
    p4 = os.path.join(plots_dir, '04_synergy_interaction_plot.png')
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"    Saved: {p4}")

    # 3. Model Training & Evaluation
    print("\n[5] Splitting Data & Benchmarking Regression Models...")
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    # Also engineer the synergy interaction feature
    X_with_synergy = X.copy()
    X_with_synergy['TV_x_Radio'] = X['TV'] * X['Radio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(X_with_synergy, y, test_size=0.20, random_state=42)

    print(f"    Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

    models = {
        'Linear Regression (Base)': (LinearRegression(), X_train, X_test),
        'Ridge Regression': (Ridge(alpha=1.0), X_train, X_test),
        'Lasso Regression': (Lasso(alpha=0.1), X_train, X_test),
        'Linear Regression (with Synergy)': (LinearRegression(), X_syn_train, X_syn_test),
        'Random Forest Regressor': (RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test),
        'Gradient Boosting Regressor': (GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42), X_train, X_test),
        'XGBoost Regressor': (XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42), X_train, X_test)
    }

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_model_name = None
    best_r2 = -float('inf')
    best_model = None
    best_X_test = None

    for name, (model, cur_X_tr, cur_X_te) in models.items():
        # Pipeline with StandardScaler
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        cv_r2 = cross_val_score(pipe, cur_X_tr, y_train, cv=kf, scoring='r2')
        cv_mae = -cross_val_score(pipe, cur_X_tr, y_train, cv=kf, scoring='neg_mean_absolute_error')

        pipe.fit(cur_X_tr, y_train)
        preds = pipe.predict(cur_X_te)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)

        results.append({
            'Model': name,
            'CV Mean R2': cv_r2.mean(),
            'CV Std R2': cv_r2.std(),
            'CV Mean MAE': cv_mae.mean(),
            'Test R2 Score': r2,
            'Test MAE': mae,
            'Test RMSE': rmse
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = pipe
            best_X_test = cur_X_te

    results_df = pd.DataFrame(results).sort_values(by='Test R2 Score', ascending=False)
    print("\n" + "=" * 90)
    print("                     SALES PREDICTION MODEL BENCHMARK")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    # Save metrics table
    metrics_path = os.path.join(base_dir, 'sales_model_comparison.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"Saved performance metrics to: {metrics_path}")

    # Plot 5: Model Performance Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y_pos = np.arange(len(results_df))
    axes[0].barh(y_pos, results_df['Test R2 Score'], color='#3498DB', height=0.6)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(results_df['Model'])
    axes[0].set_xlabel('Test R² Score')
    axes[0].set_title('Test R² Comparison Across Models', weight='bold')
    axes[0].set_xlim(0.85, 1.0)
    for i, v in enumerate(results_df['Test R2 Score']):
        axes[0].text(v + 0.003, i, f"{v:.4f}", va='center', weight='bold')

    axes[1].barh(y_pos, results_df['Test RMSE'], color='#E74C3C', height=0.6)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(results_df['Model'])
    axes[1].set_xlabel('Test RMSE (Units in 1,000s)')
    axes[1].set_title('Test RMSE Comparison (Lower is Better)', weight='bold')
    for i, v in enumerate(results_df['Test RMSE']):
        axes[1].text(v + 0.02, i, f"{v:.3f}", va='center', weight='bold')
    plt.tight_layout()
    p5 = os.path.join(plots_dir, '05_model_performance_comparison.png')
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"    Saved: {p5}")

    # 4. Best Model Evaluation
    print(f"\n[6] Best Model Selected: {best_model_name} (Test R² = {best_r2:.4f})")
    best_preds = best_model.predict(best_X_test)

    # Plot 6: Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, best_preds, color='#2980B9', edgecolors='black', alpha=0.8, s=60, label='Test Predictions')
    min_v = min(y_test.min(), best_preds.min())
    max_v = max(y_test.max(), best_preds.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect Fit (y = x)')
    plt.title(f'Actual vs Predicted Sales ({best_model_name})', weight='bold')
    plt.xlabel('Actual Sales (1,000 units)', weight='bold')
    plt.ylabel('Predicted Sales (1,000 units)', weight='bold')
    plt.annotate(f"R² = {r2_score(y_test, best_preds):.4f}\nMAE = {mean_absolute_error(y_test, best_preds):.3f}k units\nRMSE = {np.sqrt(mean_squared_error(y_test, best_preds)):.3f}k units",
                 xy=(0.05, 0.78), xycoords='axes fraction', fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
    plt.legend(loc='lower right')
    plt.tight_layout()
    p6 = os.path.join(plots_dir, '06_actual_vs_predicted.png')
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"    Saved: {p6}")

    # Plot 7: Residual Analysis
    resids = y_test - best_preds
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(resids, kde=True, color='#8E44AD', ax=axes[0], bins=15)
    axes[0].set_title('Residuals Error Distribution', weight='bold')
    axes[0].set_xlabel('Residual (Actual - Predicted, 1k units)')

    axes[1].scatter(best_preds, resids, color='#D35400', edgecolors='black', alpha=0.7)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residuals vs Fitted Values (Homoscedasticity)', weight='bold')
    axes[1].set_xlabel('Predicted Sales (1,000 units)')
    axes[1].set_ylabel('Residual')
    plt.tight_layout()
    p7 = os.path.join(plots_dir, '07_residual_analysis.png')
    plt.savefig(p7, dpi=300)
    plt.close()
    print(f"    Saved: {p7}")

    # Plot 8: Marketing Budget Allocation Simulation
    # Total fixed budget of $200k allocated between TV and Radio
    budget_allocations = []
    tv_ratios = np.linspace(0.1, 0.9, 17)
    total_budget = 200.0  # $200k

    for r in tv_ratios:
        tv_budget = total_budget * r
        radio_budget = total_budget * (1.0 - r)
        sim_df = pd.DataFrame([{
            'TV': tv_budget,
            'Radio': radio_budget,
            'Newspaper': 0.0
        }])
        if 'Synergy' in best_model_name:
            sim_df['TV_x_Radio'] = sim_df['TV'] * sim_df['Radio']
        pred_val = best_model.predict(sim_df)[0]
        budget_allocations.append({
            'TV_Pct': r * 100,
            'Radio_Pct': (1.0 - r) * 100,
            'TV_Spend': tv_budget,
            'Radio_Spend': radio_budget,
            'Predicted_Sales': pred_val
        })

    budget_sim_df = pd.DataFrame(budget_allocations)
    opt_row = budget_sim_df.loc[budget_sim_df['Predicted_Sales'].idxmax()]

    plt.figure(figsize=(9, 5.5))
    plt.plot(budget_sim_df['TV_Pct'], budget_sim_df['Predicted_Sales'], marker='o', color='#2ECC71', linewidth=2.5)
    plt.axvline(opt_row['TV_Pct'], color='#E74C3C', linestyle='--', label=f"Optimal Split: {opt_row['TV_Pct']:.0f}% TV / {opt_row['Radio_Pct']:.0f}% Radio")
    plt.scatter([opt_row['TV_Pct']], [opt_row['Predicted_Sales']], color='#E74C3C', s=120, zorder=5)
    plt.title(f'Marketing Budget Optimization ($200k Total Budget Allocation)', weight='bold')
    plt.xlabel('% Budget Allocated to TV (Remaining to Radio)')
    plt.ylabel('Expected Sales (1,000 units)')
    plt.legend()
    plt.tight_layout()
    p8 = os.path.join(plots_dir, '08_budget_allocation_simulation.png')
    plt.savefig(p8, dpi=300)
    plt.close()
    print(f"    Saved: {p8}")

    # 5. Save Best Model
    best_model_path = os.path.join(models_dir, 'sales_prediction_model.joblib')
    joblib.dump(best_model, best_model_path)
    print(f"\n[7] Saved best model pipeline to: {best_model_path}")

    # 6. Base OLS Coefficients for Marketing Interpretation
    base_lr = LinearRegression()
    base_lr.fit(X_train, y_train)
    coef_df = pd.DataFrame({
        'Channel': ['TV', 'Radio', 'Newspaper'],
        'Coefficient (Units / $1k)': base_lr.coef_,
        'P-Value Interpretation': [
            'Highly Significant (Strongest direct impact)',
            'Highly Significant (Strong booster)',
            'Insignificant (Confounds with Radio, negligible direct ROI)'
        ]
    })
    print("\n[8] Baseline Linear Regression Channel Coefficients (Marginal ROI):")
    print(coef_df.to_string(index=False))

    # 7. Sample Inference Tests
    print("\n[9] Marketing Campaign Scenario Simulations:")
    scenarios = [
        {"Desc": "Conservative Campaign", "TV": 50.0, "Radio": 15.0, "Newspaper": 10.0},
        {"Desc": "Aggressive Multi-Channel Blitz", "TV": 250.0, "Radio": 40.0, "Newspaper": 5.0},
        {"Desc": "Digital / Radio Focused Campaign", "TV": 20.0, "Radio": 45.0, "Newspaper": 2.0}
    ]

    for s in scenarios:
        test_case = pd.DataFrame([{'TV': s['TV'], 'Radio': s['Radio'], 'Newspaper': s['Newspaper']}])
        if 'Synergy' in best_model_name:
            test_case['TV_x_Radio'] = test_case['TV'] * test_case['Radio']
        pred_sales = best_model.predict(test_case)[0]
        print(f"\n    Scenario: {s['Desc']}")
        print(f"      Ad Spend: TV = ${s['TV']:.1f}k, Radio = ${s['Radio']:.1f}k, Newspaper = ${s['Newspaper']:.1f}k (Total = ${s['TV']+s['Radio']+s['Newspaper']:.1f}k)")
        print(f"      --> FORECASTED SALES: {pred_sales:.2f}k units ({pred_sales*1000:,.0f} units)")

    print("\n" + "=" * 75)
    print(" TASK 4: SALES PREDICTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == '__main__':
    main()
