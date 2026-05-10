"""
PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)
-----------------------------------------
This module performs basic EDA on the Telco Customer Churn dataset.
We generate 5 simple visualizations to understand the data:
  1. Churn Distribution (Bar Chart)
  2. Tenure vs Churn (Boxplot)
  3. Monthly Charges vs Churn (Boxplot)
  4. Contract Type vs Churn (Countplot)
  5. Correlation Heatmap
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EDA_DIR = os.path.join(PROJECT_DIR, "frontend", "static", "eda")
os.makedirs(EDA_DIR, exist_ok=True)


def load_raw_data():
    """Load the Telco Churn CSV and do minimal cleaning."""
    csv_path = os.path.join(DATA_DIR, "telco_churn.csv")
    if not os.path.exists(csv_path):
        from data_preprocessing import load_data
        return load_data()
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df


# ── Graph 1: Churn Distribution (Bar Chart) ──

def plot_churn_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['Churn'].value_counts()
    bars = ax.bar(counts.index, counts.values, color=['#4878A8', '#E07A5F'], width=0.5, edgecolor='#333', linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 80, str(val), ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Churn Distribution', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Churn Status', fontsize=11)
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "churn_distribution.png"), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print("  [1/5] Churn Distribution - saved")


# ── Graph 2: Tenure vs Churn (Boxplot) ──

def plot_tenure_vs_churn(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Churn', y='tenure', data=df, palette=['#4878A8', '#E07A5F'], ax=ax,
                flierprops={'markersize': 3}, width=0.5)
    ax.set_title('Tenure vs Churn', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Churn Status', fontsize=11)
    ax.set_ylabel('Tenure (months)', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "tenure_vs_churn.png"), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print("  [2/5] Tenure vs Churn - saved")


# ── Graph 3: Monthly Charges vs Churn (Boxplot) ──

def plot_monthly_charges_vs_churn(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette=['#4878A8', '#E07A5F'], ax=ax,
                flierprops={'markersize': 3}, width=0.5)
    ax.set_title('Monthly Charges vs Churn', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Churn Status', fontsize=11)
    ax.set_ylabel('Monthly Charges ($)', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "monthly_charges_vs_churn.png"), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print("  [3/5] Monthly Charges vs Churn - saved")


# ── Graph 4: Contract Type vs Churn (Countplot) ──

def plot_contract_vs_churn(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x='Contract', hue='Churn', data=df, palette=['#4878A8', '#E07A5F'], ax=ax,
                  edgecolor='#333', linewidth=0.5)
    ax.set_title('Contract Type vs Churn', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Contract Type', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(title='Churn', fontsize=10, title_fontsize=10)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "contract_vs_churn.png"), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print("  [4/5] Contract Type vs Churn - saved")


# ── Graph 5: Correlation Heatmap ──

def plot_correlation_heatmap(df):
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        df_enc[col] = pd.factorize(df_enc[col])[0]

    corr = df_enc.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print("  [5/5] Correlation Heatmap - saved")


def run_eda():
    print("\n" + "=" * 50)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    df = load_raw_data()
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    plot_churn_distribution(df)
    plot_tenure_vs_churn(df)
    plot_monthly_charges_vs_churn(df)
    plot_contract_vs_churn(df)
    plot_correlation_heatmap(df)

    print(f"\n[DONE] All 5 EDA charts saved to: {EDA_DIR}")
    return df


if __name__ == "__main__":
    run_eda()
