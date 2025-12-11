"""
Final Project: Diet Quality and 10-Year CVD Risk
Author: Rebecca Biddy

This script investigates the association between HEI-2015 diet scores
and predicted 10-year cardiovascular disease (CVD) risk among U.S. adults.
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.formula.api as smf


# ----------------------------
# Configuration
# ----------------------------

# Path to the cleaned dataset (relative to this script)
DATA_PATH = r"C:\Users\Michael\OneDrive\Desktop\FinalProject_report_REBIDDY\hei_cvd_clean.csv"

# Output directory for any processed files (optional)
OUTPUT_DIR = "data"


# ----------------------------
# Helper functions
# ----------------------------

def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """
    Load the cleaned HEI–CVD dataset and perform basic cleaning steps:
    - Filter age 30–74
    - Drop missing values in key variables
    - Standardize sex variable
    - Ensure numeric types
    - Create age_group categories
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Keep adults 30–74
    df = df[(df["age"] >= 30) & (df["age"] <= 74)]

    # Drop rows with missing key variables
    key_cols = ["age", "sex", "hei_score", "cvd_risk_score"]
    df = df.dropna(subset=key_cols)

    # Standardize sex values
    df["sex"] = df["sex"].astype(str).str.upper().str.strip()
    df["sex"] = df["sex"].replace({"MALE": "M", "FEMALE": "F"})

    # Ensure numeric
    df["hei_score"] = pd.to_numeric(df["hei_score"], errors="coerce")
    df["cvd_risk_score"] = pd.to_numeric(df["cvd_risk_score"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df = df.dropna(subset=["age", "hei_score", "cvd_risk_score"])

    # Create age groups
    age_bins = [30, 45, 60, 75]
    age_labels = ["30-44", "45-59", "60-74"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)

    print("Data loaded and cleaned.")
    print("First few rows:")
    print(df.head())

    return df


def descriptive_statistics(df: pd.DataFrame) -> None:
    """
    Print basic descriptive statistics overall
    and grouped by sex and age_group.
    """
    print("\n=== Descriptive Statistics (Overall) ===")
    print(df[["age", "hei_score", "cvd_risk_score"]].describe())

    print("\n=== HEI & CVD Risk by Sex ===")
    group_sex = df.groupby("sex")[["hei_score", "cvd_risk_score"]].agg(["mean", "std", "count"])
    print(group_sex)

    print("\n=== HEI & CVD Risk by Age Group ===")
    group_age = df.groupby("age_group")[["hei_score", "cvd_risk_score"]].agg(["mean", "std", "count"])
    print(group_age)

    # Optionally save summaries
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    group_sex.to_csv(os.path.join(OUTPUT_DIR, "group_summary_by_sex.csv"))
    group_age.to_csv(os.path.join(OUTPUT_DIR, "group_summary_by_age.csv"))
    print(f"\nGroup summaries saved to '{OUTPUT_DIR}' directory.")


def correlation_analysis(df: pd.DataFrame) -> None:
    """
    Compute and print Pearson correlation between HEI-2015
    and 10-year CVD risk.
    """
    print("\n=== Correlation Analysis: HEI vs CVD Risk ===")
    r, p_value = stats.pearsonr(df["hei_score"], df["cvd_risk_score"])
    print(f"Pearson r = {r:.3f}, p-value = {p_value:.3e}")


def plot_scatter_with_regline(df: pd.DataFrame) -> None:
    """
    Scatterplot of HEI vs CVD risk with a regression line.
    """
    print("\nGenerating scatterplot with regression line...")

    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df,
        x="hei_score",
        y="cvd_risk_score",
        scatter_kws={"alpha": 0.5},
        line_kws={"linewidth": 2},
    )
    plt.xlabel("HEI-2015 Diet Quality Score")
    plt.ylabel("Predicted 10-Year CVD Risk (%)")
    plt.title("Relationship Between Diet Quality and 10-Year CVD Risk")
    plt.tight_layout()
    plt.show()


def plot_stratified_scatter(df: pd.DataFrame) -> None:
    """
    Stratified scatterplots by sex and by age_group.
    """
    print("\nGenerating stratified scatterplots...")

    # By sex
    sns.lmplot(
        data=df,
        x="hei_score",
        y="cvd_risk_score",
        hue="sex",
        height=5,
        aspect=1.2,
        scatter_kws={"alpha": 0.5},
        line_kws={"linewidth": 2},
    )
    plt.xlabel("HEI-2015 Diet Quality Score")
    plt.ylabel("Predicted 10-Year CVD Risk (%)")
    plt.title("HEI-2015 vs CVD Risk by Sex")
    plt.tight_layout()
    plt.show()

    # By age group
    sns.lmplot(
        data=df,
        x="hei_score",
        y="cvd_risk_score",
        hue="age_group",
        height=5,
        aspect=1.2,
        scatter_kws={"alpha": 0.5},
        line_kws={"linewidth": 2},
    )
    plt.xlabel("HEI-2015 Diet Quality Score")
    plt.ylabel("Predicted 10-Year CVD Risk (%)")
    plt.title("HEI-2015 vs CVD Risk by Age Group")
    plt.tight_layout()
    plt.show()


def regression_analysis(df: pd.DataFrame) -> None:
    """
    Fit a multiple linear regression model:
    cvd_risk_score ~ hei_score + age + sex
    and print the summary.
    """
    print("\n=== Regression Analysis: CVD Risk ~ HEI + Age + Sex ===")

    # Ensure sex is categorical
    df["sex"] = df["sex"].astype("category")

    model = smf.ols(
        formula="cvd_risk_score ~ hei_score + age + C(sex)",
        data=df
    ).fit()

    print(model.summary())


def save_processed_data(df: pd.DataFrame) -> None:
    """
    Save a processed version of the dataset (optional).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "hei_cvd_clean_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"\nProcessed dataset saved to: {out_path}")


# ----------------------------
# Main
# ----------------------------

def main():
    # Load and clean data
    df = load_and_clean_data(DATA_PATH)

    # Descriptive statistics
    descriptive_statistics(df)

    # Correlation analysis
    correlation_analysis(df)

    # Visualizations
    plot_scatter_with_regline(df)
    plot_stratified_scatter(df)

    # Regression
    regression_analysis(df)

    # Save processed data (optional)
    save_processed_data(df)


if __name__ == "__main__":
    main()
