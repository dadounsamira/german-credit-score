# German Credit Risk: Statistical Analysis and Loan Approval Models

Which loan applicants are likely to repay? This project analyses 1,000 past applicants of a German bank, finds which applicant characteristics are really linked to credit risk using proper statistical tests, and builds classification models to support loan approval decisions.

**Status:** data preparation ✅ · statistical tests ✅ · classification models 🔄 in progress · imbalance handling ⬜ · customer segments ⬜ · Power BI dashboard ⬜

---

## The problem

A bank must decide who gets a loan. Approving a customer who will not repay costs far more than rejecting one who would have repaid. The goal is a data-driven, explainable approach that helps managers make that call.

## The data

- **German Credit dataset** (UCI Statlog), numeric-coded version: [UCI page](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **1,000 applicants**, **20 features** (account balance, credit history, loan duration and purpose, savings, employment, age, housing…)
- **Target:** `Creditability` (1 = good credit risk, 0 = bad)
- **Class balance:** 700 good / 300 bad (70 / 30)
- No missing values, no duplicate rows

## Approach

| Phase | Question | Methods |
|---|---|---|
| 1. Data preparation | What does the data look like? | Variable typing (numeric / ordinal / nominal), descriptive statistics (mean, median, std, CV, skewness), histograms, box plots, IQR outliers, class balance |
| 2. Statistical tests | Which features are really linked to credit risk? | Shapiro-Wilk, Mann-Whitney U, Welch's t-test, chi-square + Cramér's V, Spearman correlation, VIF, Weight of Evidence + Information Value, logistic regression (odds ratios, Wald tests) |
| 3. Models | Can we predict good vs bad? | Decision Tree, Naive Bayes, KNN; all features vs reduced set; confusion matrix, TP / FP rate, ROC-AUC, cost matrix |
| 4. Imbalance | Does balancing help catch bad customers? | Undersampling, oversampling, SMOTE (training data only) |
| 5. Segments | What types of customers exist? | K-Means, elbow method |
| 6. Model comparison | Is one model really better? | McNemar's test, bootstrap confidence intervals |
| 7. Dashboard | How do managers explore the results? | Power BI |

Each test was chosen for the variable type: rank-based tests for skewed numeric data, chi-square for categorical data, and never Pearson correlation on category codes.

## Key findings so far

### Data preparation
- All numeric variables are **right-skewed** (credit amount skew 1.95, CV 86%), so medians and rank-based tests are used.
- Outliers were **kept**: very long loans (> 42 months) and very large loans (> 7,882 DM) are about **twice as risky** (57% and 54% bad vs 28%), so they carry key information.
- A model predicting "good" for everyone scores **70% accuracy** while catching no bad customers, so accuracy alone is misleading.

### Statistical tests
- **Shapiro-Wilk:** duration, credit amount and age are not normal (all p < 0.001).
- **Mann-Whitney U:** bad customers take longer loans (median 24 vs 18 months) and slightly larger ones; good customers are older (median 34 vs 31). The t-test agrees.
- **Chi-square + Cramér's V / Information Value:** the strongest categorical features are

| Feature | Cramér's V | Information Value |
|---|---|---|
| Account balance | 0.35 | 0.67 (very strong) |
| Payment status of previous credit | 0.25 | 0.29 (medium) |
| Savings | 0.19 | 0.20 (medium) |
| Purpose | 0.18 | 0.16 (medium) |

- Occupation, telephone and years at current address show **no link** to credit risk.
- **No multicollinearity:** highest Spearman ρ = 0.62 (duration vs amount), all VIF ≤ 2.15.
- **Logistic regression** (holding other features fixed):
  - No checking account: **5.65×** the odds of being good vs an overdrawn account
  - Each extra month of loan: odds of good **−3.1%** (about −31% per extra year)
  - Credit amount was significant alone but **not in the model** (p = 0.29): loan duration already carries that information.
- **Surprises worth noting:** customers flagged as "critical account" are among the safest (17% bad), and used-car loans are far safer than new-car loans (16.5% vs 38% bad).

### Reduced feature set
Kept features that are strong alone **and** significant together in a logistic regression:

**Account balance, payment status, loan duration, savings, purpose.**

Compared with my original 2024 analysis (Pearson correlation + stepwise selection), 4 of 5 features are the same. Credit amount is replaced by **purpose**, which Pearson correlation on category codes could not detect.

## Repository structure

```
german-credit-score/
├── data/
│   └── german_credit.csv
├── notebooks/
│   ├── 01-data-prep.ipynb           # data quality, descriptive statistics, outliers, class balance
│   └── 02-statistical-tests.ipynb   # normality, group tests, chi-square, correlation, WoE/IV, logistic regression
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/dadounsamira/german-credit-score.git
cd german-credit-score
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook
```

Then open the notebooks in order.

## Tools

Python · pandas · NumPy · SciPy · statsmodels · scikit-learn · imbalanced-learn · matplotlib · seaborn · Power BI

## Background

Based on a course project (CIND 119, Toronto Metropolitan University) originally done in SAS and Google Colab with a Decision Tree and Naive Bayes. This version is my own re-implementation in Python, extended with formal statistical testing, Weight of Evidence / Information Value, logistic regression, KNN, imbalance handling, clustering and a Power BI dashboard.