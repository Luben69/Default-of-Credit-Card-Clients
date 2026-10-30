# Credit Card Default Prediction (Taiwan, 2005)

Predicting whether a credit card client will default on next month's payment, using 6 months of payment history, bill statements and demographics.

**Result:** tuned LightGBM, **test ROC-AUC 0.78**. A logistic regression gets 0.77 in cross-validation, so most of the signal is simple. The strongest predictor by far is the most recent repayment status (`PAY_1`).

## Data

- [UCI: Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh & Lien, 2009): 30,000 clients, 23 features, no missing values.
- Target `defpay`: 22.1% of clients default (imbalanced). A model that always predicts "no default" scores 77.9% accuracy but ROC-AUC 0.5, so accuracy is not used to compare models.

### What I found in EDA

- **Credit limit matters.** Default rate is 31.8% in the lowest `LIMIT_BAL` quartile and 14.0% in the highest.
- **Sex, education, marriage and age are weak** predictors on their own.
- **Raw bill amounts have almost no linear correlation with default** (about -0.02 to 0). A bill only means something relative to the limit.
- **Undocumented codes:** `EDUCATION` has 0, 5, 6; `MARRIAGE` has 0; `PAY_x` has -2 and 0, which the dataset documentation doesn't define. I kept them as-is and one-hot encoded them.
- **35 duplicate rows**, mostly inactive accounts (all-zero bills and payments). Kept, since the number is negligible.
- Bills contain negative values (credits/refunds), down to about -340k.

## Method

1. Stratified 80/20 train/test split (`random_state=42`). The test set was used once, at the end.
2. All preprocessing (scaling, one-hot encoding) lives inside scikit-learn `Pipeline`s, so it is fit on training folds only.
3. 5 models compared with 5-fold stratified CV and `RandomizedSearchCV`.
4. The final model was chosen from CV results, not from the test set.

## Results

| Model | CV ROC-AUC  |
|---|---|
| Logistic Regression | 0.7697 |
| Random Forest | 0.7821 |
| XGBoost | 0.7835 |
| LightGBM | 0.7853 |
| CatBoost | 0.7854 |

The top four are statistically indistinguishable (their spread is smaller than the fold-to-fold noise). I chose **LightGBM** for speed and a simple input pipeline, not because it scored higher.

### Test set (tuned LightGBM, default 0.5 threshold)

| Metric | Value |
|---|---|
| ROC-AUC | 0.781 |
| Accuracy | 0.819 |
| Precision (defaulters) | 0.67 |
| Recall (defaulters) | 0.36 |
| F1 (defaulters) | 0.47 |

Of 1,327 real defaulters in the test set, the model catches 474 and misses 853.

## Limitations

- Data is from one Taiwanese bank in 2005, with no income or transaction detail. It is a practice dataset, not a deployable credit model.
- Using sex and marital status as features would be legally and ethically problematic in real credit scoring.

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost xlrd
```

Put `default of credit card clients.xls` next to the notebook (download from the UCI link above) and run credit_default_analysis.ipynb from top to bottom. The hyperparameter search takes about 10 minutes.