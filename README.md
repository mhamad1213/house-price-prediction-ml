# House Price Prediction — ML Workflow

A complete, interpretable machine learning project: predicting residential sale prices from structured housing data, using the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset (Ames, Iowa — 1,460 houses, 79 features).

## Problem

Given a house's features — size, quality rating, location, age, garage/basement details, and 70+ others — predict its sale price. This is a supervised regression problem with the kind of mess real-world data actually has: a skewed target, missing values that mean different things in different columns, and a mix of numeric and categorical features.

## Approach

1. **Explored** the data: distribution of sale price (right-skewed → modeled `log(price)` instead), which "missing" values are actually meaningful (e.g. `NaN` in `PoolQC` means "no pool," not "unknown"), and which features correlate most with price.
2. **Preprocessed** with a leakage-safe `scikit-learn` pipeline: explicit `"None"` category for features that are genuinely absent (not missing), median imputation + scaling for numeric features, most-frequent imputation + one-hot encoding for categoricals — all fit only on the training fold during cross-validation.
3. **Trained three baselines** spanning different model families: Linear Regression, Decision Tree, and Random Forest.
4. **Evaluated** with 5-fold cross-validation for model selection, then a held-out 20% test set for the final, honest numbers — reported in both log scale (for comparing models) and dollars (for a result that actually means something).
5. **Tuned** the Random Forest with `GridSearchCV` over tree depth, leaf size, and forest size.
6. **Visualized and interpreted**: feature importance, predicted-vs-actual, and residual plots.
7. **Bonus:** reframed the problem as binary classification (above/below median price) with Logistic Regression, to also demonstrate accuracy/precision/recall/confusion-matrix evaluation — the metrics named in the original project brief, applied where they actually fit (a continuous price target isn't a classification problem; a median-split version of it is).

## Results

| Model | Held-out MAE | Held-out RMSE | Held-out R² |
|---|---|---|---|
| **Linear Regression** | **≈ $15,075** | **≈ $22,907** | **≈ 0.932** |
| Random Forest (tuned) | ≈ $17,353 | ≈ $29,601 | ≈ 0.886 |
| Decision Tree | ≈ $25,676 | ≈ $40,134 | ≈ 0.790 |

Bonus classification (above/below median price, Logistic Regression): **~90% accuracy**, with precision and recall both close to accuracy — no systematic bias toward either class.

**The honest, slightly counter-intuitive finding:** plain Linear Regression generalized better than a tuned Random Forest on this dataset. That's not the usual "just use a Random Forest" story, and it's worth stating plainly rather than hiding it — it happens because, after log-transforming the target, the strongest predictors (overall quality rating, living area, basement/garage size) relate to log-price in a fairly linear, monotonic way, so a linear model captures most of the signal cheaply while the extra flexibility of a tree ensemble mostly adds variance on a dataset this size (~1,200 training rows). The practical lesson: always compare against a simple baseline, and let held-out performance — not a model's reputation — make the call.

`OverallQual` (a 1–10 quality rating) is by a wide margin the single strongest driver of price, followed by living area and basement/garage size.

## What I learned

- Log-transforming a skewed target changes which model wins — a good reminder to check assumptions before picking a model family.
- Not all missing data is missing: reading the data dictionary mattered more than any imputation technique.
- A `ColumnTransformer` + `Pipeline` isn't just convenient — it's what prevents preprocessing (imputation, scaling, encoding) from leaking information across cross-validation folds.
- Feature importance from a tree model and coefficient inspection from a linear model tell complementary stories; using the right model for the right question (even when it's not the "winning" one) produces more useful output than always defaulting to the best single metric.

## Repo contents

- `house_price_prediction.ipynb` — the full workflow, with all cells executed and outputs/plots saved inline
- `data/` — `train.csv` / `test.csv` (Kaggle competition files; `test.csv` has no `SalePrice` and isn't used here — it's only for Kaggle leaderboard submissions)
- `outputs/` — exported PNG charts (target distribution, feature importance, predicted-vs-actual, confusion matrix)
- `requirements.txt` — Python dependencies

## Running it

```bash
pip install -r requirements.txt
jupyter notebook house_price_prediction.ipynb
```

## Possible extensions

- Target/ordinal encoding for quality-rated features instead of one-hot
- Gradient boosting (XGBoost / LightGBM) as a stronger baseline
- SHAP values for per-prediction explanations
- Wrap the trained pipeline in a small Flask/FastAPI endpoint for live predictions
