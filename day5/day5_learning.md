# Day 5: Ensemble Methods & Feature Engineering

## What I Learned

**Decision Trees:** Trees with root, branches, and leaves. Each split tries to minimize Gini impurity (measure of disorder). Trees are simple but can overfit.

**Random Forests:** Multiple decision trees voting together. Each tree is trained on random subsets of data and features. The majority vote is the final prediction. More robust than single trees.

**XGBoost:** Sequential boosting—each new tree corrects errors from previous trees. Theory was interesting, but tuning felt complex.

**Feature Engineering:** Improving model inputs through:
- Scaling: Normalizing numeric features (StandardScaler)
- Encoding: Converting categories to numbers (one-hot vs label encoding)
- Selection: Dropping irrelevant features

## Results on Titanic Dataset

- Decision Tree: 81.46%
- Random Forest (n=100, depth=5): **82.58%** ✓
- XGBoost: 80.34%

Random Forest won. More trees helped, but too many (200) caused overfitting.

## Key Insight

Not all feature engineering improves accuracy. Sometimes simpler is better. Hyperparameter tuning (`max_depth`, `n_estimators`) mattered more than fancy feature transformation.

## Next Steps

Try other algorithms. Push toward 85%+ accuracy.