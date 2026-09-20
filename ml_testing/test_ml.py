import sys
import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import TimeSeriesSplit


# =========================================================
# STERLIFLUX - ML MODEL TESTING
# =========================================================

if len(sys.argv) < 2:
    print("\nUsage:")
    print("python ml_testing/test_ml.py your_file.csv")
    sys.exit()


file_path = sys.argv[1]


# =========================================================
# 1. LOAD DATA
# =========================================================

if not os.path.exists(file_path):
    print(f"\nERROR: File not found: {file_path}")
    sys.exit()


df = pd.read_csv(file_path)

print("\n========================================")
print("     STERLIFLUX ML MODEL TEST")
print("========================================")


# =========================================================
# 2. CHECK COLUMNS
# =========================================================

df.columns = df.columns.str.strip().str.lower()

if "date" not in df.columns or "amount" not in df.columns:
    print("\nERROR: CSV must contain Date and Amount columns.")
    sys.exit()


# =========================================================
# 3. CLEAN DATA
# =========================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["amount"] = pd.to_numeric(
    df["amount"],
    errors="coerce"
)

df = df.dropna(
    subset=["date", "amount"]
)

df = df[df["amount"] >= 0]

df = df.sort_values("date")


# =========================================================
# 4. CREATE DAILY DATA
# =========================================================

daily = (
    df.groupby("date")["amount"]
    .sum()
    .reset_index()
)

daily = daily.set_index("date")

full_dates = pd.date_range(
    daily.index.min(),
    daily.index.max(),
    freq="D"
)

daily = daily.reindex(
    full_dates,
    fill_value=0
)

daily.index.name = "date"

daily = daily.reset_index()


# =========================================================
# 5. FEATURE ENGINEERING
# =========================================================

daily["day"] = daily["date"].dt.day

daily["month"] = daily["date"].dt.month

daily["weekday"] = daily["date"].dt.weekday

daily["week_of_year"] = (
    daily["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

daily["day_number"] = (
    daily["date"] - daily["date"].min()
).dt.days


# Previous day's expense
daily["previous_expense"] = (
    daily["amount"].shift(1)
)


# Previous 7 days only
daily["rolling_7"] = (
    daily["amount"]
    .shift(1)
    .rolling(
        7,
        min_periods=1
    )
    .mean()
)


features = [
    "day",
    "month",
    "weekday",
    "week_of_year",
    "day_number",
    "previous_expense",
    "rolling_7"
]


daily_model = daily.dropna(
    subset=features + ["amount"]
).copy()


X = daily_model[features]

y = daily_model["amount"]


# =========================================================
# 6. CHRONOLOGICAL TRAIN / TEST
# =========================================================

split_index = int(
    len(daily_model) * 0.8
)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


print("\nTraining records :", len(X_train))
print("Testing records  :", len(X_test))


# =========================================================
# 7. TRAIN RANDOM FOREST
# =========================================================

model = RandomForestRegressor(
    n_estimators=250,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# =========================================================
# 8. TEST PREDICTIONS
# =========================================================

predictions = model.predict(
    X_test
)


# =========================================================
# 9. PERFORMANCE METRICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n========================================")
print("          MODEL PERFORMANCE")
print("========================================")

print(f"\nMAE  : ₹{mae:.2f}")
print(f"RMSE : ₹{rmse:.2f}")
print(f"R²   : {r2:.4f}")


# =========================================================
# 10. TIME SERIES CROSS-VALIDATION
# =========================================================

print("\n========================================")
print("       CROSS-VALIDATION TEST")
print("========================================")


tscv = TimeSeriesSplit(
    n_splits=5
)

cv_mae = []
cv_rmse = []
cv_r2 = []


for fold, (train_index, test_index) in enumerate(
    tscv.split(X),
    start=1
):

    X_train_cv = X.iloc[train_index]
    X_test_cv = X.iloc[test_index]

    y_train_cv = y.iloc[train_index]
    y_test_cv = y.iloc[test_index]


    cv_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )


    cv_model.fit(
        X_train_cv,
        y_train_cv
    )


    cv_predictions = cv_model.predict(
        X_test_cv
    )


    fold_mae = mean_absolute_error(
        y_test_cv,
        cv_predictions
    )

    fold_rmse = mean_squared_error(
        y_test_cv,
        cv_predictions
    ) ** 0.5

    fold_r2 = r2_score(
        y_test_cv,
        cv_predictions
    )


    cv_mae.append(fold_mae)
    cv_rmse.append(fold_rmse)
    cv_r2.append(fold_r2)


    print(
        f"\nFold {fold}"
    )

    print(
        f"MAE  : ₹{fold_mae:.2f}"
    )

    print(
        f"RMSE : ₹{fold_rmse:.2f}"
    )

    print(
        f"R²   : {fold_r2:.4f}"
    )


# =========================================================
# 11. CROSS-VALIDATION AVERAGE
# =========================================================

print("\n----------------------------------------")

print(
    f"Average MAE  : ₹{sum(cv_mae) / len(cv_mae):.2f}"
)

print(
    f"Average RMSE : ₹{sum(cv_rmse) / len(cv_rmse):.2f}"
)

print(
    f"Average R²   : {sum(cv_r2) / len(cv_r2):.4f}"
)


# =========================================================
# 12. ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values,
    label="Actual"
)

plt.plot(
    predictions,
    label="Predicted"
)

plt.title(
    "SterliFlux - Actual vs Predicted Expenses"
)

plt.xlabel(
    "Test Days"
)

plt.ylabel(
    "Expense (₹)"
)

plt.legend()

plt.tight_layout()

graph_path = os.path.join(
    os.path.dirname(__file__),
    "actual_vs_predicted.png"
)

plt.savefig(
    graph_path,
    dpi=300
)

plt.close()


print(
    f"\nGraph saved to: {graph_path}"
)


# =========================================================
# 13. FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\n========================================")
print("          FEATURE IMPORTANCE")
print("========================================\n")

print(
    importance.to_string(index=False)
)


# =========================================================
# 14. FINAL
# =========================================================

print("\n========================================")
print("              TEST COMPLETE")
print("========================================")