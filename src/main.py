import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from warnings import filterwarnings
filterwarnings("ignore")

# Load data
train_features = pd.read_csv(
    "data/raw/dengue_features_train.csv", index_col=[0, 1, 2]
)
train_labels = pd.read_csv(
    "data/raw/dengue_labels_train.csv", index_col=[0, 1, 2]
)

# Prepare San Juan data
sj_features = train_features.loc["sj"].copy()
sj_labels = train_labels.loc["sj"].copy()

sj_features.drop("week_start_date", axis=1, inplace=True)
sj_features.fillna(method="ffill", inplace=True)

sj_features["total_cases"] = sj_labels["total_cases"]

# Features to use
base_features = [
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "station_avg_temp_c",
    "station_min_temp_c"
]

# Create lag features for each base feature (1 and 2 weeks lag)
def create_lag_features(df, features, lags=[1, 2]):
    df_lag = df.copy()
    for feature in features:
        for lag in lags:
            df_lag[f"{feature}_lag{lag}"] = df_lag[feature].shift(lag)
    df_lag.dropna(inplace=True)  # drop rows with NaNs created by lagging
    return df_lag

sj_features_lagged = create_lag_features(sj_features, base_features)

# Define features after lagging
lagged_features = base_features.copy()
for f in base_features:
    for lag in [1, 2]:
        lagged_features.append(f"{f}_lag{lag}")

# Split into train/test
train_size = 800
sj_train = sj_features_lagged.iloc[:train_size]
sj_test = sj_features_lagged.iloc[train_size:]

X_train = sj_train[lagged_features]
y_train = sj_train["total_cases"]

X_test = sj_test[lagged_features]
y_test = sj_test["total_cases"]

# Pipeline setup
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))
])

# Hyperparameter grid for RandomForest
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [5, 10, None],
    "regressor__min_samples_split": [2, 5],
}

# TimeSeries cross-validator
tscv = TimeSeriesSplit(n_splits=5)

# Grid search with time series split
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

# Fit grid search
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV MAE:", -grid_search.best_score_)

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

print(f"Test MAE: {test_mae:.2f}")

# Optional: You can switch to GradientBoostingRegressor by replacing pipeline regressor like:
# pipeline.set_params(regressor=GradientBoostingRegressor(random_state=42))
# and update param_grid accordingly

