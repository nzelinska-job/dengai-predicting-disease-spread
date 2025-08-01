import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from warnings import filterwarnings

filterwarnings("ignore")

# === Utility functions ===

def create_lag_features(df, features, lags=[1, 2]):
    df_lag = df.copy()
    for feature in features:
        for lag in lags:
            df_lag[f"{feature}_lag{lag}"] = df_lag[feature].shift(lag)
    # Instead of dropna, fill missing lag values (first rows) with forward fill or 0
    df_lag.fillna(method="ffill", inplace=True)
    df_lag.fillna(0, inplace=True)  # in case ffill fails for first rows
    return df_lag

def preprocess_features(features, labels=None):
    # Drop date column if present
    if "week_start_date" in features.columns:
        features = features.drop("week_start_date", axis=1)
    # Forward fill missing values
    features.fillna(method="ffill", inplace=True)
    features.fillna(0, inplace=True)

    # Create lag features
    base_features = [
        "reanalysis_specific_humidity_g_per_kg",
        "reanalysis_dew_point_temp_k",
        "station_avg_temp_c",
        "station_min_temp_c"
    ]
    features_lagged = create_lag_features(features, base_features)

    # Features to keep (base + lagged)
    lagged_features = base_features.copy()
    for f in base_features:
        for lag in [1, 2]:
            lagged_features.append(f"{f}_lag{lag}")

    if labels is not None:
        # Align labels by index and add
        features_lagged["total_cases"] = labels.loc[features_lagged.index]["total_cases"]

    return features_lagged, lagged_features

def train_and_save_model(city, train_features, train_labels, filename):
    # Prepare city-specific data
    city_features = train_features.loc[city].copy()
    city_labels = train_labels.loc[city].copy()

    # Preprocess features & labels
    city_features_lagged, lagged_features = preprocess_features(city_features, city_labels)

    X_train = city_features_lagged[lagged_features]
    y_train = city_features_lagged["total_cases"]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [5, 10, None],
        "regressor__min_samples_split": [2, 5]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    print(f"{city} Best parameters:", grid_search.best_params_)
    print(f"{city} Best CV MAE:", -grid_search.best_score_)

    # Train final model with best params
    best_params = grid_search.best_params_
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(
            n_estimators=best_params["regressor__n_estimators"],
            max_depth=best_params["regressor__max_depth"],
            min_samples_split=best_params["regressor__min_samples_split"],
            random_state=42,
            n_jobs=-1
        ))
    ])
    final_pipeline.fit(X_train, y_train)

    # Save model
    joblib.dump(final_pipeline, filename)
    print(f"{city} final model saved to {filename}\n")

    return lagged_features  # Return features for test processing

def preprocess_test_data(test_filepath, lagged_features_sj, lagged_features_iq):
    test_features = pd.read_csv(test_filepath, index_col=[0,1,2])

    # Separate cities
    sj_test = test_features.loc["sj"].copy()
    iq_test = test_features.loc["iq"].copy()

    # Drop date column and fill missing
    for df in [sj_test, iq_test]:
        if "week_start_date" in df.columns:
            df.drop("week_start_date", axis=1, inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(0, inplace=True)

    # Create lag features without dropping rows
    sj_test_lagged = create_lag_features(sj_test, lagged_features_sj[:4])
    iq_test_lagged = create_lag_features(iq_test, lagged_features_iq[:4])

    # Select only required features (all lagged features)
    sj_test_lagged = sj_test_lagged[lagged_features_sj]
    iq_test_lagged = iq_test_lagged[lagged_features_iq]

    return sj_test_lagged, iq_test_lagged

def main():
    # Load train data
    train_features = pd.read_csv("data/raw/dengue_features_train.csv", index_col=[0, 1, 2])
    train_labels = pd.read_csv("data/raw/dengue_labels_train.csv", index_col=[0, 1, 2])

    # Train models and get lagged features list
    lagged_features_sj = train_and_save_model("sj", train_features, train_labels, "final_best_model_sj.pkl")
    lagged_features_iq = train_and_save_model("iq", train_features, train_labels, "final_best_model_iq.pkl")

    # Load trained models
    sj_model = joblib.load("final_best_model_sj.pkl")
    iq_model = joblib.load("final_best_model_iq.pkl")

    # Preprocess test data
    sj_test, iq_test = preprocess_test_data("data/raw/dengue_features_test.csv", lagged_features_sj, lagged_features_iq)

    # Predict
    sj_preds = sj_model.predict(sj_test).astype(int)
    iq_preds = iq_model.predict(iq_test).astype(int)

    # Load submission with multi-index to align by index
    submission = pd.read_csv("data/raw/submission_format.csv", index_col=[0,1,2])

    # Assign predictions by aligning index (no length mismatch)
    submission.loc["sj", "total_cases"] = sj_preds
    submission.loc["iq", "total_cases"] = iq_preds

    # Save submission CSV (reset index if needed)
    submission.reset_index(inplace=True)
    submission.to_csv("data/raw/benchmark.csv", index=False)
    print("✅ Submission saved as data/raw/benchmark.csv")

if __name__ == "__main__":
    main()

