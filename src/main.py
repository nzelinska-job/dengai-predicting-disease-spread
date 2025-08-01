import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
import statsmodels.api as sm    
import pandas as pd
#
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class NegativeBinomialRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-8):
        self.alpha = alpha
        self.model_ = None
        self.result_ = None

    def fit(self, X, y):
        X_const = sm.add_constant(X, has_constant='add')
        model = sm.GLM(y, X_const, family=sm.families.NegativeBinomial(alpha=self.alpha))
        self.result_ = model.fit()
        return self

    def predict(self, X):
        X_const = sm.add_constant(X, has_constant='add')
        preds = self.result_.predict(X_const)
        if np.any(np.isnan(preds)):
             print("Warning: NaN values detected in predictions")
        preds = np.nan_to_num(preds, nan=0.0)
        return np.round(preds).astype(int)

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    # select features we want
    features = [
        "reanalysis_specific_humidity_g_per_kg",
        "reanalysis_dew_point_temp_k",
        "station_avg_temp_c",
        "station_min_temp_c",
    ]
    df = df[features]

    # fill missing values
    df.ffill(inplace=True)


    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc["sj"]
    iq = df.loc["iq"]

    return sj, iq

sj_train, iq_train = preprocess_data("data/raw/dengue_features_train.csv",
                                     labels_path="data/raw/dengue_labels_train.csv",)
test_features = pd.read_csv("data/raw/dengue_features_test.csv", index_col=[0, 1, 2])
sj_test = test_features.loc["sj"]
iq_test = test_features.loc["iq"]

FEATURES = [
    # перелік потрібних ознак, наприклад:
    'reanalysis_specific_humidity_g_per_kg',
    'reanalysis_dew_point_temp_k',
    'station_avg_temp_c',
    'station_min_temp_c'
]

pipe = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("neg_binom", NegativeBinomialRegressor(alpha=1e-8))
])


pipe.fit(sj_train[FEATURES], sj_train["total_cases"])
pipe.fit(iq_train[FEATURES], iq_train["total_cases"])

y_pred_sj = pipe.predict(sj_test[FEATURES]).astype(int)
y_pred_iq = pipe.predict(iq_test[FEATURES]).astype(int)


submission = pd.read_csv("data/interim/submission_format.csv", index_col=[0, 1, 2])
submission.total_cases = np.concatenate([y_pred_sj, y_pred_iq])
submission.to_csv("data/processed/benchmark.csv")
