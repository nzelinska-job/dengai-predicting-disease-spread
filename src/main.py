import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import statsmodels.formula.api as smf  
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import pandas as pd
import os
#
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class BestGLMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, formula=None, alphas=None):
        self.formula = formula or (
            "total_cases ~ 1 + "
            "reanalysis_specific_humidity_g_per_kg + "
            "reanalysis_dew_point_temp_k + "
            "station_min_temp_c + "
            "station_avg_temp_c"
        )
        if alphas is None:
            # Дефолтна сітка, як у твоїй функції
            self.alphas = 10 ** np.arange(-8, -3, dtype=np.float64)
        else:
            self.alphas = alphas
        self.best_alpha_ = None
        self.model_ = None
        self.fitted_model_ = None

    def fit(self, X, y):
        # X — DataFrame, y — Series чи масив total_cases
        if isinstance(X, str):
            data = pd.read_csv(X, index_col=[0, 1, 2])
        else:
            data = X.copy()
        data['total_cases'] = y

        # Якщо хочеш крос-валідацію, розбий data тут (наприклад, на train/test)
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data, test_size=0.2, random_state=42)

        best_score = np.inf
        best_alpha = None

        for alpha in self.alphas:
            model = smf.glm(
                formula=self.formula,
                data=train,
                family=sm.families.NegativeBinomial(alpha=alpha)
            )
            try:
                results = model.fit()
                preds = results.predict(test).astype(int)
                score = eval_measures.meanabs(preds, test['total_cases'])
                if score < best_score:
                    best_score = score
                    best_alpha = alpha
            except Exception as e:
                continue  # statsmodels може впасти на некоректних alpha

        self.best_alpha_ = best_alpha

        # Фітати модель на всіх даних
        self.model_ = smf.glm(
            formula=self.formula,
            data=data,
            family=sm.families.NegativeBinomial(alpha=best_alpha)
        )
        self.fitted_model_ = self.model_.fit()
        return self

    def predict(self, X):
        preds = self.fitted_model_.predict(X)
        return np.round(preds).astype(int)  # якщо треба int, як у твоєму коді

    def get_params(self, deep=True):
        # Необхідно для сетапу в GridSearchCV
        return {
            'formula': self.formula,
            'alphas': self.alphas
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

class DenguePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, city, labels_path=None):
        self.city = city  # 'sj' або 'iq'
        self.labels_path = labels_path
        self.features = [
            "reanalysis_specific_humidity_g_per_kg",
            "reanalysis_dew_point_temp_k",
            "station_avg_temp_c",
            "station_min_temp_c",
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Завантажити якщо передали шлях, або використовувати DataFrame напряму
        if isinstance(X, str):
            df = pd.read_csv(X, index_col=[0, 1, 2])
        else:
            df = X.copy()
       
        df = df[self.features]
        df.ffill(inplace=True)
        
        if self.labels_path:
            labels = pd.read_csv(self.labels_path, index_col=[0, 1, 2])
            df = df.join(labels)
        
        # fill missing values
        df.ffill(inplace=True)
        return df.reset_index()  # якщо потрібно скинути індекс для подальшої роботи


test_features = pd.read_csv("data/raw/dengue_features_test.csv", index_col=[0, 1, 2])
sj_test = test_features.loc["sj"]
iq_test = test_features.loc["iq"]

preprocessor_sj = DenguePreprocessor(city='sj', 
                                     labels_path='data/raw/dengue_labels_train.csv')
pipeline_sj = Pipeline([
    ('preprocess', preprocessor_sj),  # твій трансформер
    ('glm', BestGLMRegressor())
])

y_sj = pd.read_csv("data/raw/dengue_labels_train.csv")
pipeline_sj.fit("data/raw/dengue_features_train.csv", y_sj.total_cases)
y_pred_sj = pipeline_sj.predict(sj_test)


pipeline_iq = Pipeline([
    ('preprocess', DenguePreprocessor(city='iq')),  # твій трансформер
    ('glm', BestGLMRegressor())
])
pipeline_iq.set_params(preprocess__labels_path="data/raw/dengue_labels_train.csv")
pipeline_iq.fit("data/raw/dengue_features_train.csv", y_sj.total_cases)
y_pred_iq = pipeline_iq.predict(iq_test)

# Збереження результатів у форматі, який очікується для подачі
submission = pd.read_csv("data/interim/submission_format.csv", index_col=[0, 1, 2])
submission.total_cases = np.concatenate([y_pred_sj, y_pred_iq])

src_path = 'data/interim/submission_format.csv'
dst_path = 'data/processed/benchmark.csv'

# Створити папки для файлу, якщо їх немає
if not os.path.isfile(dst_path):
    # Зчитуємо шаблон (лише заголовки)
    df_template = pd.read_csv(src_path, nrows=0)
    # Записуємо пустий DataFrame із таким же форматом
    df_template.to_csv(dst_path, index=False)
    print(f"Файл '{dst_path}' створено на основі '{src_path}'.")
else:
    print(f"Файл '{dst_path}' вже існує.")

submission.to_csv("data/processed/benchmark.csv")
