from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = 'data/preprocessed_data.csv'
X_TEST_PATH = 'data/X_test_data.csv'
Y_TEST_PATH = 'data/y_test_data.csv'
MODEL_PATH = Path('models/model.pkl')

CATEGORICAL = ['Sex']
NUMERICAL = [
    'Length', 'Diameter', 'Height',
    'Whole_weight', 'Shucked_weight',
    'Viscera_weight', 'Shell_weight'
]

if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=['Rings'])
    y = data['Rings']

    bins = [0, 8, 10, np.inf]
    y_bins = pd.cut(y, bins=bins)  # 10 квантилей

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y_bins)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
            ('num', 'passthrough', NUMERICAL)
        ]
    )

    model = Pipeline([
        ('preprocess', preprocessor),
        ('rf', RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    X_test.to_csv(X_TEST_PATH)
    y_test.to_csv(Y_TEST_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
