import os
import pandas as pd
import joblib

from explainerdashboard import ExplainerDashboard
from explainerdashboard.explainers import RegressionExplainer

MODEL_PATH = 'models/model.pkl'
X_TEST_PATH = 'data/X_test_data.csv'
Y_TEST_PATH = 'data/y_test_data.csv'
DASHBOARD_YAML = 'data/dashboard.yaml'


if __name__ == '__main__':

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('Model not found. Train model first.')
    elif not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError('X_test_data not found. Train model first.')
    elif not os.path.exists(Y_TEST_PATH):
        raise FileNotFoundError('y_test_data not found. Train model first.')

    model = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(X_TEST_PATH, index_col=0)
    y_test = pd.read_csv(Y_TEST_PATH, index_col=0)

    explainer = RegressionExplainer(model, X_test, y_test)

    db = ExplainerDashboard(explainer)
    db.to_yaml(DASHBOARD_YAML, explainerfile='explainer.joblib', dump_explainer=True)
