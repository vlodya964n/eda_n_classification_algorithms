import os
import subprocess

from explainerdashboard import ExplainerDashboard

DASHBOARD_YAML = 'data/dashboard.yaml'
DASHBOARD_JOBLIB = 'data/explainer.joblib'

if __name__ == '__main__':
    if not os.path.exists(DASHBOARD_YAML):
        subprocess.run(['python', 'generate_dashboard.py'], check=True)

    db = ExplainerDashboard.from_config(DASHBOARD_JOBLIB, DASHBOARD_YAML)
    db.run(host='0.0.0.0', port=9050, use_waitress=True)
