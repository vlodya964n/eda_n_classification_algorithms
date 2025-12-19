
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)
model = joblib.load('models/model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    df = pd.DataFrame(content, index=[0])
    print(df)
    result = model.predict(df)
    result = {'Predicted_Rings': float(result[0])}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
