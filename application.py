from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "LinearRegressionModel.pkl"), 'rb'))
car = pd.read_csv(os.path.join(BASE_DIR, "cleaned car.xls"))

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))


    input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    prediction = model.predict(input_data)
    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    # Use port Render provides
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
