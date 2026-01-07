import pickle
from flask import Flask, request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


app = Flask(__name__)

@app.route('/')
def index_func():
    return render_template('index.html')

@app.route('/predict', methods= ['GET','POST'])
def predictions():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))

        )

        pred_df =data.get_data_as_df()
        print(pred_df)

        predict_pipleine = PredictPipeline()
        results  = predict_pipleine.prediction(pred_df)
        return render_template('home.html' , results =results[0])


if __name__ == "__main__":
    app.run(host = "0.0.0.0")