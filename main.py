from flask import Flask, request, jsonify
from keras import models
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# tf.disable_v2_behavior()

app = Flask(__name__)
# model = models.load_model('NN_model')

# Load models
NN_models = []
Normalizer_models = []
classifier = joblib.load(f'./models/DesTreeClassifier_NN.save')


def get_data_class(sample: pd.DataFrame):
    return classifier.predict(sample)[0]


i = 0
while i < 9:
    i += 1
    NN_models.append(models.load_model(f'./models/NN_model_{i}'))
    Normalizer_models.append(joblib.load(f'./models/NN_normalization_model_{i}.save'))


# Простые примеры
@app.route('/test')
def test():
    return 'Hello world!'


@app.route('/get_predict')
def get_predict():
    # get json
    json = request.json
    # Transform json to DataFrame
    sample = pd.DataFrame.from_dict(json, orient='index').T
    # Get data class
    data_class = get_data_class(sample)
    # Add Cluster column
    sample['Cluster'] = data_class
    # Get model and predict
    predict = NN_models[data_class].predict(sample)[0][0]
    # Create answer
    answer = {'class': int(data_class), 'predict': float(predict)}
    return jsonify(answer), 200


def start():
    app.run()


if __name__ == '__main__':
    start()
