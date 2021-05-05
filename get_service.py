import pandas as pd
import os

import sqlite3 as sql
import flask
from flask import request, jsonify
import configparser


import json
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.4f}'.format)


config = configparser.ConfigParser()
config.sections()
config.read('config.ini')

conn = sql.connect(os.getcwd()+"/src/df_all_results.db")
data = pd.read_sql("SELECT * FROM df_all_results", conn).drop(columns="index")
data.fillna("NA", inplace=True)

data['Random Forest Probability'] = data['Random Forest Probability'].apply(lambda x: '{:.5f}'.format(x))
data['Calibrated Random Forest Probability'] = data['Calibrated Random Forest Probability'].apply(lambda x: '{:.5f}'.format(x))
data['Naive Bayes'] = data['Naive Bayes'].apply(lambda x: '{:.5f}'.format(x))
data['Calibrated Naive Bayes (Isotonic)'] = data['Calibrated Naive Bayes (Isotonic)'].apply(lambda x: '{:.5f}'.format(x))
data['Calibrated Naive Bayes (Sigmoid)'] = data['Calibrated Naive Bayes (Sigmoid)'].apply(lambda x: '{:.5f}'.format(x))


df = data.to_json(orient="records")
df = json.loads(df)

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/all_result', methods=['GET'])
def total_api():
    return jsonify(df)

@app.route('/result', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'uuid' in request.args:
        uuid = request.args['uuid']
    else:
        return "Error: No uuid field provided. Please specify an id."

    # Create an empty list for our results
    results = []
    # Loop through the data and match results that fit the requested ID.

    for id_ in range(len(df)):
        if (df[id_]["uuid"] == uuid):
            results.append(df[id_])

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    if len(results)<1:
        return "uuid is not found", 404
    else:
        return jsonify(results)

app.run(host=config["Service"]["Host"], port=int(config["Service"]["Port"]), debug=True)


