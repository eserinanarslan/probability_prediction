#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os

import sqlite3 as sql

from flask import Flask
from flask_restful import Resource, Api

import json

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.4f}'.format)



conn = sql.connect(os.getcwd()+"\df_all_results.db")
data = pd.read_sql("SELECT * FROM df_all_results", conn).drop(columns="index")
data.fillna("NA", inplace=True)



data['Random Forest Probability'] = data['Random Forest Probability'].apply(lambda x: '{:.5f}'.format(x))
data['Calibrated Random Forest Probability'] = data['Calibrated Random Forest Probability'].apply(lambda x: '{:.5f}'.format(x))
data['Naive Bayes'] = data['Naive Bayes'].apply(lambda x: '{:.5f}'.format(x))
data['Calibrated Naive Bayes (Isotonic)'] = data['Calibrated Naive Bayes (Isotonic)'].apply(lambda x: '{:.5f}'.format(x))
data['Calibrated Naive Bayes (Sigmoid)'] = data['Calibrated Naive Bayes (Sigmoid)'].apply(lambda x: '{:.5f}'.format(x))



df = data.to_json(orient="records")
df = json.loads(df)



app = Flask(__name__)
api = Api(app)

class ProbabilityPrediction(Resource):
    
    def __init__(self):
        self.check_uuid=None
        self.uuid=None
    
    def get(self):
        return df, 200
    
    def post(self, uuid):
        for id_ in range(len(df)):
            if (df[id_]["uuid"] == uuid):
                return df[id_], 201


api.add_resource(ProbabilityPrediction, "/result", "/result/", "/result/<uuid>")

if __name__ == '__main__':
    app.run()


