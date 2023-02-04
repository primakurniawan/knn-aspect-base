import json
from flask import Flask, render_template, request
from markupsafe import escape

import pandas as pd

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", title='home')


@app.route("/crawling", methods=["GET"])
def crawling():
    df = pd.read_excel("./static/data/excel/reviews.xlsx")
    df = df[['content', 'score']]
    return render_template("crawling.html",
                           title='crawling',
                           data_crawling=df.to_dict(orient="records"))


@app.route("/preprocessing", methods=["GET"])
def preprocessing_():
    df = pd.read_excel("./static/data/excel/after_labelling.xlsx")
    df = df[['content', 'after_preprocessing', 'service', 'doctor', 'medicine', 'fee']]
    return render_template("preprocessing.html",
                           title='preprocessing',
                           data_preprocessing=df.to_dict(orient="records"))


@app.route("/result/<tf>", methods=["GET"])
def result(tf):
    if request.method == "GET":
        df_train = pd.read_excel(f'./static/data/excel/prediction_result_train_{tf}.xlsx')
        df_train = df_train[['content', 'after_preprocessing', 'service', 'doctor', 'medicine', 'fee']]
        df_test = pd.read_excel(f'./static/data/excel/prediction_result_test_{tf}.xlsx')
        df_test = df_test[['content', 'after_preprocessing', 'service', 'doctor', 'medicine', 'fee', 'service_pred', 'doctor_pred', 'medicine_pred', 'fee_pred']]
    return render_template("result.html",
                           title='result',
                           tf=escape(tf),
                           data_training=df_train.to_dict(orient="records"),
                           data_testing=df_test.to_dict(orient="records"))


@ app.route("/report/<tf>", methods=["GET"])
def report(tf):
    if request.method == "GET":
        df_classification_report_service = pd.read_excel(
            f"./static/data/excel/classification_report_service_{tf}.xlsx").to_dict(orient="records")
        df_classification_report_doctor = pd.read_excel(
            f"./static/data/excel/classification_report_doctor_{tf}.xlsx").to_dict(orient="records")
        df_classification_report_medicine = pd.read_excel(
            f"./static/data/excel/classification_report_medicine_{tf}.xlsx").to_dict(orient="records")
        df_classification_report_fee = pd.read_excel(
            f"./static/data/excel/classification_report_fee_{tf}.xlsx").to_dict(orient="records")
        with open(f'./static/data/json/report_{tf}.json') as json_file:
            data = json.load(json_file)
        return render_template(
            "report.html",
            title='report',
            tf=tf,
            k=data['k'],
            accuracy=data['accuracy'],
            f1_score=data['f1_score'],
            precision=data['precision_score'],
            recall=data['recall_score'],
            data_classification_report_service=df_classification_report_service,
            data_classification_report_doctor=df_classification_report_doctor,
            data_classification_report_medicine=df_classification_report_medicine,
            data_classification_report_fee=df_classification_report_fee
            )


if __name__ == "__main__":
    app.run(debug=True)
