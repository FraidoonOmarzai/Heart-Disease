from flask import Flask, render_template, url_for, redirect
import joblib
from flask import request
import numpy as np
import os

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


@app.route("/")
def cancer():
    return render_template("heart.html")


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if(size == 13):
        loaded_model = joblib.load("saved_models/model.joblib")
        result = loaded_model.predict(to_predict)

    return result[0]


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        # print(to_predict_list)
        to_predict_list = list(to_predict_list.values())
        # print(to_predict_list)
        to_predict_list = list(map(float, to_predict_list))
        # print(to_predict_list)
        if(len(to_predict_list) == 13):
            result = ValuePredictor(to_predict_list, 13)

    if(int(result) == 1):
        prediction = "Sorry! it seems getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))


if __name__ == "__main__":
    app.run(debug=True)
