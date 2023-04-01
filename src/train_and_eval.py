###### Steps:
## load the train and test
## train alg
## save the metrices, params

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from get_data import read_params
import argparse
import joblib
import json


def eval_metrics(actual, pred):
    f1 = round(f1_score(actual, pred), 3)
    precision = round(precision_score(actual, pred),3)
    recall = round(recall_score(actual, pred),3)
    return f1, precision, recall


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]
    criterion = config["estimators"]["RandomForestClassifier"]["params"]["criterion"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)


    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion=criterion,
        random_state=random_state)
    model.fit(train_x, train_y)

    predicted_qualities = model.predict(test_x)

    (f1, precision, recall) = eval_metrics(test_y, predicted_qualities)

    print(f"RandomForestClassifier model (n_estimators={n_estimators}, criterion={criterion})")
    print(f"  F1_score: {f1}")
    print(f"  precision: {precision}")
    print(f"  recall: {recall}")


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)