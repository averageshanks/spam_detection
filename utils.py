import json


def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


accuracy = {
    "decision_tree": 0.9659,
    "random_forest": 0.9749,
    "naive_bays": 0.8879,
    "svm":0.9821,
    "mlp":0.9856,
    "logistic_regression":0.9659,
    "bagging_classifier":0.96502
}

precision = {
    "decision_tree": 0.96,
    "random_forest": 1.0,
    "naive_bays": 0.56,
    "svm":0.99,
    "mlp":1.0,
    "logistic_regression":1.0,
    "bagging_classifier": 0.95
}

reacall = {
    "decision_tree": 0.7870,
    "random_forest": 0.8193,
    "naive_bays": 0.9032,
    "svm":0.8774,
    "mlp":0.8968,
    "logistic_regression":0.75489,
    "bagging_classifier":0.7870
}
