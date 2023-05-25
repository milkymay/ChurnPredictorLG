from sklearn.metrics import f1_score, roc_auc_score

from predictors.ChurnPredictor import business_score


def predict_zero(x_test):
    prediction = [0 for _ in range(len(x_test))]
    return prediction


class NoPredictor:
    def __init__(self):
        self.time = 0
        self.model_name = 'no'

    def get_metrics(self, X_test, y_test):
        pred = predict_zero(X_test)
        f1 = f1_score(y_test, pred)
        roc_auc = roc_auc_score(y_test, pred)
        business = business_score(y_test, pred)
        return f1, roc_auc, business, self.time

