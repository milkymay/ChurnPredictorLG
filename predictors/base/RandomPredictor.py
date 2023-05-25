import random
import numpy as np
import timeit

from sklearn.metrics import f1_score, roc_auc_score

from predictors.ChurnPredictor import business_score


def repeat_and_count_average(func, n=1000):
    average_list = []
    for i in range(n):
        average_list.append(func)

    return (sum(average_list) / len(average_list))


class RandomPredictor:
    def __init__(self):
        self.p = 0
        self.time = 0
        self.model_name = 'random'

    def fit(self, X_true, y_true):
        starttime = timeit.default_timer()
        ones = np.count_nonzero(y_true == 0)
        self.p = ones / len(y_true)
        endtime = timeit.default_timer()
        self.time = endtime - starttime

    def predict(self, x_test):
        values = [1, 0]
        prediction = random.choices(values, weights=[1 - self.p, self.p], k=len(x_test))
        return prediction

    def get_metrics(self, X_test, y_test):
        pred = self.predict(X_test)
        f1 = repeat_and_count_average(f1_score(y_test, pred))
        roc_auc = repeat_and_count_average(roc_auc_score(y_test, pred))
        business = repeat_and_count_average(business_score(y_test, pred))
        return f1, roc_auc, business, self.time
