import numpy as np
import timeit

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from predictors.ChurnPredictor import business_score


class LogisticPredictor:
    def __init__(self):
        self.model_name = 'logistic'
        self.best_model = None
        self.time = 0

    def fit(self, X_train, y_train, X_test, y_test):
        scores = []
        models = []
        starttime = timeit.default_timer()
        metrics = [f1_score, roc_auc_score, business_score]
        for c in np.logspace(-4, 0, 40):
            logistic = LogisticRegression(max_iter=10000, solver='liblinear', penalty='l2', C=c)
            pipe = Pipeline(steps=[("scaler", StandardScaler()),
                                   ("logistic", logistic)])
            pipe.fit(X_train, y_train)
            scores.append(tuple([m(y_test, pipe.predict(X_test)) for m in metrics]))
            models.append([pipe])
        endtime = timeit.default_timer()
        self.time = endtime - starttime
        best_model_ind = scores.index(max(scores, key=lambda x: x[2]))
        self.best_model = models[best_model_ind][0]

    def predict(self, x_test):
        return self.best_model.predict(x_test)

    def get_metrics(self, X_test, y_test):
        pred = self.best_model.predict(X_test)
        f1 = f1_score(y_test, pred)
        roc_auc = roc_auc_score(y_test, pred)
        business = business_score(y_test, pred)
        return f1, roc_auc, business, self.time
