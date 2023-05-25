import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import timeit
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler


def business_score(y_test, y_predict, discount=0.05):
    return np.dot(confusion_matrix(y_test, y_predict).ravel(), np.array([0, -discount, -1, 1 - discount]))


class ChurnPredictor:
    def __init__(self, model_name, hyperparams=None, random_state=42):
        self.model_name = model_name
        self.hyperparams = hyperparams
        self.random_state = random_state
        self.model = None
        self.results = {}
        self.time = 0

    def fit(self, X_train, y_train):
        if self.model_name == 'forest':
            self.model = RandomForestClassifier(random_state=self.random_state)
        elif self.model_name == 'catboost':
            self.model = CatBoostClassifier(random_state=self.random_state)
        elif self.model_name == 'lightgbm':
            self.model = lgb.LGBMClassifier(random_state=self.random_state)
        elif self.model_name == 'nn':
            self.model = KerasClassifier(build_fn=self.build_nn_model, input_shape=X_train.shape[1:], verbose=0)

        starttime = timeit.default_timer()
        param_grid = self.hyperparams
        self.pipe = Pipeline(steps=[('scaler', StandardScaler()), ('model', self.model)])
        self.grid_search = RandomizedSearchCV(estimator=self.pipe, param_distributions=param_grid, cv=4,
                                              n_iter=10, n_jobs=-1, scoring=make_scorer(business_score))
        # self.grid_search = GridSearchCV(estimator=self.pipe, param_grid=param_grid, cv=4,
        #                                 n_jobs=-1, scoring=make_scorer(business_score))
        self.grid_search.fit(X_train, y_train)
        self.model = self.grid_search.best_estimator_
        self.results['best_params'] = self.grid_search.best_params_
        if self.model_name == 'forest':
            self.results['feature_importances'] = self.model['model'].feature_importances_
        elif self.model_name == 'catboost':
            self.results['feature_importances'] = self.model['model'].get_feature_importance()
        elif self.model_name == 'lightgbm':
            self.results['feature_importances'] = self.model['model'].feature_importances_
        else:
            self.results['feature_importances'] = []
        endtime = timeit.default_timer()
        self.time = endtime - starttime

    def predict(self, X_test):
        if self.model_name == 'nn':
            preds = self.model.predict(X_test)
            preds = (preds > 0.5).astype(int).flatten()
        else:
            preds = self.model.predict(X_test)
        self.results['predictions'] = preds
        return preds

    def get_metrics(self, X_test, y_test):
        pred = self.predict(X_test)
        f1 = f1_score(y_test, pred)
        roc_auc = roc_auc_score(y_test, pred)
        business = business_score(y_test, pred)
        return f1, roc_auc, business, self.time

    def build_nn_model(self, input_shape):
        model = Sequential()
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        return model
