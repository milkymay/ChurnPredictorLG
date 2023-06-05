import json

from data_prepare.PrepareData import PrepareData
from logic.PredictManager import PredictManager
import numpy as np
from utils.Data import Data
import pandas as pd
from utils.PrettyLogger import PrettyLogger

conf_path = ""

data_json = json.load(open(conf_path, 'r'))

dataPrepare = PrepareData(conf_path=conf_path)
dataPrepare.prepare_data()

data_path = data_json["data_store_path"] + "/kids_lessons_monthly.csv"
log_path = data_json["log_path"]
res_path = data_json["res_path"]
actual_data_path = data_json["actual_data_path"]


def create_manager(clfs_names, text_analysis):
    return PredictManager(data_path, log_path, clfs_names=clfs_names,
                          text_analysis=text_analysis)


def store_best(manager, results):
    best_clf = max(manager.metrics, key=lambda x: manager.metrics[x][2])
    metric = manager.metrics[best_clf][2]
    model = manager.clfs[best_clf]
    results[tuple(manager.text_analysis)] = (model, metric)


results = {}

managers = [
    create_manager(['lightgbm', 'catboost', 'forest', 'nn'], []),
    create_manager(['lightgbm', 'catboost', 'forest', 'nn'], ['dostoevsky']),
    # create_manager(['lightgbm', 'catboost', 'forest', 'nn'], ['chatGPT']),
    # create_manager(['lightgbm', 'catboost', 'forest', 'nn'], ['chatGPT', 'dostoevsky'])
]

for manager in managers:
    manager.baseline()
    manager.evaluate()
    store_best(manager, results)

best_clf = max(results, key=lambda x: results[x][1])
text_analysis = []
if len(best_clf) != 0:
    text_analysis = np.array(best_clf)
data_actual = Data(actual_data_path, text_analysis=np.array(text_analysis)).X_true
best_clf = results[best_clf][0]

data_full = pd.read_csv(actual_data_path)

prediction = best_clf.predict(data_actual)
ids = pd.DataFrame(prediction, data_full['student_id'])
ids.columns = ['pred']
expected_churn_ids = list(np.unique(ids[ids['pred'] == 1].index))

imps = best_clf.results['feature_importances']
if len(imps) == len(data_actual.columns):
    exp = pd.DataFrame(imps, data_actual.columns)
    exp.columns = ['importance']
    imps = list(exp.sort_values(by='importance', ascending=False).index)

logger = PrettyLogger(res_path)
logger.print_out(expected_churn_ids)
logger.print_out(imps)
