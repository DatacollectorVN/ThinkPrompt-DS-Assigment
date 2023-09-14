from xgboost import XGBClassifier
XGBClassifier_ = {
    "xgbclassifier__learning_rate": [0.1],
    "xgbclassifier__max_depth": [10],
    "xgbclassifier__n_estimators": [30],
    "xgbclassifier__objective:": ['multi:softmax'],
    "xgbclassifier__num_classes": [3],
    "xgbclassifier__verbose": [True],
    "xgbclassifier__n_jobs": [4]
}   

from sklearn.ensemble import HistGradientBoostingClassifier
HistGradientBoostingClassifier_ = {
    'histgradientboostingclassifier__learning_rate': [0.1, 0.5]
    , 'histgradientboostingclassifier__max_iter': [100, 150, 200]
}

# from sklearn.neural_network import MLPClassifier
# MLPClassifier_ = {
#     "mlpclassifier__alpha": [0.1, 0.001],
#     "mlpclassifier__hidden_layer_sizes": [(32), (64), (64, 32)],
#     "mlpclassifier__learning_rate": ['adaptive']
# }