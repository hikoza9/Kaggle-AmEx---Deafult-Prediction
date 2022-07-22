from gc import callbacks
import os
import numpy as np
import pandas as pd
import pandas_profiling as pp
from pyarrow import csv, parquet
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler, scale
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier, log_evaluation
import optuna

from datetime import datetime
t1 = datetime.now()
path = os.path.join(os.path.dirname(__file__))

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

sample_submission = pd.read_csv(path + "/dataset/sample_submission.csv")
print("Load Train Labels")
train_labels = pd.read_csv(path + "/dataset/train_labels.csv")

print("Load Data Train")
train_data = pd.read_feather(path + "/dataset/train_data_scaled.ftr")
train_data = train_data.set_index("customer_ID")

train_labels = train_labels.set_index("customer_ID")

print("Split Data Train")
x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size = 0.3, random_state = 4222)
del train_data

y_pred_amex = pd.DataFrame(y_test.copy(deep=True))
y_pred_amex = y_pred_amex.rename(columns={'target':'prediction'})


def objective(trial):
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    # dtrain = lgb.Dataset(x_train, label=y_train)

    param = {
        'objective': 'binary',
        'metric': 'binary_losslog',
        "verbosity": -1,
        'boosting_type' : 'dart',
        'force_row_wise' : True,
        'n_estimators' : 100,
        'random_state' : 4222,
        'learning_rate': 0.18,
        'lambda_l1': trial.suggest_float("lambda_l1", 1e-8, 20.0, log=True),
        'lambda_l2': trial.suggest_float("lambda_l2", 1e-8, 20.0, log=True),
        'num_leaves': 100,
        'min_data_in_leaf' : 40,
        'feature_fraction': trial.suggest_float("feature_fraction", 0.4, 1.0),
        'bagging_fraction': trial.suggest_float("bagging_fraction", 0.4, 1.0),
        'bagging_freq': 10,
    }

    # gbm = lgb.train(param, dtrain)
    model = LGBMClassifier(**param)
    model.fit(x_train, y_train.values.ravel())
    # preds = gbm.predict(x_test)
    preds = model.predict(x_test)
    # pred_labels = np.rint(preds)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)

    # y_pred_amex = pd.DataFrame(y_test.copy(deep=True))
    # y_pred_amex = y_pred_amex.rename(columns={'target':'prediction'})
    y_pred_amex["prediction"] = model.predict_proba(x_test)[:,1]
    metric = amex_metric(y_test, y_pred_amex)
    print("Amex Metric : ", metric)
    print("Accuracy : ", accuracy*100)
    return metric

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)

# print("Number of finished trials: {}".format(len(study.trials)))

# print("Best trial:")
# trial = study.best_trial

# print("  Value: {}".format(trial.value))

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value)) 

fixed_params={
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type' : 'dart',
    'force_row_wise' : True,
    'n_jobs' : -1,
    'random_state' : 4222,
    'n_estimators': 400,
}

params0 = {
    'learning_rate' : 0.18,
    'lambda_l1': 3.2226185594082373,
    'lambda_l2': 6.452975460684419e-08,
    'num_leaves': 100,
    'min_data_in_leaf': 40,
    'feature_fraction': 0.6017653745685946,
    'bagging_fraction': 0.9632635572203325,
    'bagging_freq': 10,
    } #Amex : 0.786834841387744 n_estimator = 400 learning_rate : 0.18


#LOAD TEST DATA
# test_data = pd.read_feather(path + "/dataset/test_data_scaled.ftr")
# test_data = test_data.set_index("customer_ID")

print("Model Trial")
#MODEL TRIAL
# model = LogisticRegression(max_iter=470)
# model = MLPClassifier(hidden_layer_sizes=(200,150,100), activation='relu', solver='adam', random_state=422)
model = LGBMClassifier(**fixed_params, **params0)
model.fit(x_train, y_train.values.ravel())
del x_train, y_train


print("Amex Metric")
start = datetime.now()
# AMEX METRIC
y_pred_amex["prediction"] = model.predict_proba(x_test)[:,1]
print(y_pred_amex)
# print("Parameter LGBM : ", params2)
print("Amex Metric : ", amex_metric(y_test, y_pred_amex))
end = datetime.now()
print(f"Execution time took {end-start} seconds.")

# y_pred_amex = pd.DataFrame(sample_submission.copy(deep=True))
# y_pred_amex["prediction"] = model.predict_proba(test_data)[:,1]
# del test_data
# # print(y_pred_amex)
# # print("Parameter LGBM : ", params2)
# # print("Amex Metric : ", amex_metric(y_test, y_pred_amex))
# y_pred_amex.to_csv(path + "/dataset/sample submission competition.csv")

# #PREDICT TEST DATA

y_pred = model.predict(x_test)
model_conf_matrix = confusion_matrix(y_test, y_pred)
model_acc_score = accuracy_score(y_test, y_pred)
print("confussion matrix")
print(model_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Logistic Regression:",model_acc_score*100,'\n')
print("-------------------------------------------")
print(classification_report(y_test,y_pred))

# #PREDICT TRAIN DATA
# y_pred = model.predict(x_train)
# model_conf_matrix = confusion_matrix(y_train, y_pred)
# model_acc_score = accuracy_score(y_train, y_pred)
# print("confussion matrix")
# print(model_conf_matrix)
# print("-------------------------------------------")
# print("Accuracy of Logistic Regression:",model_acc_score*100,'\n')
# print("-------------------------------------------")
# print(classification_report(y_train,y_pred))

# print(train_data.select_dtypes(include="float64"))

#B_31 int64
#customer_ID, S_2, D_63, D_64 object
# print(train_data.select_dtypes(include=['object']))
# print(train_data.dtypes)

t2 = datetime.now()
took = t2 - t1
print(f"Execution time took {took} seconds.")