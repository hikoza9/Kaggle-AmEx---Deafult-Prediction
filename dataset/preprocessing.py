import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime

t1 = datetime.now()

path = os.path.join(os.path.dirname(__file__))

train_data = pd.read_csv(path + "/train_data.csv")
# test_data = pd.read_csv(path + "/test_data.csv")

def feature_engineering(data):
    data[data.select_dtypes(np.float64).columns] = data.select_dtypes(np.float64).astype(np.float32)

    all_features = data.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
    cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'] #Reference From Kaggle
    num_features = [col for col in all_features if col not in cat_features]

    ord_enc = OrdinalEncoder() #                                                           
    data[cat_features] = ord_enc.fit_transform(data[cat_features])

    data_num_agg = train_data.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    data_num_agg.columns = ['_'.join(x) for x in data_num_agg.columns]
    data_cat_agg = train_data.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    data_cat_agg.columns = ['_'.join(x) for x in data_cat_agg.columns]

    data = data_num_agg.merge(data_cat_agg, how = 'inner', on = 'customer_ID')

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(scaled_features, index = data.index, columns=data.columns)
    data_scaled = data_scaled.reset_index()
    return data_scaled

train_data = feature_engineering(train_data)
train_data.to_feather(path + "/train_data_scaled.ftr")
# test_data = feature_engineering(test_data)
# test_data.to_feather(path + "/test_data_scaled.ftr")

t2 = datetime.now()
took = t2 - t1
print(f"Execution time took {took} seconds.")