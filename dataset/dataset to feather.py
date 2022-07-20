import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, scale
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
t1 = datetime.now()
path = os.path.join(os.path.dirname(__file__))

# train_data = pd.read_csv(path + "/train_data.csv")
# train_data.to_feather("train_data.ftr")

# train_data = pd.read_feather(path + "/train_data.ftr")

# all_features = train_data.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
# cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'] #Reference From Kaggle
# num_features = [col for col in all_features if col not in cat_features]

# train_data[train_data.select_dtypes(np.float64).columns] = train_data.select_dtypes(np.float64).astype(np.float32)

# ord_enc = OrdinalEncoder() #
# train_data[cat_features] = ord_enc.fit_transform(train_data[cat_features])

# train_num_agg = train_data.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
# train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
# train_cat_agg = train_data.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
# train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]

# train_data = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID')

# scaler = StandardScaler()
# scaled_features_train = scaler.fit_transform(train_data)
# train_data_scaled = pd.DataFrame(scaled_features_train, index = train_data.index, columns=train_data.columns)
# train_data_scaled = train_data_scaled.reset_index()
# train_data_scaled.to_feather(path + "/train_data_scaled.ftr")

# test_data = pd.read_csv(path + "/test_data.csv")
# test_data.to_feather("test_data.ftr")

#======================================================#

# test_data = pd.read_feather(path + "/test_data.ftr")

# all_features = test_data.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
# cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'] #Reference From Kaggle
# num_features = [col for col in all_features if col not in cat_features]

# test_data[test_data.select_dtypes(np.float64).columns] = test_data.select_dtypes(np.float64).astype(np.float32)

# ord_enc = OrdinalEncoder() #
# test_data[cat_features] = ord_enc.fit_transform(test_data[cat_features])

# test_num_agg = test_data.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
# test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
# test_cat_agg = test_data.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
# test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

# test_data = test_num_agg.merge(test_cat_agg, how = 'inner', on = 'customer_ID')

# scaler = StandardScaler()
# scaled_features_test = scaler.fit_transform(test_data)
# test_data_scaled = pd.DataFrame(scaled_features_test, index = test_data.index, columns=test_data.columns)
# test_data_scaled = test_data_scaled.reset_index()
# test_data_scaled.to_feather(path + "/test_data_scaled.ftr")

test_data = pd.read_feather(path + "/test_data_scaled.ftr")
test_data.set_index("customer_ID")
train_data = pd.read_feather(path + "/train_data_scaled.ftr")
train_data.set_index("customer_ID")

feature = [col for col in test_data if col not in train_data ]
print(feature)

t2 = datetime.now()
took = t2 - t1
print(f"Execution time took {took} seconds.")