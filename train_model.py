import pandas as pd
import random

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class TrainModel():

    def __init__(self):

        self.model = None

    @staticmethod
    def __stratify_split(X, y, test_size=0.2, random_state=15):

        p_index = list(y[y==1].index)
        n_index = list(y[y==0].index)

        test_p_cnt = int(len(p_index)*test_size)
        test_n_cnt = int(len(n_index)*test_size)

        random.seed(random_state)
        p_index_test = random.sample(p_index, test_p_cnt)
        p_index_train = list(set(p_index) - set(p_index_test))

        n_index_test = random.sample(n_index, test_n_cnt)
        n_index_train = list(set(n_index) - set(n_index_test))

        test_index = p_index_test + n_index_test
        train_index = p_index_train + n_index_train

        return X.loc[train_index,], X.loc[test_index,], y[train_index], y[test_index]

    def prepare_train(self, feature, label):

        X_train, X_test, y_train, y_test  \
        =  self.__class__.__stratify_split(feature, label, test_size = 0.2, random_state=28)

        X_train, X_valid, y_train, y_valid  \
        = self.__class__.__stratify_split(X_train, y_train, test_size = 0.25, random_state=42)

        print(f"train size:{y_train.shape[0]}, positive label count: {y_train.value_counts()[1]}")
        print(f"validation size:{y_valid.shape[0]}, positive label count: {y_valid.value_counts()[1]}")
        print(f"test size:{y_test.shape[0]}, positive label count:{y_test.value_counts()[1]}")

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def train_model(self, model_type = "lr", weight=None, train_data=None, valid_data=None):

        if model_type == "lr":
            self.model = LogisticRegression(random_state=45, class_weight = weight)
            self.model.fit(train_data[0], train_data[1])

        elif model_type == "rf":
            self.model = RandomForestClassifier(n_estimators = 50, max_depth = 4, random_state = 38,  class_weight = weight)
            self.model.fit(train_data[0], train_data[1])

        elif model_type == "xgb":
            self.model = xgb.XGBClassifier(objective='binary:logistic', learning_rate =0.001, n_estimators=1000, max_depth=5, 
                subsample=0.75, colsample_bytree=0.8, scale_pos_weight= weight[1], seed=27, n_jobs=-1)
            self.model.fit(train_data[0], train_data[1], early_stopping_rounds=10, eval_metric="auc", eval_set=[valid_data])


        # validate
        y_pred_train = self.model.predict(train_data[0])       
        y_pred_valid = self.model.predict(valid_data[0])

        return y_pred_train, y_pred_valid     

    def predict(self, test_data = None, transform=False):
        """
        transform: boolean, if True, data processing and feature engineering is required
        """

        if transform:

            test_data = self.preprocess(test_data)
           
        y_pred = self.model.predict(test_data)

        return y_pred


    def preprocess(self, data=None):

        skewed_cols = ['attribute2']

        drop_cols = ['attribute8']

        cat_cols = ['attribute3', 'attribute4', 'attribute5', 'attribute7', 'attribute9', 'device_type']

        # drop redundant cols
        data.drop(drop_cols, axis =1, inplace = True)

        # transform skewed data
        for col in skewed_cols:
            data[col] = (data[col])**0.5

        # # transform device feature
        # device_fail = pd.read_csv("device_fail_prob.csv")
        # data = pd.merge(data, device_fail, how='left',on='device')
        # data["device_fail_prob"].fillna(0, inplace=True)

        # transform data feature
        data["date"] = pd.to_datetime(data["date"])
        data["weekday"] = data["date"].dt.weekday
        data["day"] = data["date"].dt.day
        data["month"] = data["date"].dt.month
        data["quarter"] = data["date"].dt.quarter

        # device type
        data["device_type"] = data.device.apply(lambda x: x[:4])

        # one hot encoding
        data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('category'))        
        data = pd.get_dummies(data, prefix=cat_cols, columns=cat_cols)

        # build features
        feature = data[data.columns.difference(["date", "device","failure"])]

        return feature
    