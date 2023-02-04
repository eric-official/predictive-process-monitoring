import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


class MachineLearningPredictor:

    def __init__(self, ml_algorithm, svr_c=1.0, svr_epsilon=0.1, knn_n_neighbors=5, knn_leaf_size=30, xgb_eta=0.3,
                 xgb_max_depth=6):
        """
        MachineLearningPredictor handles fitting, predicting and scoring of machine learning models for remaining time
        prediction
        @param ml_algorithm: machine learning algorithm for prediction (SVR, KNN, XGBoost)
        @param svr_c: SVR regularization parameter
        @param svr_epsilon: specifies the range within which no penalty is associated in the training loss
                            function with points predicted within a distance epsilon from the actual value
        @param knn_n_neighbors: Number of neighbors
        @param knn_leaf_size: controls minimum number of points in a given node
        @param xgb_eta: step size shrinkage used in update to prevent overfitting
        @param xgb_max_depth: Maximum depth of a tree
        """

        self.ml_algorithm = ml_algorithm
        self.fitted_models_dict = {}
        self.svr_c = svr_c
        self.svr_epsilon = svr_epsilon
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_leaf_size = knn_leaf_size
        self.xgb_eta = xgb_eta
        self.xgb_max_depth = xgb_max_depth

    def fit(self, prefix_dict_encoded):
        """
        fit machine learning model to data
        @param prefix_dict_encoded: dictionary with encoded prefixes
        """

        # fit machine learnings model to each bucket
        for key, value in prefix_dict_encoded.items():

            if self.ml_algorithm == "SVR":
                model = SVR(C=self.svr_c, epsilon=self.svr_epsilon)

            elif self.ml_algorithm == "KNN":
                model = KNeighborsRegressor(n_neighbors=self.knn_n_neighbors, leaf_size=self.knn_leaf_size)

            else:
                model = xgb.XGBRegressor(objective="reg:squarederror", eta=self.xgb_eta, max_depth=self.xgb_max_depth)

            x_train = value.iloc[:, : -1].values
            y_train = value.iloc[:, -1].values
            model.fit(x_train, y_train)

            self.fitted_models_dict[key] = model

    def predict(self, prefix_dict_encoded, fitted_cluster_model):
        """
        predict remaining time for prefixes in dictionary with model of corresponding bucket
        @param prefix_dict_encoded: dictionary with encoded prefixes for prediction
        @param fitted_cluster_model: dictionary with fitted machine learning models for each bucket
        @return: lists with ground truth and prediction for target attribute
        """

        y_true = []
        y_pred = []

        for key, value in prefix_dict_encoded.items():
            for index, row in value.iterrows():
                if fitted_cluster_model is not None:
                    cluster = fitted_cluster_model.predict(row.to_numpy().reshape(1, -1))[0]
                else:
                    cluster = key

                y_true.append(row[0])
                trace_prediction = self.fitted_models_dict[cluster].predict(row[1:].to_numpy().reshape(1, -1))
                y_pred.append(trace_prediction)

        return y_true, y_pred

    def score(self, scaler, y_true, y_pred):
        """
        measure accuracy score of prediction
        @param scaler: standard scaler for inverse transformation
        @param y_true: ground truth target variables
        @param y_pred: predicted target variables
        @return: mae score
        """

        rmse_score = mean_squared_error(y_true, y_pred, squared=True)

        y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, 1))
        y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))

        mae_score = mean_absolute_error(y_true, y_pred)
        return {"rmse_score": rmse_score, "mae_score": mae_score}
