from ray import tune


"""
{
    "dataset_path": absolute path to semicolon seperated CSV with process log
    "temporal_split_sort_by": name of temporal attribute for sorting dataset before splitting
    "add_age": boolean indicating inclusion of age attribute
    "add_temporal_information": boolean indicating integration of month, weekday and hour
    "bucketing_technique": method how to bucket prefixes (SingleBucket, PrefixLength, Clustering)
    "bucketing_lower_bound": minimum length for prefix length (must be equal bucketing_upper_bound for
                              bucketing techniques Clustering and SingleBucket)
    "bucketing_upper_bound": maximum length for prefix length (must be equal bucketing_lower_bound for
                              bucketing techniques Clustering and SingleBucket)
    "kmeans_n_clusters": number of desired clusters
    "encoding_technique": name of encoding technique ("Aggregation", "LastState")
    "prediction_data": data to use for prediction (val, test)
    "ml_algorithm": machine learning algorithm for prediction (SVR, KNN, XGBoost)
    "svr_c": SVR regularization parameter
    "svr_epsilon": specifies the range within which no penalty is associated in the training loss
                    function with points predicted within a distance epsilon from the actual value
    "knn_n_neighbors": Number of neighbors
    "knn_leaf_size": controls minimum number of points in a given node
    "xgb_eta": step size shrinkage used in update to prevent overfitting
    "xgb_max_depth": Maximum depth of a tree
}
"""

manual_ml_config = {
    "dataset_path": "/Users/ericnaser/Dropbox/Wirtschaftsinformatik_Master/2. Semester/Entwicklungspraktikum - Data Science/predictive-process-monitoring/data/sepsis_cases_1.csv",
    "temporal_split_sort_by": "timesincecasestart",
    "add_age": False,
    "add_temporal_information": False,
    "bucketing_technique": "SingleBucket",
    "bucketing_lower_bound": 14,
    "bucketing_upper_bound": 14,
    "kmeans_n_clusters": 10,
    "encoding_technique": "Aggregation",
    "prediction_data": "val",
    "ml_algorithm": "SVR",
    "svr_c": 1.2,
    "svr_epsilon": 0.4,
    "knn_n_neighbors": 5,
    "knn_leaf_size": 30,
    "xgb_eta": 0.3,
    "xgb_max_depth": 6
}

hyperparam_ml_config = {
    "dataset_path": "/Users/ericnaser/Dropbox/Wirtschaftsinformatik_Master/2. Semester/Entwicklungspraktikum - Data Science/predictive-process-monitoring/data/sepsis_cases_1.csv",
    "temporal_split_sort_by": tune.choice(["timesincecasestart", "time:timestamp"]),
    "add_age": tune.choice([True, False]),
    "add_temporal_information": tune.choice([True, False]),
    "bucketing_technique": tune.grid_search(["SingleBucket", "PrefixLength", "Clustering"]),
    "bucketing_lower_bound": tune.randint(2, 20),
    "bucketing_upper_bound": tune.sample_from(lambda spec: tune.randint(spec.config.bucketing_lower_bound,
                                                                        20) if spec.config.bucketing_technique == "PrefixLength" else spec.config.bucketing_lower_bound),
    "kmeans_n_clusters": tune.sample_from(lambda spec: tune.randint(
        5, 15) if spec.config.bucketing_technique == "Clustering" else None),
    "encoding_technique": tune.grid_search(["Aggregation", "LastState"]),
    "prediction_data": "val",
    "ml_algorithm": tune.grid_search(["SVR", "KNN", "XGB"]),
    "svr_c": tune.sample_from(lambda spec: tune.quniform(0.5, 1.5, 0.1) if spec.config.ml_algorithm == "SVR" else None),
    "svr_epsilon": tune.sample_from(lambda spec:tune.quniform(0.1, 1.0, 0.1) if spec.config.ml_algorithm == "SVR" else None),
    "knn_n_neighbors": tune.sample_from(lambda spec:tune.randint(5, 15) if spec.config.ml_algorithm == "KNN" else None),
    "knn_leaf_size": tune.sample_from(lambda spec:tune.randint(25, 45) if spec.config.ml_algorithm == "KNN" else None),
    "xgb_eta": tune.sample_from(lambda spec:tune.quniform(0.1, 1.0, 0.1) if spec.config.ml_algorithm == "XGB" else None),
    "xgb_max_depth": tune.sample_from(lambda spec:tune.randint(5, 15) if spec.config.ml_algorithm == "XGB" else None)
}
