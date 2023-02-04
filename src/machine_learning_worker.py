from src.dataloader import DataLoader
from src.machine_learning_predictor import MachineLearningPredictor
from src.prefix_encoder import PrefixEncoder
from src.prefix_extractor import PrefixExtractor


class MachineLearningWorker:

    def __init__(self, config=None):
        """
        MachineLearningWorker handles execution of machine learning based approach for remaining time prediction
        @param config:
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

        if config is not None:
            self.dataset_path = config["dataset_path"]
            self.temporal_split_sort_by = config["temporal_split_sort_by"]
            self.add_age = config["add_age"]
            self.add_temporal_information = config["add_temporal_information"]
            self.bucketing_technique = config["bucketing_technique"]
            self.bucketing_lower_bound = config["bucketing_lower_bound"]
            self.bucketing_upper_bound = config["bucketing_upper_bound"]
            self.kmeans_n_clusters = config["kmeans_n_clusters"]
            self.encoding_technique = config["encoding_technique"]
            self.prediction_data = config["prediction_data"]
            self.ml_algorithm = config["ml_algorithm"]
            self.svr_c = config["svr_c"]
            self.svr_epsilon = config["svr_epsilon"]
            self.knn_n_neighbors = config["knn_n_neighbors"]
            self.knn_leaf_size = config["knn_leaf_size"]
            self.xgb_eta = config["xgb_eta"]
            self.xgb_max_depth = config["xgb_max_depth"]
            self.num_additional_attributes = 0
            self.last_state_offset = 1
            self.add_age_offset = 1

            if self.encoding_technique == "LastState":
                self.num_additional_attributes += 1
            if self.add_age is True:
                self.last_state_offset += 1
                self.num_additional_attributes += 1
            if self.add_temporal_information is True:
                self.add_age_offset += 3
                self.last_state_offset += 3
                self.num_additional_attributes += 3

    def test_settings(self):
        """
        test if chosen settings are possible
        """

        if self.bucketing_technique not in ["SingleBucket", "PrefixLength", "Clustering"]:
            raise ValueError("Defined bucketing technique is not available (", self.bucketing_technique, ")!")

        if self.ml_algorithm not in ["SVR", "KNN", "XGB"]:
            raise ValueError("Defined machine learning algorithm is not available!")

        if self.bucketing_upper_bound < self.bucketing_lower_bound:
            raise ValueError("Upper bound of bucket length must greater or equal lower bound of bucket length!")

        if (
                self.bucketing_technique == "SingleBucket" or self.bucketing_technique == "Clustering") and self.bucketing_lower_bound != self.bucketing_upper_bound:
            raise ValueError("Buckets must have same length for selected bucketing technique!")

    def execute_workflow(self, config):
        """
        execution of machine learning based approach for remaining time prediction
        @param config: see documentation at __init__
        @return: dictionary with r2 and mae score of prediction
        """

        self.__init__(config=config)
        self.test_settings()

        # Load data
        dataloader = DataLoader(dataset_path=self.dataset_path, temporal_split_sort_by=self.temporal_split_sort_by,
                                add_age=self.add_age, add_temporal_information=self.add_temporal_information,
                                encoding_technique=self.encoding_technique)
        dataloader.calculate_remaining_time()
        dataloader.get_last_state()
        dataloader.split()
        if self.encoding_technique == "LastState":
            dataloader.add_last_state()
        scaler = dataloader.scale_numeric_attributes()
        dataloader.log_to_stream()
        activity_list = dataloader.get_activity_list()

        # Extract prefixes
        prefix_extractor = PrefixExtractor(bucketing_lower_bound=self.bucketing_lower_bound,
                                           bucketing_upper_bound=self.bucketing_upper_bound, add_age=self.add_age,
                                           add_temporal_information=self.add_temporal_information,
                                           encoding_technique=self.encoding_technique,
                                           num_additional_attributes=self.num_additional_attributes,
                                           last_state_offset=self.last_state_offset, add_age_offset=self.add_age_offset)
        prefix_extractor.extract_prefixes(dataloader.stream_train, dataloader.stream_val, dataloader.stream_test)
        prefix_extractor.transform_prefix_to_df()

        # Encode prefixes of training data
        prefix_encoder_train = PrefixEncoder(prefix_dict_original=prefix_extractor.prefixes_train,
                                             categories=activity_list, add_age=self.add_age,
                                             add_temporal_information=self.add_temporal_information,
                                             encoding_technique=self.encoding_technique,
                                             num_additional_attributes=self.num_additional_attributes,
                                             last_state_offset=self.last_state_offset,
                                             add_age_offset=self.add_age_offset)
        prefix_encoder_train.encode()
        prefix_encoder_train.aggregate_same_columns()
        if self.bucketing_technique == "Clustering":
            prefix_encoder_train.cluster_streams(n_clusters=self.kmeans_n_clusters)

        # Train ML model
        machine_learning_predictor = MachineLearningPredictor(ml_algorithm=self.ml_algorithm, svr_c=self.svr_c,
                                                              svr_epsilon=self.svr_epsilon,
                                                              knn_n_neighbors=self.knn_n_neighbors,
                                                              knn_leaf_size=self.knn_leaf_size, xgb_eta=self.xgb_eta,
                                                              xgb_max_depth=self.xgb_max_depth)
        machine_learning_predictor.fit(prefix_encoder_train.prefix_dict_encoded)

        # Get encoded running traces
        if self.prediction_data == "val":
            prefix_encoder_pred = PrefixEncoder(prefix_dict_original=prefix_extractor.prefixes_val,
                                                categories=activity_list, add_age=self.add_age,
                                                add_temporal_information=self.add_temporal_information,
                                                encoding_technique=self.encoding_technique,
                                                num_additional_attributes=self.num_additional_attributes,
                                                last_state_offset=self.last_state_offset,
                                                add_age_offset=self.add_age_offset)
        else:
            prefix_encoder_pred = PrefixEncoder(prefix_dict_original=prefix_extractor.prefixes_test,
                                                categories=activity_list, add_age=self.add_age,
                                                add_temporal_information=self.add_temporal_information,
                                                encoding_technique=self.encoding_technique,
                                                num_additional_attributes=self.num_additional_attributes,
                                                last_state_offset=self.last_state_offset,
                                                add_age_offset=self.add_age_offset)
        prefix_encoder_pred.encode()
        prefix_encoder_pred.aggregate_same_columns()

        # Do prediction
        y_true, y_pred = machine_learning_predictor.predict(prefix_encoder_pred.prefix_dict_encoded,
                                                            prefix_encoder_train.fitted_cluster_model)

        # Do scoring
        score = machine_learning_predictor.score(scaler, y_true, y_pred)

        return score
