import copy

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class PrefixEncoder:

    def __init__(self, prefix_dict_original, categories, num_additional_attributes=0, last_state_offset=0,
                 add_age_offset=0, add_age=False, add_temporal_information=False, encoding_technique="Aggregation"):
        """
        PrefixesEncoder one hot encodes dictionary with prefixes
        @param prefix_dict_original: dictionary with prefixes
        @param categories: list of categories to encode (in this case activities)
        @param num_additional_attributes: number of additional attributes
                                          (LastState, add_age, add_temporal_information)
        @param last_state_offset: number of attributes from last column to last state attribute
        @param add_age_offset: number of attributes from last column to age attribute
        @param add_age: boolean indicating inclusion of age attribute
        @param add_temporal_information: boolean indicating integration of month, weekday and hour
        @param encoding_technique: name of encoding technique ("Aggregation", "LastState")
        """

        self.prefix_dict_original = prefix_dict_original
        self.categories = np.array(categories)
        self.add_age = add_age
        self.add_temporal_information = add_temporal_information
        self.encoding_technique = encoding_technique
        self.num_additional_attributes = num_additional_attributes
        self.last_state_offset = last_state_offset
        self.add_age_offset = add_age_offset
        self.prefix_dict_encoded = {}
        self.fitted_cluster_model = None

    def encode(self):
        """
        encodes all dataframes in the dictionary with prefixes
        """

        # calculate position of age column
        offset = 1
        if self.add_temporal_information is True:
            offset += 3
        if self.add_age is True:
            offset += 1

        for key, value in self.prefix_dict_original.items():

            # define categories and column indices to encode
            encode_indices = [i for i in range(0, len(value.columns) - offset)]
            categories_list = [self.categories for _ in range(0, len(value.columns) - offset)]

            # define one hot encoder to transform categorical columns with
            one_hot = ('OneHot', OneHotEncoder(categories=categories_list, handle_unknown='ignore', sparse=False),
                       encode_indices)

            # configure Column transformer
            transformer = ColumnTransformer(transformers=[one_hot], remainder="passthrough")
            encoded_prefixes = transformer.fit_transform(value)

            # name last column with remaining time as target
            column_names = transformer.get_feature_names_out()
            column_names[len(column_names) - 1] = "target"

            # add numerical column age
            if self.add_age is True:
                column_names[len(column_names) - offset] = "age"

            # add numerical columns with temporal information
            if self.add_temporal_information is True:
                column_names[len(column_names) - 2] = "hour"
                column_names[len(column_names) - 3] = "weekday"
                column_names[len(column_names) - 4] = "month"

            # add encoded dataframe to dictionary with prefixes
            encoded_df = pd.DataFrame(data=encoded_prefixes, columns=column_names)
            self.prefix_dict_encoded[key] = encoded_df

    def aggregate_same_columns(self):
        """
        aggregate one hot encoded activity columns for dimensionality reduction
        @return: dictionary with aggregated dataframes
        """

        # for all buckets
        for key, value in self.prefix_dict_encoded.items():

            # copy additional and target attributes since they should not be aggregated
            aggregated_df = value[value.columns[- (1 + self.add_age_offset):]].copy()

            # iterate over one hot encoded activity columns
            for column in value.columns[:- (1 + self.add_age_offset)]:

                # extract activity name
                activity_name = column.split("_")[3]

                # add numbers if column with activity already exists
                if activity_name in aggregated_df.columns:
                    aggregated_df[activity_name] += value[column]

                # else add new activity column to aggregated dataframe
                else:
                    aggregated_df.insert(loc=0, column=activity_name, value=value[column].values)

            # drop unnecessary columns
            aggregated_df.drop("None", axis=1, inplace=True)
            aggregated_df.drop("other", axis=1, inplace=True)

            self.prefix_dict_encoded[key] = aggregated_df.copy()

    def cluster_streams(self, n_clusters):
        """
        cluster prefixes into buckets if bucketing technique == "Clustering"
        @param n_clusters: number of desired clusters
        @return: dictionary with cluster indices as keys and clustered dataframes as values
        """

        # get dataframe with prefixes
        encoded_streams = self.prefix_dict_encoded[list(self.prefix_dict_encoded.keys())[0]]

        # do KMeans clustering
        kmeans_model = KMeans(n_clusters=n_clusters)
        kmeans_model.fit(encoded_streams.values)
        self.fitted_cluster_model = kmeans_model

        # assign clusters to each row / prefix of dataframe
        labels = kmeans_model.labels_
        encoded_streams["cluster"] = labels

        # split dataframe to buckets in dictionary
        prefix_dict_clustered = {}
        for label in sorted(encoded_streams["cluster"].unique()):
            prefix_dict_clustered[label] = encoded_streams[encoded_streams["cluster"] == label]

        # delete cluster column in dataframe
        for key, value in prefix_dict_clustered.items():
            value.drop("cluster", axis=1, inplace=True)

        self.prefix_dict_encoded = copy.deepcopy(prefix_dict_clustered)
