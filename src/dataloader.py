import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class DataLoader:

    def __init__(self, dataset_path, temporal_split_sort_by="timesincecasestart", add_age=False,
                 add_temporal_information=False, encoding_technique="Aggregation", val_cut=0.7, test_cut=0.9):
        """
        class responsible for importing and preparing data
        @param dataset_path: absolute path to semicolon seperated CSV with process log
        @param temporal_split_sort_by: name of temporal attribute for sorting dataset before splitting
        @param add_age: boolean indicating inclusion of age attribute
        @param add_temporal_information: boolean indicating integration of month, weekday and hour
        @param encoding_technique: name of encoding technique ("Aggregation", "LastState")
        @param val_cut: percentage at which validation data begins
        @param test_cut: percentage at which test data begins
        """

        self.dataset_path = dataset_path
        self.temporal_split_sort_by = temporal_split_sort_by
        self.add_age = add_age
        self.add_temporal_information = add_temporal_information
        self.encoding_technique = encoding_technique
        self.val_cut = val_cut
        self.test_cut = test_cut
        self.dataset = self.load_dataset()
        self.log_train = None
        self.log_val = None
        self.log_test = None
        self.stream_train = None
        self.stream_val = None
        self.stream_test = None
        self.last_state_by_case = None
        self.max_case_events = 0

    def load_dataset(self):
        """
        read CSV
        @return: dataframe with process log
        """

        dataset = pd.read_csv(self.dataset_path, sep=";")
        return dataset

    def calculate_remaining_time(self):
        """
        add column with calculated remaining time for each case
        """

        self.dataset["case_duration"] = self.dataset.groupby(["Case ID"])["timesincecasestart"].transform("max")
        self.dataset["remaining_time"] = self.dataset["case_duration"] - self.dataset["timesincecasestart"]

    def get_last_state(self):
        """
        get last state of case for LastState encoding
        """

        self.last_state_by_case = self.dataset.sort_values(by=["Case ID", "timesincecasestart"]).groupby(["Case ID"])["Activity"]

    def split(self):
        """
        split dataset into train, validation and test data
        """

        val_index = int(self.val_cut * len(self.dataset))
        test_index = int(self.test_cut * len(self.dataset))

        self.log_train, self.log_val, self.log_test = np.array_split(
            self.dataset.sort_values(by=[self.temporal_split_sort_by]), [val_index, test_index])

    def add_last_state(self):
        """
        add column with last state of case for LastState encoding
        """

        self.log_train["last_state"] = self.last_state_by_case.transform("last")
        self.log_val["last_state"] = self.last_state_by_case.transform("last")
        self.log_test["last_state"] = self.last_state_by_case.transform("last")

    def scale_numeric_attributes(self):
        """
        scale numeric attributes in dataset
        @return: return scaler for remaining time for inverse transformation
        """

        # scale age, month, weekday and hour
        for column in ["Age", "month", "weekday", "hour"]:
            additional_information_scaler = StandardScaler()
            additional_information_scaler.fit(self.log_train[column].values.reshape(-1, 1))
            self.log_train[column] = additional_information_scaler.transform(
                self.log_train[column].values.reshape(-1, 1))
            self.log_val[column] = additional_information_scaler.transform(self.log_val[column].values.reshape(-1, 1))
            self.log_test[column] = additional_information_scaler.transform(self.log_test[column].values.reshape(-1, 1))

        # scale remaining time
        remaining_time_scaler = StandardScaler()
        remaining_time_scaler.fit(self.log_train["remaining_time"].values.reshape(-1, 1))
        self.log_train["remaining_time"] = remaining_time_scaler.transform(
            self.log_train["remaining_time"].values.reshape(-1, 1))
        self.log_val["remaining_time"] = remaining_time_scaler.transform(
            self.log_val["remaining_time"].values.reshape(-1, 1))
        self.log_test["remaining_time"] = remaining_time_scaler.transform(
            self.log_test["remaining_time"].values.reshape(-1, 1))
        return remaining_time_scaler

    def log_to_stream(self):

        current_case = None
        case_activities = []
        streams = []

        for log in [self.log_train, self.log_val, self.log_test]:

            stream_of_split = []

            for index, row in log.sort_values(by=["Case ID", "timesincecasestart"]).iterrows():

                if len(case_activities) > self.max_case_events:
                    self.max_case_events = len(case_activities)

                if current_case != row["Case ID"]:
                    current_case = row["Case ID"]
                    case_activities = []

                case_activities.append(row["Activity"])

                append_stream = case_activities.copy()

                if self.encoding_technique == "LastState":
                    append_stream.append(row["last_state"])
                if self.add_age is True:
                    append_stream.append(row["Age"])

                if self.add_temporal_information is True:
                    append_stream.append(row["month"])
                    append_stream.append(row["weekday"])
                    append_stream.append(row["hour"])

                append_stream.append(row["remaining_time"])
                stream_of_split.append(append_stream)

            streams.append(stream_of_split)

        self.stream_train, self.stream_val, self.stream_test = streams[0], streams[1], streams[2]

    def get_activity_list(self):
        activity_list = self.dataset["Activity"].unique()
        activity_list = np.append(activity_list, "None")
        return activity_list


class TorchDataset(Dataset):

    def __init__(self, df):
        self.df = df.iloc[:, : -1]
        self.df_labels = df.iloc[:, -1]

        self.dataset = torch.tensor(self.df.to_numpy()).float()
        self.labels = torch.tensor(self.df_labels.to_numpy().reshape(-1)).float()

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]
