import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataloader import DataLoader, TorchDataset
from src.deep_learning_predictor import DeepLearningPredictor
from src.prefix_encoder import PrefixEncoder
from src.prefix_extractor import PrefixExtractor


class DeepLearningWorker:

    def __init__(self, config=None):
        """
        DeepLearningWorker handles execution of deep learning based approach for remaining time prediction
        @param config:
            {
            "dataset_path": absolute path to semicolon seperated CSV with process log
            "temporal_split_sort_by": name of temporal attribute for sorting dataset before splitting
            "add_age": boolean indicating inclusion of age attribute
            "add_temporal_information": boolean indicating integration of month, weekday and hour
            "batch_size": integer with size of batch for training deep learning model
            "hidden_dim": integer with hidden size of LSTM / Linear layer
            "num_layers": integer with number of LSTM layers
            "dropout": decimal number with dropout rate between LSTM layers
            "learning_rate": decimal number with learning rate for training
            "max_epochs" integer with training epochs
            "network_architecture": name of desired network architecture (DFNN, LSTM)
            }
        """

        if config is not None:
            self.dataset_path = config["dataset_path"]
            self.temporal_split_sort_by = config["temporal_split_sort_by"]
            self.add_age = config["add_age"]
            self.add_temporal_information = config["add_temporal_information"]
            self.batch_size = config["batch_size"]
            self.hidden_dim = config["hidden_dim"]
            self.num_layers = config["num_layers"]
            self.dropout = config["dropout"]
            self.learning_rate = config["learning_rate"]
            self.max_epochs = config["max_epochs"]
            self.network_architecture = config["network_architecture"]
            self.num_additional_attributes = 0
            self.add_age_offset = 1

            if self.add_age is True:
                self.num_additional_attributes += 1
            if self.add_temporal_information is True:
                self.add_age_offset += 3
                self.num_additional_attributes += 3

    def test_settings(self):
        """
        test if chosen settings are possible
        """

        if self.network_architecture not in ["LSTM", "DFNN"]:
            raise ValueError("Defined network architecture is not available!")

    def execute_workflow(self):
        """
        execution of deep learning based approach for remaining time prediction
        @return: test_accuracy: decimal number with r2 score of deep learning model on test data
        """

        # Load data
        print("Import and preprocess data...")
        dataloader = DataLoader(dataset_path=self.dataset_path, temporal_split_sort_by=self.temporal_split_sort_by,
                                add_age=self.add_age, add_temporal_information=self.add_temporal_information)
        dataloader.calculate_remaining_time()
        dataloader.split()
        scaler = dataloader.scale_numeric_attributes()
        dataloader.log_to_stream()
        activity_list = dataloader.get_activity_list()

        # Extract prefixes
        print("Extract prefixes...")
        prefix_extractor = PrefixExtractor(bucketing_lower_bound=dataloader.max_case_events,
                                           bucketing_upper_bound=dataloader.max_case_events,
                                           add_age=self.add_age,
                                           add_temporal_information=self.add_temporal_information,
                                           num_additional_attributes=self.num_additional_attributes,
                                           add_age_offset=self.add_age_offset
                                           )
        prefix_extractor.extract_prefixes(dataloader.stream_train, dataloader.stream_val, dataloader.stream_test)
        prefix_extractor.transform_prefix_to_df()

        # Encode prefixes of training data
        print("Encode training data...")
        prefix_encoder_train = PrefixEncoder(prefix_dict_original=prefix_extractor.prefixes_train,
                                             categories=activity_list,
                                             add_age=self.add_age,
                                             add_temporal_information=self.add_temporal_information,
                                             num_additional_attributes=self.num_additional_attributes,
                                             add_age_offset=self.add_age_offset
                                             )
        prefix_encoder_train.encode()
        prefix_encoder_train.aggregate_same_columns()
        torch_dataset_train = TorchDataset(prefix_encoder_train.prefix_dict_encoded[dataloader.max_case_events])
        torch_train_loader = torch.utils.data.DataLoader(torch_dataset_train, batch_size=self.batch_size,
                                                         shuffle=False, num_workers=3)

        # encode prefixes of validation data
        print("Encode validation data...")
        prefix_encoder_val = PrefixEncoder(prefix_dict_original=prefix_extractor.prefixes_val,
                                           categories=activity_list,
                                           add_age=self.add_age,
                                           add_temporal_information=self.add_temporal_information,
                                           num_additional_attributes=self.num_additional_attributes,
                                           add_age_offset=self.add_age_offset
                                           )
        prefix_encoder_val.encode()
        prefix_encoder_val.aggregate_same_columns()
        torch_dataset_val = TorchDataset(prefix_encoder_val.prefix_dict_encoded[dataloader.max_case_events])
        torch_val_loader = torch.utils.data.DataLoader(torch_dataset_val, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=3)

        # encode prefixes of test data
        print("Encode test data...")
        prefix_encoder_test = PrefixEncoder(prefix_dict_original=prefix_extractor.prefixes_test,
                                            categories=activity_list,
                                            add_age=self.add_age,
                                            add_temporal_information=self.add_temporal_information,
                                            num_additional_attributes=self.num_additional_attributes,
                                            add_age_offset=self.add_age_offset
                                            )
        prefix_encoder_test.encode()
        prefix_encoder_test.aggregate_same_columns()
        torch_dataset_test = TorchDataset(prefix_encoder_test.prefix_dict_encoded[dataloader.max_case_events])
        torch_test_loader = torch.utils.data.DataLoader(torch_dataset_test, batch_size=self.batch_size,
                                                        shuffle=False, num_workers=3)

        # initialize deep learning model
        model = DeepLearningPredictor(input_dim=len(torch_dataset_train.df.columns),
                                      hidden_dim=self.hidden_dim,
                                      num_layers=self.num_layers,
                                      output_dim=1,
                                      dropout=self.dropout,
                                      learning_rate=self.learning_rate,
                                      network_architecture=self.network_architecture,
                                      scaler=scaler)
        # configure trainer
        logger = TensorBoardLogger("tensorboard_logs", name=self.network_architecture)
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            logger=logger,
            log_every_n_steps=1,
            gpus=None,
            accelerator="cpu"
        )

        # perform training and prediction with model
        trainer.fit(model, torch_train_loader, torch_val_loader)
        test_rmse = trainer.test(model=model, dataloaders=torch_test_loader)[0]["test_rmse"]
        return test_rmse
