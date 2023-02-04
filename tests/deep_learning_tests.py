from ray import tune


"""
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
    "max_epochs": integer with training epochs
    "network_architecture": name of desired network architecture (DFNN, LSTM)
}
"""

manual_dl_config = {
    "dataset_path": "/Users/ericnaser/Dropbox/Wirtschaftsinformatik_Master/2. Semester/Entwicklungspraktikum - Data Science/predictive-process-monitoring/data/sepsis_cases_1.csv",
    "temporal_split_sort_by": "timesincecasestart",
    "add_age": True,
    "add_temporal_information": True,
    "batch_size": 50,
    "hidden_dim": 100,
    "num_layers": 1,
    "dropout": 0.2,
    "learning_rate": 0.01,
    "max_epochs": 50,
    "network_architecture": "LSTM"
}
