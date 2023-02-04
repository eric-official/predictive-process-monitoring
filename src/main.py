import os

import pandas as pd
from IPython.display import display
from ray import tune

from src.analyser import explore_data, explain_approaches, save_results
from src.deep_learning_worker import DeepLearningWorker
from src.machine_learning_worker import MachineLearningWorker
from tests.deep_learning_tests import manual_dl_config
from tests.machine_learning_tests import manual_ml_config, hyperparam_ml_config
from tabulate import tabulate


def print_config_menu():
    """
    Print menu to choose configuration for machine learning based approach
    """

    print(" ")
    print("Which configuration for the machine learning based approach would you like to choose?")
    menu_options = {
        1: 'Use configuration with manual hyperparameters',
        2: 'Use configuration with hyperparameter search',
    }

    for key in menu_options.keys():
        print(key, '--', menu_options[key])


def print_main_menu():
    """
    Print menu to choose functionality for predictive process monitoring
    """

    print("")
    menu_options = {
        1: 'Analyse data',
        2: 'Forecast remaining time with machine learning',
        3: 'List best machine learning approaches',
        4: 'Explain machine learning model',
        5: 'Forecast remaining time with deep learning',
        6: "Exit"
    }

    for key in menu_options.keys():
        print(key, '--', menu_options[key])


def select_data_analysis(dataset_path):
    """
    Call method to analyse dataset
    @param dataset_path: absolute path to semicolon seperated CSV with process log
    """

    explore_data(dataset_path)


def select_ml_forecast(ml_params, tune_params):
    """
    Run hyperparameter optimization of workflow for machine learning based approach
    """

    # perform hyperparameter optimization
    ml_worker_tuning = MachineLearningWorker()
    tuner = tune.Tuner(ml_worker_tuning.execute_workflow, param_space=ml_params, tune_config=tune_params)
    result_grid = tuner.fit()

    # save results grid of hyperparameter search
    results_df = result_grid.get_dataframe()
    results_df.sort_values(by=["rmse_score"], ascending=True, inplace=True)
    save_results(results_df)

    # run on test data with best configuration from validation
    optimal_config = result_grid.get_best_result(metric="rmse_score", mode="min").config
    optimal_config["prediction_data"] = "test"
    ml_worker_optimized = MachineLearningWorker(optimal_config)
    score = ml_worker_optimized.execute_workflow(optimal_config)

    # show overview of test results
    print(tabulate(results_df.head(3), headers='keys', tablefmt='psql'))
    print("RMSE score for best config on test data :", score["rmse_score"])
    print("MAE score for best config on test data :", score["mae_score"])


def select_list_best_approaches():
    """
    Visualize explanation of results by plotting every attribute against MAE
    """

    current_path = os.path.abspath(os.curdir)
    project_path = current_path.split("/src")[0]
    dataset_path = os.path.join(project_path, "data", "result_grid.csv")

    if os.path.exists(dataset_path):
        results_df = pd.read_csv(dataset_path)
        results_df.sort_values(by=["rmse_score"], ascending=True, inplace=True)
        print(tabulate(results_df.head(3), headers='keys', tablefmt='psql'))
    else:
        print("You must have trained a model first (Option 2), before you can use the explainability function!")


def select_model_explanation():
    """
    Visualize explanation of results by plotting every attribute against MAE
    """

    current_path = os.path.abspath(os.curdir)
    project_path = current_path.split("/src")[0]
    dataset_path = os.path.join(project_path, "data", "result_grid.csv")

    if os.path.exists(dataset_path):
        results_df = pd.read_csv(dataset_path)
        explain_approaches(results_df)
    else:
        print("You must have trained a model first (Option 2), before you can use the explainability function!")


def select_dl_forecast(config):
    """
    Run workflow for deep learning based approach
    """
    dl_worker_tuning = DeepLearningWorker(config=config)
    dl_worker_tuning.execute_workflow()


if __name__ == "__main__":

    ml_param_space = None
    ml_tune_config = None
    dl_param_space = None

    # Manage menu for machine learning configuration
    exit_selected = False
    while exit_selected is False:
        print_config_menu()
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a valid number ...')
        # Check what choice was entered and act accordingly
        if option == 1:
            print("Machine learning configuration with manual hyperparameters has been selected!")
            ml_param_space = manual_ml_config
            dl_param_space = manual_dl_config
            ml_tune_config = tune.TuneConfig(num_samples=1)
            exit_selected = True
        elif option == 2:
            print("Machine learning configuration with hyperparameter search has been selected!")
            ml_param_space = hyperparam_ml_config
            dl_param_space = manual_dl_config
            ml_tune_config = tune.TuneConfig(num_samples=10)
            exit_selected = True
        else:
            print('Invalid option. Please enter a number between 1 and 2.')

    # Manage menu for functionality
    exit_selected = False
    while exit_selected is False:
        print_main_menu()
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a valid number ...')
        # Check what choice was entered and act accordingly
        if option == 1:
            select_data_analysis(ml_param_space["dataset_path"])
        elif option == 2:
            select_ml_forecast(ml_param_space, ml_tune_config)
        elif option == 3:
            select_list_best_approaches()
        elif option == 4:
            select_model_explanation()
        elif option == 5:
            select_dl_forecast(dl_param_space)
        elif option == 6:
            print('Goodbye!')
            exit_selected = True
        else:
            print('Invalid option. Please enter a number between 1 and 5.')
