import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def explore_data(dataset_path):
    """
    function to print statistics of data
    @param dataset_path: absolute path to semicolon seperated CSV with process log
    """

    dataset = pd.read_csv(dataset_path, sep=";")

    num_cases = len(dataset["Case ID"].unique())
    print("Number of cases: ", num_cases)

    num_activities = len(dataset["Activity"].unique())
    print("Number of activities: ", num_activities)

    num_events = len(dataset)
    print("Number of events: ", num_events)

    trace_duration_sorted = dataset.groupby(["Case ID"])["timesincecasestart"].max().sort_values()

    shortest_value = trace_duration_sorted.iloc[0]
    shortest_case = trace_duration_sorted[trace_duration_sorted == shortest_value].index[0]
    print("Shortest trace duration: ", shortest_case, " - ", round(shortest_value, 2))

    longest_value = trace_duration_sorted.iloc[len(trace_duration_sorted) - 1]
    longest_case = trace_duration_sorted[trace_duration_sorted == longest_value].index[0]
    print("Longest trace duration: ", longest_case, " - ", round(longest_value, 2))

    print("Average trace duration: ", round(trace_duration_sorted.mean(), 2))


def save_results(dataset):
    """
    function to save hyperparameters of machine learning based workflow and corresponding results for further analysis
    @param dataset: pd.Dataframe with tune.ResultsGrid of hyperparameters for machine larning based workflow
    """

    dataset.drop(labels=["time_this_iter_s", "done", "timesteps_total", "episodes_total", "training_iteration",
                         "trial_id", "experiment_id", "date", "timestamp", "time_total_s", "pid", "hostname", "node_ip",
                         "time_since_restore", "timesteps_since_restore", "iterations_since_restore", "warmup_time",
                         "logdir", "config/dataset_path", "config/prediction_data"], axis=1, inplace=True)
    dataset.columns = [name.replace("config/", "") for name in dataset.columns]

    current_path = os.path.abspath(os.curdir)
    project_path = current_path.split("/src")[0]
    dataset_path = os.path.join(project_path, "data", "result_grid.csv")
    dataset.to_csv(dataset_path)


def explain_approaches(dataset):
    """
    visualize results of hyperparameter optimization for machine learning based approach
    @param dataset: pd.Dataframe with tune.ResultsGrid of hyperparameters for machine larning based workflow
    @return:
    """

    dataset["mae_score"] = dataset["mae_score"] / 60

    categorical_subplot_titles = ["temporal_split_sort_by", "bucketing_technique", "ml_algorithm", "add_age",
                                  "add_temporal_information", "encoding_technique"]
    numerical_subplot_titles = ["bucketing_upper_bound", "kmeans_n_clusters", "svr_c",
                                "svr_epsilon", "knn_n_neighbors", "knn_leaf_size", "xgb_eta", "xgb_max_depth"]
    fig = make_subplots(rows=7, cols=2, subplot_titles=categorical_subplot_titles + numerical_subplot_titles)

    row = 1
    col = 1
    for title in categorical_subplot_titles:
        fig.add_trace(go.Box(x=dataset[title], y=dataset["mae_score"], showlegend=False), row=row, col=col)
        if col == 1:
            col = 2
        else:
            col = 1
            row += 1

    row = 4
    col = 1
    for title in numerical_subplot_titles:

        # noinspection PyTypeChecker
        fig.add_trace(go.Scatter(x=dataset[title], y=dataset["mae_score"], showlegend=False, mode="markers",
                                 marker_color=dataset["mae_score"], marker_colorscale="thermal",
                                 marker_coloraxis="coloraxis"), row=row, col=col)
        if col == 1:
            col = 2
        else:
            col = 1
            row += 1

    fig['layout'].update(height=1800,
                         title='Machine learning results explanation: <br> Effects of parameters on the deviation of '
                               'predicted remaining time from real remaining time (MAE Score) in hours.',
                         title_x=0.5)
    fig.update_layout(coloraxis=dict(colorscale='thermal', colorbar=dict(len=0.25)))
    fig.show()
