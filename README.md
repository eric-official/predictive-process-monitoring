# Predictive Process Monitoring 

## Installation
To run this program, you have the install the necessary libraries. 
This can be achieved by setting Conda as the Environment and running the following command in the terminal:
```
conda install -c conda-forge --file docs/conda-requirements.txt
```
Since not all libraries are available with Conda, 
you have to install the other packages with Pip by running the following command in the terminal:
```
pip install -r docs/pip-requirements.txt
```
If problems with the installation of grpcio arise, run the following command and try again:
```
conda install -c anaconda grpcio
```

## Usage
Before running the main.py file, you can define hyperparameters for 
the machine learning and deep learning based approach. This can be done in 
tests/deep_learning_based_tests.py and machine_learning_tests.py. Note that for the machine learning based approach, 
you can define either a configuration with manual selected parameters or a search space 
for hyperparameter search. These different configurations can later be selected in the user menu.
The user menu will guide you through the five following functionalities of this program. The functionalities correspond
to the task of the PPM project.
- Analyse data: shows statistical measures for the provided dataset
- Forecast remaining time with machine learning: performs training and prediction of machine learning models for the 
  remaining time of processes
- List best machine learning approaches: shows top 3 configurations from hyperparameter search for machine learning 
  based approach
- Explain machine learning model: shows explainability dashboard for machine learning based approach
- Forecast remaining time with deep learning: performs training and prediction of deep learning models for the 
  remaining time of processes

## Folder structure
- /data: storage folder for event logs, hyperparameter search results and splitted dataset to hand in
- /docs: storage folder for requirements as well as project presentation and report
- /src: storage folder for source code and tensorboard logs
- /tests: storage folder for configuration files of machine learning and deep learning based approach

## Tensorboard
Tensorboard logs can be shown with the following command in the terminal:
```
tensorboard --logdir src/tensorboard_logs  
```
After executing this command, click the link to the tensorboard server or 
insert the following link in your web browser: 
```
http://localhost:6006/
```
This GitLab already conists of tensorboard logs for the underdone experiments. The
following experiments were underdone:
- version 0: first run with initial network architectures
- version 1: change from frequency encoding to one-hot encoding
- version 2: batch normalization added
- version 3: number of layers decreased
- version 4: batch size decreased from 50 to 20
- version 5: batch size increased from 50 to 70
- version 6: hidden_dim decreased from 100 to 50
- version 7: hidden_dim increased from 100 to 150