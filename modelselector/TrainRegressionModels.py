from os import listdir
from time import time

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import ExtraTreesRegressor as ExTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor as DTR
from tqdm import tqdm


def trainregressionmodels(
    training_data_folder: str,
    test_data_folder: str,
    performance_output_folder: str = None,
    name: str = None,
    models: dict = None,
):
    """Train regression models on the datasets found in the path given by *_data_folder.
    Saves R^2 values in csv file.

    Note: Only the final objective (named 'f*' where * is an integer) is trained.

    Parameters
    ----------
    training_data_folder : str, optional
        Path to the datasets to be used for training, by default None
    test_data_folder : str, optional
        Path to the datasets to be used for testing, by default None
    name : str
        Name of the output files
    models : dict, optional
        Dictionary of sklearn models, by default None.
        Should be in the format: {"<Model_name>": [model object, {parameters for fit}]}.
        <model object> is the class that has .fit and .predict methods.
    """
    if performance_output_folder is None:
        performance_output_folder = "./surrogate_performance"
    training_data_files = listdir(training_data_folder)
    test_data_files = listdir(test_data_folder)
    test_data_files = [file.split("/")[-1].split("_") for file in test_data_files]
    test_data_files = pd.DataFrame(
        test_data_files, columns=["problem_name", "num_var", "num_samples", "dist"]
    )
    # DO some magic to get num_samples easily
    if models is None:
        models = {
            "svm_linear": [SVR, {"kernel": "linear"}],
            "svm_rbf": [SVR, {"gamma": "scale"}],
            "MLP": [MLPR, {}],
            "GPR_rbf": [GPR, {"kernel": kernels.RBF()}],
            "GPR_matern3/2": [GPR, {"kernel": kernels.Matern(nu=1.5)}],
            "GPR_matern5/2": [GPR, {"kernel": kernels.Matern(nu=2.5)}],
            # "GPR_ExpSinSq": [GPR, {"kernel": kernels.ExpSineSquared()}],
            "DecisionTree": [DTR, {}],
            "RandomForest_10": [RFR, {"n_estimators": 10}],
            "RandomForest_100": [RFR, {"n_estimators": 100}],
            "AdaBoost_10": [ABR, {"n_estimators": 10}],
            "AdaBoost_100": [ABR, {"n_estimators": 100}],
            "ExtraTrees_10": [ExTR, {"n_estimators": 10}],
            "ExtraTrees_100": [ExTR, {"n_estimators": 100}],
        }
    metrics = {"R^2": r2_score, "MSE": mean_squared_error, "time": []}
    model_types = models.keys()
    metrics_types = metrics.keys()
    performance = {
        metric: pd.DataFrame(
            columns=model_types, index=training_data_files, dtype=float
        )
        for metric in metrics_types
    }
    # oldfile = ""
    for file in tqdm(training_data_files):
        # use the magic above to get validation data efficiently
        training_data = pd.read_csv(training_data_folder + file)
        filename = file.split("/")[-1].split(".")[0].split("_")
        problem_name = filename[0]
        num_var = filename[1]
        test_data_file = test_data_files[
            (test_data_files["problem_name"] == problem_name)
            & (test_data_files["num_var"] == num_var)
        ].values
        test_data_file = "_".join(test_data_file[0].tolist())
        test_data = pd.read_csv(test_data_folder + test_data_file)
        columns = training_data.columns
        x_columns = [column for column in columns if "x" in column]
        y_columns = [column for column in columns if "f" in column]
        X_train = training_data[x_columns].values
        y_train = training_data[y_columns[-1]].values
        X_test = test_data[x_columns].values
        y_test = test_data[y_columns[-1]].values
        for model_name, (model_type, model_parameters) in models.items():
            model = model_type(**model_parameters)
            time_init = time()
            model.fit(X_train, y_train)
            time_delta = time() - time_init
            y_pred = model.predict(X_test)
            performance["time"].at[file, model_name] = time_delta
            performance["R^2"].at[file, model_name] = r2_score(y_test, y_pred)
            performance["MSE"].at[file, model_name] = mean_squared_error(y_test, y_pred)
    for metric, performance_data in performance.items():
        performance_data.to_csv(
            performance_output_folder + "/" + name + metric + ".csv"
        )


def trainregressionmodelsCV(
    training_data_folder: str,
    performance_output_folder: str = None,
    name: str = None,
    models: dict = None,
    num_splits: int = 5,
):
    """Train regression models on the datasets found in the path given by *_data_folder,
    by doing cross validation.
    Saves R^2 values in csv file.

    Note: Only the final objective (named 'f*' where * is an integer) is trained.

    Parameters
    ----------
    training_data_folder : str, optional
        Path to the datasets to be used for training, by default None
    test_data_folder : str, optional
        Path to the datasets to be used for testing, by default None
    name : str
        Name of the output files
    models : dict, optional
        Dictionary of sklearn models, by default None.
        Should be in the format: {"<Model_name>": [model object, {parameters for fit}]}.
        <model object> is the class that has .fit and .predict methods.
    num_splits : int, optional
        Number of splits for cross validation.
    """
    if performance_output_folder is None:
        performance_output_folder = "./surrogate_performance"
    training_data_files = listdir(training_data_folder)
    if models is None:
        models = {
            "svm_linear": [SVR, {"kernel": "linear"}],
            "svm_rbf": [SVR, {"gamma": "scale"}],
            "MLP": [MLPR, {}],
            "GPR_rbf": [GPR, {"kernel": kernels.RBF()}],
            "GPR_matern3/2": [GPR, {"kernel": kernels.Matern(nu=1.5)}],
            "GPR_matern5/2": [GPR, {"kernel": kernels.Matern(nu=2.5)}],
            # "GPR_ExpSinSq": [GPR, {"kernel": kernels.ExpSineSquared()}],
            "DecisionTree": [DTR, {}],
            "RandomForest_10": [RFR, {"n_estimators": 10}],
            "RandomForest_100": [RFR, {"n_estimators": 100}],
            "AdaBoost_10": [ABR, {"n_estimators": 10}],
            "AdaBoost_100": [ABR, {"n_estimators": 100}],
            "ExtraTrees_10": [ExTR, {"n_estimators": 10}],
            "ExtraTrees_100": [ExTR, {"n_estimators": 100}],
        }
    model_types = models.keys()
    CV_score_mean = pd.DataFrame(
        columns=model_types, index=training_data_files, dtype=float
    )
    CV_score_max = pd.DataFrame(
        columns=model_types, index=training_data_files, dtype=float
    )
    scorer = make_scorer(r2_score)
    # oldfile = ""
    for file in tqdm(training_data_files):
        # use the magic above to get validation data efficiently
        training_data = pd.read_csv(training_data_folder + file)
        columns = training_data.columns
        x_columns = [column for column in columns if "x" in column]
        y_columns = [column for column in columns if "f" in column]
        X_train = training_data[x_columns].values
        y_train = training_data[y_columns[-1]].values
        for model_name, (model_type, model_parameters) in models.items():
            model = model_type(**model_parameters)
            score = cross_val_score(
                model, X_train, y_train, scoring=scorer, n_jobs=-1, cv=num_splits
            )
            CV_score_mean.at[file, model_name] = score.mean()
            CV_score_max.at[file, model_name] = score.max()
    CV_score_max.to_csv(performance_output_folder + "/" + name + "CV-score-max.csv")
    CV_score_mean.to_csv(performance_output_folder + "/" + name + "CV-score-mean.csv")
