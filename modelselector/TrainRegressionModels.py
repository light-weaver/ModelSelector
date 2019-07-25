from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import ExtraTreesRegressor as ExTR
from sklearn.metrics import r2_score, mean_squared_error
from glob import glob
import pandas as pd
from time import time


def trainregressionmodels(
    training_data_folder: str = None,
    validation_data_folder: str = None,
    models: dict = None,
):
    """Train regression models on the datasets found in the path given by *_data_folder.
    Saves R^2 values in csv file.

    Note: Only the final objective (named 'f*' where * is an integer) is trained.
    
    Parameters
    ----------
    training_data_folder : str, optional
        Path to the datasets to be used for training, by default None
    training_data_folder : str, optional
        Path to the datasets to be used for validation, by default None
    models : dict, optional
        Dictionary of sklearn models, by default None.
        Should be in the format: {"<Model_name>": [model object, {parameters for fit}]}.
    """
    training_data_files = glob(training_data_folder + "/*.csv")
    validation_data_files = glob(validation_data_folder + "/*.csv")
    # DO some magic to get num_samples easily
    if models is None:
        models = {
            "svm_linear": [SVR, {"kernel": "linear"}],
            "svm_rbf": [SVR, {}],
            "MLP": [MLPR, {}],
            "GPR_rbf": [GPR, {"kernel": kernels.RBF()}],
            "GPR_matern3/2": [GPR, {"kernel": kernels.Matern(nu=1.5)}],
            "GPR_matern5/2": [GPR, {"kernel": kernels.Matern(nu=2.5)}],
            "GPR_ExpSinSq": [GPR, {"kernel": kernels.ExpSineSquared()}],
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
    multilevel_columns = pd.MultiIndex.from_product(
        [metrics_types, model_types], names=["metrics", "models"]
    )
    performance_all = pd.DataFrame(columns=multilevel_columns, dtype=float)
    for file in training_data_files:
        # use the magic above to get validation data efficiently
        training_data = pd.read_csv(file)
        validation_data = pd.read_csv(file_magic)
        X, y = training_data[magic2]
        performance_single = {file: dict()}
        for model_name, (model_type, model_parameters) in models.items():
            model = model_type(**model_parameters)
            time_init = time()
            model.fit(X, y)
            time_delta = time() - time_init
            performance_single[file][("R^2", model_name)] = r2_score(
                X_validate, y_validate
            )
            performance_single[file][("MSE", model_name)] = r2_score(
                X_validate, y_validate
            )
            performance_single[file][(time, model_name)] = time_delta

