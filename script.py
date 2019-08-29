# %%
# from modelselector.CalculateFeatures import CalculateFeaturesForAllData

# from modelselector.generateData import create_datasets
# import pandas as pd


# %%
# create_datasets()


# %%
"""from modelselector.TrainRegressionModels import trainregressionmodels as trm

trm(
    training_data_folder="./datasets/engineering_train/",
    test_data_folder="./datasets/engineering_test/",
    performance_output_folder="./surrogate_performance/",
    name="engineering"
)"""

from modelselector.TrainRegressionModels import trainregressionmodelsCV as trmCV

trmCV(
    training_data_folder="./datasets/datasets_benchmark_train/",
    performance_output_folder="./surrogate_performance/",
    name="benchmark"
)

"""CalculateFeaturesForAllData(
    inputfolder="./datasets/engineering_train/", name="engineering"
)"""
