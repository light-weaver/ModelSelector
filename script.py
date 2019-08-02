# %%
from modelselector.CalculateFeatures import CalculateFeaturesForAllData
#from modelselector.generateData import create_datasets
import pandas as pd


# %%
# create_datasets()


#%%
"""from modelselector.TrainRegressionModels import trainregressionmodels as trm

trm(
    training_data_folder="./datasets_benchmark_train/",
    test_data_folder="./datasets_benchmark_test/",
)"""

CalculateFeaturesForAllData(inputfolder='./datasets_benchmark_train/')