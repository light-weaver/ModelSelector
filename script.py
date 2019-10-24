# %%
from modelselector.generateData import create_datasets

# %% Train
create_datasets(
    folder="./datasets/benchmark_train"
)

# %% Test
create_datasets(
    folder="./datasets/benchmark_test", distribution=["uniform"], num_samples=[10000],
)

# %% optimal
create_datasets(
    folder="./datasets/benchmark_optimal", distribution=["optimal"], num_samples=[10000]
)


# %%
from modelselector.TrainRegressionModels import trainregressionmodels as trm

trm(
    training_data_folder="./datasets/benchmark_train/",
    test_data_folder="./datasets/benchmark_test/",
    performance_output_folder="./surrogate_performance/",
    optimal_data_folder="./datasets/benchmark_optimal/",
    name="benchmark",
)

# %%
from modelselector.TrainRegressionModels import trainregressionmodelsCV as trmCV

trmCV(
    training_data_folder="./datasets/datasets_benchmark_train/",
    performance_output_folder="./",
    name="benchmark",
)

"""CalculateFeaturesForAllData(
    inputfolder="./datasets/engineering_train/", name="engineering"
)"""
