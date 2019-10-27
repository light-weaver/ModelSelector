# %%
from modelselector.generateData import create_datasets

# %% Create training benchmark data
create_datasets(folder="./datasets/benchmark_train")

# %% create testing benchmark data
create_datasets(
    folder="./datasets/benchmark_test", distribution=["uniform"], num_samples=[10000]
)

# %% create optimal benchmark data
create_datasets(
    folder="./datasets/benchmark_optimal", distribution=["optimal"], num_samples=[10000]
)
# %% Train regression models for benchmark data
from modelselector.TrainRegressionModels import trainregressionmodels as trm

trm(
    training_data_folder="./datasets/benchmark_train/",
    test_data_folder="./datasets/benchmark_test/",
    performance_output_folder="./",
    optimal_data_folder="./datasets/benchmark_optimal/",
    name="benchmark",
)

# %%
from modelselector.TrainRegressionModels import trainregressionmodelsCV as trmCV

trmCV(
    training_data_folder="./datasets/datasets_benchmark_train/",
    performance_output_folder="./surrogate_performance/",
    name="benchmark",
)
# %% Features benchmark
from modelselector.CalculateFeatures import CalculateFeaturesForAllData

CalculateFeaturesForAllData(
    inputfolder="./datasets/benchmark_train/", name="benchmark_train"
)


# %% Create training engineering data
from modelselector.generateengineeringdata import main

main(name="train")

# %% Create testing engineering data
from modelselector.generateengineeringdata import main

main(num_samples_options=[10000], distribution=["lhs"], name="test")


# %% Features engineering
from modelselector.CalculateFeatures import CalculateFeaturesForAllData

CalculateFeaturesForAllData(
    inputfolder="./datasets/engineering_train/", name="engineering_train"
)

# %% Train regression models for engineering data
from modelselector.TrainRegressionModels import trainregressionmodels as trm

trm(
    training_data_folder="./datasets/engineering_train/",
    test_data_folder="./datasets/engineering_test/",
    name="engineering",
)

# %%
from modelselector.TrainRegressionModels import trainregressionmodelsCV as trmCV

trmCV(
    training_data_folder="./datasets/engineering_train/",
    performance_output_folder="./surrogate_performance/",
    name="engineering",
)

# %%
