# import pandas.rpy.common as com
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP
from rpy2.rinterface_lib.embedded import RRuntimeError
from os import listdir, path, mkdir
import pandas as pd
from tqdm import tqdm

numpy2ri.activate()
pandas2ri.activate()

r_code_single = """
library(flacco)

calculatefeatures <- function(inputs, outputs){
  features = data.frame(row.names=1)
  inputs = apply(inputs, 2 , as.numeric)
  outputs = apply(outputs, 1, as.numeric)
  feat.object = createFeatureObject(X = inputs, y = outputs)
  subsets = listAvailableFeatureSets(allow.additional_costs = FALSE)
  features_that_work = list(1, 3, 4, 6, 7, 8, 9, 10, 11, 14)
  subsets = subsets[unlist(features_that_work)]
  for (subset in subsets){
    nextfeatures = calculateFeatureSet(feat.object, set = subset)
    nextfeatures = head(nextfeatures,-2)
    features = cbind(features, nextfeatures)
  }
  return(features)
}
"""

flacco_features = STAP(r_code_single, "test")


def calculatefeatures(inputs, outputs) -> list:
    """Calculates ELA features for one dataset using the R Flacco package.

        Parameters
        ----------
        inputs : np.ndarray
            The input array.
        outputs : np.ndarray
            The outputs (Just one objective)

        Returns
        -------
        list
            list of features
    """
    features = flacco_features.calculatefeatures(inputs, outputs)
    return features


def CalculateFeaturesForAllData(inputfolder: str, name: str):
    """Calculate features for all csv files (objective name = 'f2') in inputfolder.

    Parameters
    ----------
    inputfolder : str
        Path to input folder
    name :str
        Name of the output file, located at ./features/name.csv
    """
    data_files = listdir(inputfolder)
    output_dir = "./features/"
    if not path.exists(output_dir):
        mkdir(output_dir)
    features_all = pd.DataFrame()
    for file in tqdm(data_files):
        data = pd.read_csv(inputfolder + file)
        X = [x for x in data.columns if "x" in x]
        y = [y for y in data.columns if "f" in y][-1]
        try:
            features_single = calculatefeatures(data[X], data[[y]])
            features_single.rename({"1": file}, inplace=True)
            features_all = features_all.append(features_single)
        except RRuntimeError as exception:
            print(exception)
            print("FILE: " + file)
    features_all.to_csv(output_dir + name + ".csv")
