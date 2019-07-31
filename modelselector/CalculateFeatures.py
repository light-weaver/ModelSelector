# import pandas.rpy.common as com
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP

numpy2ri.activate()
pandas2ri.activate()

r_code = """
library(flacco)

calculatefeatures <- function(inputs, outputs){
  features = data.frame(row.names=1)
  inputs = apply(inputs, 2 , as.numeric)
  outputs = apply(outputs, 1, as.numeric)
  feat.object = createFeatureObject(X = inputs, y = outputs)
  subsets = listAvailableFeatureSets(allow.additional_costs = FALSE)
  features_that_work = list(1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14)
  subsets = subsets[unlist(features_that_work)]
  for (subset in subsets){
    nextfeatures = calculateFeatureSet(feat.object, set = subset)
    nextfeatures = head(nextfeatures,-2)
    features = cbind(features, nextfeatures)
  }
  return(features)
}
"""

flacco_features = STAP(r_code, "test")


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
