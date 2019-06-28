from modelselector.CalculateFeatures import calculatefeatures
import numpy as np

inputs = np.random.rand(500, 10)
outputs = np.random.rand(500, 1)

print(calculatefeatures(inputs, outputs))
