"""Create synthetic datasets for surrogate training."""
# %%
import pickle as pk
from os import getcwd

import numpy as np
import pandas as pd
from optproblems import wfg, cec2005, cec2007, dtlz, zdt
from pyDOE import lhs
from scipy.stats.distributions import norm

problems = {
    "WFG1": wfg.WFG1,
    "WFG2": wfg.WFG2,
    "WFG3": wfg.WFG3,
    "WFG4": wfg.WFG4,
    "WFG5": wfg.WFG5,
    "WFG6": wfg.WFG6,
    "WFG7": wfg.WFG7,
    "WFG8": wfg.WFG8,
    "WFG9": wfg.WFG9,
    "ZDT1": zdt.ZDT1,
    "ZDT2": zdt.ZDT2,
    "ZDT3": zdt.ZDT3,
    "ZDT4": zdt.ZDT4,
    "ZDT6": zdt.ZDT6,
    "DTLZ1": dtlz.DTLZ1,
    "DTLZ2": dtlz.DTLZ2,
    "DTLZ3": dtlz.DTLZ3,
    "DTLZ4": dtlz.DTLZ4,
    "DTLZ5": dtlz.DTLZ5,
    "DTLZ6": dtlz.DTLZ6,
    "DTLZ7": dtlz.DTLZ7,
}


def generatedata(
    *,
    problemname: str,
    num_var: int,
    num_samples: int,
    distribution: str,
    noise: bool,
    missing_data: bool,
):
    """Generate random dataset from known benchmark problems or engineering problems.

    Parameters
    ----------
    problemname : str
        Name of the problem
    num_var : int
        number of decision variables
    num_samples : int
        number of samples
    distribution : str
        Normal or uniform distribution
    noise : bool
        Presence or absence of noise in data
    missing_data : bool
        Presence or absence of missing chunks of data
    """
    if "DTLZ" in problemname:
        generateDTLZ(
            problemname, num_var, num_samples, distribution, noise, missing_data
        )
    elif "WFG" in problemname:
        generateWFG(
            problemname, num_var, num_samples, distribution, noise, missing_data
        )
    elif "ZDT" in problemname:
        generateZDT(
            problemname, num_var, num_samples, distribution, noise, missing_data
        )
    pass
