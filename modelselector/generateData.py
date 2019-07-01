"""Create synthetic datasets for surrogate training."""
# %%
import pickle as pk
from os import getcwd

import numpy as np
import pandas as pd
from optproblems import wfg, cec2005, cec2007, dtlz, zdt
from pyDOE import lhs
from scipy.stats.distributions import norm

mean = 0.5
std_dev = 0.1
noise_mean = 0
noise_std = 0.05
num_obj = 2
num_var_zdt = {
    "ZDT1": 30,
    "ZDT2": 30,
    "ZDT3": 30,
    "ZDT4": 10,
    "ZDT6": 10,
}


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
    save_folder: str,
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
    save_folder : str
        Path to the save folder
    """
    if "DTLZ" in problemname:
        generateDTLZ(
            problemname,
            num_var,
            num_samples,
            distribution,
            noise,
            missing_data,
            save_folder,
        )
        return
    elif "WFG" in problemname:
        generateWFG(
            problemname,
            num_var,
            num_samples,
            distribution,
            noise,
            missing_data,
            save_folder,
        )
        return
    elif "ZDT" in problemname:
        generateZDT(
            problemname,
            num_var,
            num_samples,
            distribution,
            noise,
            missing_data,
            save_folder,
        )
        return
    else:
        print("Error with Problem name")
    return


def generate_var_0_1(
    problemname: str,
    num_var: int,
    num_samples: int,
    distribution: str,
    noise: bool,
    missing_data: bool,
    save_folder: str,
):
    filename = (
        save_folder
        + "/"
        + problemname
        + "_"
        + str(num_var)
        + "_"
        + str(num_samples)
        + "_"
        + distribution
    )
    if distribution == "Uniform":
        var = lhs(num_var, num_samples)
    elif distribution == "Normal":
        means = [mean] * num_var
        cov = np.eye(num_var) * np.square(std_dev)
        var = np.random.multivariate_normal(means, cov, num_samples)
    if noise:
        noise_means = [noise_mean] * num_var
        noise_cov = np.eye(num_var) * np.square(noise_std)
        noise_var = np.random.multivariate_normal(noise_means, noise_cov, num_samples)
        filename = filename + "_noisy"
        var = var + noise_var
    # To keep values between 0 and 1
    var[var > 1] = 1
    var[var < 0] = 0
    return (var, filename)



def generateDTLZ(
    problemname: str,
    num_var: int,
    num_samples: int,
    distribution: str,
    noise: bool,
    missing_data: bool,
    save_folder: str,
):
    """Generate and save DTLZ datasets as csv.

    Parameters
    ----------
    problemname : str
        Name of the problem
    num_var : int
        number of variables
    num_samples : int
        Number of samples
    distribution : str
        Uniform or normal distribution
    noise : bool
        Presence or absence of noise
    missing_data : bool
        Presence or absence of missing data
    save_folder : str
        Path to the folder to save csv files
    """
    objective = problems[problemname](num_obj, num_var)
    var_names = ["x{0}".format(x) for x in range(num_var)]
    obj_names = ["f1", "f2"]
    var, filename = generate_var_0_1(
        problemname,
        num_var,
        num_samples,
        distribution,
        noise,
        missing_data,
        save_folder,
    )
    obj = [objective(x) for x in var]
    data = np.hstack((var, obj))
    data = pd.DataFrame(data, columns=var_names + obj_names)
    filename = filename + ".csv"
    data.to_csv(filename, index=False)
    return


def generateWFG(
    problemname: str,
    num_var: int,
    num_samples: int,
    distribution: str,
    noise: bool,
    missing_data: bool,
    save_folder: str,
):
    """Generate and save WFG datasets as csv.

    Parameters
    ----------
    problemname : str
        Name of the problem
    num_var : int
        number of variables
    num_samples : int
        Number of samples
    distribution : str
        Uniform or normal distribution
    noise : bool
        Presence or absence of noise
    missing_data : bool
        Presence or absence of missing data
    save_folder : str
        Path to the folder to save csv files
    """
    objective = problems[problemname](num_obj, num_var, k=4)
    var_names = ["x{0}".format(x) for x in range(num_var)]
    obj_names = ["f1", "f2"]
    var, filename = generate_var_0_1(
        problemname,
        num_var,
        num_samples,
        distribution,
        noise,
        missing_data,
        save_folder,
    )
    obj = [objective(x) for x in var]
    data = np.hstack((var, obj))
    data = pd.DataFrame(data, columns=var_names + obj_names)
    filename = filename + ".csv"
    data.to_csv(filename, index=False)
    return


def generateZDT(
    problemname: str,
    num_var: int,
    num_samples: int,
    distribution: str,
    noise: bool,
    missing_data: bool,
    save_folder: str,
):
    """Generate and save ZDT datasets as csv.

    Parameters
    ----------
    problemname : str
        Name of the problem
    num_var : int
        number of variables
    num_samples : int
        Number of samples
    distribution : str
        Uniform or normal distribution
    noise : bool
        Presence or absence of noise
    missing_data : bool
        Presence or absence of missing data
    save_folder : str
        Path to the folder to save csv files
    """
    objective = problems[problemname](num_obj, num_var)
    var_names = ["x{0}".format(x) for x in range(num_var)]
    obj_names = ["f1", "f2"]
    num_var = num_var_zdt[problemname]
    var, filename = generate_var_0_1(
        problemname,
        num_var,
        num_samples,
        distribution,
        noise,
        missing_data,
        save_folder,
    )
    obj = [objective(x) for x in var]
    data = np.hstack((var, obj))
    data = pd.DataFrame(data, columns=var_names + obj_names)
    filename = filename + ".csv"
    data.to_csv(filename, index=False)
    return
