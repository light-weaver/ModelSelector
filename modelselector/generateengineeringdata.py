# %%
import numpy as np
import pandas as pd
from pyDOE import lhs
import os

mean = 0.5
std_dev = 0.1
noise_mean = 0
noise_std = 0.05


def generate_var_0_1(num_var: int, num_samples: int, distribution: str):
    if distribution == "lhs":
        var = lhs(num_var, num_samples)
    elif distribution == "normal":
        means = [mean] * num_var
        cov = np.eye(num_var) * np.square(std_dev)
        var = np.random.multivariate_normal(means, cov, num_samples)
    elif distribution == "uniform":
        var = np.random.rand(num_samples, num_var)
    else:
        print("Error with distribution")
        return
    # To keep values between 0 and 1
    var[var > 1] = 1
    var[var < 0] = 0
    return var


def kursawe(num_samples: int = 100, distribution: str = "lhs") -> list:
    """Kursawe Test function.
    As found in Multiobjective structural optimization using a microgenetic algorithm.
    Parameters
    ----------
    x : list or ndarray
        x is a vector with 3 components. -5 < Xi < 5
    Returns
    -------
    list
        Returns a list of f1 and f2.
    """
    num_var = 3
    f1 = 0
    f2 = 0
    x = generate_var_0_1(num_var, num_samples, distribution)
    for i in range(2):
        f1 = f1 - 10 * np.exp(
            -0.2 * np.sqrt(x[:, i] * x[:, i] + x[:, i + 1] * x[:, i + 1])
        )
    for i in range(3):
        f2 = f2 + np.power(np.abs(x[:, i]), 0.8) + 5 * np.power(np.sin(x[:, i]), 3)
    return (x, np.asarray([f1, f2]).T)


def four_bar_plane_truss(num_samples: int = 100, distribution: str = "lhs") -> list:
    """Four bar plane truss problem.
    As found in Multiobjective structural optimization using a microgenetic algorithm.
    Parameters
    ----------
    x : list or ndarray
        Should have 4 elements
    Returns
    -------
    list
        (f1, f2)
    """
    num_var = 4
    x = generate_var_0_1(num_var, num_samples, distribution)
    F = 10
    E = 200000
    L = 200
    sigma = 10
    # bounds
    x[:, 0] = x[:, 0] * 2 * F / sigma + F / sigma
    x[:, 1] = (x[:, 1] * (3 - np.sqrt(2)) + np.sqrt(2)) * F / sigma
    x[:, 2] = (x[:, 2] * (3 - np.sqrt(2)) + np.sqrt(2)) * F / sigma
    x[:, 3] = x[:, 3] * 2 * F / sigma + F / sigma
    f1 = L * (2 * x[:, 0] + np.sqrt(2 * x[:, 1]) + np.sqrt(x[:, 2]) + x[:, 3])
    f2 = (
        F
        * (L / E)
        * 2
        * (1 / x[:, 0] + 1 / x[:, 3] + np.sqrt(2) * (1 / x[:, 1] - 1 / x[:, 2]))
    )
    return (x, np.asarray([f1, f2]).T)


def _gear_train_design(num_samples: int = 100) -> float:
    """Gear Train Design
    As found in Augmented Lagrange multiplier...
    Parameters
    ----------
    x : list or ndarray
        Should have 4 elements, integers in the range [12, 60].
    Returns
    -------
    float
    """
    x = np.random.randint(12, 61, (num_samples, 4))
    return (
        x,
        np.asarray(
            [np.square((1 / 6.931) - (x[:, 0] * x[:, 1]) / (x[:, 2] * x[:, 3]))]
        ).T,
    )


def _pressure_vessel(num_samples: int = 100) -> float:
    """Pressure Vessel design problem.
    As found in An augmented lagrange multiplier....
    Parameters
    ----------
    x : list or ndarray
        should contain 4 elements. First two should be discrete multiples or 0.0625.
        Last two should be continuous.
    Returns
    -------
    float
        cost
    """
    x = np.hstack(
        (
            np.random.randint(1, 10, (num_samples, 2)) * 0.0625,
            np.random.random((num_samples, 2)) * 80 + 50,
        )
    )
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    F = (
        0.6224 * x1 * x3 * x4
        + 1.7781 * x2 * x3 * x3
        + 3.1661 * x1 * x1 * x4
        + 19.84 * x1 * x1 * x3
    )
    G = -np.pi * x3 * x3 * x4 + (4 / 3) * np.pi * x3 * x3 * x3 + 1296000
    return (x, np.asarray([F, G]).T)


def speed_reducer(num_samples: int = 100, distribution: str = "lhs") -> list:
    """Speed reducer problem. Biobjective.
    As found in Multiobjective structural optimization using a microgenetic algorithm.
    Parameters
    ----------
    x : list or ndarray
        7 element vector.
    Returns
    -------
    list
        weight and stress
    """
    num_var = 7
    x = generate_var_0_1(num_var, num_samples, distribution)
    x[:, 0] = x[:, 0] * (3.6 - 2.6) + 2.6
    x[:, 1] = x[:, 1] * (0.8 - 0.7) + 0.7
    x[:, 2] = x[:, 2] * (11) + 17
    x[:, 3] = x[:, 3] * (1) + 7.3
    x[:, 4] = x[:, 4] * (1) + 7.3
    x[:, 5] = x[:, 5] * (1) + 2.9
    x[:, 6] = x[:, 6] * (0.5) + 5
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]
    x6 = x[:, 5]
    x7 = x[:, 6]
    f1 = (
        0.7854 * x1 * x2 * x2 * (10 * x3 * x3 / 3 + 14.933 * x3 - 43.0934)
        - 1.508 * x1 * (x6 * x6 + x7 * x7)
        + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7)
        + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    )
    f2 = np.sqrt(np.power(745 * x4 / (x2 * x3), 2) + 1.69 * np.power(10, 7)) / (
        0.1 * x6 * x6 * x6
    )
    f3 = np.sqrt(np.power(745 * x5 / (x2 * x3), 2) + 1.575 * np.power(10, 8)) / (
        0.1 * x7 * x7 * x7
    )
    return (x, np.asarray([f1, f2, f3]).T)


def welded_beam_design(num_samples: int = 100, distribution: str = "lhs"):
    """The Welded beam design

    AS found in An improved harmony search algo...

    Parameters
    ----------
    num_samples : int, optional
        Number of samples (the default is 100, which generates 100 datapoints)

    """
    num_var = 4
    x = generate_var_0_1(num_var, num_samples, distribution)
    x[:, 0] = x[:, 0] * (0.125)
    x[:, 1] = x[:, 1] * (9.9) + 0.1
    x[:, 2] = x[:, 2] * (9.9) + 0.1
    x[:, 3] = x[:, 3] * (4.9) + 0.1
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    f = 1.10471 * x1 * x1 * x2 + 0.04811 * x3 * x4 * (14 + x2)
    return (x, np.asarray([f]).T)


def unconstrained_f(num_samples: int = 100, distribution: str = "lhs"):
    num_var = 2
    x = generate_var_0_1(num_var, num_samples, distribution) * 100 - 50
    x1 = x[:, 0]
    x2 = x[:, 1]
    """f1 = (
        np.exp(0.5 * np.square(x1 * x1 + x2 * x2 - 25))
        + np.power(np.sin(4 * x1 - 3 * x2), 4)
        + 0.5 * (2 * x1 + x2 - 10)
    )"""
    f2 = (
        1
        + np.square(x1 + x2 + 1)
        * (19 - 14 * x1 + 3 * x1 * x1 - 14 * x2 + 6 * x1 * x2 + 3 * x2 * x2)
    ) * (
        30
        + np.square(2 * x1 - 3 * x2)
        * (18 - 32 * x1 + 12 * x1 * x1 + 48 * x2 - 36 * x1 * x2 + 27 * x2 * x2)
    )
    return (x, np.asarray([f2]).T)


def electric_power_dispatch(num_samples: int = 100, distribution: str = "lhs"):
    """The electric power dispatch problem described in:
    M. A. Abido, "Multiobjective evolutionary algorithms for electric power dispatch
    problem," in IEEE Transactions on Evolutionary Computation, vol. 10, no. 3, pp.
    315-329, June 2006.
    doi: 10.1109/TEVC.2005.857073

    Args:
        num_samples (int, optional): Number of samples. Defaults to 100.
        distribution (str, optional): Distribution of samples. Defaults to "lhs".
    """
    num_var = 6
    var = generate_var_0_1(num_var, num_samples, distribution)
    # Cost coefficients
    a = [10, 10, 20, 10, 20, 10]
    b = [200, 150, 180, 100, 180, 150]
    c = [100, 120, 40, 60, 40, 100]
    # Emission coefficients
    alpha = [4.091, 2.543, 4.258, 5.326, 4.258, 6.131]
    beta = [-5.554, -6.047, -5.094, -3.550, -5.094, -5.555]
    gamma = [6.490, 5.638, 4.586, 3.380, 4.586, 5.151]
    xi = [2.0e-4, 5.0e-4, 1.0e-6, 2.0e-3, 1.0e-6, 1.0e-5]
    lambd = [2.857, 3.333, 8.000, 2.000, 8.000, 6.667]
    # Fuel cost
    f1_1 = a
    f1_2 = b * var
    f1_3 = c * var * var
    f1 = (f1_1 + f1_2 + f1_3).sum(axis=1).reshape(-1, 1)
    # Emissions
    f2_1 = alpha
    f2_2 = beta * var
    f2_3 = gamma * var * var
    f2_4 = xi * np.exp(lambd * var)
    f2 = (((f2_1 + f2_2 + f2_3) / 100) + f2_4).sum(axis=1).reshape(-1, 1)
    return (var, np.hstack((f1, f2)))


def water_resource_management(num_samples: int = 100, distribution: str = "lhs"):
    """The water resource management problem as found in:
    A New Decomposition-Based NSGA-IIfor Many-Objective Optimization
    Maha Elarbi, Slim Bechikh, Abhishek Gupta, Lamjed Ben Said, and Yew-Soon Ong

    Args:
        num_samples (int, optional): Number of samples. Defaults to 100.
        distribution (str, optional): Distribution of samples. Defaults to "lhs".
    """
    num_var = 3
    var = generate_var_0_1(num_var, num_samples, distribution)
    # bounds
    lb = np.asarray([0.01, 0.01, 0.01])
    ub = np.asarray([0.45, 0.1, 0.1])
    var = (var * (ub - lb)) + lb
    # Objectives
    f1 = (106780.37 * (var[:, 1] + var[:, 2]) + 61704.67).reshape(-1, 1)
    f2 = (3000 * var[:, 0]).reshape(-1, 1)
    f3 = ((305700 * 2289 * var[:, 1]) / ((0.06 * 2289) ** 0.65)).reshape(-1, 1)
    f4 = (250 * 2289 * np.exp(-39.75 * var[:, 1] + 9.9 * var[:, 2] + 2.74)).reshape(
        -1, 1
    )
    f5 = (25 * (1.39 / (var[:, 0] * var[:, 1])) + 4940 * var[:, 2] - 80).reshape(-1, 1)
    # Constraints
    g1 = ((0.00139 / (var[:, 0] * var[:, 1])) + 4.94 * var[:, 2] - 0.08 - 1).reshape(
        -1, 1
    )
    g2 = (
        (0.000306 / (var[:, 0] * var[:, 1])) + 1.082 * var[:, 2] - 0.0986 - 1
    ).reshape(-1, 1)
    g3 = (
        (12.307 / (var[:, 0] * var[:, 1])) + 49408.24 * var[:, 2] + 4051.02 - 50000
    ).reshape(-1, 1)
    g4 = (
        (2.098 / (var[:, 0] * var[:, 1])) + 8046.33 * var[:, 2] - 696.71 - 16000
    ).reshape(-1, 1)
    g5 = (
        (2.138 / (var[:, 0] * var[:, 1])) + 7883.39 * var[:, 2] - 705.04 - 10000
    ).reshape(-1, 1)
    g6 = (
        (0.417 / (var[:, 0] * var[:, 1])) + 1721.26 * var[:, 2] - 136.54 - 2000
    ).reshape(-1, 1)
    g7 = ((0.164 / (var[:, 0] * var[:, 1])) + 631.13 * var[:, 2] - 54.48 - 550).reshape(
        -1, 1
    )

    return (var, np.hstack((f1, f2, f3, f4, f5, g1, g2, g3, g4, g5, g6, g7)))


# %%
def main(name: str, num_samples_options: list = None, distribution: list = None):
    # decision_vars = np.random.randint(12, 61, (100, 4))
    """decision_vars = np.hstack(
        (np.random.randint(1, 10, (100, 2))*0.0625, np.random.random((100, 2))*80+50)
    )"""
    if num_samples_options is None:
        num_samples_options = [100, 250, 500, 1000, 2000]
    if distribution is None:
        distribution = ["lhs", "normal"]
    for dist in distribution:
        for num_samples in num_samples_options:
            folderpath = "./datasets/engineering_" + name + "/"
            folderfull = "./datasets/engineeringfull/"
            if not os.path.exists(folderpath):
                os.mkdir(folderpath)
            if not os.path.exists(folderfull):
                os.mkdir(folderfull)
            problems = {
                "kursawe": kursawe,  # 2 obj
                "four-bar": four_bar_plane_truss,  # 2 obj
                # "gear-train": _gear_train_design,
                # "pressure": _pressure_vessel,
                "speed-reducer": speed_reducer,  # 3 obj
                "welded-beam": welded_beam_design,  # 1 obj
                "unconstrained-f": unconstrained_f,  # 1 obj
                "electric-power-dispatch": electric_power_dispatch,  # 2 obj
                "water-resource-management": water_resource_management,  # 12 obj
            }
            for problem in problems:
                x, f = problems[problem](num_samples)
                x = pd.DataFrame(
                    x, columns=["x" + str(i) for i in range(1, x.shape[1] + 1)]
                )
                f = pd.DataFrame(
                    f, columns=["f" + str(i) for i in range(1, f.shape[1] + 1)]
                )
                for f_ind in f:
                    data = pd.concat([x, f[f_ind]], axis=1)
                    data.to_csv(
                        folderpath
                        + problem
                        + "-"
                        + f_ind
                        + "_"
                        + str(len(x.columns))
                        + "_"
                        + str(num_samples)
                        + "_"
                        + dist
                        + ".csv",
                        index=False,
                    )
                datafull = pd.concat([x, f], axis=1)
                datafull.to_csv(
                    folderfull
                    + problem
                    + "_"
                    + str(len(x.columns))
                    + "_"
                    + str(num_samples)
                    + "_"
                    + dist
                    + ".csv",
                    index=False,
                )


if __name__ == "__main__":
    main()
