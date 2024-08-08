import unittest
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.compose import ColumnTransformer


def reliable_indices(
    gp_s1: GaussianProcessRegressor,
    gp_cheap_s_m: GaussianProcessRegressor,
    X_cheap_s: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Computes the indices of the reliable points for a single cheap source and single objective according to the equation (8).

    ------

    :param gp_s1: The expensive full dataset GP (called "ground truth" in the paper)
    :param gp_cheap_s: A GP for a cheap source and some objective
    :param X_cheap_s: The queried points from the cheap source s
    :param alpha: The hyperparamter for reliability
    :param transformer: The transformer used to transform the data for better GPs
    :return: The indices of the reliable points
    """
    mu_s1, sigma_s1 = gp_s1.predict(X_cheap_s, return_std=True)
    mu_cheap_s = gp_cheap_s_m.predict(X_cheap_s)
    reliable_points_indices = np.where(np.abs(mu_s1 - mu_cheap_s) <= alpha * sigma_s1)[
        0
    ]
    return reliable_points_indices


def agp_for_objective_m(
    gps: list[GaussianProcessRegressor],
    X: list[np.ndarray],
    Y: list[np.ndarray],
    alpha: float,
    rng_gen: np.random.Generator,
):
    """
    Creates an augmented GP for a single objective given the GPs of the different sources and their corresponding data.

    --------
    :param gps: The GPs of the different sources for one objective
    :param X: The queried data of the different sources
    :param Y: The target values for the given objective for X
    :param alpha: The hyperparameter for reliability
    :param transformer: The transformer used to transform the data for better GPs
    :return: The augmented GP for the given objective
    """
    gp_s1 = gps[0]
    X_hat = X[0]
    X_hat_source_distribution = np.zeros(len(X))
    X_hat_source_distribution[0] = X[0].shape[0]
    Y_hat = Y[0]
    for s in range(1, len(gps)):
        gp_cheap_s = gps[s]
        reliable_idxs = reliable_indices(gp_s1, gp_cheap_s, X[s], alpha)
        X_hat = np.vstack((X_hat, X[s][reliable_idxs]))
        X_hat_source_distribution[s] = len(reliable_idxs)
        Y_hat = np.vstack((Y_hat, Y[s][reliable_idxs]))
    return (
        GaussianProcessRegressor(Matern(), random_state=rng_gen()).fit(X_hat, Y_hat),
        X_hat_source_distribution,
    )


def create_all_agps(
    gps: list[list[GaussianProcessRegressor]],
    X: list[np.ndarray],
    Y: list[np.ndarray],
    alpha: float,
    rng_gen: np.random.Generator,
):
    agps = []
    reliable_point_source_distribution = np.zeros((len(gps), len(X)))
    for m in range(len(gps[0])):
        gps_m = [gps[s][m] for s in range(len(gps))]
        agp_m, reliable_sources = agp_for_objective_m(
            gps_m, X, [y[:, m : m + 1] for y in Y], alpha, rng_gen
        )
        reliable_point_source_distribution[m] = reliable_sources
        agps.append(agp_m)
    return agps, reliable_point_source_distribution


class TestMethods(unittest.TestCase):

    def test_reliable_points_indices(self):
        X_s1 = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
        y_s1 = np.array(
            [[0.25], [0.5], [0.75], [0.8], [0.75], [0.5], [0.25], [0.1], [0.2], [0.3]]
        )

        X_cheap_s = np.array([[0], [2], [4], [6], [8]])
        y_cheap_s = np.array([[0.2], [0.7], [0.75], [0.25], [0.02]])

        gp_s1 = GaussianProcessRegressor().fit(X_s1, y_s1)
        gp_cheap_s = GaussianProcessRegressor().fit(X_cheap_s, y_cheap_s)
        indices = reliable_indices(gp_s1, gp_cheap_s, X_cheap_s, 1.0)
        self.assertEqual(indices.tolist(), [2, 3])

    def test_agp(self):
        X_s1 = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
        y_s1 = np.array(
            [[0.25], [0.5], [0.75], [0.8], [0.75], [0.5], [0.25], [0.1], [0.2], [0.3]]
        )

        X_cheap_s = np.array([[0], [2], [4], [5.9999], [8]])
        y_cheap_s = np.array([[0.2], [0.7], [0.75], [0.25], [0.02]])

        gp_s1 = GaussianProcessRegressor().fit(X_s1, y_s1)
        gp_cheap_s = GaussianProcessRegressor().fit(X_cheap_s, y_cheap_s)
        gps = [gp_s1, gp_cheap_s]
        X = [X_s1, X_cheap_s]
        Y = [y_s1, y_cheap_s]
        agps, indices = agp_for_objective_m(
            gps,
            X,
            Y,
            1.0,
            np.random.default_rng(42),
        )
        print(indices)

    def identity_column_transformer(self):
        return ColumnTransformer(
            transformers=[("identity", "passthrough", [0])],
            remainder="passthrough",
        ).fit(
            [[1]], [1]
        )  # Has to match test dimensions


if __name__ == "__main__":
    unittest.main()
