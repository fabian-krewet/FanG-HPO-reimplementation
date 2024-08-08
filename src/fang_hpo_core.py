# import from sklearn the GaussianProcess and Maternal Kernel
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from deepcave import Recorder, Objective
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from ConfigSpace import ConfigurationSpace
import time
import logging

from .utils import evaluate_model, convert_layer_values_to_single_tuple
from .ehvi import ehvi_wfg
from .agp import create_all_agps
from .pareto import HV, update_pareto_front, compute_initial_pareto_set_front

logger = logging.getLogger(__name__)


# TODO: Keep in mind furhter hyperparameters for BO (e.g., noise, smoothness)
# TODO: Retrain
def fang_hpo(
    classifier: BaseEstimator,
    data_x: pd.DataFrame,
    data_y: pd.Series,
    protected_features,  # TODO
    budget: int,
    n_initial_points: int = 5,
    random_state: int = 42,
    other_information_sources_shares: list = [0.5],
    cv: int = 5,
    config_space: ConfigurationSpace = None,
    save_path: str = "./data/results/DeepCAVE",
    alpha: float = 1.0,
    compute_HV: bool = False,
):
    """ """

    print("started new run with parameters: ", classifier)

    # ensure randomness:
    rng = np.random.default_rng(random_state)
    cv_random_state = random_state + 7823
    rng_gen = lambda: rng.integers(
        low=0, high=2**31 - 1, dtype=int
    )  # TODO do this the more modern way
    config_space.seed(rng_gen())

    # helping variables
    n_information_sources = len(other_information_sources_shares) + 1
    objectives = ["MCE", "DSP"]

    # define source costs (c_s) as per the paper page 4 - TODO: multiply by
    source_costs = np.array([1] + [share for share in other_information_sources_shares])

    # Sample a few configurations to fit the column transformer for the Gaussian Process
    all_columns, transformer = create_transformer_for_gp(config_space)

    mce_metric = Objective("MCE", lower=0, upper=1, optimize="lower")
    dsp_metric = Objective("DSP", lower=0, upper=1, optimize="lower")

    info_sources = {0: {"data_X": data_x, "data_y": data_y}}
    for s in range(1, n_information_sources):
        data_split_seed = rng_gen()
        (
            data_x_low_budget,
            _,
            data_y_low_budget,
            _,
        ) = train_test_split(
            data_x,
            data_y,
            train_size=source_costs[s],
            random_state=data_split_seed,
        )
        info_sources[s] = {"data_X": data_x_low_budget, "data_y": data_y_low_budget}

    gps, queries_X, queries_Y, hvs = [], [], [], []

    with Recorder(
        config_space, objectives=[mce_metric, dsp_metric], save_path=save_path
    ) as r:

        # initialization
        for s in range(n_information_sources):
            configs = [
                config_space.sample_configuration().get_dictionary()
                for _ in range(n_initial_points)
            ]
            configs_vectorized = transformer.transform(
                pd.DataFrame(configs, columns=all_columns)
            )

            scores = [
                train_model(
                    classifier,
                    data_X=info_sources[s]["data_X"],
                    data_y=info_sources[s]["data_y"],
                    protected_features=["race", "sex"],  # TODO
                    cv_random_state=cv_random_state,
                    config_space=config_space,
                    config=config,
                    seed_model_training=rng_gen(),
                    cv=cv,
                )
                for config in configs
            ]
            # iterate over the objectives to create multple GPs
            objective_scores_list = np.array(scores)
            gps_for_s = [
                GaussianProcessRegressor(kernel=Matern(), random_state=rng_gen()).fit(
                    X=configs_vectorized, y=objective_scores_list[:, m]
                )
                for m in range(len(objectives))
            ]

            gps.append(gps_for_s)
            queries_X.append(configs_vectorized)
            queries_Y.append(objective_scores_list)

        pareto_set, pf = compute_initial_pareto_set_front(
            np.vstack(queries_X), np.vstack(queries_Y)
        )
        print("Initial Pf", pf)

        # set the current budget according to the number of initial points
        current_budget = np.array(source_costs).sum() * n_initial_points

        while current_budget <= budget:  # maybe needs adjustment as well
            print("Start Iteration")
            start_time = time.perf_counter()

            agps, reliable_point_distribution = create_all_agps(
                gps, queries_X, queries_Y, alpha, rng_gen
            )
            time_ehvi = time.perf_counter()
            next_config = select_next_query_config(
                config_space, agps, pf, transformer, n_samples=10000
            )
            print("Time for EHVI", time.perf_counter() - time_ehvi)
            next_config_vectorized = transformer.transform(
                pd.DataFrame([next_config.get_dictionary()])
            )

            next_source = select_next_information_source(
                next_config_vectorized,
                agps,
                gps,
                source_costs,
                reliable_point_distribution,
            )

            model_seed = rng_gen()

            r.start(
                next_config,
                budget=source_costs[next_source],
                seed=model_seed,
            )

            config_score = train_model(
                classifier,
                data_X=info_sources[next_source]["data_X"],
                data_y=info_sources[next_source]["data_y"],
                protected_features=["race.", "sex."],  # TODO
                cv_random_state=cv_random_state,
                config_space=config_space,
                config=next_config,
                seed_model_training=model_seed,
                cv=cv,
            )

            r.end(
                costs=config_score.tolist(),
                budget=source_costs[next_source],
            )

            pareto_set, pf = update_pareto_front(
                pareto_set, pf, next_config_vectorized, config_score
            )

            queries_X[next_source] = np.vstack(
                (queries_X[next_source], next_config_vectorized)
            )
            queries_Y[next_source] = np.vstack((queries_Y[next_source], config_score))

            gps[s] = [
                GaussianProcessRegressor(Matern(), random_state=rng_gen()).fit(
                    queries_X[next_source], queries_Y[next_source][:, m]
                )
                for m in range(len(objectives))
            ]

            # TODO update the pareto front and check if new element is in it - otherwise we don't need to compute the full_data_score
            # if next_source != 0:
            #     full_data_score = train_model(
            #         classifier,
            #         data_X=info_sources[0]["data_X"],
            #         data_y=info_sources[0]["data_y"],
            #         protected_features=["race", "sex"],  # TODO
            #         cv_random_state=cv_random_state,
            #         config_space=config_space,
            #         config=next_config,
            #         seed_model_training=model_seed,
            #         cv=cv,
            #     )
            # else:
            #     full_data_score = config_score

            logger.info("Iteration %s: %s ", current_budget, config_score)

            logger.info("Time for iteration %f:", time.perf_counter() - start_time)

            if compute_HV:
                hv = HV(pf)
                hvs.append(hv)
                print("hvs", hvs)

            # step 6
            current_budget += source_costs[next_source]

    return pf, transformer, hvs


def select_next_query_config(
    config_space: ConfigurationSpace,
    agps: list[GaussianProcessRegressor],
    pf,
    transformer: ColumnTransformer,
    n_samples: int = 10000,
):
    """
    Selects the next query configuration by maximizing the EHVI as aquisition function.
    For that we sample n_samples configurations and evaluate the EHVI for each of them.

    ------
    :param config_space: The configuration space
    :param agps: The AGPs for the different objectives
    :param pf: The current approximation of the Pareto front
    :param transformer: The transformer used to transform the data for better GPs
    :param n_samples: The number of samples to evaluate the EHVI
    :return: The configuration with the highest EHVI
    """
    configurations = config_space.sample_configuration(n_samples)
    configurations_vectorized = transformer.transform(
        pd.DataFrame([config.get_dictionary() for config in configurations])
    )

    mus = np.zeros((n_samples, len(agps)))
    sigmas = np.zeros((n_samples, len(agps)))
    for m, agp in enumerate(agps):
        mus[::, m], sigmas[::, m] = agp.predict(
            configurations_vectorized, return_std=True
        )

    acquisition_values = Parallel(n_jobs=-1)(
        delayed(ehvi_wfg)(
            pf,
            mu,
            sigma,
            np.array([1, 1]),
        )
        for mu, sigma in zip(mus, sigmas)
    )

    # ehvis = np.zeros(n_samples)
    # for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
    #     ehvis[i] = ehvi_wfg(pf, mu, sigma, np.array([1, 1]))

    return configurations[np.argmax(acquisition_values)]


def select_next_information_source(
    config_vectorized: np.ndarray,
    agps: list[GaussianProcessRegressor],
    gps: list[GaussianProcessRegressor],
    source_costs: np.ndarray,
    reliable_points_distribution: np.ndarray,
):

    if any(
        [
            reliable_points_distribution[m][0]
            < np.sum(reliable_points_distribution[m][1:])
            for m in range(len(reliable_points_distribution))
        ]
    ):
        return 0

    print(config_vectorized.shape, config_vectorized)
    discrepancies = [
        sum(
            [
                abs(
                    agps[m].predict(config_vectorized)
                    - gps[s][m].predict(config_vectorized)
                )
                for m in range(len(agps))
            ]
        )
        for s in range(len(gps))
    ]
    query_costs = [sum([discrepancies[s] * source_costs[s]]) for s in range(len(gps))]
    return np.argmin(query_costs)


def train_model(
    classifier: BaseEstimator,
    config: dict,
    data_X: np.ndarray,
    data_y: np.ndarray,
    protected_features: list[str],
    cv_random_state: int,
    seed_model_training: int,
    config_space: ConfigurationSpace,
    cv: int = 5,
) -> np.ndarray:
    """
    trains the ML model with the given HP conig and returns a tuple of the scores
    """

    if config_space.name == "MLP_config_space":
        config_adapted = convert_layer_values_to_single_tuple(config)
    else:
        config_adapted = config

    classifier.set_params(**config_adapted, random_state=seed_model_training)

    scores = np.zeros((cv, 2))
    kf = KFold(n_splits=cv, shuffle=True, random_state=cv_random_state)

    for i, (train_index, val_index) in enumerate(kf.split(data_X)):
        training_data_X, val_data_X = data_X.iloc[train_index], data_X.iloc[val_index]
        training_data_y, val_data_y = data_y.iloc[train_index], data_y.iloc[val_index]

        classifier.fit(training_data_X, training_data_y)
        scores[i] = evaluate_model(
            classifier, val_data_X, val_data_y, protected_features
        )

    return np.mean(scores, axis=0)


def create_transformer_for_gp(config_space):
    configs_for_transformer_fit = pd.DataFrame(
        [config.get_dictionary() for config in config_space.sample_configuration(10000)]
    )

    all_columns = configs_for_transformer_fit.columns.tolist()
    numerical_columns = configs_for_transformer_fit.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_columns = configs_for_transformer_fit.select_dtypes(
        include=[object]
    ).columns.tolist()

    transformer = tranformer_definition(numerical_columns, categorical_columns)
    transformer.fit(configs_for_transformer_fit)
    return all_columns, transformer


def tranformer_definition(numerical_columns, categorical_columns):
    """
    Helper function that returns a ColumnTransformer that adds a constant for missing values (i.e. to deal with
    the hierarchical search spaces) and scales numerical columns and one-hot encodes categorical columns.

    ==================

    :param numerical_columns: List of numerical columns
    :param categorical_columns: List of categorical columns
    :return: ColumnTransformer
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                    ]
                ),
                numerical_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="passthrough",
    )
