import pandas as pd
import numpy as np
import math
import re
import itertools
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter


def remove_specific_keys(dict_of_dict: dict, keys_to_remove) -> dict:
    """
    Removes specified keys from sub-dictionaries in a given dictionary.

    Parameters:
    dict_of_dict: The dictionary containing sub-dictionaries.
    keys_to_remove (list): List of keys to remove from sub-dictionaries.

    Returns:
    dict: The dictionary with specified keys removed from sub-dictionaries.
    """
    for sub_dict in dict_of_dict.values():
        for key in keys_to_remove:
            if key in sub_dict:
                del sub_dict[key]
    return dict_of_dict


def create_dataframe_and_dict_from_dict(data, seed, cls_name, data_set_name):
    """
    Creates a pandas DataFrame from a nested dictionary structure.
    and a dict with the configs
    """
    new_dict = {}
    df_rows = []
    for info_source, sub_dict in data.items():
        num_configs = len(sub_dict["queried_configs_x"])
        if "c_s" in sub_dict:
            c_s = sub_dict["c_s"]
        else:
            c_s = None
        for i in range(num_configs):
            row = {
                "Info_source": info_source,
                "seed": seed,
                "data_set": data_set_name,
                "cls_name": cls_name,
                "c_s": c_s,
                "ith_point": i,
                "queried_configs_y_mce": sub_dict["queried_configs_y_mce"][i],
                "queried_configs_y_DSP": sub_dict["queried_configs_y_DSP"][i],
            }
            df_rows.append(row)
            new_dict[f"{info_source}_{seed}_{data_set_name}_{cls_name}_{c_s}_{i}"] = (
                sub_dict["queried_configs_x"][i]
            )

    return pd.DataFrame(df_rows), new_dict


def create_dataframe_from_dict_2(data):

    for info_source, sub_dict in data.items():
        num_configs = len(sub_dict["queried_configs_x"])
        for i in range(num_configs):
            row = {
                "Info_source": info_source,
            }
            if "queried_configs_x" in sub_dict:
                for key, value in sub_dict["queried_configs_x"][i].items():
                    row[f"queried_configs_x_{key}"] = value
            if "queried_configs_y_mce" in sub_dict:
                row["queried_configs_y_mce"] = sub_dict["queried_configs_y_mce"][i]
            if "queried_configs_y_DSP" in sub_dict:
                row["queried_configs_y_DSP"] = sub_dict["queried_configs_y_DSP"][i]
            for key, value in sub_dict.items():
                if key not in [
                    "queried_configs_x",
                    "queried_configs_y_mce",
                    "queried_configs_y_DSP",
                ]:
                    if not isinstance(value, (list, np.ndarray, dict)):
                        row[key] = value
            rows.append(row)


def create_dataframe_from_dict(data):
    """
    Creates a pandas DataFrame from a nested dictionary structure.

    Parameters:
    data (dict): The outer dictionary with sub-dictionaries containing lists and numpy arrays.

    Returns:
    pd.DataFrame: A DataFrame with the specified structure.
    """
    # Initialize a list to collect rows for the DataFrame
    rows = []
    for outer_key, sub_dict in data.items():
        num_configs = len(sub_dict.get("queried_configs_x", []))

        for i in range(num_configs):
            row = {
                "Info_source": outer_key,
            }

            # Add the i-th entry from the lists of sub-dictionaries if they exist
            if "queried_configs_x" in sub_dict:
                for key, value in sub_dict["queried_configs_x"][i].items():
                    row[f"queried_configs_x_{key}"] = value

            # Add the i-th entry from the numpy arrays if they exist
            if "queried_configs_y_mce" in sub_dict:
                row["queried_configs_y_mce"] = sub_dict["queried_configs_y_mce"][i]
            if "queried_configs_y_DSP" in sub_dict:
                row["queried_configs_y_DSP"] = sub_dict["queried_configs_y_DSP"][i]

            # Add scalar values from the sub-dictionary
            for key, value in sub_dict.items():
                if key not in [
                    "queried_configs_x",
                    "queried_configs_y_mce",
                    "queried_configs_y_DSP",
                ]:
                    if not isinstance(value, (list, np.ndarray, dict)):
                        row[key] = value

            rows.append(row)

    # Create the DataFrame from the list of rows
    df = pd.DataFrame(rows)

    return df


def reevaluate_not_dominated_set(
    old_not_dominated_set: set, new_value: list = None
):  # TODO: double check if this is the correct way to do this
    if new_value is not None:
        old_not_dominated_set.add(tuple(new_value))
    new_not_dominated_set = set()
    for value in old_not_dominated_set:
        dominated = False
        for other in old_not_dominated_set:
            if value != other and all(other[i] <= value[i] for i in range(len(value))):
                dominated = True
                break
        if not dominated:
            new_not_dominated_set.add(value)
    return new_not_dominated_set


def convert_layer_values_to_single_tuple(config: dict):
    output = config.copy()
    n_layers = output["n_layers"]
    layers = [output[f"n_neurons_layer{i}"] for i in range(1, n_layers + 1)]
    output["hidden_layer_sizes"] = tuple(layers)
    output.pop("n_layers")
    for i in range(1, 5):
        if f"n_neurons_layer{i}" in output:
            output.pop(f"n_neurons_layer{i}")
    return output


def load_fair_data(file_path, target_column, sensitive_columns):
    df = pd.read_csv(file_path, delimiter=",")
    sensitive_columns = [col for col in df.columns if re.search(sensitive_columns, col)]
    data_x = df.drop(columns=[target_column])
    data_y = df[target_column]
    data_sensitive = df[sensitive_columns]
    return (data_x, data_y, data_sensitive)


def calculate_differential_statistical_parity(
    predictions: pd.Series, data: pd.DataFrame, sensitive_features: list[str]
):
    """
    Calculate Differential Statistical Parity (DSP).

    Returns:
    float: Differential Statistical Parity (DSP).
    """
    # Create a dictionary to store the separate dataframes
    sensitive_feature_dfs = {}
    # Group columns by prefix and create separate dataframes
    for sens_feature in sensitive_features:
        columns = [col for col in data.columns if col.startswith(sens_feature)]
        sensitive_feature_dfs[sens_feature] = data[columns]

    dsps = []
    n = predictions.shape[0]

    probability_positive = predictions.mean()
    for sensitive_feature_df in sensitive_feature_dfs.values():
        sensitive_feature_dsps = []
        for one_hot_feat in sensitive_feature_df.columns:
            one_hot_feat_df = sensitive_feature_df[one_hot_feat]
            prob_y_given_s = []
            for value in [0, 1]:
                idx = np.where(one_hot_feat_df == value)[0]
                if idx.shape[0] == 0:
                    return 0  # TODO check
                prob_s = idx.shape[0] / n
                prob_y_s = (
                    np.where((predictions == 1) | (one_hot_feat_df == value))[0].shape[
                        0
                    ]
                    / n
                )
                prob_y_given_s.append(
                    (probability_positive + prob_s - prob_y_s) / prob_s
                )
            sensitive_feature_dsps.append(abs(prob_y_given_s[0] - prob_y_given_s[1]))
        dsps.append(np.mean(sensitive_feature_dsps))

    return np.max(dsps)


def calculate_mce(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Calculate the misclassification error (MCE).
    """
    # Calculate the number of incorrect predictions
    incorrect_predictions = np.sum(true_labels != predicted_labels)

    # Calculate the total number of predictions
    total_predictions = len(true_labels)

    # Calculate the misclassification error
    mce = incorrect_predictions / total_predictions

    return mce


def evaluate_model(
    classifier: BaseEstimator,
    data_x: pd.DataFrame,
    data_y: pd.Series,
    sensitive_features: list[str],
):

    prediction_y = classifier.predict(data_x)

    mce = calculate_mce(data_y.to_numpy(), prediction_y)
    dsp = calculate_differential_statistical_parity(
        prediction_y, data_x, sensitive_features
    )

    return mce, dsp


"""
seperates new from legacy functions
"""


def create_comparable_random_search(benchmark_set, bo_params: dict, seeds: list):
    rs_list = []
    for seed in seeds:
        rs = random_search(
            benchmark_set,
            budget=bo_params["budget"],
            minimize_scoring=False,
            seed=seed,
        )
        rs["seed"] = seed
        rs_list.append(rs)
    return pd.concat(rs_list, ignore_index=True)


def random_search(
    benchmark_set,
    budget: int,
    minimize_scoring: bool = True,
    seed: int = 42,
):
    configs = [
        config.get_dictionary()
        for config in benchmark_set.get_opt_space(seed=seed).sample_configuration(
            budget
        )
    ]
    scores = [
        benchmark_set.objective_function(config)[0][benchmark_set.targets[0]]
        for config in configs
    ]
    results_df = pd.DataFrame(configs)
    results_df["score"] = scores
    results_df["iteration"] = range(budget)
    results_df["best_score"] = (
        results_df["score"].cummin()
        if minimize_scoring
        else results_df["score"].cummax()
    )
    return results_df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the dataframe.
    """
    # TOFIX: Do this properly
    return (df - df.mean()) / df.std()


def calculate_average_and_error_on_dict(data):
    results = {}

    for key, (measurements, errors) in data.items():
        n = len(measurements)
        if n == 0:
            raise ValueError("The list of measurements should not be empty.")

        # Calculate the average
        average = sum(measurements) / n

        # Calculate the combined error
        combined_error = math.sqrt(sum(error**2 for error in errors)) / math.sqrt(n)

        results[key] = (average, combined_error)

    return results


def plot_averages_with_errors(data_dict_with_errors):

    keys = list(data_dict_with_errors.keys())
    averages = [data_dict_with_errors[key][0] for key in keys]
    errors = [data_dict_with_errors[key][1] for key in keys]

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(keys, averages, yerr=errors, capsize=5, alpha=0.7, color="skyblue")
    ax.set_xlabel("hyperaparameters")
    ax.set_ylabel("Average Importance")
    ax.set_title("Average Measurements with Errors")
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.grid(True)

    return fig, ax


def create_transformer(ConfigSpace: ConfigurationSpace):
    """ """
    # TODO remove fidelity parameters

    # This can probably be done better
    categorical_features_list = [
        param
        for param in ConfigSpace.get_hyperparameters()
        if isinstance(param, CategoricalHyperparameter)
    ]
    # categorical_features_list.remove(['repl', 'trainsize'])
    categorical_features = {
        param.name: param.choices for param in categorical_features_list
    }
    numerical_features_list = [
        param
        for param in ConfigSpace.get_hyperparameters()
        if not isinstance(param, CategoricalHyperparameter)
    ]
    # numerical_features_list.remove(['repl', 'trainsize'])
    numerical_features = {
        param.name: (param.lower, param.upper) for param in numerical_features_list
    }
    numerical_features.pop("repl")
    numerical_features.pop("trainsize")
    print(numerical_features_list)

    # Create a DataFrame for categorical features using the Cartesian product of all possible values
    cat_product = list(itertools.product(*categorical_features.values()))
    categorical_df = pd.DataFrame(cat_product, columns=categorical_features.keys())

    # Create a DataFrame for numerical features by generating values within the specified bounds
    num_data = {
        feature: np.linspace(bounds[0], bounds[1], num=len(cat_product))
        for feature, bounds in numerical_features.items()
    }
    numerical_df = pd.DataFrame(num_data)

    # Combine the categorical and numerical DataFrames
    df = pd.concat([categorical_df, numerical_df], axis=1)

    # Preprocessing for categorical features: OneHotEncoder with predefined categories
    categorical_transformers = []

    for feature, categories in categorical_features.items():
        transformer = (
            feature,
            OneHotEncoder(categories=[categories], handle_unknown="ignore"),
            [feature],
        )
        categorical_transformers.append(transformer)

    # Preprocessing for numerical features: MinMaxScaler with specified bounds
    numerical_transformers = []

    for feature, bounds in numerical_features.items():
        transformer = (feature, MinMaxScaler(feature_range=bounds), [feature])
        numerical_transformers.append(transformer)

    preprocessor = ColumnTransformer(
        transformers=numerical_transformers + categorical_transformers
    )

    preprocessor.fit(df)

    return preprocessor


def lower_dimensional_representation_MDS(
    df: pd.DataFrame, target_dim: int = 2
) -> np.array:
    """
    Reduce the dimensionality of the dataframe.
    """
    df = df.to_numpy()

    pairwise_distances_condensed = pdist(df, metric="euclidean")
    pairwise_distances = squareform(pairwise_distances_condensed)

    # Create an instance of MDS
    mds = MDS(
        n_components=2
    )  # Specify the number of dimensions for the lower-dimensional space

    # Perform MDS
    lower_dimensional_points = mds.fit_transform(pairwise_distances)
    return lower_dimensional_points


def change_categorical_to_one_hot_encoded(
    df: pd.DataFrame, ConfigSpace: ConfigurationSpace
) -> pd.DataFrame:
    """
    Change the categorical columns to numerical columns.
    """

    # Fill NaN with eihther 'None' or -1
    categorical_columns = (df.dtypes == object)[df.dtypes == object].index
    df[categorical_columns] = df[categorical_columns].fillna("None")
    not_categorical_columns = (df.dtypes != object)[df.dtypes != object].index
    df[not_categorical_columns] = df[not_categorical_columns].fillna(-1)

    return pd.get_dummies(
        df,
        columns=categorical_columns,
        # [entry.name
        #    for entry in ConfigSpace.get_hyperparameters()
        #    if isinstance(entry, CategoricalHyperparameter)],
        dummy_na=True,
        dtype=np.int32,
    )


def PCA_plotting(
    df: pd.DataFrame, ConfigSpace: ConfigurationSpace, target_dim: int = 2
) -> None:
    """
    Plot the dataframe in lower dimensions.
    """
    # Generate data to fit PCA and normalize it
    space = ConfigurationSpace.sample_configuration(10000)

    # Fit PCA
    pca = PCA(n_components=2)

    # lower_dimensional_points = lower_dimensional_representation_PCA(df, target_dim) #TODO: This function is missing???
    # plt.scatter(lower_dimensional_points[:, 0], lower_dimensional_points[:, 1])
    # plt.show()


class evaluated_params_data:
    def __init__(self, benchmarks: list, seeds: list):
        self.data = {}
        # self.dataframe = dataframe #TODO evaluate if this is the best way to store the data (dict of dataframes better performance wise?)
        self.benchmarks = benchmarks
        self.seeds = seeds

    def add(self, benchmark: str, df: pd.DataFrame):
        """
        Add a dataframe to the data.
        """
        if benchmark not in self.benchmarks:
            raise ValueError(f"Unknown benchmark {benchmark}")
        self.data[benchmark] = df

    def simple_iterator(
        self, group_by_seeds: bool = True, return_seperate: list = ["score"]
    ):  # TODO: list = ['score', 'mu','std']):
        """
        Yield dataframe grouped by benchmarks (and optionally seeds) and the score column as a separate series.
        """

        for benchmark, df in self.data.items():
            if group_by_seeds:
                grouped = df.groupby("seed")
                for name, group in grouped:
                    yield benchmark, name, group.drop(
                        columns=return_seperate + ["seed"]
                    ), group[return_seperate]
            else:
                yield benchmark, name, df, df["score"]

    def dimension_iterator(self, group_by_seeds: bool = True):
        """
        Yield dataframe grouped by benchmarks (and optionally seeds) and the score column as a separate series but with reduced dimensionality.
        """
        pass

    def best(self, group_by_seeds: bool = True):
        """
        Yield the best configuration for each benchmark (and optionally seeds).
        """
        for benchmark, df in self.dataframes.items():
            grouped = df.groupby("seeds")
            for name, group in grouped:
                # Find the column with the highest score, excluding 'benchmarks', 'seeds', and 'score'
                best_config = group.drop(columns=["seeds", "score"]).loc(
                    group["score"].idxmax()
                )
                yield (benchmark, name), best_config, group["score"].max()

    def save(self) -> None:
        """
        Save the results to a file.
        """
        for benchmark, df in self.dataframes.items():
            df.to_csv("./data/{}_results.csv".format(benchmark))

    def load(self) -> None:
        """
        Load the results from a file.
        """
        for benchmark in self.benchmarks:
            self.data[benchmark] = pd.read_csv(
                "./data/{}_results.csv".format(benchmark)
            )
