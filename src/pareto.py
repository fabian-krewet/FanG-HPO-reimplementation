import numpy as np
import unittest


def HV(nondominated_set: np.ndarray, reference_point: np.ndarray = np.array([1, 1])):
    """
    Calculate the hypervolume of a nondominated set of points. - here assume two dimensions only
    """
    if (
        np.max(nondominated_set[:, 0]) > reference_point[0]
        or np.max(nondominated_set[:, 1]) > reference_point[1]
    ):
        raise ValueError("Reference point not worst point!")

    nondominated_set = nondominated_set[np.argsort(nondominated_set[:, 0])]
    if nondominated_set[0, 1] < reference_point[1]:
        nondominated_set = np.vstack(
            ([nondominated_set[0, 0], reference_point[1]], nondominated_set)
        )
    if nondominated_set[-1, 0] < reference_point[0]:
        nondominated_set = np.vstack(
            (nondominated_set, [reference_point[0], nondominated_set[-1, 1]])
        )

    hv = 0
    for i in range(1, nondominated_set.shape[0]):
        hv += (nondominated_set[i - 1, 1] - nondominated_set[i, 1]) * (
            reference_point[0] - nondominated_set[i, 0]
        )
    return hv


def compute_initial_pareto_set_front(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the initial pareto set and front
    """
    pareto_set = np.array([X[0]])
    pareto_front = np.array([Y[0]])
    for i in range(1, len(X)):
        pareto_set, pareto_front = update_pareto_front(
            pareto_set, pareto_front, X[i], Y[i]
        )
    return pareto_set, pareto_front


def update_pareto_front(
    pareto_set: np.ndarray,
    pareto_front: np.ndarray,
    new_config: np.ndarray,
    new_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a new pareto front with a new configuration and score
    """
    if any([dominance_check(old_y, new_score) for old_y in pareto_front]):
        return pareto_set, pareto_front
    pareto_indices = [
        i
        for i, old_score in enumerate(pareto_front)
        if not dominance_check(new_score, old_score)
    ]
    pareto_set = np.vstack([x for i, x in enumerate(pareto_set) if i in pareto_indices])
    pareto_set = np.vstack([pareto_set, new_config])
    pareto_front = np.vstack([pareto_front[pareto_indices], new_score])
    return pareto_set, pareto_front


def dominance_check(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Check if x pareto dominates y
    """
    return all(x <= y) and any(x < y)


class ParetoTestClass(unittest.TestCase):
    def test_HV(self):
        ref = np.array([1, 1])
        PF = np.array([[0.5, 0.5], [0.25, 0.75]])
        self.assertEqual(HV(PF, ref), 0.3125)

    def test_dominance(self):
        bad_value = np.array([1, 1])
        good_value = np.array([0.5, 0.5])
        self.assertFalse(dominance_check(bad_value, good_value))
        self.assertTrue(dominance_check(good_value, bad_value))
        self.assertFalse(dominance_check(bad_value, bad_value))

    def test_pareto_update_additional(self):
        pareto_set = np.array([[1, 1], [0.5, 0.5]])
        pareto_front = np.array([[0.8, 0.8], [0.6, 0.9]])
        new_config = np.array([0.25, 0.75])
        new_score = np.array([0.25, 0.92])
        pareto_set, pareto_front = update_pareto_front(
            pareto_set, pareto_front, new_config, new_score
        )
        self.assertTrue(
            np.all(pareto_set == np.array([[1, 1], [0.5, 0.5], [0.25, 0.75]]))
        )
        self.assertTrue(
            np.all(pareto_front == np.array([[0.8, 0.8], [0.6, 0.9], [0.25, 0.92]]))
        )

    def test_pareto_update_subsitute(self):
        pareto_set = np.array([[1, 1], [0.5, 0.5]])
        pareto_front = np.array([[0.8, 0.8], [0.6, 0.9]])
        new_config = np.array([0.25, 0.75])
        new_score = np.array([0.25, 0.82])
        pareto_set, pareto_front = update_pareto_front(
            pareto_set, pareto_front, new_config, new_score
        )
        self.assertTrue(np.all(pareto_set == np.array([[1, 1], [0.25, 0.75]])))
        self.assertTrue(np.all(pareto_front == np.array([[0.8, 0.8], [0.25, 0.82]])))


if __name__ == "__main__":
    unittest.main()
    # example
    ref = np.array([1, 1])
    PF = np.array([[0.5, 0.5], [0.25, 0.75]])
    print(HV(PF, ref))
