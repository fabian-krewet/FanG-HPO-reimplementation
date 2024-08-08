import numpy as np
from scipy.stats import norm


def delta_box(u: np.ndarray, l: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    dim_m = len(l)
    res = 1.0
    for j in range(dim_m):
        if sigma[j] < 1e-16:  # in case of numerical errors
            print(f"Numerical error: sd[{j}]={sigma[j]}")
            return 0.0
        # Eq. 10 from Fast Exact Computation of Expected HyperVolume Improvement by Zhang et al.
        # TODO: Compare with Github Repo: There defined differently
        res *= Psi(u[j], mu[j], sigma[j]) - Psi(l[j], mu[j], sigma[j])
    return res


def Psi(a: float, mu: float, sigma: float) -> float:
    """
    Large Psi according to  Eq. 10 from Fast Exact Computation of Expected HyperVolume Improvement by Zhang et al.
    """
    if a == np.NINF:
        return 0
    return (a - mu) * norm.cdf((a - mu) / sigma) + sigma * norm.pdf((a - mu) / sigma)


def limit(s: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Limit function as described in Fast Exact Computation of Expected HyperVolume Improvement by Zhang et al.
    """
    PF_limited = np.maximum(s, a)
    return PF_limited


def ehvi_wfg(
    nondominated_set: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    def recursive_ehvi(A) -> float:
        if len(A) == 0:
            return 0
        ai = A[0]
        S = A[1:]
        limited_set = [limit(s, ai) for s in S]
        non_dominated_set = [
            s
            for i, s in enumerate(limited_set)
            if not any(np.all(s >= x) for x in limited_set[:i] + limited_set[i + 1 :])
        ]
        return delta_box(l=ai, u=reference_point, mu=mu, sigma=sigma) - recursive_ehvi(
            non_dominated_set
        )

    m = len(reference_point)
    return delta_box(
        l=reference_point, u=np.full(m, np.NINF), mu=mu, sigma=sigma
    ) - sum(
        [
            recursive_ehvi(nondominated_set[: i + 1])
            for i in range(len(nondominated_set))
        ]
    )


if __name__ == "__main__":
    pass
    # # Example
    # mu = np.array([1.5, 2.5])
    # sigma = np.array([0.1, 0.1])
    # ref_point = np.array([4, 4])
    # candidate = np.array([1.5, 2.5])
    # PF = np.array([[2, 3], [3, 2]])
    # print(compute_ehvi(PF, mu, sigma, ref_point))
    # # expected to be roughly 0.75
