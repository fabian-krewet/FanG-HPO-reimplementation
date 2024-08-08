import numpy as np
import pandas as pd
import string
import random
import pickle
import time

def transform_data(data: dict) -> pd.DataFrame:
    rows = []
    for key, value in data.items():
        rows.append([value])
    df = pd.DataFrame(rows)
    return pd.DataFrame(df)

def writing_with_pandas(data: dict, path) -> None:
    data_df = transform_data(data)
    data_df.to_csv(path + "test.csv")

def writing_with_pickle(data: dict, path) -> None:
    with open(path + "test.pickle", "wb") as f:
        pickle.dump(data, f)

def generate_pseudo_data(n, m, l, seed) -> dict:
    random.seed(seed)
    labels = ["x" + str(i) for i in range(m)]
    data = {}
    for i in range(n):
        for label in labels:
            data[label] = ''.join(random.choice(string.ascii_letters) for _ in range(l))
    return data

if __name__ == "__main__":  
    
    rng = np.random.default_rng(543)
    rng_gen = lambda: rng.integers(low=0, high=2**31-1, dtype=int)
    seed = rng_gen()

    path = "hpo-lab/assignments/assignment_3/src/test_files/"
    
    write_methods = [writing_with_pandas, writing_with_pickle]
    if True:
        start = time.time()
        data = generate_pseudo_data(10**3, 100, 500, seed)
        end = time.time()
        print(f"Generating data took {end - start} seconds")
    
    if False:    
        for n in [10**3, 10**4, 10**5]:
            for m in [10, 100, 1000]:
                for l in [10, 100, 1000]:
                    data = generate_pseudo_data(n, m, l, seed)

                    for write_method in write_methods:
                        start = time.time()
                        for _ in range(10):
                            write_method(data, path)
                        end = time.time()
                        print(f"Writing method {write_method.__name__} took {end - start} seconds for n={n}, m={m}, l={l}")
