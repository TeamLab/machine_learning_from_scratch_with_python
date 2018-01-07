import numpy as np
import pandas as pd


def get_rating_matrix(filename, dtype=np.float32):
    dataset = open(filename, "r")
    data = pd.read_csv(dataset)
    critic_dict = {k: i for i, k in enumerate(
                    sorted(set(data.source.tolist())))}
    title_dict = {k: i for i, k in enumerate(
                    sorted(set(data.target.tolist())))}
    data["source"] = data.source.map(critic_dict)
    data["target"] = data.target.map(title_dict)

    result = np.zeros(shape=(len(critic_dict), len(title_dict)),
                      dtype=np.float32)
    result[data.source, data.target] = data.rating
    return result




def get_frequent_matrix(filename, dtype=np.float32):
    dataset = open(filename, "r")
    data = pd.read_csv(dataset)
    source_dict = {k: i for i, k in enumerate(
                sorted(set(data.source.tolist())))}
    target_dict = {k: i for i, k in enumerate(
                sorted(set(data.target.tolist())))}
    data["source"] = data.source.map(source_dict)
    data["target"] = data.target.map(target_dict)
    data["rating"] = 1
    data["temp"] = data["source"].map(str) + "-" + data["target"].map(str)
    from collections import Counter
    temp = Counter(data["temp"])
    temp = np.array([
        [int(k.split("-")[0]), int(k.split("-")[1]), v]
        for k, v in temp.items()])

    dataset.close()
    result = np.zeros(shape=(len(source_dict), len(target_dict)),
                      dtype=np.float32)
    result[temp[:, 0], temp[:, 1]] = temp[:, 2]
    return result
