import numpy as np

def zscore_exclude_index(lst, u):
    """
    Applies Z-score normalization to the input data while excluding
    specified indices from the normalization process.
    """

    arr = np.asarray(lst, dtype=float)
    result = np.zeros_like(arr)

    excluded_indices = set(u)
    valid_indices = [i for i in range(arr.shape[0])
                        if i not in excluded_indices]

    if not valid_indices:
        result[list(excluded_indices)] = 0
        return result

    for f in range(arr.shape[1]):
        vals = arr[valid_indices, f]
        mean = vals.mean()
        std = vals.std()

        if std == 0:
            result[valid_indices, f] = 0
        else:
            result[valid_indices, f] = (vals - mean) / std

    for i in excluded_indices:
        result[i, :] = 999999

    return result
