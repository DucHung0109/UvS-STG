import pickle
import numpy as np

def load_data(file_path):

    train = np.load(file_path + "/train.npy").transpose((1, 2, 0))
    # (N, F, T)
    train = train.astype(np.double)
    # Normalization using Z-score method
    means = np.mean(train, axis=(0, 2))
    train = train - means.reshape(1, -1, 1)
    stds = np.std(train, axis=(0, 2))
    train = train / stds.reshape(1, -1, 1)
    
    test = np.load(file_path + "/test.npy").transpose((1, 2, 0))
    # (N, F, T)
    test = test.astype(np.double)
    # Normalization using Z-score method
    test = test - means.reshape(1, -1, 1)
    test = test / stds.reshape(1, -1, 1)
    
    try:
        with open(file_path + "/adj_mx.pkl", 'rb') as f:
            A = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file_path + "/adj_mx.pkl", 'rb') as f:
            A = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file_path + "/adj_mx.pkl", ':', e)
        raise

    return A, train, test, means, stds