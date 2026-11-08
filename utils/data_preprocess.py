import numpy as np
import torch

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32)) # Convert to float32
    D = np.array(np.sum(A, axis=1)).reshape((-1,))  # Sum of row
    D[D <= 10e-5] = 10e-5    # Prevent divde by 0
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float() 

    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1)]
    # (start : start + num_input + num_output) 

    # Save samples
    if indices == []:
        return torch.from_numpy(np.array([])), torch.from_numpy(np.array([]))
    
    features, targets = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].permute(
                (0, 2, 1)))
        targets.append(X[:, :, i + num_timesteps_input: j].permute((0, 2, 1)))

    return torch.stack(features), torch.stack(targets)