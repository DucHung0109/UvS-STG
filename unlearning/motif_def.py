from tslearn.metrics import dtw

def discover_motifs_proxy(dataset, u, faulty_node_idx, threshold):
    """
    This function identifies a "concept" as any data involving a USER PROVIDES INFO about what to forget.
    
    Params:
        dataset: (num_nodes, num_features, num_timesteps).
        u: unlearn data.
        faulty_node_idx (int): The index of the sensor node to be "forgotten".
        threshold (float): DTW distance threshold.
                           
    Returns:
        forget_indices (list of lists): [[start, end], ...] of subsets similar with u
        retain_indices (list of lists): [[start, end], ...] of subsets not similar with u
    """
    S = dataset[:, 0, :]
    _, time_step = S.shape
    forget_indices = []
    node_idx = faulty_node_idx[0]

    # Using a simpler sliding window approach to find motifs
    window_size = len(u)
    for i in range(time_step - window_size + 1):
        segment = S[node_idx, i:i+window_size]
        if dtw(u, segment) <= threshold:
            # Merge overlapping or adjacent motifs
            if forget_indices and i <= forget_indices[-1][1]:
                forget_indices[-1][1] = i + window_size
            else:
                forget_indices.append([i, i + window_size])

    retain_indices = []
    # Handle case where no motifs are found
    if not forget_indices:
        print("Warning: No motifs found with the given threshold. Entire dataset is considered 'retain'.")
        retain_indices.append([0, time_step])
        return forget_indices, retain_indices

    # Calculate retain indices based on forget indices
    last_forget_end = 0
    for start, end in forget_indices:
        if start > last_forget_end:
            retain_indices.append([last_forget_end, start])
        last_forget_end = end
    
    if last_forget_end < time_step:
        retain_indices.append([last_forget_end, time_step])
    
    return forget_indices, retain_indices

def find_forget(dataset, faulty_node_idx, threshold = 0.5):
    S = dataset[faulty_node_idx, 0, :]
    time_step = S.shape[1]
    mask = (S >= threshold).any(axis=0) 
    forget_indices = []
    retain_indices = []

    start = -1
    end = 0
    count = 0
    for i in range(time_step):
        if mask[i]:
            count += 1
            if start == -1:
                start = i
                
                end = i + 1
            else:
                end = i + 1
        else:
            if start != -1:
                forget_indices.append([start, end])
                start = -1
                end = 0

    if start != -1:
        forget_indices.append([start, time_step])

    if not forget_indices:
        print("Warning: No motifs found with the given threshold. Entire dataset is considered 'retain'.")
        retain_indices.append([0, time_step])
        return forget_indices, retain_indices

    last_forget_end = 0
    for start, end in forget_indices:
        if start > last_forget_end:
            retain_indices.append([last_forget_end, start])
        last_forget_end = end
    
    if last_forget_end < time_step:
        retain_indices.append([last_forget_end, time_step])
    
    return forget_indices, retain_indices
