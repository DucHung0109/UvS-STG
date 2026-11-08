def replace_target(forget_target, surrogates, Df_indices, forget_indices, node_idxs, num_timesteps_input, num_timesteps_output):
    """
    Replaces the target values of forgotten samples with surrogate values
    for the specified nodes and time intervals.
    """
    new_forget_target = forget_target.clone()

    for i, node_idx in enumerate(node_idxs):
        surrogate = surrogates[i]
        
        for w in range(new_forget_target.shape[0]):
            global_start = Df_indices[w].item()
            global_target_start = global_start + num_timesteps_input

            for seg_idx, (f_start, f_end) in enumerate(forget_indices):
                surrogate_seg = surrogate[seg_idx] 
                
                overlap_start = max(global_target_start, f_start)
                overlap_end   = min(global_target_start + num_timesteps_output, f_end)
                
                if overlap_start < overlap_end:
                    local_start = overlap_start - global_target_start
                    local_end   = overlap_end   - global_target_start
                    
                    surrogate_offset = overlap_start - max(num_timesteps_input, f_start)

                    values_to_paste = surrogate_seg[surrogate_offset : surrogate_offset + (local_end - local_start)]

                    actual_len = values_to_paste.shape[0]

                    new_forget_target[w, node_idx, local_start:local_start+actual_len, :] = values_to_paste       
            
    return new_forget_target