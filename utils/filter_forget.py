import torch
def filter_forget(train_input, train_target, forget_indices, num_timesteps_input, num_timesteps_output):
    """
    Separates the training dataset into retained and forgotten samples
    based on the specified forget time intervals.
    """
    nums_window = train_input.shape[0]
    window_starts = torch.arange(nums_window)
    window_ends = window_starts + num_timesteps_input + num_timesteps_output
    mask_forget = torch.zeros(nums_window, dtype=torch.bool)
    for item in forget_indices:
        start = item[0]
        end = item[1]
        overlap = ~((window_ends < start) | (window_starts > end))
        mask_forget |= overlap

    Df_indices = torch.where(mask_forget)[0]
    Dr_indices = torch.where(~mask_forget)[0]

    retain_input = train_input[Dr_indices]
    retain_target = train_target[Dr_indices]

    forget_input = train_input[Df_indices]
    forget_target = train_target[Df_indices]

    return retain_input, retain_target, Dr_indices, forget_input, forget_target, Df_indices