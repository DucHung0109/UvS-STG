import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

class TemporalGenerativeReplay:
    """
    Self-Contained Temporal Generative Replay (T-GR) implementation
    Uses the model's own capabilities for reconstruction and neutralization
    """
    def __init__(self, model_type: str = "stgcn"): # Change the model type for different tests
        """
        Params:
            model_type: Type of model ("stgcn", "", "")
        """
        self.model_type = model_type

    def surrogate(self, model: nn.Module, forget_input, 
                        forget_indices: list, faulty_node_idx: int, num_timesteps_input, num_timesteps_output, Df_indices,
                        device, A_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Frozen-teacher predictions at the removed node(s), averaged over overlapping
        windows, one segment per forget interval. Returns {node: [segment (F, T)]}."""
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'forward_unlearning_compatible'):
                project_output = model.forward_unlearning_compatible(forget_input)
            elif A_hat is not None:
                batch_size = 256
                all_outputs = []
                num_samples = forget_input.size(0)

                for i in range(0, num_samples, batch_size):
                    batch = forget_input[i:i+batch_size].to(device)
                    batch_out = model(A_hat, batch)
                    all_outputs.append(batch_out.detach())

                project_output = torch.cat(all_outputs, dim=0).detach().cpu()
                # (B, N, T, F)
            else:
                raise ValueError("Either model must have forward_unlearning or A_hat must be provided")
            
        B, N, T_out, F = project_output.shape

        faulty_nodes = torch.as_tensor(
            faulty_node_idx, dtype=torch.long
        ).reshape(-1)
        Df_indices = torch.as_tensor(
            Df_indices,
            dtype=torch.long
        )
        row_ids = Df_indices.unsqueeze(1)

        col_ids = torch.arange(
            T_out,
            dtype=torch.long
        ).unsqueeze(0)

        global_times = (
            row_ids
            + num_timesteps_input
            + col_ids
        )

        global_times_flat = global_times.reshape(-1)

        node_outputs = project_output[
            :,
            faulty_node_idx,
            :,
            :
        ]

        # [B, T_out, F]
        outputs_flat = node_outputs.reshape(
            -1,
            F
        )

        surrogates_by_node = {}
        for node in faulty_nodes.tolist():
            # (B, T_out, F) -> (B*T_out, F)
            node_outputs = project_output[:, node, :, :]
            outputs_flat = node_outputs.reshape(-1, F)

            node_surrogate = []

            for start, end in forget_indices:
                interval_mask = ((global_times_flat >= start) & (global_times_flat < end))

                times = global_times_flat[interval_mask]
                outputs = outputs_flat[interval_mask]

                if times.numel() == 0:
                    print(
                        f"[WARNING] No prediction found for node {node}, "
                        f"interval [{start}, {end})"
                    )
                    continue

                unique_times = torch.sort(torch.unique(times)).values

                seg = torch.zeros(
                    F,
                    unique_times.numel(),
                    dtype=outputs.dtype
                )

                for idx, t in enumerate(unique_times):
                    seg[:, idx] = outputs[times == t].mean(dim=0)

                node_surrogate.append(seg)

            surrogates_by_node[node] = node_surrogate

        return surrogates_by_node        

    def add_error_minimizing_noise(self, data: list, 
                                  forget_indices: Union[int, list],
                                  noise_scale: float = 0.01) -> torch.Tensor:
        """
        Add imperceptible error-minimizing noise to promote unlearning
        Inspired by unlearnable examples literature
        What this will do is make the model less sensitive to the forgotten patterns because it introduces noise in a controlled manner.
        """      

        noisy_data = []

        for i, seq in enumerate(data):
            seq_copy = seq.clone() if isinstance(seq, torch.Tensor) else torch.tensor(seq)

            if i in forget_indices:
                # Thêm noise Gaussian cùng shape
                noise = torch.randn_like(seq_copy) * noise_scale
                seq_copy = seq_copy + noise

                # Optionally: smooth theo chiều thời gian (axis=-1)
                if seq_copy.ndim >= 2:
                    kernel = torch.ones(3, device=seq_copy.device) / 3.0
                    padding = (1,)  # same padding cho conv1d
                    # reshape về (batch=features, channel=1, length=time)
                    smoothed = torch.nn.functional.conv1d(
                        seq_copy.unsqueeze(1), 
                        kernel.view(1, 1, -1), 
                        padding=padding
                    )
                    seq_copy = smoothed.squeeze(1)

            noisy_data.append(seq_copy)

        return noisy_data
    

    def perform_temporal_generative_replay_subset(self, model: nn.Module, 
                                         forget_input,
                                         forget_indices: Union[int, list],
                                         faulty_node_idx: int,
                                         num_timesteps_input,
                                         num_timesteps_output,
                                         Df_indices,
                                         device,
                                         A_wave: Optional[torch.Tensor] = None) -> list:
        """
        Main T-GR function implementing Reconstruction and Neutralization
        
        Params:
            model: The model being unlearned
            forget_input: Input of faulty node
            forget_indices: Indices to be neutralized
            faulty_node_idx: Index of faulty node
            num_timesteps_input: number of sample input
            num_timesteps_output: number of sample output
            Df_indices: Indices of the forget input
            device,
            A_wave: Graph edges (for STGCN)
            
        Returns:
            Neutralized surrogate data
        """
        
        surrogate_sample = self.surrogate(model, forget_input, forget_indices, faulty_node_idx, num_timesteps_input, num_timesteps_output, Df_indices, device, A_wave)

        # [(F, T)]
        

        surrogate_sample = self.add_error_minimizing_noise(
            surrogate_sample[faulty_node_idx[0]], forget_indices, noise_scale=1.5 # Change accordingly to fit the desired 
        )
        # [(F, T)]
        return [i.transpose(0, 1) for i in surrogate_sample]
    
    def perform_temporal_generative_replay_node(self, model: nn.Module,
                                         forget_input,
                                         forget_indices,
                                         faulty_node_idx: int,
                                         num_timesteps_input,
                                         num_timesteps_output,
                                         Df_indices,
                                         device,
                                         impacts,
                                         k, 
                                         A_wave: Optional[torch.Tensor] = None) -> list:
        """
        Main T-GR function implementing Reconstruction and Neutralization
        
        Params:
            model: The model being unlearned
            forget_input: Input of faulty node
            forget_indices: Indices to be neutralized
            faulty_node_idx: Index of faulty node
            num_timesteps_input: number of sample input
            num_timesteps_output: number of sample output
            Df_indices: Indices of the forget input
            device,
            impacts: Impact scores for each node when removing the faulty node
            k: Number of top nodes to consider for neutralization
            A_wave: Graph edges (for STGCN)
            
        Returns:
            Neutralized surrogate data
        """

        alpha = 0.8
        beta = 0.1

        # +1: feature increases when traffic becomes more congested
        # -1: feature decreases when traffic becomes more congested
        # s = torch.tensor([1, -1])
        s = torch.tensor([1, -1], dtype=torch.float32, device=device)

        impacts = np.array(impacts) 
        impact_tensor = torch.as_tensor(impacts, dtype=torch.float32, device=device)

        # Get the top-k positive(the same with delete node(decrease load)) and negative(contrast with delete node(increase load)) nodes based on impact
        negative_nodes = np.argsort(impacts)[:k]
        positive_nodes = np.argsort(impacts)[-(k+len(faulty_node_idx)):]

        # remove faulty node if somehow included
        faulty_nodes = np.asarray(faulty_node_idx, dtype=int).reshape(-1)

        negative_nodes = negative_nodes[~np.isin(negative_nodes, faulty_nodes)]
        positive_nodes = positive_nodes[~np.isin(positive_nodes, faulty_nodes)]

        surrogates = []
        forget_nodes = []

        origin_surrogate = self.surrogate(model, forget_input, forget_indices, faulty_node_idx, num_timesteps_input, num_timesteps_output, Df_indices, device, A_wave)

        for node in positive_nodes:
            node_data = []

            for faulty_node in faulty_node_idx:
                for sur in origin_surrogate[int(faulty_node)]:
                    surrogate = torch.tensor(sur.T, dtype=torch.float32, device=device)

                    impact_strength = impact_tensor[node]
                    delta = alpha * impact_strength
                    noise_std = beta * abs(impact_strength)
                    noise = torch.randn_like(surrogate) * noise_std

                    surrogate = surrogate - delta * s + noise
                    node_data.append(surrogate)

            surrogates.append(node_data)
            forget_nodes.append(node)

        for node in negative_nodes:
            node_data = []
            
            for faulty_node in faulty_node_idx:
                for sur in origin_surrogate[int(faulty_node)]:
                    surrogate = torch.tensor(sur.T, dtype=torch.float32, device=device)

                    impact_strength = impact_tensor[node]
                    delta = alpha * impact_strength
                    noise_std = beta * abs(impact_strength)
                    noise = torch.randn_like(surrogate) * noise_std

                    surrogate = surrogate - delta * s + noise
                    node_data.append(surrogate)

                surrogates.append(node_data)
                forget_nodes.append(node)

        return surrogates, list(set(forget_nodes))