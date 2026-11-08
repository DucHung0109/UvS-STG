""" Run the 3 Orders from the Initial Models using files from the unlearning folder
- Combine Components: Load the pre-trained model ($θ*$).
- Partition Data: Use your PEPA implementation to get $D_f$ and $D_r$.
- Calculate FIM: Compute the PA-FIM ($F^T$) using $D_r$ and your PA-EWC module. """
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import copy
import argparse
import json
# Components
from models.stgcn import STGCN
from models.stgat import STGAT
from models.gwn import gwnet
from utils.data_loader import load_data
from utils.filter_forget import filter_forget
from unlearning.pa_ewc import PopulationAwareEWC
from unlearning.t_gr import TemporalGenerativeReplay
from unlearning.motif_def import discover_motifs_proxy, find_forget
from utils.VAR import VAR_IRF
from utils.data_preprocess import get_normalized_adj, generate_dataset
from utils.replace_surrogate import replace_target

class SATimeSeries:
    """Complete SA-TS Framework Integration"""
    def __init__(self, model, A_wave, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # --- Store the original model state for evaluation ---
        self.original_model = copy.deepcopy(model).to(self.device)
        self.original_model.eval()
        
        self.model = copy.deepcopy(model).to(self.device) # This is the model that will be modified during unlearning
        self.new_A_wave = copy.deepcopy(A_wave).to(self.device)
        # Ensure model parameters are float32
        for param in self.model.parameters():
            param.data = param.data.float()
        
        # Store original parameters for EWC
        self.original_params = {}
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.data.clone().float()
        
        # Initialize components
        self.pa_ewc = PopulationAwareEWC("stgcn", device)
        self.t_gr = TemporalGenerativeReplay("stgcn")
        self.fim_diagonal = None
        
    def unlearn_faulty_subset(self, dataset, forget_ex, faulty_node_idx, A_wave, means, stds, 
                           num_timesteps_input, num_timesteps_output, threshold=0.1, num_epochs=50, learning_rate=5e-5, 
                           lambda_ewc=10.0, lambda_surrogate=1.0, lambda_retain=1.0, lambda_forget=1.0, forget_margin=0.1, batch_size=128):
        """
        Main unlearning process for faulty node
        """
        print(f"Starting unlearning subset for faulty node {faulty_node_idx}")
        
        # Ensure dataset is float32
        dataset = dataset.astype(np.float32)
        train_input, train_target = generate_dataset(dataset, num_timesteps_input, num_timesteps_output)
        nums_window = train_input.shape[0]
        window_starts = torch.arange(nums_window)
        forget_ex = forget_ex.astype(np.float32)
        
        # --- Partition data
        forget_indices, retain_indices = discover_motifs_proxy(
            dataset, forget_ex, faulty_node_idx, threshold
        )
        print(f"Forget samples: {len(forget_indices)}, Retain samples: {len(retain_indices)}")
        print(forget_indices)
        
        global forget_loader, retain_loader

        if not forget_indices:
            print("No forget samples found to unlearn. Skipping training.")
            forget_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0))) # Empty dataloader
            # Create a loader with all training data for retain_loader as nothing is forgotten
            retain_loader = DataLoader(TensorDataset(train_input, train_target), batch_size=batch_size, shuffle=True)
            # Same 3-tuple shape as the trained path so callers can unpack safely.
            return {'total_loss': [], 'surrogate_loss': [], 'ewc_penalty': [], 'retain_loss': []}, retain_loader, forget_loader

        retain_input, retain_target, Dr_indices, forget_input, forget_target, Df_indices = filter_forget(train_input, train_target, forget_indices, num_timesteps_input, num_timesteps_output)

        if not retain_indices:
            retain_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=batch_size, shuffle=True)
        else:
            retain_loader = DataLoader(TensorDataset(retain_input, retain_target), batch_size=batch_size, shuffle=True)

        forget_loader = DataLoader(TensorDataset(forget_input, forget_target), batch_size=batch_size, shuffle=True)
        print("TARGET: ", forget_target.shape)
                
        # --- Compute FIM
        print("Computing Population-Aware FIM...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.original_model, retain_loader, A_wave, max_samples=500
        )

        # --- Create surrogate data using T-GR
        print("Creating surrogate data...")
        surrogate_data = self.t_gr.perform_temporal_generative_replay_subset(
            self.original_model, forget_input, forget_indices, faulty_node_idx,
            num_timesteps_input, num_timesteps_output, Df_indices, self.device, A_wave
        )
        
        forget_target = replace_target(forget_target, [surrogate_data], Df_indices, forget_indices, [faulty_node_idx], num_timesteps_input, num_timesteps_output)
        surrogate_loader = DataLoader(TensorDataset(forget_input, forget_target), batch_size=batch_size, shuffle=True)
        
        history = self._training_loop(surrogate_loader, retain_loader, forget_loader, num_epochs, learning_rate, lambda_ewc, lambda_surrogate, 
                                      lambda_retain, lambda_forget, forget_margin, faulty_node_idx=faulty_node_idx, mask_faulty_in_retain=False)

        return history, retain_loader, forget_loader
    
    
    def _isolate_extension(self, faulty_node_idx):
        """Isolate the faulty node in the graphs the model keeps internally:
        GWNet's fixed supports and learned adjacency, STGAT's cached edge list."""

        if hasattr(self.model, "isolate_nodes"):
            self.model.isolate_nodes(faulty_node_idx)

        if hasattr(self.model, "edge_index"):
            self.model.edge_index = None   # rebuilt from new_A_wave on the next forward
                

    def unlearn_faulty_node(self, dataset, faulty_node_idx, A_wave, means, stds,
                            num_timesteps_input, num_timesteps_output, threshold = 0.5, num_epochs=100, learning_rate=1e-4,
                            lambda_ewc=5.0, lambda_surrogate=1.0, lambda_retain=2.0, lambda_forget=1.0, forget_margin=0.5, batch_size=128):
        """Node unlearning: remove a sensor's influence from the model."""
        print(f"Starting node unlearning for node {faulty_node_idx}")
        dataset = dataset.astype(np.float32)
        print(f"Dataset shape: {dataset.shape}")
        # --- 1. Isolate the faulty node in the adjacency BEFORE any forward pass ---
        print(f"Isolating node {faulty_node_idx} in adjacency...")
        self.new_A_wave[faulty_node_idx, :] = 0.0
        self.new_A_wave[:, faulty_node_idx] = 0.0
        self._isolate_extension(faulty_node_idx)

        # --- 2. Build full-graph training windows ---
        train_input, train_target = generate_dataset(dataset, num_timesteps_input, num_timesteps_output)

        forget_indices, retain_indices = find_forget(dataset, faulty_node_idx, threshold)
        print(f"Forget samples: {len(forget_indices)}, Retain samples: {len(retain_indices)}")
        print(forget_indices)
        global forget_loader, retain_loader

        if not forget_indices:
            print("No forget samples found to unlearn. Skipping training.")
            forget_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0))) # Empty dataloader
            # Create a loader with all training data for retain_loader as nothing is forgotten
            retain_loader = DataLoader(TensorDataset(train_input, train_target), batch_size=batch_size, shuffle=True)
            return {'total_loss': [], 'surrogate_loss': [], 'ewc_penalty': [], 'retain_loss': []}, retain_loader, forget_loader
        
        retain_input, retain_target, Dr_indices, forget_input, forget_target, Df_indices = filter_forget(train_input, train_target, forget_indices, num_timesteps_input, num_timesteps_output)

        if not retain_indices:
            retain_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=batch_size, shuffle=True)
        else:
            retain_loader = DataLoader(TensorDataset(retain_input, retain_target), batch_size=batch_size, shuffle=True)
            
        forget_loader = DataLoader(TensorDataset(forget_input, forget_target), batch_size=batch_size, shuffle=True)

        # --- Compute FIM
        print("Computing Population-Aware FIM...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(self.original_model, retain_loader, A_wave, max_samples=500)

        impacts = VAR_IRF(dataset, faulty_node_idx)

        # --- Create surrogate data using T-GR
        print("Creating surrogate data...")
        surrogate_datas, forget_nodes = self.t_gr.perform_temporal_generative_replay_node(
            self.original_model, forget_input, forget_indices, faulty_node_idx, 
            num_timesteps_input, num_timesteps_output, Df_indices, self.device, impacts, k=5, A_wave=A_wave
        )

        forget_target = replace_target(forget_target, surrogate_datas, Df_indices, forget_indices, forget_nodes, num_timesteps_input, num_timesteps_output)
        surrogate_loader = DataLoader(TensorDataset(forget_input, forget_target), batch_size=batch_size, shuffle=True)

        history = self._training_loop(surrogate_loader, retain_loader, forget_loader, num_epochs, learning_rate, lambda_ewc, lambda_surrogate, 
                                      lambda_retain, lambda_forget, forget_margin, faulty_node_idx=faulty_node_idx, mask_faulty_in_retain=True)

        return history, retain_loader, forget_loader


    def _training_loop(self, surrogate_loader, retain_loader, forget_loader_tr, num_epochs, learning_rate, 
                       lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget, forget_margin, faulty_node_idx=None, mask_faulty_in_retain=False):
        """
        Shared loop for subset + node unlearning.

        Objective:
            L = λ_surr·L_surr + λ_retain·L_retain + λ_ewc·L_ewc
                + λ_forget·max(0, margin − L_forget)     ← BOUNDED ascent

        The bounded forget term stops pushing once L_forget exceeds the
        margin, preventing catastrophic collapse on MSE regression.
        """
        print("\n[TRAIN] starting unified unlearning loop")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mse = nn.MSELoss()
        history = {
            "total_loss": [], "surrogate_loss": [], "retain_loss": [],
            "forget_loss": [], "forget_term": [], "ewc_penalty": [],
        }

        new_A = self.new_A_wave.to(self.device)
        N = None

        for epoch in range(num_epochs):
            self.model.train()
            stats = {k: 0.0 for k in history}
            n_batches = 0

            forget_iter = iter(forget_loader_tr) if forget_loader_tr is not None else None

            for (surr_X, surr_y), (ret_X, ret_y) in zip(surrogate_loader, retain_loader):
                optimizer.zero_grad()

                surr_X = surr_X.float().to(self.device)
                surr_y = surr_y.float().to(self.device)
                ret_X = ret_X.float().to(self.device)
                ret_y = ret_y.float().to(self.device)

                if N is None:
                    N = surr_X.shape[1]

                # ---- Surrogate loss ----
                surr_pred = self.model(new_A, surr_X)
                if faulty_node_idx is not None and surr_pred.shape[1] > 1 and surr_y.shape[1] == 1:
                    surr_pred = surr_pred[:, faulty_node_idx:faulty_node_idx + 1, :, :]
                l_surr = mse(surr_pred, surr_y)

                # ---- Retain loss ----
                ret_pred = self.model(new_A, ret_X)
                if mask_faulty_in_retain and faulty_node_idx is not None:
                    mask = torch.ones(N, dtype=torch.bool, device=self.device)
                    mask[faulty_node_idx] = False
                    l_retain = mse(ret_pred[:, mask, :, :], ret_y[:, mask, :, :])
                else:
                    l_retain = mse(ret_pred, ret_y)

                # ---- Bounded forget term ----
                l_forget = torch.tensor(0.0, device=self.device)
                l_forget_term = torch.tensor(0.0, device=self.device)
                if forget_iter is not None:
                    try:
                        fgt_X, fgt_y = next(forget_iter)
                    except StopIteration:
                        forget_iter = iter(forget_loader_tr)
                        fgt_X, fgt_y = next(forget_iter)

                    fgt_X = fgt_X.float().to(self.device)
                    fgt_y = fgt_y.float().to(self.device)
                    fgt_pred = self.model(new_A, fgt_X)
                    # Slice BOTH prediction and target to the faulty node so
                    # L_forget (and hence the margin m) is a per-node quantity;
                    # a full-graph mean dilutes the signal ~1/N.  Guard on >1 so
                    # an already single-node target is never sliced out of range.
                    if faulty_node_idx is not None and fgt_pred.shape[1] > 1:
                        fgt_pred = fgt_pred[:, faulty_node_idx, :, :]
                        if fgt_y.shape[1] > 1:
                            fgt_y = fgt_y[:, faulty_node_idx, :, :]

                    l_forget = mse(fgt_pred, fgt_y)
                    # Margin loss: push L_forget up to `margin`, then stop.
                    l_forget_term = torch.clamp(forget_margin - l_forget, min=0.0)

                # ---- EWC penalty ----
                l_ewc = torch.tensor(0.0, device=self.device)
                if self.fim_diagonal is not None:
                    l_ewc = self.pa_ewc.apply_ewc_penalty(
                        self.model, self.fim_diagonal, self.original_params, lambda_ewc
                    )

                total = (
                    lambda_surrogate * l_surr
                    + lambda_retain * l_retain
                    + lambda_ewc * l_ewc
                    + lambda_forget * l_forget_term
                )

                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                stats["total_loss"] += total.item()
                stats["surrogate_loss"] += l_surr.item()
                stats["retain_loss"] += l_retain.item()
                stats["forget_loss"] += l_forget.item()
                stats["forget_term"] += l_forget_term.item()
                stats["ewc_penalty"] += l_ewc.item()
                n_batches += 1

            if n_batches > 0:
                for k in history:
                    history[k].append(stats[k] / n_batches)

            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1}/{num_epochs}  "
                      f"total={history['total_loss'][-1]:.4f}  "
                      f"surr={history['surrogate_loss'][-1]:.4f}  "
                      f"retain={history['retain_loss'][-1]:.4f}  "
                      f"L_forget={history['forget_loss'][-1]:.4f}  "
                      f"forget_term={history['forget_term'][-1]:.4f}  "
                      f"ewc={history['ewc_penalty'][-1]:.4f}")

        # ---- Post-training diagnostic: did the forget term ever fire? ----
        if forget_loader_tr is not None and history["forget_loss"]:
            max_lf = max(history["forget_loss"])
            final_lf = history["forget_loss"][-1]
            final_term = history["forget_term"][-1]
            print(f"\n[DIAG] forget-set MSE  max={max_lf:.4f}  final={final_lf:.4f}  "
                  f"margin={forget_margin}  final_term={final_term:.4f}")
            if max_lf < forget_margin:
                print(f"  WARNING: L_forget never reached margin ({forget_margin}). "
                      f"Bounded ascent stayed active the whole run but produced no "
                      f"forgetting above the target. Consider:")
                print(f"    - raising --lambda-forget (current contribution may be "
                      f"drowned by EWC / retain / surrogate terms)")
                print(f"    - lowering --lambda-ewc (EWC pins params near original)")
                print(f"    - lowering --forget-margin toward a realistic target "
                      f"(e.g. 2-3x the original forget MSE)")
            elif final_term > 0.0:
                print(f"  NOTE: forget term still non-zero at end — more epochs may "
                      f"continue pushing L_forget up.")
            else:
                print(f"  OK: L_forget reached margin and the bounded term saturated.")

        return history

def unlearn(model, A_wave, train_original_data, means, stds, num_timesteps_input, num_timesteps_output, forget_set=None):
    sa_ts = SATimeSeries(model, A_wave, args.device)
    
    if args.unlearn_node:
        history, retain_loader, forget_loader = sa_ts.unlearn_faulty_node(
            train_original_data, args.node_idx, A_wave, means, stds,
            num_timesteps_input, num_timesteps_output,threshold=args.threshold_node, num_epochs=args.num_epochs, learning_rate=args.learning_rate,
            lambda_ewc=args.lambda_ewc, lambda_surrogate=args.lambda_surrogate, lambda_retain=args.lambda_retain,
            lambda_forget=args.lambda_forget, forget_margin=args.forget_margin, batch_size=args.batch_size
        )
    else:
        history, retain_loader, forget_loader = sa_ts.unlearn_faulty_subset(
            train_original_data, forget_set, args.node_idx, A_wave, means, stds,
            num_timesteps_input, num_timesteps_output, threshold=args.threshold_subset, num_epochs=args.num_epochs, learning_rate=args.learning_rate,
            lambda_ewc=args.lambda_ewc, lambda_surrogate=args.lambda_surrogate, lambda_retain=args.lambda_retain,
            lambda_forget=args.lambda_forget, forget_margin=args.forget_margin, batch_size=args.batch_size
        )


    if args.unlearn_node:
        path = args.model + f"/Unlearn node {args.node_idx}"
    else:
        path = args.model + f"/Unlearn subset on node {args.node_idx}"
        
    if not os.path.exists(path):
        os.makedirs(path)
    save_dict = {
        "model_state_dict": sa_ts.model.state_dict(),
        "config": sa_ts.model.config
    }
    torch.save(save_dict, path + f"/unlearn_{args.type}_model.pt")

    print("Unlearning completed!")


def main():
    """Main execution function"""
    torch.cuda.empty_cache()
    # Load data
    print("Loading data...")
    A, train_original_data, test_original_data, means, stds = load_data(args.data_full)

    print(train_original_data.shape)
    # N, F, T
    means = means.astype(np.float32)
    stds = stds.astype(np.float32)
    forget_set = None
    if args.forget_set and not args.unlearn_node:
        with open(args.forget_set, 'r', encoding='utf8') as f:
            forget_set_json = json.load(f)
        for key, value in forget_set_json.items():
            # Only 1 element(subset on 1 node)
            node_idx = int(key)
            for item in value:
                forget_set = train_original_data[node_idx, 0, item[0]:item[1]]

            args.node_idx = [node_idx]
            break
    
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float().to(args.device)


    if not args.all:
        checkpoint = torch.load(args.model + f"/{args.type}_model.pt", map_location=args.device)
        config = checkpoint["config"]
        num_timesteps_input = config["nums_step_in"]
        num_timesteps_output = config["nums_step_out"]
        
        if args.type == 'stgcn':
            model_class = STGCN
        elif args.type == 'stgat':
            model_class = STGAT
        elif args.type == 'gwnet':
            model_class = gwnet
            config['device'] = args.device

        model = model_class(**config).to(args.device)
        model.load_state_dict({k: v.float() for k, v in checkpoint["model_state_dict"].items()})

        unlearn(model, A_wave, train_original_data, means, stds, num_timesteps_input, num_timesteps_output, forget_set)

    else:
        list_models = ['stgcn', 'stgat', 'gwnet']
        
        for model_name in list_models:
            args.type = model_name

            print(f"UNLEARNING ON MODEL: {model_name}")
            args.type = model_name
            
            # Load original model
            print("Loading original model...")
            checkpoint = torch.load(args.model + f"/{args.type}_model.pt", map_location=args.device)
            config = checkpoint["config"]
            
            if args.type == 'stgcn':
                model_class = STGCN
            elif args.type == 'stgat':
                model_class = STGAT
            elif args.type == 'gwnet':
                model_class = gwnet
                config['device'] = args.device
            else:
                raise ValueError(f"Unknown model type: {args.type}")

            num_timesteps_input = config["nums_step_in"]
            num_timesteps_output = config["nums_step_out"]

            model = model_class(**config).to(args.device)
            model.load_state_dict({k: v.float() for k, v in checkpoint["model_state_dict"].items()})

            unlearn(model, A_wave, train_original_data, means, stds, num_timesteps_input, num_timesteps_output, forget_set)

            del model
            torch.cuda.empty_cache()   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unlearning')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--unlearn-node', action='store_true', help='Enable unlearn node')
    parser.add_argument('--all', action='store_true', help='Unlearning all model')
    parser.add_argument('--node-idx', type=int, nargs='+', default=[], help='Node index need to be unlearned')
    parser.add_argument('--data-full', type=str, required=True, help='Path to the directory containing dataset')
    parser.add_argument('--data-standard', type=str, required=False, help='Path to the directory containing standard dataset')
    parser.add_argument('--type', type=str, required=False, help='Type of model')
    parser.add_argument('--model', type=str, required=True, help='Path to the directory containing weights of origin model')
    parser.add_argument('--threshold-subset', type=float, default=5, help='Threshold for subset unlearning')
    parser.add_argument('--threshold-node', type=float, default=0.1, help='Threshold for node unlearning')
    parser.add_argument('--forget-set', type=str, help='Path to the directory containing forget dataset')
    parser.add_argument('--lambda-ewc', type=float, default=5.0, help='Weight for EWC loss')
    parser.add_argument('--lambda-surrogate', type=float, default=1.0, help='Weight for surrogate loss')
    parser.add_argument('--lambda-retain', type=float, default=1.0, help='Weight for retain loss')
    parser.add_argument('--lambda-forget', type=float, default=1.0, help='Weight for forget loss')
    parser.add_argument('--forget-margin', type=float, default=0.1, help='Margin for forget loss')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs for unlearning')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate for unlearning')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for unlearning')
    args = parser.parse_args()

    if args.unlearn_node:
        if args.forget_set is not None:
            print("Warning: --forget_set will be ignored when --unlearn-node is enabled.")
    else:
        if args.forget_set is None:
            parser.error("--forget_set is required unless --unlearn-node is specified.")

    args.device = None
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    main()

