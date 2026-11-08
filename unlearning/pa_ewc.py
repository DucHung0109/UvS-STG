import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

class PopulationAwareEWC:
    """
    Population-Aware Elastic Weight Consolidation (PA-EWC).
    Diagonal Fisher information of the original model on the retain set D_r
    (per-sample MSE gradients, clipped, each tensor normalised by its own mean),
    used as a quadratic anchor that keeps the edited parameters near theta*.
    """
    
    def __init__(self, model_type: str = "stgcn", device: str = "cuda"):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

# ------------------- FIM Calculation -------------------
    def calculate_pa_fim(self, model: nn.Module, retain_data_loader: DataLoader,
                        A_wave: torch.Tensor, max_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Calculate diagonal of Population-Aware FIM
        """
        model.eval()
        
        fim_diagonal = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fim_diagonal[name] = torch.zeros_like(param, device=self.device, dtype=torch.float32)
        
        sample_count = 0
        loss_fn = nn.MSELoss(reduction='mean')
        dead_grads = set()  # param names that never received a gradient

        try:
            for data_batch in retain_data_loader:
                if sample_count >= max_samples:
                    break

                if isinstance(data_batch, (list, tuple)):
                    X_batch, y_batch = data_batch
                else:
                    X_batch = data_batch
                    y_batch = data_batch

                # Ensure float32
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                A_wave = A_wave.float().to(self.device)


                model.zero_grad()

                for i in range(X_batch.shape[0]):
                    if sample_count >= max_samples:
                        break

                    # Single sample forward pass
                    X_single = X_batch[i:i+1].float()
                    y_single = y_batch[i:i+1].float()

                    output = model(A_wave, X_single)

                    loss = loss_fn(output, y_single)
                    loss.backward()

                    # Clip gradients for FIM stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            fim_diagonal[name] += param.grad.data.pow(2)
                        else:
                            dead_grads.add(name)

                    model.zero_grad()  # Clear gradients after each sample
                    sample_count += 1

            if sample_count > 0:
                for name in fim_diagonal:
                    fim_diagonal[name] /= sample_count
                    # Normalize FIM to prevent tiny values
                    fim_diagonal[name] = fim_diagonal[name] / (fim_diagonal[name].mean() + 1e-8)
                    fim_diagonal[name] += 1e-8  # Small constant for stability
            else:
                print("FIM calc - No samples processed")
                for name in fim_diagonal:
                    fim_diagonal[name].fill_(1e-6)

            if dead_grads:
                # Dead params (never received a grad) get FIM=0, so EWC does not
                # protect them. This is expected for gwnet's residual_convs and
                # the final-layer bn/gconv (see models/gwn.py forward pass).
                preview = sorted(dead_grads)
                shown = preview[:6]
                suffix = f" ... (+{len(preview) - len(shown)} more)" if len(preview) > len(shown) else ""
                print(f"[FIM] {len(dead_grads)} param(s) had no gradient (unused in "
                      f"forward); EWC will not constrain them: {shown}{suffix}")
        
        except Exception as e:
            print(f"Error in FIM calculation: {e}")
            for name in fim_diagonal:
                fim_diagonal[name].fill_(1e-6)

        return fim_diagonal
    
    def apply_ewc_penalty(self, model: nn.Module, 
                         fim_diagonal: Dict[str, torch.Tensor],
                         original_params: Dict[str, torch.Tensor],
                         lambda_ewc: float = 10.0) -> torch.Tensor:
        """
        Apply the EWC penalty term
        """
        penalty = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        for name, param in model.named_parameters():
            if name in fim_diagonal and name in original_params:
                diff = (param - original_params[name]).pow(2)
                weighted_diff = fim_diagonal[name] * diff
                penalty += weighted_diff.sum()
        
        penalty = lambda_ewc * penalty / 2.0 
        return penalty
    