import torch.nn as nn
import torch.nn.functional as F
import torch as th
from src.algos.common.utils import mlp


class DistributionalCritic(nn.Module):
    def __init__(
            self, 
            obs_dim, 
            act_dim, 
            num_bins: int = 51, 
            vmin = -10, 
            vmax = 10, 
            net_arch: tuple[int, ...] = (512, 512),
    ):
        super().__init__()
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.delta = (vmax - vmin) / num_bins  # bin width
        # Pre-compute bin centers as a buffer (shape: [num_bins])
        self.register_buffer('z', th.linspace(vmin + self.delta/2, vmax - self.delta/2, num_bins))
        self.net = mlp([obs_dim + act_dim] + list(net_arch) + [num_bins], nn.ReLU, nn.Identity)
        
    def forward(self, obs, act):
        # Concatenate observation and action
        x = th.cat([obs, act], dim=-1)
        logits = self.net(x)  # shape: [batch_size, num_bins]
        return logits
    
    def get_q_value(self, logits):
        # Convert logits to a probability distribution via softmax
        probs = F.softmax(logits, dim=-1)
        # Compute the expected value (Q) as the dot product with the bin centers
        q = th.sum(probs * self.z, dim=-1, keepdim=True)
        return q

def scalar_to_categorical_target(y, critic, sigma):
    """
    y: TD target (shape: [batch_size, 1])
    critic: an instance of DistributionalCritic (to get self.delta and self.z)
    sigma: standard deviation for HL-Gauss smoothing
    Returns: target probabilities of shape [batch_size, num_bins]
    """
    # Expand y to match the bin shape
    delta = critic.delta
    z = critic.z.unsqueeze(0)  # shape: [1, num_bins]
    
    # Calculate the boundaries for integration
    lower_bound = z - delta / 2
    upper_bound = z + delta / 2
    
    # Create a normal distribution centered at y
    normal = th.distributions.Normal(y, sigma)
    
    # Compute CDF values at the boundaries (ensure proper broadcasting)
    cdf_upper = normal.cdf(upper_bound)
    cdf_lower = normal.cdf(lower_bound)
    
    # The target probability mass for each bin:
    p_target = cdf_upper - cdf_lower
    # (Optional: renormalize in case of numerical issues)
    p_target = p_target / (p_target.sum(dim=1, keepdim=True) + 1e-8)
    
    return p_target  # shape: [batch_size, num_bins]

