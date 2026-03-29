import torch
from torch.distributions.categorical import Categorical

class BaseMask:
    def __init__(self):
        """Stateless base; subclasses define their own parameters.
        No implicit global 'mask_percentage'.
        """
        pass

    def sample_mask(self, shape):
        """
        Sample a mask of the given shape from the distribution
        """
        raise NotImplementedError("Must implement sample_mask.")
    
    def apply_mask(self, data, mask):
        """
        Apply mask by stacking mask and data into (2, L, T_plus_4) format
        
        Args:
            data: (L, T_plus_4) float tensor
            mask: (L, T_plus_4) float tensor (1.0 = masked, 0.0 = unmasked)
        
        Returns:
            input_tensor: (2, L, T_plus_4) stacked as [mask, value] along dim 0
        """
        # apply mask to data
        masked_data = data * (1 - mask)
        # Stack mask and value: [value, mask]
        return torch.stack([masked_data, mask], dim=0)
    
    def sample_and_apply_mask(self, data):
        """
        Sample mask and create input tensor in format: (2, L, T_plus_4)
        
        Args:
            data: (L, T_plus_4) original data tensor
        
        Returns:
            input_tensor: (2, L, T_plus_4) stacked as [mask, value] along dim 0
        """
        # Sample mask
        mask = self.sample_mask(data.shape)
        
        # Apply mask (returns stacked tensor)
        return self.apply_mask(data, mask), mask
    
