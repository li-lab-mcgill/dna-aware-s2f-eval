import torch
import torch.nn as nn
import torch.nn.functional as F


def multimodal_masked_unet_input_transform_fn(
        X_tracks,
        X_seq,
        mask,
        clm_input_len=1000
    ) -> torch.Tensor:
    """
    Form CLM input
    
    Args:
        X_tracks (torch.tensor): Observed track data, shape (B, L, T)
        X_seq (torch.tensor): Central sequence input, shape (B, L, 4)
        mask (torch.tensor): Mask for LM input, shape (B, clm_input_len, 4+T)
        clm_input_len (int): Length of CLM input sequence
        
    Returns:
        torch.tensor: Predicted track data formatted for 
            CLM input (B, 2, clm_input_len, 4+T)
    """

    # Symmetric crop to tracks if needed
    if X_tracks.shape[1] > clm_input_len:
        crop_width = (predicted_track.shape[1] - clm_input_len) // 2
        predicted_track = predicted_track[:, crop_width:-crop_width, :]  
        # (B, clm_input_len, T)

    # Symmetric crop to seq if needed
    if X_seq.shape[1] > clm_input_len:
        crop_width = (X_seq.shape[1] - clm_input_len) // 2
        X_seq = X_seq[:, crop_width:-crop_width, :]  
        # (B, clm_input_len, 4)

    # Concatenate sequence and track data
    X_lm = torch.cat([X_seq, X_tracks], dim=-1)
    # (B, clm_input_len, 4+T) 

    # multiply by mask to zero out unmasked positions
    X_lm_masked = X_lm * (1 - mask)  
    # (B, clm_input_len, 4+T)
    X_lm_masked = torch.stack([X_lm_masked, mask], dim=1)  
    # (B, 2, clm_input_len, 4+T)

    # return transposed version for CLM input
    return X_lm_masked


def bpnetlite_output_transform_fn(
        s2f_outputs,
    ) -> torch.Tensor:
    """
    Transform S2F model outputs to get predicted track data for CLM input.
    
    Args:
        s2f_outputs (tuple): Outputs from S2F model, expected to contain:
            - pred_log1p_counts: Predicted log1p counts (B, T=1)
            - pred_profile_logits: Predicted profile logits (B, L_s2f, T=1)
    
    Returns:
        torch.tensor: Predicted track data formatted for CLM input (B, L_lm, T=1)
    """
    # Unpack S2F outputs
    pred_profile_logits, pred_log1p_counts = s2f_outputs

    # Handle different output formats
    if pred_profile_logits.dim() == 2:
        # Shape is (B, L) not (B, T, L) - add track dimension
        pred_profile_logits = pred_profile_logits.unsqueeze(1)  # (B, 1, L)
    
    if pred_log1p_counts.dim() == 1:
        # Shape is (B,) not (B, T) - add track dimension  
        pred_log1p_counts = pred_log1p_counts.unsqueeze(1)  # (B, 1)

    # Apply softmax to profile logits along sequence dimension  
    profile_probs = F.softmax(pred_profile_logits, dim=2)  # (B, T, L)

    # Apply inverse log1p transform: exp(x) - 1, then clamp at 0
    total_counts = torch.clamp(torch.exp(pred_log1p_counts) - 1, min=0.0)  # (B, T)

    # construct the predicted track
    total_counts_expanded = total_counts.unsqueeze(2) # (B, T, L)
    predicted_track = profile_probs * total_counts_expanded  # (B, T, L)

    # return transposed version for CLM input
    return predicted_track.transpose(1, 2)  # (B, clm_input_len, T)


class cgLM_S2F_pipeline(nn.Module):
    """
    Wrapper model that combines S2F and CLM models for knowledge distillation.

    The model takes masked data as input and produces both gtCLM and pCLM outputs:
    - Teacher branch: Uses observed track data with CLM
    - Student branch: Uses S2F predicted track data with CLM
    """

    def __init__(self, s2f_model, teacher_clm_model, s2f_output_transform_fn, clm_input_transform_fn,
                 compute_grad_teacher=True):
        """
        Args:
            s2f_model: Pre-trained S2F model that predicts tracks from DNA sequences
            teacher_clm_model: Pre-trained CLM model (frozen) for knowledge distillation
            s2f_output_transform_fn: Function that transforms S2F outputs to CLM-compatible format
            clm_input_transform_fn: Function that prepares CLM input from (mask, dna_seq, track_data)
            compute_grad_teacher: Whether to build autograd graph through teacher (default True)
        """
        super(cgLM_S2F_pipeline, self).__init__()

        self.s2f_model = s2f_model
        self.teacher_clm_model = teacher_clm_model
        self.s2f_output_transform_fn = s2f_output_transform_fn
        self.clm_input_transform_fn = clm_input_transform_fn
        self.compute_grad_teacher = compute_grad_teacher

        # Ensure teacher model is frozen
        for param in self.teacher_clm_model.parameters():
            param.requires_grad = False
        self.teacher_clm_model.eval()

    def forward(self, x_s2f_inputs, x_lm_seq, x_lm_obs_track, mask_lm):
        """
        Forward pass combining S2F and CLM models.

        Args:
            x_s2f_inputs (tuple):
                - seq (torch.tensor):
                    Central sequence input + context for S2F
                    Shape (B, L_s2f, 4) or (B, 4, L_s2f)
                - Control tracks (torch.tensor) [Optional] for ChIP-seq:
                    Control track data for S2F, if any
                - Unroll to apply
            x_lm_seq (torch.tensor):
                Central sequence input for CLM, shape (B, L_lm=1000, 4)
            x_lm_obs_track (torch.tensor):
                Observed track input, shape (B, L_lm, T=1)
            mask_lm (torch.tensor):
                Mask for LM input: (B, L_lm, 4+T)

        Returns:
            tuple: (s2f_outputs, y_lm_out_obs, y_lm_out_pred)
                - s2f_outputs: Raw outputs from S2F model
                - y_lm_out_obs: Teacher CLM output on observed track
                    (B, L_lm, 4+T) or (B, L_lm, 4)
                - y_lm_out_pred: Teacher CLM output on predicted track
                    (B, L_lm, 4+T) or (B, L_lm, 4)
        """
        # Step 1: Run S2F model on DNA sequence input
        s2f_outputs = self.s2f_model(*x_s2f_inputs)
        s2f_outputs = (s2f_outputs,) if not isinstance(s2f_outputs, tuple) else s2f_outputs

        # Step 2: Transform S2F outputs to get predicted track data
        # The transform function handles tuple indexing and format conversion
        x_lm_pred_track = self.s2f_output_transform_fn(s2f_outputs)

        # Step 3: Prepare CLM input for student branch (predicted track)
        x_lm_pred_masked = self.clm_input_transform_fn(x_lm_pred_track, x_lm_seq, mask_lm)

        # Step 4: Prepare CLM input for teacher branch (observed track)
        x_lm_obs_masked = self.clm_input_transform_fn(x_lm_obs_track, x_lm_seq, mask_lm)

        # Step 5: Run teacher CLM model on both prepared inputs
        if self.compute_grad_teacher:
            # Teacher branch: observed track
            y_lm_out_obs = self.teacher_clm_model(x_lm_obs_masked)  # (B, L_lm, 4+T)
            # Teacher branch: predicted track
            y_lm_out_pred = self.teacher_clm_model(x_lm_pred_masked)  # (B, L_lm, 4+T)
        else:
            with torch.no_grad():
                # Teacher branch: observed track
                y_lm_out_obs = self.teacher_clm_model(x_lm_obs_masked)  # (B, L_lm, 4+T)
                # Teacher branch: predicted track
                y_lm_out_pred = self.teacher_clm_model(x_lm_pred_masked)  # (B, L_lm, 4+T)

        return s2f_outputs, y_lm_out_obs, y_lm_out_pred
