from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class CriticAdapter:
    """
    Lightweight critic adapter for GT-AE and S2F-DAE training.

    Responsibilities
    ----------------
    - Take DNA, a teacher profile-shape (counts), and one or more student profile-shapes (probs).
    - Globally equalize each student's total counts to the teacher (per track channel).
    - Build critic inputs (value + mask) for a single MLM head.
    - Run the critic once for teacher (no grad) and once per student (with grad).
    - Return MLM logits + embeddings for teacher and each student.

    This adapter is agnostic to masking regime: the caller is responsible for
    constructing `critic_input_mask` (e.g. DNA-visible vs DNA-absent) and calling this
    module once per regime.
    """

    def __init__(
        self,
        *,
        critic: nn.Module,
        head_name: str = "GT",
    ):
        super().__init__()
        self.critic = critic
        self.head_name = str(head_name)

    # ---------- helpers ----------

    @staticmethod
    def _global_total_equalize(
        student: torch.Tensor,
        teacher: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Scale `student` so its total counts match `teacher` over the full window (per track channel).

        Parameters
        ----------
        student : (B, L, T)
            Student track (typically probabilities over positions or unnormalized counts).
        teacher : (B, L, T)
            Teacher track (counts).
        eps : float
            Numerical stability constant.

        Returns
        -------
        student_eq : (B, L, T)
            Student track scaled so sum_L student_eq == sum_L teacher (per track channel).
        """
        if student.shape != teacher.shape:
            raise ValueError(
                f"_global_total_equalize shape mismatch: "
                f"student={student.shape}, teacher={teacher.shape}"
            )

        student_sum = student.sum(dim=1, keepdim=True)  # (B, 1, T)
        teacher_sum = teacher.sum(dim=1, keepdim=True)  # (B, 1, T)
        scale = (teacher_sum + eps) / (student_sum + eps)  # (B, 1, T)
        return student * scale

    @staticmethod
    def _build_critic_input(
        dna: torch.Tensor,           # (B, L, 4)
        track: torch.Tensor,         # (B, L, T)
        critic_input_mask: torch.Tensor,    # (B, L, 4+T)
    ) -> torch.Tensor:
        """
        Build critic input tensor (B, 2, L, 4+T) from DNA, track, and mask.

        Geometry:
        ---------
        - values: [dna(4), tracks(T)]  -> (B, L, 4+T)
        - mask  : [dna_mask(4), track_mask(T)] -> (B, L, 4+T)
        - stacked as:
              x[:, 0] = masked values (value * (1 - mask))
              x[:, 1] = mask (1.0 = masked, 0.0 = visible)
        """
        if dna.dim() != 3:
            raise ValueError(f"dna must be (B, L, 4), got {dna.shape}")
        if track.dim() != 3:
            raise ValueError(f"track must be (B, L, T), got {track.shape}")
        if critic_input_mask.dim() != 3:
            raise ValueError(f"critic_input_mask must be (B, L, 4+T), got {critic_input_mask.shape}")

        B, L, C_dna = dna.shape
        if C_dna != 4:
            raise ValueError(f"Expected dna channels=4, got {C_dna}")

        B_t, L_t, T = track.shape
        if B_t != B or L_t != L:
            raise ValueError(
                f"track shape {track.shape} incompatible with dna {(B, L, 4)}"
            )

        B_m, L_m, C_m = critic_input_mask.shape
        if B_m != B or L_m != L:
            raise ValueError(
                f"critic_input_mask shape {critic_input_mask.shape} incompatible with dna {(B, L, 4)}"
            )
        if C_m != 4 + T:
            raise ValueError(
                f"critic_input_mask channels {C_m} != 4 + T ({4+T}); "
                f"expected mask over [dna(4), profile_shape({T})]"
            )

        values = torch.cat([dna, track], dim=-1)         # (B, L, 4+T)
        if values.shape != critic_input_mask.shape:
            raise ValueError(
                f"values shape {values.shape} != critic_input_mask {critic_input_mask.shape}"
            )

        # mask=1.0 => masked; zero out those values
        masked_values = values.masked_fill(critic_input_mask > 0, 0.0)  # (B, L, 4+T)
        x = torch.stack([masked_values, critic_input_mask], dim=1)      # (B, 2, L, 4+T)
        return x

    @staticmethod
    def _to_BLF(logits: torch.Tensor) -> torch.Tensor:
        """
        Ensure critic head logits are (B, L, 4).
        Accepts (B, L, 4) or (B, 4, L).
        """
        if logits.dim() != 3:
            raise ValueError(f"Critic head logits must be 3D, got {logits.shape}")
        B, A, C = logits.shape
        if C == 4:
            # (B, L, 4)
            return logits
        if A == 4:
            # (B, 4, L) -> (B, L, 4)
            return logits.permute(0, 2, 1).contiguous()
        raise ValueError(
            f"Unexpected logits shape {logits.shape}; expected (B, L, 4) or (B, 4, L)"
        )

    @staticmethod
    def _to_BLC(features: torch.Tensor, L: int) -> torch.Tensor:
        """
        Ensure features are (B, L_feat, C) for some L_feat.
        Accepts (B, C, L_feat) or (B, L_feat, C) and infers which dim is length.
        The `L` argument is kept for backward compatibility but is not used.
        """
        if features.dim() != 3:
            raise ValueError(f"Features must be 3D, got {features.shape}")

        B, C0, L0 = features.shape

        # Heuristic: treat the larger of (C0, L0) as the length dimension.
        # For typical conv features we have C0 << L0, so features are (B, C, L).
        if L0 >= C0:
            # (B, C, L0) -> (B, L0, C)
            return features.transpose(1, 2).contiguous()
        else:
            # (B, L0, C)
            return features

    # ---------- main API ----------

    def compute_logits_and_embeddings(
        self,
        *,
        dna: torch.Tensor,           # (B, L, 4)
        teacher_track: torch.Tensor, # (B, L, T) GT track (counts)
        student_tracks: Dict[str, torch.Tensor], # dict[name] -> (B, L, T) student tracks (probs)
        critic_input_mask: torch.Tensor,  # (B, L, 4+T) mask over [DNA, profile_shape]
    ) -> Dict[str, torch.Tensor]:
        """
        Run critic on teacher (no grad) and one or more student tracks (with grad),
        returning MLM logits and embeddings needed for loss construction.

        Parameters
        ----------
        dna : (B, L, 4)
            One-hot DNA.
        teacher_track : (B, L, T)
            Ground-truth experimental track (counts).
        student_tracks : dict[str, (B, L, T)]
            Mapping from a student name to a student track (probabilities).
        critic_input_mask : (B, L, 4+T)
            Input mask, 1.0 for masked positions, 0.0 for visible, over [DNA(4), profile_shape(T)].

        Returns
        -------
        dict with keys:
            - "teacher__logits"      : (B, L, 4)
            - "teacher__track_embed" : (B, L, C)
            - "student__{name}__logits"      : (B, L, 4)
            - "student__{name}__track_embed" : (B, L, C)
        """
        if not isinstance(student_tracks, dict) or len(student_tracks) == 0:
            raise ValueError("student_tracks must be a non-empty dict[name -> (B,L,T) tensor].")

        B, L, T = teacher_track.shape
        if dna.shape[0] != B or dna.shape[1] != L:
            raise ValueError(
                f"dna shape {dna.shape} incompatible with track shape {(B, L, T)}"
            )
        if critic_input_mask.shape[0] != B or critic_input_mask.shape[1] != L:
            raise ValueError(
                f"critic_input_mask shape {critic_input_mask.shape} incompatible with track shape {(B, L, T)}"
            )

        # Validate student tracks and equalize totals to teacher per (B,T).
        student_eq: Dict[str, torch.Tensor] = {}
        for name, track in student_tracks.items():
            if track.shape != teacher_track.shape:
                raise ValueError(
                    f"student_tracks['{name}'] shape {track.shape} != teacher_track shape {teacher_track.shape}"
                )
            student_eq[name] = self._global_total_equalize(track, teacher_track)

        # ----- Teacher path (no grad) -----
        x_teacher = self._build_critic_input(dna, teacher_track, critic_input_mask)
        with torch.no_grad():
            track_teacher, _, logits_teacher_dict = (
                self.critic.forward_with_stem_features(
                    x_teacher, heads=[self.head_name]
                )
            )
        logits_teacher = self._to_BLF(logits_teacher_dict[self.head_name])
        # (B, C, L) -> (B, L, C)
        track_teacher_BLC = self._to_BLC(track_teacher, L)
        del x_teacher, track_teacher, logits_teacher_dict

        out: Dict[str, torch.Tensor] = {
            "teacher__logits": logits_teacher,           # (B, L, 4)
            "teacher__track_embed": track_teacher_BLC,   # (B, L, C_trk)
        }

        # ----- Student paths (with grad) -----
        for name, track_eq in student_eq.items():
            x_student = self._build_critic_input(dna, track_eq, critic_input_mask)
            track_student, _, logits_student_dict = (
                self.critic.forward_with_stem_features(
                    x_student, heads=[self.head_name]
                )
            )
            logits_student = self._to_BLF(logits_student_dict[self.head_name])
            track_student_BLC = self._to_BLC(track_student, L)
            del x_student, track_student, logits_student_dict

            out[f"student__{name}__logits"] = logits_student           # (B, L, 4)
            out[f"student__{name}__track_embed"] = track_student_BLC   # (B, L, C_trk)

        return out
