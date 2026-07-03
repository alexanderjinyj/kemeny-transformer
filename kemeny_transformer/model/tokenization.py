import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
from kemeny_transformer.model.architecture import kemeny_transformer, EmbeddingType

class KemenyTransformerTokenization:
    """
    A class to handle the tokenization and batch processing for the KemenyTransformer model.

    This class encapsulates the two-step padding process required to convert a list of
    base rankings arrays (with variable voters and candidates) into a single padded tensor
    and its corresponding attention mask.

    Args:
        max_voters (int): The exact number of features (voters) to pad to. This
            value MUST match the input dimension of the model's embedding layer.
            Ignored when embedding_type is "conv".
        pad_value (float): The value to use for all padding. Defaults to 0.0.
    """
    def __init__(self, max_voters: int, pad_value: float = 0.0, normalize_input: bool = False):
        self.max_voters = max_voters
        self.pad_value = pad_value
        self.normalize_input = normalize_input


    def tokenize(self, batch_base_rankings: List[np.ndarray], embedding_type: EmbeddingType = "linear") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a raw batch of Kemeny rankings into a padded tensor and mask.

        Args:
            batch_base_rankings (List[np.ndarray]): A list where each element is a NumPy array
                representing a set of base rankings. Each array can have a different
                shape of (num_voters, num_candidates).
            embedding_type (str): The type of embedding ("linear" or "conv").
                When "conv", voter dimension padding is skipped. Defaults to "linear".

        Returns:
            - padded_batch (torch.Tensor): A single tensor for the entire batch.
              Shape: (batch_size, max_num_candidates, max_voters) for linear.
              Shape: (batch_size, max_num_candidates, num_voters) for conv.
            - padding_mask (torch.Tensor): A boolean mask for the sequence padding.
              Shape: (batch_size, max_num_candidates).
            - voter_mask (torch.Tensor): A boolean mask for the feature (voter) padding.
              Shape: (batch_size, max_features).

        Raises:
            ValueError: If embedding_type is "linear" and any item has more voters than `self.max_voters`.
        """
        if not batch_base_rankings:
            return torch.empty(0), torch.empty(0), torch.empty(0)

        # Handle both string and EmbeddingType enum
        if hasattr(embedding_type, 'value'):
            embedding_type = embedding_type.value  # Extract string from enum
        embedding_type = embedding_type.lower()


        # --- Step 1: Transpose and collect rankings ---
        # We treat candidates as the sequence and voters as features.
        # Shape after transpose: (num_candidates, num_voters)
        # NORMALIZE: Divide by (num_candidates - 1) to scale ranks to [0, 1]
        processed_rankings = []
        for rank_set in batch_base_rankings:
            num_candidates = rank_set.shape[1]
            if self.normalize_input:
                # Avoid division by zero for single-candidate case (unlikely but safe)
                scale_factor = max(1, num_candidates - 1)
                normalized_ranks = rank_set.T / scale_factor
                processed_rankings.append(torch.from_numpy(normalized_ranks).float())
            else:
                 processed_rankings.append(torch.from_numpy(rank_set.T).float())

        # --- Step 2: Pad the feature (voter) dimension (linear only) ---
        if embedding_type == "conv":
            # For conv embedding, skip voter padding - conv layers handle variable input sizes

            feature_padded_rankings = processed_rankings
        else:
            # For linear embedding, pad each tensor's feature dimension to max_voters

            feature_padded_rankings = []
            for base_rankings in processed_rankings:
                num_voters = base_rankings.size(1)
                if num_voters > self.max_voters:
                    raise ValueError(
                        f"An item in the batch has {num_voters} voters, which is more than "
                        f"the allowed max_voters of {self.max_voters}. This would result in data loss."
                    )

                # `pad` works on the last dimension first, so (0, max_voters - num_voters)
                # pads the second dimension (dim=1).
                padded_r = F.pad(base_rankings, (0, self.max_voters - num_voters), "constant", self.pad_value)
                feature_padded_rankings.append(padded_r)

        # --- Step 3: Pad the sequence (candidate) dimension ---
        # Now that the feature dimension is uniform, we can use the original function
        # to pad the sequence length.
        return pad_rankings_and_create_mask(feature_padded_rankings, self.pad_value)


def pad_rankings_and_create_mask(
    batch_rankings: List[torch.Tensor],
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads a batch of variable-length rankings (sequence dimension) and creates a mask.
    (This is the second stage of the preprocessing pipeline).
    """
    if not batch_rankings:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    # Determine max sequence length and max feature dimension
    batch_size = len(batch_rankings)
    max_len = max(base_rankings.size(0) for base_rankings in batch_rankings) # Max num of candidates
    max_features = max(base_rankings.size(1) for base_rankings in batch_rankings) # Max num of voters

    # Initialize the tensors for the padded data and the mask
    padded_batch = torch.full((batch_size, max_len, max_features), pad_value, dtype=batch_rankings[0].dtype)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    voter_mask = torch.zeros((batch_size, max_features), dtype=torch.bool)

    # Copy the data from the original tensors into the padded tensors
    for i, base_rankings in enumerate(batch_rankings):
        num_candidates = base_rankings.size(0)
        num_voters = base_rankings.size(1)
        padded_batch[i, :num_candidates, :num_voters] = base_rankings
        padding_mask[i, :num_candidates] = False
        voter_mask[i, :num_voters] = True

    return padded_batch, padding_mask, voter_mask
