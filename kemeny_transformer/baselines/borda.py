import numpy as np

def borda_from_ranking_position(base_rankings: np.ndarray) -> np.ndarray:
    """
    Computes Borda Count where the input values are the ranking positions
    of the candidates (i.e., base_rankings[voter, candidate_id] = rank_position).
    
    Returns a consensus ranking of the same format:
    consensus_ranking[candidate_id] = rank_position.
    """
    n_voters, n_candidates = base_rankings.shape
    
    # Points for candidate c from voter v is: (n_candidates - 1) - rank_position
    # Summing over all voters:
    scores = n_voters * (n_candidates - 1) - np.sum(base_rankings, axis=0)
    
    # Sort candidates by score descending (using stable sort to resolve ties deterministically)
    sorted_candidates = np.argsort(-scores, kind='stable')
    
    # Map candidate IDs back to their rank positions
    consensus_ranking = np.argsort(sorted_candidates, kind='stable')
    return consensus_ranking


def borda_from_candidate(base_rankings: np.ndarray) -> np.ndarray:
    """
    Computes Borda Count where the input values are the candidate IDs
    themselves (i.e., base_rankings[voter, rank_position] = candidate_id).
    
    Returns a consensus ranking where the values represent the ranking position
    of each candidate (i.e., consensus_ranking[candidate_id] = rank_position).
    """
    n_voters, n_candidates = base_rankings.shape
    
    unique_candidates = np.unique(base_rankings)
    num_unique = len(unique_candidates)
    
    # Fast path if candidate IDs are exactly 0 to num_unique - 1
    if np.array_equal(unique_candidates, np.arange(num_unique)):
        scores = np.zeros(num_unique, dtype=np.float64)
        for pos in range(n_candidates):
            points = n_candidates - 1 - pos
            np.add.at(scores, base_rankings[:, pos].astype(np.intp), points)
        sorted_candidates = np.argsort(-scores, kind='stable')
        consensus_ranking = np.argsort(sorted_candidates, kind='stable')
        return consensus_ranking
        
    # General path for arbitrary candidate IDs
    cand_to_idx = {cand: idx for idx, cand in enumerate(unique_candidates)}
    mapped_rankings = np.vectorize(cand_to_idx.get)(base_rankings)
    
    scores = np.zeros(num_unique, dtype=np.float64)
    for pos in range(n_candidates):
        points = n_candidates - 1 - pos
        np.add.at(scores, mapped_rankings[:, pos], points)
        
    sorted_indices = np.argsort(-scores, kind='stable')
    consensus_ranking_mapped = np.argsort(sorted_indices, kind='stable')
    
    try:
        max_cand = int(np.max(unique_candidates))
        if max_cand < 100000:
            consensus_ranking = np.zeros(max_cand + 1, dtype=np.int32)
            for cand, idx in cand_to_idx.items():
                consensus_ranking[cand] = consensus_ranking_mapped[idx]
            return consensus_ranking
    except (ValueError, TypeError):
        pass
        
    return {cand: int(consensus_ranking_mapped[idx]) for cand, idx in cand_to_idx.items()}


def borda_from_ranking_position_batch(batch_rankings) -> np.ndarray:
    """
    Batch version of borda_from_ranking_position.

    Accepts either:
      - a 3D numpy array of shape (n_instances, n_voters, n_candidates), or
      - a list/sequence of 2D arrays (each n_voters x n_candidates).

    Values are ranking positions: base_rankings[voter, candidate_id] = rank_position.

    Returns a 2D numpy array of shape (n_instances, n_candidates) where
    result[i, candidate_id] = rank_position in the i-th consensus ranking.

    Instances must all share the same n_candidates (voter counts may differ).
    """
    # Fast path: uniform shapes -> single 3D array, fully vectorized.
    if isinstance(batch_rankings, np.ndarray) and batch_rankings.ndim == 3:
        arr = batch_rankings
    else:
        instances = [np.asarray(r) for r in batch_rankings]
        shapes = {r.shape for r in instances}
        arr = np.stack(instances) if len(shapes) == 1 else None

    if arr is not None:
        n_instances, n_voters, n_candidates = arr.shape
        # Points per voter for candidate c = (n_candidates - 1) - rank_position.
        scores = n_voters * (n_candidates - 1) - arr.sum(axis=1)        # (B, C)
        sorted_candidates = np.argsort(-scores, axis=1, kind='stable')  # (B, C)
        consensus = np.argsort(sorted_candidates, axis=1, kind='stable')
        return consensus

    # General path: ragged instances (e.g. differing voter counts per instance).
    results = [borda_from_ranking_position(np.asarray(r)) for r in batch_rankings]
    return np.stack(results)


def borda_from_candidate_batch(batch_rankings) -> np.ndarray:
    """
    Batch version of borda_from_candidate.

    Accepts a list/sequence of 2D arrays (or a 3D array), where each instance is
    base_rankings[voter, rank_position] = candidate_id.

    Returns a 2D numpy array of shape (n_instances, n_candidates) where
    result[i, candidate_id] = rank_position in the i-th consensus ranking.

    Note: each instance is processed independently. If any instance uses
    non-contiguous candidate IDs that fall back to a dict result, the instances
    cannot be stacked and a ValueError is raised.
    """
    results = []
    for r in batch_rankings:
        consensus = borda_from_candidate(np.asarray(r))
        if isinstance(consensus, dict):
            raise ValueError(
                "borda_from_candidate returned a dict for an instance with "
                "non-contiguous candidate IDs; results cannot be stacked into an array."
            )
        results.append(consensus)
    return np.stack(results)

