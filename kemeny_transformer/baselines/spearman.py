import numpy as np
from scipy.optimize import linear_sum_assignment

def footrule_consensus(base_rankings):
    """
    Spearman footrule-optimal consensus ranking via
    minimum-cost bipartite perfect matching.

    Parameters
    ----------
    base_rankings : np.ndarray, shape (m, n)
        m base rankings over n items.
        base_rankings[k, i] = position of item i in the k-th ranking.

    Returns
    -------
    consensus : np.ndarray, shape (n,)
        consensus[i] = position assigned to item i.
    """
    R = np.asarray(base_rankings)
    m, n = R.shape

    # Detect indexing format (0-indexed or 1-indexed) automatically
    min_rank = np.min(R)
    positions = np.arange(min_rank, min_rank + n)


    # cost[i, p] = sum_k |p - R[k, i]|  (total footrule cost of putting item i at position p)
    cost = np.abs(positions[None, None, :] - R[:, :, None]).sum(axis=0)  # shape (n_items, n_positions)

    item_ind, pos_ind = linear_sum_assignment(cost)
    consensus = np.empty(n, dtype=int)
    consensus[item_ind] = positions[pos_ind]
    return consensus


def footrule_consensus_batch(batch_rankings) -> np.ndarray:
    """
    Batch version of footrule_consensus.

    Accepts a list/sequence of 2D arrays (or a 3D array), where each instance is
    base_rankings[voter, item] = position of that item in the voter's ranking.

    Returns a 2D numpy array of shape (n_instances, n_items) where
    result[i, item] = position assigned to that item in the i-th consensus.

    Each instance is an independent optimal-assignment problem and is solved
    separately; instances may differ in voter count but must share n_items.
    """
    results = [footrule_consensus(r) for r in batch_rankings]
    return np.stack(results)
