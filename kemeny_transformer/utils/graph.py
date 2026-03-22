import numpy as np
from numba import njit
from itertools import combinations


@njit()
def build_graph_margin(ranks):
    """
    Construct an edge-weighted graph from voter rankings using preference comparisons.
    Stores the net margin (difference) for each pair - only the winner's margin is stored.

    Used by gurobi solver for Kemeny optimal ranking computation.

    Args:
        ranks: numpy array of shape (n_voters, n_candidates)
    Returns:
        edge_weights: numpy array of shape (n_candidates, n_candidates) with net margins
    """
    n_voters, n_candidates = ranks.shape
    edge_weights = np.zeros((n_candidates, n_candidates))
    for i in range(n_candidates):
        for j in range(i+1, n_candidates):
            preference = ranks[:, i] - ranks[:, j]
            h_ij = np.sum(preference < 0)  # prefers i to j
            h_ji = np.sum(preference > 0)  # prefers j to i
            if h_ij > h_ji:
                edge_weights[i, j] = h_ij - h_ji
            elif h_ij < h_ji:
                edge_weights[j, i] = h_ji - h_ij
    return edge_weights


def build_graph_counts(rankings):
    """
    Construct an edge-weighted graph from voter rankings storing raw preference counts
    for both directions (not just the margin).

    Used by KwikSort algorithm for tournament graph construction.

    Args:
        rankings: numpy array of shape (n_voters, n_candidates)
    Returns:
        edge_weights: numpy array of shape (n_candidates, n_candidates) with raw counts
    """
    n_voters, size_candidate = rankings.shape
    edge_weights = np.zeros((size_candidate, size_candidate))
    for i, j in combinations(range(size_candidate), 2):
        preference = rankings[:, i] - rankings[:, j]
        preference_i_j = np.sum(preference < 0)  # prefers i to j
        preference_j_i = np.sum(preference > 0)  # prefers j to i
        edge_weights[i, j] = preference_i_j
        edge_weights[j, i] = preference_j_i
    return edge_weights
