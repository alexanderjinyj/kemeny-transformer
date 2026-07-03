import numpy as np
from time import time
from numba import njit


@njit()
def kemeny_dist_paralle(base_rankings, candidate_ranking):
    """
    Compute average Kemeny distance (Kendall tau distance) between a candidate ranking
    and a set of base rankings, averaged over all voters.

    Args:
        base_rankings: numpy array of shape (nb_voters, nb_candidates)
        candidate_ranking: numpy array of shape (nb_candidates,)
    Returns:
        kemeny_dist: float, average Kemeny distance
    """
    nb_voters = base_rankings.shape[0]
    nb_candidates = base_rankings.shape[1]
    kemeny_dist = 0
    for i in range(nb_voters):
        tau = 0
        base_ranking = base_rankings[i]
        for j in range(nb_candidates):
            for k in range(j+1, nb_candidates):
                if (np.sign(candidate_ranking[j] - candidate_ranking[k]) == -np.sign(base_ranking[j] - base_ranking[k])):
                    tau += 1
        kemeny_dist += tau
    kemeny_dist = kemeny_dist / nb_voters
    return kemeny_dist


def compute_kemeny_distance_parallel(batch_base_rankings, final_rankings):
    """
    Compute Kemeny distances for a batch of rankings.

    Args:
        batch_base_rankings: numpy array of shape (bsz, n_voters, n_candidates)
        final_rankings: numpy array of shape (bsz, n_candidates)
    Returns:
        kemeny_distances: numpy array of shape (bsz,)
    """
    bsz = batch_base_rankings.shape[0]
    kemeny_distances = np.empty(bsz)
    for i in range(bsz):
        base_rankings = batch_base_rankings[i]
        final_ranking = final_rankings[i]
        kemeny_distance = kemeny_dist_paralle(base_rankings, final_ranking)
        kemeny_distances[i] = kemeny_distance
    return kemeny_distances


def compute_kemeny_distance_parallel_greedy(batch_base_rankings, final_rankings):
    """
    Compute Kemeny distances for greedy-decoded permutations (2D arrays).

    Args:
        batch_base_rankings: numpy array of shape (bsz, n_voters, n_candidates)
        final_rankings: numpy array of shape (bsz, n_candidates)
    Returns:
        kemeny_distances: numpy array of shape (bsz,)
    """
    bsz = batch_base_rankings.shape[0]
    kemeny_distances = np.empty(bsz)
    for i in range(bsz):
        base_rankings = batch_base_rankings[i]
        final_ranking = final_rankings[i]
        kemeny_distance = kemeny_dist_paralle(base_rankings, final_ranking)
        kemeny_distances[i] = kemeny_distance
    return kemeny_distances


def compute_kemeny_distance_parallel_beam_search(batch_base_rankings, final_rankings_beam_search):
    """
    Compute Kemeny distances for beam search results (3D arrays with multiple hypotheses),
    selecting the best ranking per sample.

    Args:
        batch_base_rankings: numpy array of shape (bsz, n_voters, n_candidates)
        final_rankings_beam_search: numpy array of shape (bsz, beam_size, n_candidates)
    Returns:
        kemeny_distances: numpy array of shape (bsz,)
        final_rankings: numpy array of shape (bsz, n_candidates) - best ranking per sample
    """
    print(f'final rankings shape: {final_rankings_beam_search.shape}')
    print(f'base rankings shape: {batch_base_rankings.shape}')
    bsz = final_rankings_beam_search.shape[0]
    beam_size = final_rankings_beam_search.shape[1]
    ranking_size = batch_base_rankings.shape[2]
    print(f'size:{bsz},{beam_size},{ranking_size}')
    kemeny_distances = np.empty(bsz)
    final_rankings = np.empty(shape=(bsz, ranking_size))
    for i in range(bsz):
        base_rankings = batch_base_rankings[i]
        final_ranking_beam_search = final_rankings_beam_search[i]
        kemeny_distance_beam_search = np.empty(beam_size)
        for j in range(beam_size):
            final_ranking = final_ranking_beam_search[j]
            kemeny_distance_beam_search[j] = kemeny_dist_paralle(base_rankings, final_ranking)
        idx_minimal_kemeny_distance = np.argmin(kemeny_distance_beam_search)
        kemeny_distance = kemeny_distance_beam_search[idx_minimal_kemeny_distance]
        final_ranking_optimal = final_ranking_beam_search[idx_minimal_kemeny_distance]
        kemeny_distances[i] = kemeny_distance
        final_rankings[i] = final_ranking_optimal
    return kemeny_distances, final_rankings


def permutation_to_ranking_greedy(batch_permutation):
    """Convert batch permutation to ranking via argsort (axis=1)."""
    ranking = np.argsort(batch_permutation, axis=1)
    return ranking


def permutation_to_ranking_beam_search(batch_permutation):
    """Convert batch permutation to ranking via argsort (axis=2, for beam dim)."""
    ranking = np.argsort(batch_permutation, axis=2)
    return ranking
