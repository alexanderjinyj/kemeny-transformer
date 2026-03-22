
from time import time

import numpy as np

def aggregate_rank_mc(rankings):
    precedence_matrix=build_precedence_matrix(rankings)
    stationary_distribution = stationary_distribute(rankings,precedence_matrix)
    temp_ranking = np.argsort(stationary_distribution)
    aggregate_ranking = np.empty_like(temp_ranking)
    aggregate_ranking[temp_ranking] = np.arange(len(stationary_distribution))[::-1]
    return aggregate_ranking


def stationary_distribute(rankings,precedence_matrix):
    transition_matrix = []
    transition_matrix = (1-0.05)*generate_transition_matrix(rankings,precedence_matrix)  + 0.05/(rankings.shape[1])
    # transition_matrix = transition_matrix*(1-alpha)+(alpha/transition_matrix.shape[0])
    transition_matrix_trans = transition_matrix.T
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix_trans)
    close_to_1_idx = np.isclose(eigenvalues,1, rtol=0.001)
    target_eigenvector = eigenvectors[:,close_to_1_idx]
    target_eigenvector = target_eigenvector[:,0]
# Turn the eigenvector elements into probabilities
    stationary_distribute = target_eigenvector / sum(target_eigenvector)
    return stationary_distribute




def generate_transition_matrix(rankings,precedence_matrix):
    total_pairs = rankings.shape[0] * rankings.shape[1]
    # Use vectorized transpose and division
    transition_matrix = precedence_matrix.T / float(total_pairs)
    # Zero out diagonal
    np.fill_diagonal(transition_matrix, 0.0)
    # Give remaining probabilities to diagonal
    np.fill_diagonal(transition_matrix, 1.0 - np.sum(transition_matrix, axis=1))
    return transition_matrix

def build_precedence_matrix(rankings):
    # Use array broadcasting to compute all combinations at once: (voters, candidates, 1) - (voters, 1, candidates)
    preference = rankings[:, :, None] - rankings[:, None, :]
    return np.sum(preference < 0, axis=0)