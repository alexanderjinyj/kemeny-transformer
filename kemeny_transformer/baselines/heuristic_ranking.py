import numpy as np

class HeuristicConsensusRanker:
    """
    A class to calculate consensus rankings using heuristics from the 1983 paper
    by Beck and Lin, "Some Heuristics for the Consensus Ranking Problem".

    Attributes:
        agreement_matrix (np.ndarray): The matrix where A[i, j] is the number
                                       of raters preferring object i to object j.
        num_objects (int): The number of objects to be ranked.
    """

    def __init__(self, base_rankings=None,batch_base_rankings=None,agreement_matrix=None):
        """
        Initializes the ConsensusRanker with an agreement matrix.

        Args:
            agreement_matrix (list or np.ndarray): A square matrix where A[i, j]
                                                  is the number of raters
                                                  preferring object i to object j.
        """
        self.base_ranking=base_rankings
        self.agreement_matrix=agreement_matrix
        self.batch_base_ranking=batch_base_rankings

    def maximize_agreement(self):
        """
        computer the consensus ranking by agrement matrix.
        first computer the agreement matrix by methods agreement_matrix()function
        then compute the positive P and nagative N
        get max(|P[i]-N[i]|) if P[i]-N[i]>0 assign to top else assign to bottom
        Returns: list of indices representing ranking from top (best) to bottom (worst),value is the candidate.
        """
        # Work on a copy of the matrix
        if self.base_ranking is None and self.agreement_matrix is None:
            raise ValueError("neither base rankings nor agreement_matrix")

        if self.agreement_matrix is None and self.base_ranking is not None:
            self.agreement_matrix=self.build_agreement_matrix(self.base_ranking)
        agreement_matrix=self.agreement_matrix
        num_objects =self.agreement_matrix.shape[0]
        remaining = np.arange(num_objects)
        top_slots = []
        bottom_slots = []

        while remaining.size > 0:
            #print(f"{remaining.size=}")

            # Create a submatrix of the remaining objects to calculate P and N
            sub_agreement_matrix =agreement_matrix[np.ix_(remaining, remaining)]

            # Calculate Positive (P) and Negative (N) preference vectors
            P = np.sum(sub_agreement_matrix, axis=1)
            N = np.sum(sub_agreement_matrix, axis=0)
            #find the max abs diff
            diff = P - N
            abs_diff = np.abs(diff)
            max_idx = np.argmax(abs_diff)
            #if the diff[max_idx] >0 assign it on the top else assign it on the bottom
            if diff[max_idx]>0:
                top_slots.append(remaining[max_idx])
            else:
                bottom_slots.append(remaining[max_idx])
            #delet max_idx from remaining
            remaining = np.delete(remaining, max_idx)
        #
        return top_slots + bottom_slots[::-1]

    def minimize_regret(self):
        """
        Calculates a consensus ranking using the Minimize Regret Heuristic.

        This method implements the algorithm described in Section 4 of the paper.
        It iteratively selects the object that minimizes "rater regret" to place
        at the top of the remaining ranks.

        Returns:
            list: A list of object indices representing the consensus ranking
                  from best (rank 1) to worst.
        """
        # Work on a copy of the matrix
        if self.base_ranking is None and self.agreement_matrix is None:
            raise ValueError("neither base rankings nor agreement_matrix")

        if self.agreement_matrix is None and self.base_ranking is not None:
            self.agreement_matrix=self.build_agreement_matrix(self.base_ranking)
        agreement_matrix=self.agreement_matrix
        regret_matrix=self.get_regret_matrix_from_agreement()
        n_candidate = regret_matrix.shape[0]
        remaining = np.arange(n_candidate)
        consesus_ranking=[]
        while remaining.size > 0:
            sub_matrix_idx=np.ix_(remaining, remaining)
            sub_agreement_matrix=agreement_matrix[sub_matrix_idx]
            sub_regret_matrix = regret_matrix[sub_matrix_idx]
            #Compute P and N from the sliced sub-matrices.
            P = np.sum(sub_agreement_matrix, axis=1)
            N = np.sum(sub_regret_matrix, axis=0) # Sum of columns of the regret sub-matrix
            diff = P - N
            max_idx = np.argmax(diff)
            consesus_ranking.append(remaining[max_idx])
            remaining = np.delete(remaining, max_idx)

        return consesus_ranking



    def build_regret_matrix(self, base_rankings):
        num_voters,num_objects = base_rankings.shape
        edge_weights = np.zeros((num_objects, num_objects))
        for i in range(num_objects):
            for j in range(i+1,num_objects):
                preference = base_rankings[:, i] - base_rankings[:, j]
                h_ij = np.sum(preference < 0)  # prefers i to j
                h_ji = np.sum(preference > 0)  # prefers j to i
                if h_ij > h_ji:
                    edge_weights[i, j] = h_ij - h_ji
                elif h_ij < h_ji:
                    edge_weights[j, i] = h_ji - h_ij
        return edge_weights

    def build_agreement_matrix(self, base_rankings):
        num_voters,num_objects = base_rankings.shape
        edge_weights = np.zeros((num_objects, num_objects))
        for i in range(num_objects):
            for j in range(i+1,num_objects):
                preference = base_rankings[:, i] - base_rankings[:, j]
                h_ij = np.sum(preference < 0)  # prefers i to j
                h_ji = np.sum(preference > 0)  # prefers j to i
                edge_weights[i, j] = h_ij
                edge_weights[j, i] = h_ji
        return edge_weights

    def get_regret_matrix_from_agreement(self):
        """
        Computes and returns the full regret matrix from the agreement matrix.

        The regret matrix R is defined such that for any pair of objects (i, j):
        - R[i, j] = a_ij - a_ji, if a_ij > a_ji
        - R[j, i] = a_ji - a_ij, if a_ji >= a_ij
        - Otherwise, the value is 0.

        This represents the "cost" or "regret" of ranking one object over another
        against the group's preference.

        Returns:
            np.ndarray: The computed regret matrix.
        """
        if self.agreement_matrix is None:
             raise ValueError("neither base rankings nor agreement_matrix")
        agreement_matrix = self.agreement_matrix
        regret_matrix = np.zeros_like(agreement_matrix)
        num_objects=agreement_matrix.shape[0]

        # Iterate through each unique pair of objects exactly once
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                agreement_ij = agreement_matrix[i, j]
                agreement_ji = agreement_matrix[j, i]

                # Assign the net regret to the cell corresponding to the "losing" choice
                if agreement_ij > agreement_ji:
                    # Ranking j over i would cause regret
                    regret_matrix[i, j] = agreement_ij - agreement_ji
                else: # agreement_ji >= agreement_ij
                    # Ranking i over j would cause regret
                    regret_matrix[j, i] = agreement_ji - agreement_ij

        return regret_matrix



# --- Demonstration ---
if __name__ == "__main__":
    # Example: 4 objects (0, 1, 2, 3) and 10 raters.
    # A[i, j] = number of raters preferring i over j.
    # e.g., A[0, 1] = 7 means 7 raters prefer object 0 to 1.

    agreement_matrix_A = np.array([
        [0, 7, 8, 6],  # Preferences for object 0
        [3, 0, 4, 2],  # Preferences for object 1
        [2, 6, 0, 5],  # Preferences for object 2
        [4, 8, 5, 0]   # Preferences for object 3
    ])

    # Create an instance of the ranker
    ranker = HeuristicConsensusRanker(agreement_matrix=agreement_matrix_A)

    print("--- Maximize Agreement Heuristic (Heuristic 1) ---")
    print("Agreement Matrix:\n", ranker.agreement_matrix)

    h1_ranking = ranker.maximize_agreement()

    print("\nCalculated Consensus Ranking (raw):", h1_ranking)

    print("\n" + "="*60 + "\n")

    print("--- Minimize Regret Heuristic (Heuristic 2) ---")
    print("Agreement Matrix:\n", ranker.agreement_matrix)

    h2_ranking = ranker.minimize_regret()

    print("\nCalculated Consensus Ranking (raw):", h2_ranking)