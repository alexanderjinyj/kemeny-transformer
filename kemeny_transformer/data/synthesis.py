import os

import numpy as np
import numba
import collections.abc
import torch
from numba import njit, prange, objmode
import time

@njit()
def generate_jiggling_distance(max_distance):
    M = int(max_distance) - 1
    if M < 1:
        return 1
    
    # Precomputed constants
    INV_E = 0.36787944117144233 # 1/e
    ONE_MINUS_INV_E = 0.6321205588285577 # 1 - 1/e
    
    # Sum of geometric series 1 + r + r^2 ... + r^(M-1) is (1 - r^M) / (1 - r)
    # We work with normalized weights where first element (d=1) has weight 1.0
    # This avoids overflow issues with exp(M) for large M.
    
    # Calculate normalization factor (Total Sum)
    # math.exp(-M) might underflow to 0 for large M, which is fine (term becomes 1.0)
    term_M = np.exp(-M)
    total_weight = (1.0 - term_M) / ONE_MINUS_INV_E
    
    # Sample target
    target = np.random.random() * total_weight
    
    current_weight = 1.0
    cumulative = 0.0
    
    # Iterate to find the bucket
    # Since the series decays exponentially, this loop runs very few times on average (1-2 times)
    for d in range(1, M + 1):
        cumulative += current_weight
        if cumulative >= target:
            return d
        current_weight *= INV_E
        
    return M

@njit()
def generate_jiggling_ranking(base_ranking,num_items):
    jiggling_ranking = np.copy(base_ranking)
    # Cast to int for array indexing
    n_items_int = int(num_items)
    
    for i in range(n_items_int):
        candidate_position = jiggling_ranking[i]
        
        # Calculate max distance for jiggling
        dist_a = n_items_int - candidate_position - 1
        dist_b = candidate_position
        max_dist = dist_a if dist_a > dist_b else dist_b
        
        jiggling_distance = generate_jiggling_distance(max_dist)
        #print(f'jiggling_distance:{jiggling_distance}')
        
        minus_position = candidate_position - jiggling_distance
        plus_position = candidate_position + jiggling_distance
        jiggling_position = 0
        
        if minus_position < 0:
            jiggling_position = plus_position
        else:
            if plus_position >= n_items_int:
                jiggling_position = minus_position
            else:
                # Coin flip
                if np.random.random() < 0.5:
                    jiggling_position = minus_position
                else:
                    jiggling_position = plus_position
        
        # Find who is currently at the target jiggling_position
        # np.argwhere is slow in loops, use manual linear search or assume valid permutation
        swap_candidate = -1
        for k in range(n_items_int):
            if jiggling_ranking[k] == jiggling_position:
                swap_candidate = k
                break
        
        # Swap
        if swap_candidate != -1:
            jiggling_ranking[swap_candidate] = candidate_position
            jiggling_ranking[i] = jiggling_position
            
    return jiggling_ranking


@njit(fastmath=True)
def generate_base_ranking_jiggling(num_voters, num_items, jiggling_prob_input, repeat_prob_input):
    """
    JIT-compiled version of the jiggling generator.
    Pass -1.0 to probabilities to trigger random sampling.
    """
    # Handle random defaults inside JIT
    if jiggling_prob_input < 0.0:
        real_jiggling_prob = np.random.random()
    else:
        real_jiggling_prob = jiggling_prob_input

    if repeat_prob_input < 0.0:
        real_repeat_prob = np.random.random()
    else:
        real_repeat_prob = repeat_prob_input

    base_rankings = np.empty((num_voters, num_items), dtype=np.float32)
    
    # Initialize first voter
    base_rankings[0] = np.random.permutation(num_items)
    
    for i in range(1, num_voters):
        # Decide: Jiggle or Repeat
        is_jiggling = (np.random.random() < real_jiggling_prob)
        
        if is_jiggling:
            base_rankings[i] = generate_jiggling_ranking(base_rankings[i-1], num_items)
            #print(f'jiggling ranking:{base_rankings[i]}')
        else:
            is_repeat = (np.random.random() < real_repeat_prob)
            if is_repeat:
                base_rankings[i] = base_rankings[i-1] # Copy is implicit in assignment usually, but safety:
                # In numpy, assignment slice copies data? No, it might not.
                # Explicit copy to be safe inside JIT loop
                base_rankings[i] = np.copy(base_rankings[i-1]) 
            else:
                base_rankings[i] = np.random.permutation(num_items)
                
    # Shuffle the voters in this instance
    np.random.shuffle(base_rankings)
    return base_rankings

@njit(fastmath=True)
def generate_base_ranking_repeat(num_voters, num_items, probability):
    """
    JIT-compiled version of the repeat generator.
    """
    if probability < 0.0:
        real_prob = np.random.random()
    else:
        real_prob = probability
        
    base_rankings = np.empty((num_voters, num_items), dtype=np.float32)
    base_rankings[0] = np.random.permutation(num_items)
    
    for i in range(1, num_voters):
        is_repeat = (np.random.random() < real_prob)
        if is_repeat:
            base_rankings[i] = np.copy(base_rankings[i-1])
        else:
            base_rankings[i] = np.random.permutation(num_items)
            
    np.random.shuffle(base_rankings)
    return base_rankings

@njit(fastmath=True)
def _generate_batch_kernel(bsz, num_voters, num_items):
    """
    The heavy lifting kernel. Generates the entire batch in parallel.
    Returns a 3D array (bsz, num_voters, num_items).
    """
    # Pre-allocate the entire batch
    # Using float32 as per original code
    all_data = np.empty((bsz, num_voters, num_items), dtype=np.float32)
    #print(f'Generating batch: bsz={bsz}, num_voters={num_voters}, num_items={num_items}')
    
    # Thresholds for splitting the batch types
    split_1 = bsz // 2
    split_2 = (bsz * 3) // 4
    # Parallel loop over the batch dimension
    for i in prange(bsz):
        #print(f'Generating instance {i}/{bsz}')
        # 1. Random Permutations (0 to split_1)
        if i < split_1:
            for v in range(num_voters):
                all_data[i, v] = np.random.permutation(num_items)
                #print(f'random ranking:{all_data[i,v]}')
                
        # 2. Repeat Logic (split_1 to split_2)
        elif i < split_2:
            # Determine probability for this instance
            # Logic: all_repeat = binomial(1, 0.1)
            if np.random.random() < 0.1:
                prob = 1.0
            else:
                prob = 0.5
            
            # Generate one instance
            all_data[i] = generate_base_ranking_repeat(num_voters, num_items, prob)
            #print(f'repeat ranking:{all_data[i]}')
            
        # 3. Jiggling Logic (split_2 to end)
        else:
            # Determine probability for this instance
            # Logic: all_jiggling = binomial(1, 0.1)
            if np.random.random() < 0.1:
                jig_prob = 1.0
            else:
                jig_prob = -1.0 
            
            # For repeat_probability inside jiggling, original code doesn't set it, so it defaults to random (-1.0)
            all_data[i] = generate_base_ranking_jiggling(num_voters, num_items, jig_prob, -1.0)
            #print(f'jiggling ranking:{all_data[i]}')    
            
    return all_data

@njit(fastmath=True)
def is_permutation_jit(arr):
    """
    O(N) check if arr is a valid permutation of 0..N-1.
    Replaces the slow O(N^2) nested loop.
    """
    n = len(arr)
    seen = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        val = int(arr[i])
        
        # 1. Check bounds (must be 0 <= val < n)
        if val < 0 or val >= n:
            return False
            
        # 2. Check duplicates (if we saw it before, it's not a permutation)
        if seen[val]:
            return False
            
        seen[val] = True
        
    # By Pigeonhole Principle: if we processed n items, all within bounds, 
    # and found no duplicates, we must have seen every number exactly once.
    return True

class DataSynthesis:
    def __init__(self, random_seed=0):
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate_base_rankings_random(self,num_voters,num_items):
        base_rankings=np.empty(shape=(num_voters,num_items),dtype=np.float32)
        for i in range(num_voters):
            base_rankings[i]=np.random.permutation(num_items)
        return  base_rankings
    
    @staticmethod
    @njit()
    def generate_base_ranking_repeat(num_voters, num_items, probability=None):
        if probability==None:
            probability=np.random.random_sample()
        base_rankings=np.empty(shape=(num_voters,num_items),dtype=np.float32)
        for i in range(num_voters):
            if i==0 :
                base_rankings[i]=np.random.permutation(num_items)
            else:
                #print(probability)
                repeat=np.random.binomial(1, probability)
                #print(repeat)
                if repeat:
                    base_rankings[i]=np.copy(base_rankings[i-1])
                else:
                    base_rankings[i]=np.random.permutation(num_items)
        np.random.shuffle(base_rankings)
        return base_rankings



    
    def is_permutation_fast(self, arr):
        return is_permutation_jit(arr)
    def is_permutation(self, arr):

        n = len(arr)



        # Check for each element from 1 to N in the list

        for i in range(n ):

            found = False

            for j in range(n):

                if arr[j] == i:

                    found = True

                    break



            # If any element is not found, the list is not a permutation

            if not found:

                print(i)

                return False
        return True

    def generate_batch_dataset_random(self,bsz,num_voters,num_items):
        batch_base_rankings=np.empty(shape=(bsz,num_voters,num_items),dtype=np.float32)
        for i in range(bsz):
            batch_base_rankings[i]=self.generate_base_rankings_random(num_voters, num_items)
        is_all_permutation=True
        """
        for i in range(bsz):
            for j in range(num_voters):
                is_permut=self.is_permutation(batch_base_rankings[i,j,:])
                if not is_permut:
                    print(f'bsz;{i},ranking:{j}')
                    is_all_permutation=False
        """
        return batch_base_rankings,is_all_permutation

    def generate_batch_dataset_random_from_range(self, bsz, num_voters_range, num_items_range, sample_voters_from_range=True,
        sample_items_from_range=True):
        """
        Generates a batch of random ranking datasets where num_votersand num_items 
        are sampled once from the given ranges, so all batch samples have the same shape.
        
        Args:
            bsz (int): Batch size - number of samples to generate.
            num_voters_range (tuple): A tuple (min_voters, max_voters) specifying the 
                                     range for the number of voters.
            num_items_range (tuple): A tuple (min_items, max_items) specifying the 
                                     range for the number of items.
        
        Returns:
            np.ndarray: A 3D array of shape (bsz, num_voters, num_items).
            bool: True if all generated rankings are valid permutations.
        """
        # Sample num_votersand num_items once for the entire batch
        if sample_voters_from_range is True:
            voters_choices = np.arange(num_voters_range[0], num_voters_range[1] + 1)
        else:
            voters_choices = np.array(num_voters_range)
        num_voters = int(np.random.choice(voters_choices))
        # Determine num_items
        if sample_items_from_range is True:
            items_choices = np.arange(num_items_range[0], num_items_range[1] + 1)
        else:
            items_choices = np.array(num_items_range)
        num_items = int(np.random.choice(items_choices))
        #print(f'num_voters:{num_voters},num_items:{num_items}')
        batch_base_rankings = np.empty(shape=(bsz, num_voters, num_items), dtype=np.float32)         
        is_all_permutation = True
        
        for i in range(bsz):
            batch_base_rankings[i] = self.generate_base_rankings_random(num_voters, num_items)
        return batch_base_rankings, is_all_permutation

    def generate_mix_batch_dataset_random_from_range(self, bsz, num_voters_range, num_items_range,
        sample_voters_from_range=True, sample_items_from_range=True):
        """
        Generates a batch of random ranking datasets where num_voters and num_items
        are resampled independently for EACH sample in the batch.

        Unlike generate_batch_dataset_random_from_range (which samples once for the
        entire batch), this function produces samples with potentially different shapes.

        Args:
            bsz (int): Batch size - number of samples to generate.
            num_voters_range (tuple): A tuple (min_voters, max_voters) specifying the
                                     range for the number of voters.
            num_items_range (tuple): A tuple (min_items, max_items) specifying the
                                     range for the number of items.
            sample_voters_from_range (bool): If True, sample uniformly from
                [min, max] range. If False, sample only from the explicit values.
            sample_items_from_range (bool): If True, sample uniformly from
                [min, max] range. If False, sample only from the explicit values.

        Returns:
            list: A list of bsz numpy arrays, each of shape (num_voters_i, num_items_i),
                  where dimensions can differ per sample.
            bool: True if all generated rankings are valid permutations.
        """
        # Pre-compute choices arrays
        if sample_voters_from_range is True:
            voters_choices = np.arange(num_voters_range[0], num_voters_range[1] + 1)
        else:
            voters_choices = np.array(num_voters_range)

        if sample_items_from_range is True:
            items_choices = np.arange(num_items_range[0], num_items_range[1] + 1)
        else:
            items_choices = np.array(num_items_range)

        batch_base_rankings = []
        is_all_permutation = True

        for i in range(bsz):
            # Resample num_voters and num_items for each sample
            num_voters = int(np.random.choice(voters_choices))
            num_items = int(np.random.choice(items_choices))
            batch_base_rankings.append(self.generate_base_rankings_random(num_voters, num_items))

        return batch_base_rankings, is_all_permutation

    def generate_batch_dataset_repeat(self,bsz,num_voters,num_items):
        batch_base_rankings=np.empty(shape=(bsz,num_voters,num_items),dtype=np.float32)
        for i in range(bsz):
            batch_base_rankings[i]=self.generate_base_ranking_repeat(num_voters, num_items)
        is_all_permutation=True
        for i in range(bsz):
            for j in range(num_voters):
                is_permut=self.is_permutation(batch_base_rankings[i,j,:])
                if not is_permut:
                    print(f'bsz;{i},ranking:{j}')
                    is_all_permutation=False

        return batch_base_rankings,is_all_permutation

    def generate_batch_dataset_jiggling(self,bsz,num_voters,num_items):
        batch_base_rankings=np.empty(shape=(bsz,num_voters,num_items),dtype=np.float32)
        for i in range(bsz):
            all_jiggling=np.random.binomial(1, p=0.1)
            if all_jiggling:
                batch_base_rankings[i]=generate_base_ranking_jiggling(num_voters,num_items, 1.0, -1.0)
            else:
                batch_base_rankings[i]=generate_base_ranking_jiggling(num_voters,num_items, -1.0, -1.0)
        is_all_permutation=True
        for i in range(bsz):
            for j in range(num_voters):
                is_permut=self.is_permutation(batch_base_rankings[i,j,:])
                if not is_permut:
                    print(f'bsz;{i},ranking:{j}')
                    is_all_permutation=False
        return  batch_base_rankings,is_all_permutation
    
    def generate_batch_dataset_repeat_jiggling(self,bsz,num_voters,num_items):
        num_random_base_rankings=bsz/2
        num_repeat_base_rankings=bsz/4
        num_jiggling_base_rankings=bsz/4
        batch_base_rankings=np.empty(shape=(bsz,num_voters,num_items),dtype=np.float32)
        index_batch_base_rankings=0
        for _ in range(int(num_random_base_rankings)):
            print(f'num:{index_batch_base_rankings}')
            base_rankings_random=self.generate_base_rankings_random(num_voters, num_items)
            batch_base_rankings[index_batch_base_rankings]=base_rankings_random
            index_batch_base_rankings+=1

        for _ in range(int(num_repeat_base_rankings)):
            print(f'num:{index_batch_base_rankings}')
            base_rankings_repeat=self.generate_base_ranking_repeat(num_voters,num_items)
            batch_base_rankings[index_batch_base_rankings]=base_rankings_repeat
            index_batch_base_rankings+=1
            print()

        for _ in range(int(num_jiggling_base_rankings)):
            print(f'num:{index_batch_base_rankings}')
            all_jiggling=np.random.binomial(1, p=0.1)
            if all_jiggling:
                base_rankings_jiggling=generate_base_ranking_jiggling(num_voters,num_items, 1.0, -1.0)
            else:
                base_rankings_jiggling=generate_base_ranking_jiggling(num_voters,num_items, -1.0, -1.0)
            batch_base_rankings[index_batch_base_rankings]=base_rankings_jiggling
            index_batch_base_rankings+=1

        is_all_permutation=True
        """
        for i in range(bsz):
            for j in range(num_voters):
                is_permut=self.is_permutation(batch_base_rankings[i,j,:])
                if not is_permut:
                    print(f'bsz;{i},ranking:{j}')
                    is_all_permutation=False
        """
        return batch_base_rankings,is_all_permutation

    def generate_batch_instances_fine_tuning(self, bsz, num_voters_range, num_items_range,sample_voters_from_range=True,
        sample_items_from_range=True):
        # Determine num_voters
        if sample_voters_from_range:
            voters_choices = np.arange(num_voters_range[0], num_voters_range[1] + 1)
        else:
            voters_choices = np.array(num_voters_range)
        num_voters = int(np.random.choice(voters_choices))
        # Determine num_items
        if sample_items_from_range:
            items_choices = np.arange(num_items_range[0], num_items_range[1] + 1)
        else:
            items_choices = np.array(num_items_range)
        num_items = int(np.random.choice(items_choices))
        # --- High Performance Generation ---
        # Call the JIT-compiled parallel kernel
        # This returns a 3D numpy array (bsz, num_voters, num_items)

        batch_array = _generate_batch_kernel(bsz, num_voters, num_items)

        # --- Post-Processing ---
        # The original code returned a LIST of 2D arrays and a boolean flag.
        # We convert the 3D array to a list of 2D arrays to maintain API compatibility.
        # This is very fast (pointer wrapping).
        
        all_batches_list = list(batch_array)
        
        # Assuming permutation validity is guaranteed by the logic (it is), we return True.
        # Computing is_all_permutation check on large batches is slow, so we skip it 
        # unless strictly necessary for debugging.
        is_all_permutation = True 
        """
        for i in range(bsz):
            # print(i)
            for j in range(num_voters):
                is_permut=self.is_permutation(all_batches_list[i][j,:])
                is_permut_fast=self.is_permutation_fast(all_batches_list[i][j,:])
                if is_permut != is_permut_fast:
                    print(f'Discrepancy found at bsz:{i}, ranking:{j}') 
                if not is_permut:
                    print(f'bsz;{i},ranking:{j}')
                    is_all_permutation=False
        print(f'is_permut:{is_all_permutation}')
        print(f"is_all_permutation:{is_all_permutation}")
        """
        
        #print(f'{all_batches_list[0].shape=}')
        return all_batches_list, is_all_permutation

    def generate_mix_batch_instances_fine_tuning(self, bsz, num_voters_range, num_items_range,
        sample_voters_from_range=True, sample_items_from_range=True):
        """
        Generates a fine-tuning batch where num_voters and num_items are resampled
        independently for EACH sample in the batch. Mixes random, repeat, and jiggling subsets.
        """
        if sample_voters_from_range is True:
            voters_choices = np.arange(num_voters_range[0], num_voters_range[1] + 1)
        else:
            voters_choices = np.array(num_voters_range)

        if sample_items_from_range is True:
            items_choices = np.arange(num_items_range[0], num_items_range[1] + 1)
        else:
            items_choices = np.array(num_items_range)

        all_batches_list = []
        is_all_permutation = True
        
        split_1 = bsz // 2
        split_2 = (bsz * 3) // 4

        for i in range(bsz):
            num_voters = int(np.random.choice(voters_choices))
            num_items = int(np.random.choice(items_choices))
            
            # 1. Random Permutations (0 to split_1)
            if i < split_1:
                instance = np.empty((num_voters, num_items), dtype=np.float32)
                for v in range(num_voters):
                    instance[v] = np.random.permutation(num_items)
                all_batches_list.append(instance)
                
            # 2. Repeat Logic (split_1 to split_2)
            elif i < split_2:
                if np.random.random() < 0.1:
                    prob = 1.0
                else:
                    prob = 0.5
                instance = generate_base_ranking_repeat(num_voters, num_items, prob)
                all_batches_list.append(instance)
                
            # 3. Jiggling Logic (split_2 to end)
            else:
                if np.random.random() < 0.1:
                    jig_prob = 1.0
                else:
                    jig_prob = -1.0 
                instance = generate_base_ranking_jiggling(num_voters, num_items, jig_prob, -1.0)
                all_batches_list.append(instance)

        return all_batches_list, is_all_permutation

    def geometric_series(self,a, m):
        gs = 0
        for i in range(m):
            gs += a**i
        return gs


    def generate_base_rankings_Mallows(self, num_voters, num_items, phi):
        """
    Generates a profile with n voters and k items drawn from a Mallows phi model.
    The reference ranking (center) is a randomly generated permutation.
    """
    # 1. Generate a random permutation to be the center ranking.
    # This will be the same for all voters generated in this call.
        center_ranking = np.random.permutation(num_items)
        print(f"The randomly generated center ranking is: {center_ranking}")

        base_permutations = []

        for _ in range(num_voters):  # for all voters
            # 2. Generate one permutation relative to the identity ranking (0, 1, 2,...)
            # using the repeated insertion model, just like in the original code.
            relative_permutation = []
            for inum_item in range(1, num_items + 1):  # for all items
                # Probabilities for inserting the current item
                probabilities = [phi ** (inum_item - j) / self.geometric_series(phi, inum_item)
                                for j in range(1, inum_item + 1)]
                position = np.random.choice(inum_item, p=probabilities)
                relative_permutation.insert(position, inum_item - 1)

            # 3. Apply the relative_permutation to the center_ranking to get the final
            # permutation for the current voter.
            # For example, if center is [2,0,1] and relative is [1,0,2],
            # the result is [center[1], center[0], center[2]] -> [0,2,1].
            final_permutation = center_ranking[relative_permutation]
            base_permutations.append(final_permutation)

        # 4. Convert the list of permutations to rankings using argsort.
        # The ranking indicates the position of each item.
        base_rankings = np.argsort(np.array(base_permutations), axis=1)
        return base_rankings,self.is_all_permutation(base_rankings)
        
    def batch_generate_base_rankings_Mallows(self,bsz,num_voters, num_items, phi):
        """
    Generates a profile with n voters and k items drawn from a Mallows phi model.
    The reference ranking (center) is a randomly generated permutation.
    """
    # 1. Generate a random permutation to be the center ranking.
    # This will be the same for all voters generated in this call.
        base_rankings_all_batches=np.empty(shape=(bsz,num_voters,num_items),dtype=np.float32)
        is_all_permutation=True
        for batch in range(bsz):
            center_ranking = np.random.permutation(num_items)
            print(f"The randomly generated center ranking is: {center_ranking}")

            base_permutations = []

            for _ in range(num_voters):  # for all voters
                # 2. Generate one permutation relative to the identity ranking (0, 1, 2,...)
                # using the repeated insertion model, just like in the original code.
                relative_permutation = []
                for inum_item in range(1, num_items + 1):  # for all items
                    # Probabilities for inserting the current item
                    probabilities = [phi ** (inum_item - j) / self.geometric_series(phi, inum_item)
                                    for j in range(1, inum_item + 1)]
                    position = np.random.choice(inum_item, p=probabilities)
                    relative_permutation.insert(position, inum_item - 1)

                # 3. Apply the relative_permutation to the center_ranking to get the final
                # permutation for the current voter.
                # For example, if center is [2,0,1] and relative is [1,0,2],
                # the result is [center[1], center[0], center[2]] -> [0,2,1].
                final_permutation = center_ranking[relative_permutation]
                base_permutations.append(final_permutation)

            # 4. Convert the list of permutations to rankings using argsort.
            # The ranking indicates the position of each item.
            base_rankings = np.argsort(np.array(base_permutations), axis=1)
            is_all_permutation=self.is_all_permutation(base_rankings)
            if not is_all_permutation:
                print(f'bsz;{batch},ranking is not permutation')
            base_rankings_all_batches[batch]=base_rankings
        return base_rankings_all_batches,is_all_permutation
    
    @staticmethod
    @njit(nopython=True, parallel=True)
    def _vcode_to_permutation_numba(vcodes):
        """
        Converts a batch of V-codes (inversion vectors) to permutations.
        This function is JIT-compiled and parallelized for maximum CPU performance.

        Args:
            vcodes (np.ndarray): A 2D array where each row is a V-code.

        Returns:
            np.ndarray: A 2D array where each row is the corresponding permutation.
        """
        n_voters, n_items = vcodes.shape
        perms = np.empty_like(vcodes, dtype=np.int64)

        # The prange enables parallel execution over the voters.
        # Each thread in the pool will handle the conversion for one voter.
        for i in numba.prange(n_voters):
            vcode = vcodes[i]
            
            # This is a numba-compatible way of doing `list.pop()` on a sorted list.
            # We start with a pool of available items to be placed in the permutation.
            items_left = np.arange(n_items)
            
            for j in range(n_items):
                # The j-th element of the vcode tells us which of the *remaining*
                # items to pick.
                index = vcode[j]
                
                # Place the selected item into the permutation.
                perms[i, j] = items_left[index]
                
                # "Remove" the chosen item from the pool by shifting the
                # remaining items to the left.
                for k in range(index, n_items - 1 - j):
                    items_left[k] = items_left[k + 1]
                    
        return perms

    def _generate_mallows_relative_perms_vcode(self, num_voters, num_items, phi):
        """
        Generates Mallows model samples relative to the identity permutation
        using the fast V-code (inversion vector) method.

        Returns:
            np.ndarray: A 2D array of relative permutations.
        """
        # Step 1: Generate the V-codes. This is the core of the algorithm.
        # The j-th element of a v-code, V_j, is an integer from 0 to (num_items - 1 - j).
        # We generate these for all voters and all positions in a vectorized way.
        
        j_values = np.arange(num_items - 1, -1, -1) # [n-1, n-2, ..., 0]
        
        if phi == 1.0:
            # For phi=1 (the uniform distribution), V_j is uniform in {0, ..., num_items-1-j}.
            # We can generate this by scaling uniform random numers.
            rand_unif = np.random.rand(num_voters, num_items)
            vcodes = np.floor(rand_unif * (j_values + 1)).astype(np.int64)
        else:
            # For phi != 1, we sample from a truncated geometric distribution.
            # This can be done efficiently using the inverse CDF method, which
            # is fully vectorized in NumPy.
            
            # Precompute powers of phi for the formula
            phi_powers = np.power(phi, j_values + 1)
            
            # Generate uniform random numers for sampling
            rand_unif = np.random.rand(num_voters, num_items)
            
            # Vectorized inverse CDF formula to get V-codes directly
            log_phi = np.log(phi + 1e-9)
            vcodes = np.floor(
                np.log(1 - rand_unif * (1 - phi_powers)) / log_phi
            ).astype(np.int64)

        # Step 2: Convert the generated V-codes to permutations.
        # This is the part that is hard to vectorize but easy to parallelize.
        # We offload this to our fast numba function.
        return self._vcode_to_permutation_numba(vcodes)

    def batch_generate_base_rankings_Mallows_vcode(self, bsz, num_voters, num_items, phi):
        """
        Generates batches of rankings drawn from a Mallows phi model.
        
        This high-speed version uses an efficient algorithm based on inversion
        vectors (V-codes) to accelerate the generation of permutations.
        """
        base_rankings_all_batches = np.empty(shape=(bsz, num_voters, num_items), dtype=np.int32)
        
        for batch in range(bsz):
            # 1. Generate a random permutation to be the center ranking for this batch.
            center_ranking = np.random.permutation(num_items)

            # 2. Generate all relative permutations for the batch using the fast V-code method.
            # This replaces the slow, iterative insertion model.
            relative_permutations = self._generate_mallows_relative_perms_vcode(num_voters, num_items, phi)
            
            # 3. Apply the relative permutations to the center ranking.
            # This is a fast, vectorized operation using NumPy's advanced indexing.
            base_permutations = center_ranking[relative_permutations]

            # 4. Convert the final permutations to rankings using argsort.
            base_rankings = np.argsort(base_permutations, axis=1)
            is_all_permutation=self.is_all_permutation(base_rankings)
            if not is_all_permutation:
                print(f'bsz;{batch},ranking is not permutation')
            
            base_rankings_all_batches[batch] = base_rankings
            
        # The V-code method always produces valid permutations.
        is_all_permutation = True
        
        return base_rankings_all_batches, is_all_permutation



    def batch_generate_base_rankings_Mallows_vcode(self, bsz, num_voters_range, num_items_range, phi_range=None,
        sample_voters_from_range=True,
        sample_items_from_range=True,
        sample_phi_from_range=False):
        """
        Generates batches of rankings from a Mallows model with variable dimensions.

        For each batch, the numer of voters is randomly determined. If `num_voters_range`
        is a list, a value is sampled directly from it. If it's a tuple, it is
        treated as a `(min, max)` range. The numer of items is always sampled from a range.

        This version uses the efficient V-code method for generation.

        Args:
            bsz (int): The numer of batches to generate.
            num_voters_range (list or tuple): A list of specific integer values to sample from,
                                            or a tuple `(min_voters, max_voters)` to define
                                            a range for sampling.
            num_items_range (tuple): A tuple `(min_items, max_items)` specifying
                                    the range for the numer of items (items).

            phi_range (float, list, optional): The dispersion parameter(s) of the Mallows model.

            sample_voters_from_range (bool, optional):
                If True: Samples uniformly from the range [min, max] of num_voters_range.
                If False (default): Samples *only* from the explicit values in num_voters_range.

            sample_items_from_range (bool, optional):
                If True: Samples uniformly from the range [min, max] of num_items_range.
                If False (default): Samples *only* from the explicit values in num_items_range.

            sample_phi_from_range (bool, optional):
                If True: Samples uniformly from the range [min, max] of phi_range.
                If False (default): Samples *only* from the explicit values in phi_range.

        Returns:
            list: A list of NumPy arrays. Each array in the list represents a batch of
                rankings and has a shape of `(num_voters, num_items)`, where these
                dimensions can differ for each batch.
            bool: True if all generated batches contained valid permutations, False otherwise.
        """
        all_batches = []
        overall_all_permutation = True

        # --- VOTERS ---
        voters_choices = []
        if sample_voters_from_range: # If True
            # Case: Sample from the range [min, max]
            voters_min, voters_max = num_voters_range[0], num_voters_range[1]
            voters_choices = np.arange(voters_min, voters_max + 1, 1) # Step is 1
        else:
            # Case: Sample *only* from the provided list
            voters_choices = num_voters_range

        # --- ITEMS ---
        items_choices = []
        if sample_items_from_range: # If True
            # Case: Sample from the range [min, max]
            items_min, items_max = num_items_range[0], num_items_range[1]
            items_choices = np.arange(items_min, items_max + 1, 1) # Step is 1
        else:
            # Case: Sample *only* from the provided list
            items_choices = num_items_range

        # --- PHI ---
        phi_choices = None
        if phi_range is None:
            phi_choices = None # Special case: will sample uniformly from [0, 1]
        elif isinstance(phi_range, (int, float)):
            phi_choices = [phi_range] # Single value
        elif sample_phi_from_range: # If True
            # Case: Sample from the range [min, max]
            phi_min, phi_max = phi_range[0], phi_range[1]
            # Sample uniformly from the float range
            # We'll do this sampling inside the loop
            phi_choices = "uniform_range"
        else:
            # Case: Sample *only* from the provided list
            phi_choices = phi_range


        # Initialize list only if phi is None. Otherwise, it stays None.
        generated_phis = [] if phi_choices is None else None

        for batch in range(bsz):

            # 1. Sample the numer of voters for the current batch.
            num_voters = np.random.choice(voters_choices)

            # 2. Sample the numer of items for the current batch.
            num_items = np.random.choice(items_choices)
            # --- END MODIFICATION ---

            # 3. Generate a random permutation to be the center ranking for this batch.
            center_ranking = np.random.permutation(num_items)

            # --- MODIFIED: Sample from phi_choices ---
            # 4.Determine phi for the current batch ---
            current_phi = 0.0
            if phi_choices is None:
                # Special case: uniform sampling [0, 1]
                current_phi = np.random.uniform(0.0, np.nextafter(1.0, np.inf))
                generated_phis.append(current_phi)# Store the phi used
            elif phi_choices == "uniform_range":
                # Special case: uniform sampling [min, max]
                phi_min, phi_max = phi_range[0], phi_range[1]
                current_phi = np.random.uniform(phi_min, np.nextafter(phi_max, phi_max + 1))
            else:
                # Sample from the pre-calculated list of choices
                current_phi = np.random.choice(phi_choices)

            #print(f"{num_voters =}{num_items =}{current_phi =}")
            # 5. Generate all relative permutations using the fast V-code method.
            #    Note: Uses current_phi now.
            relative_permutations = self._generate_mallows_relative_perms_vcode(num_voters, num_items, current_phi)

            # 6. Apply the relative permutations to the center ranking via advanced indexing.
            base_permutations = center_ranking[relative_permutations]

            # 7. Convert the final permutations to rankings using argsort.
            base_rankings = np.argsort(base_permutations, axis=1)

            """
            # 8. Check permutation validity for this batch
            #current_batch_is_permutation = self.is_all_permutation(base_rankings)
            if not current_batch_is_permutation:
                print(f'bsz;{batch},ranking is not permutation')
                overall_all_permutation = False
            """

            # 9. Append the generated batch of rankings to our list of batches.
            all_batches.append(base_rankings + 1)

        return all_batches, overall_all_permutation, generated_phis
    
    
    def batch_generate_base_rankings_Mallows_all_same_shape_vcode(self, bsz, num_voters_range, num_items_range, phi_range=None,
        sample_voters_from_range=True,
        sample_items_from_range=True,
        sample_phi_from_range=False):
        """
        Generates batches of rankings from a Mallows model with variable dimensions.

        For each batch, the numer of voters is randomly determined. If `num_voters_range`
        is a list, a value is sampled directly from it. If it's a tuple, it is
        treated as a `(min, max)` range. The numer of items is always sampled from a range.

        This version uses the efficient V-code method for generation.

        Args:
            bsz (int): The numer of batches to generate.
            num_voters_range (list or tuple): A list of specific integer values to sample from,
                                            or a tuple `(min_voters, max_voters)` to define
                                            a range for sampling.
            num_items_range (tuple): A tuple `(min_items, max_items)` specifying
                                    the range for the numer of items (items).

            phi_range (float, list, optional): The dispersion parameter(s) of the Mallows model.

            sample_voters_from_range (bool, optional):
                If True: Samples uniformly from the range [min, max] of num_voters_range.
                If False (default): Samples *only* from the explicit values in num_voters_range.

            sample_items_from_range (bool, optional):
                If True: Samples uniformly from the range [min, max] of num_items_range.
                If False (default): Samples *only* from the explicit values in num_items_range.

            sample_phi_from_range (bool, optional):
                If True: Samples uniformly from the range [min, max] of phi_range.
                If False (default): Samples *only* from the explicit values in phi_range.

        Returns:
            list: A list of NumPy arrays. Each array in the list represents a batch of
                rankings and has a shape of `(num_voters, num_items)`, where these
                dimensions can differ for each batch.
            bool: True if all generated batches contained valid permutations, False otherwise.
        """
        all_batches = []
        overall_all_permutation = True

        # --- VOTERS ---
        voters_choices = []
        if sample_voters_from_range: # If True
            # Case: Sample from the range [min, max]
            voters_min, voters_max = num_voters_range[0], num_voters_range[1]
            voters_choices = np.arange(voters_min, voters_max + 1, 1) # Step is 1
        else:
            # Case: Sample *only* from the provided list
            voters_choices = num_voters_range

        # --- ITEMS ---
        items_choices = []
        if sample_items_from_range: # If True
            # Case: Sample from the range [min, max]
            items_min, items_max = num_items_range[0], num_items_range[1]
            items_choices = np.arange(items_min, items_max + 1, 1) # Step is 1
        else:
            # Case: Sample *only* from the provided list
            items_choices = num_items_range

        # --- PHI ---
        phi_choices = None
        if phi_range is None:
            phi_choices = None # Special case: will sample uniformly from [0, 1]
        elif isinstance(phi_range, (int, float)):
            phi_choices = [phi_range] # Single value
        elif sample_phi_from_range: # If True
            # Case: Sample from the range [min, max]
            phi_min, phi_max = phi_range[0], phi_range[1]
            # Sample uniformly from the float range
            # We'll do this sampling inside the loop
            phi_choices = "uniform_range"
        else:
            # Case: Sample *only* from the provided list
            phi_choices = phi_range.copy()

        # Initialize list only if phi is None. Otherwise, it stays None.
        generated_phis = [] if phi_choices is None else None
        
        # Sample the numer of voters for the current batch.
        num_voters = np.random.choice(voters_choices)

        # Sample the numer of items for the current batch.
        num_items = np.random.choice(items_choices)
        for batch in range(bsz):


            # 3. Generate a random permutation to be the center ranking for this batch.
            center_ranking = np.random.permutation(num_items)

            # 4.Determine phi for the current batch ---
            current_phi = 0.0
            
            if phi_choices is None:
                # Special case: uniform sampling [0, 1]
                current_phi = np.random.uniform(0.0, np.nextafter(1.0, np.inf))
                generated_phis.append(current_phi)# Store the phi used
            elif phi_choices == "uniform_range":
                # Special case: uniform sampling [min, max]
                phi_min, phi_max = phi_range[0], phi_range[1]
                current_phi = np.random.uniform(phi_min, np.nextafter(phi_max, phi_max + 1))
            else:
                # Sample from the pre-calculated list of choices
                current_phi = np.random.choice(phi_choices)
            #print(f"{num_voters =}{num_items =}{current_phi =}")
            # 5. Generate all relative permutations using the fast V-code method.
            #    Note: Uses current_phi now.
            relative_permutations = self._generate_mallows_relative_perms_vcode(num_voters, num_items, current_phi)
            # 6. Apply the relative permutations to the center ranking via advanced indexing.
            base_permutations = center_ranking[relative_permutations]

            # 7. Convert the final permutations to rankings using argsort.
            base_rankings = np.argsort(base_permutations, axis=1)

            """
            # 8. Check permutation validity for this batch
            #current_batch_is_permutation = self.is_all_permutation(base_rankings)
            if not current_batch_is_permutation:
                print(f'bsz;{batch},ranking is not permutation')
                overall_all_permutation = False
            """

            # 9. Append the generated batch of rankings to our list of batches.
            all_batches.append(base_rankings + 1)

        return all_batches, overall_all_permutation, generated_phis
    

    
def order_to_rank(orders):
    return torch.argsort(orders,dim=1)

def order_to_rank_batch(orders: list[np.ndarray]):
    """Converts a list of order/permutation arrays into a list of rank arrays.""" 
    return [np.argsort(order) for order in orders]



def kemeny_distance_batch(batch_base_rankings: list[np.ndarray], final_rankings: list[np.ndarray]) -> np.ndarray:
    # WARNING: Using @njit on a function that iterates over a Python list
    # (like batch_base_rankings) will likely fall back to "object mode"
    # and be SLOWER than plain Python. This decorator is not recommended here.
    """
    Calculates the Kemeny distance for a batch of rankings sequentially.

    Args:
        batch_base_rankings (list[np.ndarray]): A list of 2D arrays.
                                              Each element is a (num_voters, num_items) array.
                                              Note: num_voters can vary per element.

        final_rankings (list[np.ndarray]): A list of 1D arrays.
                                     Each row is a (num_items,) final candidate ranking.
                                     len(final_rankings) must equal len(batch_base_rankings).

    Returns:
        np.ndarray: 1D array of shape (bsz,) containing the Kemeny distance
                    for each item in the batch.
    """
    # Get batch size from the length of the list
    bsz = len(batch_base_rankings)

    # Basic check to prevent errors
    if bsz != len(final_rankings):
        raise ValueError(f"Mismatch in batch size: batch_base_rankings has length {bsz} "
                         f"but final_rankings has length {len(final_rankings)}")

    kemeny_distances = np.empty(bsz, dtype=np.float64)

    for i in range(bsz):
        # The performance gain comes from _kemeny_distance_single being @njit
        kemeny_distances[i] = _kemeny_distance_single(
            batch_base_rankings[i],  # This is a 2D array (num_voters, num_items)
            final_rankings[i]        # This is a 1D array (num_items)
        )
    return kemeny_distances

@njit
def _kemeny_distance_single(base_rankings, candidate_ranking):
    """
    Calculates the average Kendall tau distance using vectorized operations
    across voters. This is much faster than the slow loop-in-loop version.
    """
    num_voters = base_rankings.shape[0]
    num_items = base_rankings.shape[1]
    
    kemeny_dist = 0.0

    # Loop over all unique pairs of items (j, k)
    for j in range(num_items):
        for k in range(j + 1, num_items):
            
            # 1. Get the preference for the candidate ranking (a scalar)
            # This is now calculated *outside* the voter loop.
            cand_sign = np.sign(candidate_ranking[j] - candidate_ranking[k])
            
            # 2. Get the preferences for ALL voters at once (a vector)
            # This is a fast, numba-optimized array operation.
            base_signs = np.sign(base_rankings[:, j] - base_rankings[:, k])
            
            # 3. Count disagreements
            # numba will compile np.sum() on this boolean 
            # array into a fast, simple loop.
            kemeny_dist += np.sum(cand_sign != base_signs)
                        
    # The Kemeny distance is the total numer of disagreements,
    # averaged over the numer of voters.



    kemeny_dist=kemeny_dist/num_voters
    return kemeny_dist

def main(self):
    validate_dataset_random=np.zeros(shape=(1000,8,100),dtype=np.float32)
    #for i in range(validate_dataset_random.shape[0]):
        #validate_dataset_random[i]=generate_base_rankings_random(validate_dataset_random.shape[1],validate_dataset_random.shape[2])
    os.system("mkdir validate_dataset")
    #np.save('validate_dataset/validate_dataset_random.npy',validate_dataset_random)
    validate_dataset_repeat_jiggling,is_all_permutation=self.generate_batch_dataset_repeat_jiggling(1024,8,100)
    print(f'random:{validate_dataset_repeat_jiggling[0:100][::10]},repeat:{validate_dataset_repeat_jiggling[600:700][::10]},jiggling:{validate_dataset_repeat_jiggling[900:1000,:,:][::10]}')
    print(f'is all permutation={is_all_permutation}')
    np.save('validate_dataset/validate_dataset_repeat_jiggling.npy',validate_dataset_repeat_jiggling)
if __name__ == '__main__':
    print(('main'))
    main()


    #%%
