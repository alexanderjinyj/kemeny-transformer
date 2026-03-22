import math

import numpy as np
import pandas as pd


def pairwise_statistical_parity(ranking_1, ranking_2):
    """
    compute the pairwise_statistical_parity for ranking. compute the pairwise_statistical_parity for ranking.
    parameter ranking_1 is A numpy 2d_array of ranking, 1 first column is the id of the candidates which
    have same attribute value,2nd is the ranking value of the candidates.
    parameter ranking_2 is A numpy 2d_array of ranking, 1 first column is the id of the candidates which
    have another same attribute value,2nd is the ranking value of the candidates.
    
    """
    # return the parity value
    parity = 0
    # calculate parity
    for x in ranking_1:
        for y in ranking_2:
            # if in the ranking candidate x in ranking_1 is smaller than y in ranking_2 parity plus 1
            if x[1] < y[1]:
                parity += 1
            # if in the ranking candidate x in ranking_1 is greater than y in ranking_2 parity minus 1
            elif x[1] > y[1]:
                parity += -1
    parity = parity / (len(ranking_1) * len(ranking_2))
    return parity

def top_k_fairness(ranking,k,attributes,proportions,threshold):
    satisfied=True
    counts_of_groups=np.zeros(proportions.shape[0])
    #print(f'ranking:{ranking.shape},attribute:{attributes.shape},proportion:{proportions.shape}')
    for i in range(k):
        #print(f'ranking:{ranking[i]}')
        #print(f'attribute:{attributes[int(ranking[i])]}')
        try:
            counts_of_groups[int(attributes[int(ranking[i])])] =counts_of_groups[int(attributes[int(ranking[i])])] +1
        except:
            print(f'ranking:{ranking[i]},i:{i}')
            #print(f'attribute:{attributes[int(ranking[i])]}')
    for i in range(proportions.shape[0]):
        lower_boundary=max(math.floor(round((proportions[i]-threshold),4)*k),0)
        upper_boundary=min(math.ceil(round((proportions[i]+threshold),4)*k),k)
        if counts_of_groups[i] < lower_boundary or counts_of_groups[i] > upper_boundary:
            satisfied=False
        #print(str(k)+" "+str(counts_of_groups[i]) +'  '+str(lower_boundary) + '  ' +str(upper_boundary)+" " +str(i)+" "+str(satisfied))
    return satisfied

def top_k_parity(groups, k,threshold):
    """
        computer whether the ranking with mutually exclusive groups satisfies top-k parity.
        groups are the arrays of  group,
        the group in the groups is a 2d array first column is the id of the candidates and
        2nd is the ranking value of the candidates.
        k is the top k.
        calculate the number of candidate(rank smaller than k) in each group then
        counter the Proportion: number of candidate(rank smaller than k)/number of group
        then calculate whether the proportion on each group is equaled if True satisfied top-k parity,
        if False not satisfied.
    """

    # result value of satisfied top-k parity
    satisfied = True
    # an array, element is number of candidate(rank smaller than k) in one group
    cand_counters = []
    # counter the sum of candidate
    sum_candidates = 0
    # the counter of number of candidate in one group
    num_candidates = []
    # count the number of candidate(rank smaller than k) in each group,and total number of all candidates
    for group in groups:
        cand_counter = 0
        num_candidate = group.shape[0]
        num_candidates.append(num_candidate)
        sum_candidates = sum_candidates + num_candidate
        # count the number of candidate(rank smaller than k)
        for candidate in group:
            if candidate[1] <k:
                cand_counter += 1
        cand_counters.append(cand_counter)
    # check whether the candidate(whose rank is smaller than k) in each group is proportional to number of candidate
    # in each group
    for i in range(0, groups.shape[0]):
        proportion = num_candidates[i] / sum_candidates
        lower_boundary=max(math.floor((proportion-threshold)*k),0)
        upper_boundary=min(math.ceil((proportion+threshold)*k),sum_candidates)
        if cand_counters[i] < lower_boundary or cand_counters[i] >upper_boundary:
            satisfied = False
            break
    return satisfied


def rank_equality_error(rank_1, rank_2, group_1, group_2):
    """
    computer the Rank Equality Error ratio the pair which one element for group_1 and another form group_2 has different
    favoring in 2 ranks. This means one has lower rank value than other in one rank but inverted in another rank
    param rank_1 is a 2d array with candidate and rank
    param rank_2 is a 2d array with candidate and rank
    param group_1 is an array with candidates
    param group_2 is an array with candidates

    """
    # count_pairs is-the number of pairs has different favoring
    count_pairs = 0
    # calculate the count_pairs
    for candidate_1 in group_1:
        for candidate_2 in group_2:
            rank_1_of_candidate_1 = rank_1[np.argwhere(rank_1[:, 0] == candidate_1), 1]
            rank_1_of_candidate_2 = rank_1[np.argwhere(rank_1[:, 0] == candidate_2), 1]
            rank_2_of_candidate_1 = rank_2[np.argwhere(rank_2[:, 0] == candidate_1), 1]
            rank_2_of_candidate_2 = rank_2[np.argwhere(rank_2[:, 0] == candidate_2), 1]
            # if candidate1 and candidate2 has different favoring in 2 rankings then count_pairs +1
            if np.sign(rank_1_of_candidate_1 - rank_1_of_candidate_2) == np.sign(
                    rank_2_of_candidate_2 - rank_2_of_candidate_1):
                count_pairs += 1
    # the number of mix pairs
    number_pairs = group_1.shape[0] * group_2.shape[0]
    # the ratio of Rank Equality Error
    rqe = count_pairs / number_pairs
    return rqe


def attribute_rank_parity(rank, attributes, k_attribute):
    """
    calculate arp value for the protected attribute p in rank 
    param rank is a 2d array with candidate and attribute value and rank value
    param attributes is a 2d numpy array with candidate and attributes values of the candidates
    param k_attribute is the kth attribute k
    """
    # get the all kth attributes value  +1 because attributes[0] is candidates
    group_of_values = np.unique(attributes[:, k_attribute + 1])
    # get the number of attributes in attributes prepare the parament attributes_values for function
    # favored_pair_representation
    number_of_attributes = attributes.shape[1] - 1
    # array of fpr score of all value of kth attribute
    fprs = []
    # compute teh fpr of every value of kth attribute
    for value in group_of_values:
        attributes_values = np.full(number_of_attributes, np.nan, dtype=object)
        attributes_values[k_attribute] = value
        fpr = favored_pair_representation(rank, attributes, attributes_values)
        fprs.append([value, fpr])

    # calculate arp score
    fprs = np.array(fprs)
    fprs_scores = fprs[:, 1].astype(np.float64)
    max_fpr = np.max(fprs_scores)
    min_fpr = np.min(fprs_scores)
    max_value = fprs[np.argwhere(fprs[:, 1].astype(np.float64) == max_fpr), 0]
    min_value = fprs[np.argwhere(fprs[:, 1].astype(np.float64) == min_fpr), 0]
    arp = max_fpr - min_fpr
    # return a tuple of max_value ,min_value, arp
    return max_value, min_value, arp


def favored_pair_representation(rank, attributes, attributes_values):
    """
    calculate fps score
    param rank is a 2d array with candidate and attribute value and rank value .
    param attributes is a 2d numpy array with candidate and attributes values of the candidates
    param attributes_values is an array of values of attributes. the interested attributes have value and
    the uninterested attribute's value is nan.
    """
    # groups is an array of 2d arrays, first  array is an array of wanted candidates,
    # second is the array of other candidates.
    groups = group_by(attributes, attributes_values)
    # group of candidates whose attributes values are  equal to attributes_values
    wanted_group = groups[0]
    # group of candidates whose attributes values are not equal to attributes_values
    other_group = groups[1]

    # number of pair in mixed pairs equals to (number of candidate in wanted_group*number of candidate in
    # other_group)
    mixed_pairs = len(wanted_group) * len(other_group)
    # number of count pairs
    count_pairs = 0
    # compute the number of count pair
    for wanted_candidate in wanted_group:
        for other_candidate in other_group:
            rank_of_wanted = rank[np.argwhere(rank[:, 0] == wanted_candidate), 1]
            rank_of_other = rank[np.argwhere(rank[:, 0] == other_candidate), 1]
            if rank_of_wanted < rank_of_other:
                count_pairs += 1

    # fpr score is the result
    fpr = count_pairs / mixed_pairs
    return fpr


def group_by(attributes, attributes_values):
    """
    select the candidates, whose attributes values are equal to attributes_values,then group them into wanted_group,
    group other candidate into other_group.
    param attributes is a 2d numpy array with candidate and attributes values of the candidates
    param attributes_values is an array of values of attributes. the interested attributes have value and
    the uninterested attribute's value is nan.
    """
    # group of candidates whose attributes values are  equal to attributes_values
    wanted_group = []
    # group of candidates whose attributes values are not equal to attributes_values
    other_group = []
    # divide candidates into 2 groups
    for attribute in attributes:
        selected_candidate = True
        # check whether the candidates' attributes values are  equal to attributes_values
        for value in range(len(attributes_values)):
            # if the value of one attribute in attributes_values is nan don't check it
            # if it is not nan check the value
            if ~pd.isnull(attributes_values[value]):
                # check value if is not equal selected_candidate become False and end loop.
                if attribute[value + 1] != attributes_values[value]:
                    selected_candidate = False
                    break
        # if candidate is qualified add it to qualified_candidates,if not add it to other_group
        if selected_candidate:
            wanted_group.append(attribute[0])
        else:
            other_group.append(attribute[0])

    return wanted_group, other_group
