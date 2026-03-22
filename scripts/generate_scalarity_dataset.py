import os

import numpy as np

from kemeny_transformer.data import synthesis as data_syn


def main():
    for nb_candidates in [20,50,100,150,200]:
        test_dataset_random=np.zeros(shape=(500,8,nb_candidates),dtype=np.float32)
        test_dataset_repeat=np.zeros(shape=(500,8,nb_candidates),dtype=np.float32)
        test_dataset_jiggling=np.zeros(shape=(500,8,nb_candidates),dtype=np.float32)
        test_dataset_random,is_all_permutation=data_syn.generate_batch_dataset_random(500,8,nb_candidates)
        print(f'test dataset random is all permutation={is_all_permutation}')

        test_dataset_repeat,is_all_permutation=data_syn.generate_batch_dataset_repeat(500,8,nb_candidates)
        print(f'test dataset repeat is all permutation={is_all_permutation}')

        test_dataset_jiggling,is_all_permutation=data_syn.generate_batch_dataset_jiggling(500,8,nb_candidates)
        print(f'test dataset repeat is all permutation={is_all_permutation}')
        os.system("mkdir test_dataset")
        np.save(f"test_dataset/test_dataset_{nb_candidates}_random.npy",test_dataset_random)
        np.save(f"test_dataset/test_dataset_{nb_candidates}_repeat.npy",test_dataset_repeat)
        np.save(f"test_dataset/test_dataset_{nb_candidates}_jiggling.npy",test_dataset_jiggling)
if __name__ == '__main__':
    print(('main'))
    main()
