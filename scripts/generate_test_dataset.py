import os

import numpy as np

from kemeny_transformer.data import synthesis as data_syn


def main():
    num_items=[25,50,75,100]
    for num in num_items:
        test_dataset_random=np.zeros(shape=(500,8,num),dtype=np.float32)
        test_dataset_repeat=np.zeros(shape=(1000,8,num),dtype=np.float32)
        test_dataset_jiggling=np.zeros(shape=(1000,8,num),dtype=np.float32)
        test_dataset_random,is_all_permutation=data_syn.generate_batch_dataset_random(2000,8,num)
        print(f'test dataset random is all permutation={is_all_permutation}')

        test_dataset_repeat,is_all_permutation=data_syn.generate_batch_dataset_repeat(10000,8,num)
        print(f'test dataset repeat is all permutation={is_all_permutation}')

        test_dataset_jiggling,is_all_permutation=data_syn.generate_batch_dataset_jiggling(10000,8,num)
        print(f'test dataset repeat is all permutation={is_all_permutation}')
        os.system("mkdir -p test_dataset/ablation_test_dataset")
        np.save(f'test_dataset/ablation_test_dataset/test_dataset_random_{num}.npy',test_dataset_random)
        np.save(f'test_dataset/ablation_test_dataset/test_dataset_repeat_{num}.npy',test_dataset_repeat)
        np.save(f'test_dataset/ablation_test_dataset/test_dataset_jiggling_{num}.npy',test_dataset_jiggling)
if __name__ == '__main__':
    print(('main'))
    main()
