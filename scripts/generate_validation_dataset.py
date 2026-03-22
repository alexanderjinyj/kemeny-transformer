import os

import numpy as np

from kemeny_transformer.data.synthesis import DataSynthesis as dsy


def main():
    num_items=100
    data_synthesis=dsy(random_seed=1234)
    for num_voters in [6,10]:
        validation_dataset_random=np.zeros(shape=(500,num_voters,num_items),dtype=np.float32)
        validation_dataset_repeat=np.zeros(shape=(1000,num_voters,num_items),dtype=np.float32)
        validation_dataset_jiggling=np.zeros(shape=(1000,num_voters,num_items),dtype=np.float32)
        validation_dataset_random,is_all_permutation=data_synthesis.generate_batch_dataset_random(500,num_voters,num_items)
        print(f'validation dataset random is all permutation={is_all_permutation}')

        validation_dataset_repeat,is_all_permutation=data_synthesis.generate_batch_dataset_repeat(1000,num_voters,num_items)
        print(f'validation dataset repeat is all permutation={is_all_permutation}')

        validation_dataset_jiggling,is_all_permutation=data_synthesis.generate_batch_dataset_jiggling(1000,num_voters,num_items)
        print(f'validation dataset jiggling is all permutation={is_all_permutation}')
        folder_name=f'validate_dataset/{num_voters}_voters_100_items'
        os.makedirs(folder_name,exist_ok=True)
        np.save(f'{folder_name}/validate_dataset_random.npy',validation_dataset_random)
        np.save(f'{folder_name}/validate_dataset_repeat.npy',validation_dataset_repeat)
        np.save(f'{folder_name}/validate_dataset_jiggling.npy',validation_dataset_jiggling)
if __name__ == '__main__':
    print(('main'))
    main()
