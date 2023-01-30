import nibabel as nib
from scipy.io import loadmat
import numpy as np

import json
import os
import glob
import re
import natsort

import argparse


import os
from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle
from collections import OrderedDict

import glob
import natsort
import json

from sklearn.model_selection import GroupKFold

def main():
    parser = argparse.ArgumentParser(description='dataset conversion to decathlon format')
    parser.add_argument('-t', '--task_name', type=str)
    parser.add_argument('-s', '--seed', type=int, default=2023)
    parser.add_argument('-n', '--n_splits', type=int, default=5)


    args = parser.parse_args()

    seed = args.seed
    task = args.task_name

    print(f'Make {args.n_splits} splits, for task: {task}, with seed {seed}')
    pkl_path = os.path.join(os.environ['nnUNet_preprocessed'], task, 'splits_final.pkl')
    data_path = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task, 'labelsTr', '*.nii.gz')
    data_path = natsort.natsorted(glob.glob(data_path))
    data_path = np.array([os.path.basename(d)[:-7] for d in data_path])
    data_idx = np.arange(len(data_path))

    cases_fn = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task, 'cases.json')
    with open(cases_fn, 'r') as f:
        cases = json.load(f)
    patients = np.array(list(cases.values()))

    group_kfold = GroupKFold(n_splits=args.n_splits)

    splits = []
    for i, (train_index, test_index) in enumerate(group_kfold.split(data_path, groups=patients)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}, group={patients[train_index]}, names={data_path[train_index]}")
        print(f"  Test:  index={test_index}, group={patients[test_index]}, names={data_path[test_index]}")
        split = OrderedDict()
        split['train'] = data_path[train_index]
        split['val'] = data_path[test_index]
        splits.append(split)

    save_pickle(splits, pkl_path)

if __name__ == '__main__':
    main()