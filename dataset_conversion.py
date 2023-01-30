import nibabel as nib
from scipy.io import loadmat
import numpy as np

import json
import os
import glob
import re
import natsort

import argparse


def main():
    parser = argparse.ArgumentParser(description='dataset conversion to decathlon format')
    parser.add_argument('target_folder', type=str)
    parser.add_argument('data_folder', type=str)
    parser.add_argument('-l', '--labels', action='store_true')
    parser.add_argument('-p', '--postfix', type=str)

    args = parser.parse_args()

    cwd = os.getcwd()
    target_folder = args.target_folder
    image_folder = os.path.join(cwd, target_folder, 'imagesT' + args.postfix)
    mask_folder = os.path.join(cwd, target_folder, 'labelsT' + args.postfix)

    print(f'Image folder: {image_folder}')
    print(f'Mask folder: {mask_folder}')

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    labels = args.labels
    print(f"Make labels: {labels}")

    name = "wrist_"

    datafolder = args.data_folder

    files = natsort.natsorted(glob.glob(datafolder))

    pattern = r"Patient(\d{3})"
    matches = re.finditer(pattern, ''.join(files), re.MULTILINE)
    patients = [int(m.group(1)) for m in matches]
    patients = list(set(patients))
    N_patients = patients[-1]

    number = 0
    cases = {}
    cartilages = {}
    for p in patients:
        filt_files = list(filter(lambda x: f"Patient{p:#03d}" in x, files))
        for file in filt_files:
            number += 1

            mat = loadmat(file)
            if 'Slice' in mat.keys():
                mat = mat['Slice']
                image = mat[..., 0]
                mask = mat[..., 1]
            else:
                image = mat['original']
                mask = mat['mask']

            image = np.expand_dims(np.swapaxes(image, 0, 1), -1)  # for nibabel lib it is needed to swapaxes
            image_name = name + f"{number:#03d}" + "_0000"
            nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
            nib.save(nifti_image, os.path.join(image_folder, image_name + '.nii.gz'))

            if labels:
                mask = np.expand_dims(np.swapaxes(mask, 0, 1), -1)
                mask_name = name + f"{number:#03d}"
                nifti_mask = nib.Nifti1Image(mask, affine=np.eye(4))
                nib.save(nifti_mask, os.path.join(mask_folder, mask_name + '.nii.gz'))
            cartilage = np.sum(mask.flatten(), dtype=float)
            cartilages[f"{number:#03d}"] = cartilage
            cases[f"{number:#03d}"] = p

    with open(os.path.join(cwd, target_folder, 'cases.json'), 'w') as f:
        json.dump(cases, f)
    with open(os.path.join(cwd, target_folder, 'cartilages.json'), 'w') as f:
        json.dump(cartilages, f)

    image_folder = os.path.join(cwd, target_folder, 'imagesTr')
    mask_folder = os.path.join(cwd, target_folder, 'labelsTr')

    train_patient_images = [os.path.basename(i)[:-12] + '.nii.gz' for i in
                            natsort.natsorted(glob.glob(image_folder + '/*'))]
    # train_patient_images = [os.path.basename(i) for i in natsort.natsorted(glob.glob(image_folder+'/*'))]

    train_patient_masks = [os.path.basename(i) for i in natsort.natsorted(glob.glob(mask_folder + '/*'))]
    test_patient_names = [os.path.basename(i)[:-12] + '.nii.gz' for i in
                          natsort.natsorted(glob.glob(image_folder[:-1] + 's/*'))]

    # test_patient_names = [os.path.basename(i) for i in natsort.natsorted(glob.glob(image_folder[:-1]+'s/*'))]

    json_dict = {}
    json_dict['name'] = name[:-1]
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "3DVIBE",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "wrist"
    }

    json_dict['numTraining'] = len(train_patient_images)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % j} for i, j in
                             zip(train_patient_images, train_patient_masks)]
    json_dict['test'] = ["./imagesTs/%s" % i for i in test_patient_names]

    with open(os.path.join(cwd, target_folder, 'dataset.json'), 'w') as f:
        json.dump(json_dict, f)

if __name__ == '__main__':
    main()