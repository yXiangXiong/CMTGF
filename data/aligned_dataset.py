import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_labelset
from PIL import Image
import random
import torch
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.interpolation import rotate


def max_min_norm(arr, min_value, max_value):
    arr_std = (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr))
    arr_scaled = arr_std * (max_value - min_value) + min_value
    return arr_scaled


def Crop_HU(arr, Min, Max):
    arr[arr < Min] = Min
    arr[arr > Max] = Max
    return arr


def image_augmentation_3d(image, label, contrast):
    I = image[:, :, :]
    G = label[:, :, :]
    K = contrast[:, :, ]

    angle = random.uniform(-10, 10)
    Num = random.sample(range(3), 1)
    if Num == 0:
        I = rotate(I, angle, mode='nearest', axes=(0, 1), reshape=False)
        G = rotate(G, angle, mode='nearest', axes=(0, 1), reshape=False)
        K = rotate(K, angle, mode='nearest', axes=(0, 1), reshape=False)
    elif Num == 1:
        I = rotate(I, angle, mode='nearest', axes=(1, 2), reshape=False)
        G = rotate(G, angle, mode='nearest', axes=(1, 2), reshape=False)
        K = rotate(K, angle, mode='nearest', axes=(1, 2), reshape=False)
    elif Num == 2:
        I = rotate(I, angle, mode='nearest', axes=(0, 2), reshape=False)
        G = rotate(G, angle, mode='nearest', axes=(0, 2), reshape=False)
        K = rotate(K, angle, mode='nearest', axes=(0, 2), reshape=False)
    return I, G, K  # {'volume': image, 'mask': label_new}

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.csv_path = os.path.join(opt.dataroot, 'ground_truth_classification.csv')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.C_paths = make_dataset(self.dir_C)
        self.labels = []

        patients, labels = make_labelset(self.csv_path, self.dir_A)
        labels = list(map(int, labels))

        self.A_paths = sorted(self.A_paths)

        for ap in self.A_paths:
            for i in range(len(patients)):
                if patients[i] == ap:
                    self.labels.append(labels[i])

        assert (len(self.A_paths) == len(self.labels))

        self.B_paths = sorted(self.B_paths)
        self.C_paths = sorted(self.C_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        C_path = self.C_paths[index % self.C_size]
        L = self.labels[index % self.A_size]
        # 'nii.gz' in A_path:  # Very hacky 2d/3d npy load  TODO: proper data loading/normalization
        a_itk = sitk.ReadImage(A_path)
        b_itk = sitk.ReadImage(B_path)
        c_itk = sitk.ReadImage(C_path)

        A_img = sitk.GetArrayFromImage(a_itk)  # aorta non-contrast
        B_img = sitk.GetArrayFromImage(b_itk)  # aorta cta
        C_img = sitk.GetArrayFromImage(c_itk)  # aorta seg

        A_img = Crop_HU(A_img, 0, 200)
        B_img = Crop_HU(B_img, 0, 800)

        A_img = max_min_norm(A_img, -1, 1)
        B_img = max_min_norm(B_img, -1, 1)
        C_img = max_min_norm(C_img, -1, 1)

        A_img, B_img, C_img = image_augmentation_3d(A_img, B_img, C_img)  # if opt.isTrain
        if len(A_img.shape) == 3:
            A = torch.from_numpy(A_img).unsqueeze(0).float()
            B = torch.from_numpy(B_img).unsqueeze(0).float()
            C = torch.from_numpy(C_img).unsqueeze(0).float()
            B = torch.cat([C, B])
        else:
            raise NotImplementedError('Unknown number of data dimensions:', A_img.shape)

        return {'A': A, 'B': B, 'L': L,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'AlignedDataset'
