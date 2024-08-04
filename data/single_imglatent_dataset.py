"""The class for dataset in EditNet."""
import os
import random
import torch
import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import random

_LATENT_EXTENSIONS = ['npy']
_IMG_EXTENSIONS = ['jpg', 'png']


def load_img(img_path, size):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    img[:,:,:3] = img[:,:,:3][:,:,::-1]
    img = img.astype(np.float32) / 127.5 - 1
    img = img.transpose((2, 0, 1)).clip(-1, 1)
    return torch.from_numpy(img).unsqueeze(0)


def latent_check(name):
    return '.' in name and name.split('.')[-1] in _LATENT_EXTENSIONS and 'real_feature' not in name


def img_check(name):
    return '.' in name and name.split('.')[-1] in _IMG_EXTENSIONS


def make_dataset(folder, data_type):
    paths = []
    assert os.path.isdir(folder), f'{folder} is not a valid directory!'
    check = img_check if data_type == 'img' else latent_check
    for root, _, fnames in sorted(os.walk(folder)):
        for fname in fnames:
            if not check(fname):
                continue
            path = os.path.join(root, fname)
            paths.append(path)

    return paths


def sample(l, indices):
    return list(map(lambda idx: l[idx], indices))


class ImgLatentDataset(data.Dataset):
    """The class for loading paired latent codes and images."""
    def __init__(self, opt):
        self.opt = opt
        self.img_dir = opt.img_dir
        self.latent_dir = opt.latent_dir
        self.img_paths = sorted(make_dataset(self.img_dir, 'img'))
        self.latent_paths = sorted(make_dataset(self.latent_dir, 'latent'))
        if len(self.img_paths) != len(self.latent_paths):
            raise ValueError(f'The images and latent codes should be paired, '
                             f'i.e, `len(self.img_paths)` and '
                             f'`len(self.latent_paths)` should be the same, '
                             f'however, ({len(self.img_paths)}, '
                             f'{len(self.latent_paths)}) is received!')
        self.length = len(self.img_paths)

        if opt.num_shot is not None:
            indices = list(range(self.length))
            if opt.num_shot < self.length:
                indices = random.sample(indices, opt.num_shot)
                
            self.img_paths = sample(self.img_paths, indices)
            self.latent_paths = sample(self.latent_paths, indices)

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        img_path = self.img_paths[index % len(self.img_paths)]
        latent_path = self.latent_paths[index % len(self.latent_paths)]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.opt.img_size, self.opt.img_size),
                         interpolation=cv2.INTER_CUBIC)
        img[:,:,:3] = img[:,:,:3][:,:,::-1]
        img = img.astype(np.float32) / 127.5 - 1
        img = img.transpose((2, 0, 1)).clip(-1, 1)
        latent = np.load(latent_path)

        return dict(latent=latent.squeeze(0), img=img,
                    latent_path=latent_path, img_path=img_path)

    def __len__(self):
        return len(self.img_paths)


class ImgLatentDataset_ht(data.Dataset):
    """The class for loading paired latent codes and images."""

    def __init__(self, opt):
        self.opt = opt
        print(opt.img_dir)
        self.img_dir = opt.img_dir
        self.latent_dir = opt.latent_dir
        self.img_paths = sorted(make_dataset(self.img_dir, 'img'))
        self.length = len(self.img_paths)

        if opt.num_shot is not None:
            indices = list(range(self.length))
            if opt.num_shot < self.length:
                indices = random.sample(indices, opt.num_shot)

            self.img_paths = sample(self.img_paths, indices)

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        img_path = self.img_paths[index % len(self.img_paths)]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.opt.img_size, self.opt.img_size),
                         interpolation=cv2.INTER_CUBIC)
        img[:, :, :3] = img[:, :, :3][:, :, ::-1]
        img = img.astype(np.float32) / 127.5 - 1
        img = img.transpose((2, 0, 1)).clip(-1, 1)

        return dict( img=img,img_path=img_path)

    def __len__(self):
        return len(self.img_paths)

class ImgLatentDataset_ht2(data.Dataset):
    """The class for loading paired latent codes and images."""

    def __init__(self, opt):
        self.opt = opt
        print(opt.img_dir)
        self.img_dir = opt.img_dir
        self.latent_dir = opt.latent_dir
        self.img_paths=[]
        dir1='dataset/celeba-test/%s'%opt.attr1
        dir2='dataset/celeba-test/%s'%opt.attr2
        print(opt.attr1,opt.attr2)
        self.img_paths.append(sorted(make_dataset(dir1, 'img')))
        self.img_paths.append(sorted(make_dataset(dir2, 'img')))
        self.length = len(self.img_paths[0])

        if opt.num_shot is not None:
            indices = list(range(self.length))
            if opt.num_shot < self.length:
                indices = random.sample(indices, opt.num_shot)

            self.img_paths = sample(self.img_paths, indices)

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        if random.random()>0.5:
            img_label=1   #pale_skin
        else:
            img_label=0  #gray_hair
        img_path = self.img_paths[img_label][index % len(self.img_paths[img_label])]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.opt.img_size, self.opt.img_size),
                         interpolation=cv2.INTER_CUBIC)
        img[:, :, :3] = img[:, :, :3][:, :, ::-1]
        img = img.astype(np.float32) / 127.5 - 1
        img = img.transpose((2, 0, 1)).clip(-1, 1)
        return dict( img=img,img_path=img_path,label=img_label)

    def __len__(self):
        return len(self.img_paths[0])

class ImgLatentDataset_ht3(data.Dataset):
    """The class for loading paired latent codes and images."""

    def __init__(self, opt):
        self.opt = opt
        print(opt.img_dir)
        self.img_dir = opt.img_dir
        self.latent_dir = opt.latent_dir
        self.img_paths=[]
        dir1='dataset/celeba-test/%s'%opt.attr1
        dir2='dataset/celeba-test/%s'%opt.attr2
        dir3='dataset/celeba-test/%s'%opt.attr3
        print(opt.attr1,opt.attr2,opt.attr3)
        self.img_paths.append(sorted(make_dataset(dir1, 'img')))
        self.img_paths.append(sorted(make_dataset(dir2, 'img')))
        self.img_paths.append(sorted(make_dataset(dir3, 'img')))
        self.length = len(self.img_paths[0])

        if opt.num_shot is not None:
            indices = list(range(self.length))
            if opt.num_shot < self.length:
                indices = random.sample(indices, opt.num_shot)

            self.img_paths = sample(self.img_paths, indices)

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        rand = random.random()
        if rand>0.6666666667:
            img_label=2   #pale_skin
        elif rand>0.3333333333:
            img_label=1  #gray_hair
        else:
            img_label=0
        img_path = self.img_paths[img_label][index % len(self.img_paths[img_label])]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.opt.img_size, self.opt.img_size),
                         interpolation=cv2.INTER_CUBIC)
        img[:, :, :3] = img[:, :, :3][:, :, ::-1]
        img = img.astype(np.float32) / 127.5 - 1
        img = img.transpose((2, 0, 1)).clip(-1, 1)
        return dict( img=img,img_path=img_path,label=img_label)

    def __len__(self):
        return len(self.img_paths[0])