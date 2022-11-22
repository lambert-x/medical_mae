import copy
import os
import random
import numpy as np
import torchvision.transforms.functional
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import torch
import torchvision.transforms as transforms
import pandas as pd
import medpy.io

NORMALIZATION_STATISTICS = {"self_learning_cubes_32": [[0.11303308354465243, 0.12595135887180803]],
                            "self_learning_cubes_64": [[0.11317437834743148, 0.12611378817031038]],
                            "lidc": [[0.23151727, 0.2168428080133056]],
                            "luna_fpr": [[0.18109835972793722, 0.1853707675313153]],
                            "lits_seg": [[0.46046468844492944, 0.17490586272419967]],
                            "pe": [[0.26125720740546626, 0.20363551346695796]]}


# -------------------------------------Data augmentation-------------------------------------
class Augmentation():
    def __init__(self, normalize):
        if normalize.lower() == "imagenet":
            self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif normalize.lower() == "chestx-ray":
            self.normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        elif normalize.lower() == "none":
            self.normalize = None
        else:
            print("mean and std for [{}] dataset do not exist!".format(normalize))
            exit(-1)

    def get_augmentation(self, augment_name, mode):
        try:
            aug = getattr(Augmentation, augment_name)
            return aug(self, mode)
        except:
            print("Augmentation [{}] does not exist!".format(augment_name))
            exit(-1)

    def basic(self, mode):
        transformList = []
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def _basic_crop(self, transCrop, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomCrop(transCrop))
        else:
            transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_crop_224(self, mode):
        transCrop = 224
        return self._basic_crop(transCrop, mode)

    def _basic_resize(self, size, mode="train"):
        transformList = []
        transformList.append(transforms.Resize(size))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_resize_224(self, mode):
        size = 224
        return self._basic_resize(size, mode)

    def _basic_crop_rot(self, transCrop, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomCrop(transCrop))
            transformList.append(transforms.RandomRotation(7))
        else:
            transformList.append(transforms.CenterCrop(transCrop))

        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_crop_rot_224(self, mode):
        transCrop = 224
        return self._basic_crop_rot(transCrop, mode)

    def _full(self, transCrop, transResize, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomResizedCrop(transCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.RandomRotation(7))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "val":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "test":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(
                    transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def full_224(self, mode):
        transCrop = 224
        transResize = 256
        return self._full(transCrop, transResize, mode)

    def full_448(self, mode):
        transCrop = 448
        transResize = 512
        return self._full(transCrop, transResize, mode)

    def _full_colorjitter(self, transCrop, transResize, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomResizedCrop(transCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.RandomRotation(7))
            transformList.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "val":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "test":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(
                    transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def full_colorjitter_224(self, mode):
        transCrop = 224
        transResize = 256
        return self._full_colorjitter(transCrop, transResize, mode)


from torch.utils.data import Dataset


# --------------------------------------------Downstream ChestX-ray14-------------------------------------------
class ChestX_ray14(Dataset):
    def __init__(self, data_dir, file, augment,
                 num_class=14, img_depth=3, heatmap_path=None,
                 pretraining=False):
        self.img_list = []
        self.img_label = []

        with open(file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(data_dir, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        self.augment = augment
        self.img_depth = img_depth
        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')
        else:
            self.heatmap = None
        self.pretraining = pretraining

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        file = self.img_list[index]
        label = self.img_label[index]

        imageData = Image.open(file).convert('RGB')
        if self.heatmap is None:
            imageData = self.augment(imageData)
            img = imageData
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return img, label
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            imageData, heatmap = self.augment(imageData, heatmap)
            img = imageData
            # heatmap = torch.tensor(np.array(heatmap), dtype=torch.float)
            heatmap = heatmap.permute(1, 2, 0)
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return [img, heatmap], label


class Covidx(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transform):
        self.data_dir = data_dir
        self.phase = phase

        self.classes = ['normal', 'positive', 'pneumonia', 'COVID-19']
        self.class2label = {c: i for i, c in enumerate(self.classes)}

        # collect training/testing files
        if phase == 'train':
            with open(os.path.join(data_dir, 'train_COVIDx9A.txt'), 'r') as f:
                lines = f.readlines()
        elif phase == 'test':
            with open(os.path.join(data_dir, 'test_COVIDx9A.txt'), 'r') as f:
                lines = f.readlines()
        lines = [line.strip() for line in lines]
        self.datalist = list()
        for line in lines:
            patient_id, fname, label, source = line.split(' ')
            if phase in ('train', 'val'):
                self.datalist.append((os.path.join(data_dir, 'train', fname), label))
            else:
                self.datalist.append((os.path.join(data_dir, 'test', fname), label))

        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        fpath, label = self.datalist[index]
        image = Image.open(fpath).convert('RGB')
        image = self.transform(image)
        label = self.class2label[label]
        label = torch.tensor(label, dtype=torch.long)
        return image, label


class Node21(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transform):
        self.data_dir = data_dir
        self.phase = phase

        if phase == 'train':
            with open(os.path.join(data_dir, 'train_mae.txt')) as f:
                fnames = f.readlines()
        elif phase == 'test':
            with open(os.path.join(data_dir, 'test_mae.txt')) as f:
                fnames = f.readlines()
        fnames = [fname.strip() for fname in fnames]

        self.datalist = list()
        for line in fnames:
            fname, label = line.split(' ')
            self.datalist.append((os.path.join(data_dir, 'images', fname), int(label)))
        # metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        # self.datalist = list()
        # for i in range(len(metadata)):
        #     fname = metadata.loc[i, 'img_name']
        #     if fname in fnames:
        #         label = metadata.loc[i, 'label']
        #         self.datalist.append((os.path.join(data_dir, 'images', fname), label))

        # transforms
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        fpath, label = self.datalist[index]
        image, _ = medpy.io.load(fpath)
        image = image.astype(np.float)
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
        image = image.transpose(1, 0)
        image = Image.fromarray(image).convert('RGB')
        image = self.transform(image)
        label = torch.tensor([label], dtype=torch.float32)
        return image, label


class CheXpert(Dataset):
    '''
    Reference:
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''

    def __init__(self,
                 csv_path,
                 image_root_path='',
                 class_index=0,
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 transform=None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 mode='train',
                 heatmap_path=None,
                 pretraining=False
                 ):

        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

            # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')

        else:
            self.heatmap = None

        # impute missing values
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

            # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 5 classes
            print('Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:  # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.mode = mode
        self.class_index = class_index

        self.transform = transform

        self._images_list = [image_root_path + path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()

        if verbose:
            if class_index != -1:
                print('-' * 30)
                if flip_label:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[-1] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                else:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[0] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                print('-' * 30)
            else:
                print('-' * 30)
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    imratio = self.value_counts_dict[class_key][1] / (
                            self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                    print()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                print('-' * 30)
        self.pretraining = pretraining

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        # image = cv2.imread(self._images_list[idx], 0)
        # image = Image.fromarray(image)
        # if self.mode == 'train':
        #     image = self.transform(image)
        # image = np.array(image)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #
        # # resize and normalize; e.g., ToTensor()
        # image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        # image = image / 255.0
        # __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        # __std__ = np.array([[[0.229, 0.224, 0.225]]])
        # image = (image - __mean__) / __std__

        if self.heatmap is None:
            image = Image.open(self._images_list[idx]).convert('RGB')

            image = self.transform(image)

            # image = image.transpose((2, 0, 1)).astype(np.float32)

            if self.class_index != -1:  # multi-class mode
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
                # label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
            else:
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

            if self.pretraining:
                label = -1

            return image, label
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            image = Image.open(self._images_list[idx]).convert('RGB')
            image, heatmap = self.transform(image, heatmap)
            heatmap = heatmap.permute(1, 2, 0)
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            if self.class_index != -1:  # multi-class mode
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
                # label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
            else:
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

            if self.pretraining:
                label = -1

            return [image, heatmap], label


'''
 NIH:(train:75312  test:25596)
 0:A 1:Cd 2:Ef 3:In 4:M 5:N 6:Pn 7:pnx 8:Co 9:Ed 10:Em 11:Fi 12:PT 13:H
 Chexpert:(train:223415 val:235)
 0:NF 1:EC 2:Cd 3:AO 4:LL 5:Ed 6:Co 7:Pn 8:A 9:Pnx 10:Ef 11:PO 12:Fr 13:SD
 combined:
 0: Airspace Opacity(AO)	1: Atelectasis(A)	2:Cardiomegaly(Cd)	3:Consolidation(Co)
 4:Edema(Ed)	5:Effusion(Ef)	6:Emphysema(Em)	7:Enlarged Card(EC)	8:Fibrosis(Fi)	
 9:Fracture(Fr)	10:Hernia(H)	11:Infiltration(In)	12:Lung lession(LL)	13:Mas(M)	
 14:Nodule(N)	15:No finding(NF)	16:Pleural thickening(PT)	17:Pleural other(PO)	18:Pneumonia(Pn)	
 19:Pneumothorax(Pnx)	20:Support Devices(SD)
'''


class combine(Dataset):
    def __init__(self, path_image_1, path_image_2, path_list_1, path_list_2, transform1, transform2, reduct_ratio=1):

        self.path_image_1 = path_image_1
        self.path_image_2 = path_image_2
        self.path_list_1 = path_list_1
        self.path_list_2 = path_list_2
        self.transform1 = transform1
        self.transform2 = transform2
        self.num_class = 21

        self.img_list = []
        self.img_label = []
        self.source = []
        self.dict = [{'1.0': 1.0, '': 0.0, '0.0': 0.0, '-1.0': -1.0},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]

        self.dict_nih2combine = {0: 1, 1: 2, 2: 5, 3: 11, 4: 13, 5: 14, 6: 18, 7: 19, 8: 3, 9: 4, 10: 6, 11: 8, 12: 16,
                                 13: 10}
        self.dict_chex2combine = {0: 15, 1: 7, 2: 2, 3: 0, 4: 12, 5: 4, 6: 3, 7: 18, 8: 1, 9: 19, 10: 5, 11: 17, 12: 9,
                                  13: 20}

        with open(self.path_list_1, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(self.path_image_1, lineItems[0])
                    imageLabel = lineItems[1:14 + 1]
                    self.img_list.append(imagePath)
                    tmp_label = [-1] * 21
                    for i in range(14):
                        tmp_label[self.dict_nih2combine[i]] = float(imageLabel[i])
                    self.img_label.append(tmp_label)
                    self.source.append(0)

        # random.seed(1)
        # self.reduct_ratio = reduct_ratio
        # self.img_list = np.array(self.img_list)
        # self.img_label = np.array(self.img_label)
        # self.source=np.array(self.source)
        # index = sample(range(len(self.img_list)), len(self.img_list) // reduct_ratio)
        # self.img_list = self.img_list[index]
        # self.img_label = self.img_label[index]
        # self.source = self.source[index]
        # self.img_list = self.img_list.tolist()
        # self.img_label = self.img_label.tolist()
        # self.source=self.source.tolist()
        # index=sample(range(166739), len(self.img_list))
        cnt = -1

        with open(self.path_list_2, "r") as fileDescriptor:
            line = fileDescriptor.readline()
            line = True
            while line:
                line = fileDescriptor.readline()
                cnt += 1
                if line:  # and cnt in index:
                    lineItems = line.strip('\n').split(',')
                    imagePath = os.path.join(self.path_image_2, lineItems[0])
                    imageLabel = lineItems[5:5 + 14]
                    self.img_list.append(imagePath)
                    tmp_label = [-1] * 21
                    for idx, _ in enumerate(imageLabel):
                        # if idx not in [5,8,2,6,10]:
                        #     continue
                        # if idx in [5,8]:
                        #     imageLabel[idx]=self.dict[0][imageLabel[idx]]
                        # elif idx in [2,6,10]:
                        #     imageLabel[idx]=self.dict[1][imageLabel[idx]]
                        # labels.append(float(imageLabel[idx]))
                        tmp_label[self.dict_chex2combine[idx]] = self.dict[0][imageLabel[idx]]
                    self.img_label.append(tmp_label)
                    self.source.append(1)
        self.img_label = torch.tensor(self.img_label)
        self.source = torch.tensor(self.source)

    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')

        if self.transform1 is not None:
            img = self.transform1(img)
        # label = torch.zeros((self.num_class),dtype=torch.float)
        #
        # for i in range(0, self.num_class):
        #     label[i] = self.img_label[idx][i]

        return img, self.img_label[idx], self.source[idx]

    def __len__(self):
        return len(self.img_list)


class combine_semi(Dataset):
    def __init__(self, path_image_1, path_image_2, path_list_1, path_list_2, transform1, transform2, transform_semi,
                 reduct_ratio=1):

        self.path_image_1 = path_image_1
        self.path_image_2 = path_image_2
        self.path_list_1 = path_list_1
        self.path_list_2 = path_list_2
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform_semi = transform_semi
        self.num_class = 21

        self.img_list = []
        self.img_label = []
        self.source = []
        self.dict = [{'1.0': 1.0, '': 0.0, '0.0': 0.0, '-1.0': -1.0},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]

        self.dict_nih2combine = {0: 1, 1: 2, 2: 5, 3: 11, 4: 13, 5: 14, 6: 18, 7: 19, 8: 3, 9: 4, 10: 6, 11: 8, 12: 16,
                                 13: 10}
        self.dict_chex2combine = {0: 15, 1: 7, 2: 2, 3: 0, 4: 12, 5: 4, 6: 3, 7: 18, 8: 1, 9: 19, 10: 5, 11: 17, 12: 9,
                                  13: 20}

        with open(self.path_list_1, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(self.path_image_1, lineItems[0])
                    imageLabel = lineItems[1:14 + 1]
                    self.img_list.append(imagePath)
                    tmp_label = [-1] * 21
                    for i in range(14):
                        # if i not in [0,9,2,8,7]:
                        #     continue
                        tmp_label[self.dict_nih2combine[i]] = float(imageLabel[i])
                    self.img_label.append(tmp_label)
                    self.source.append(0)

        # random.seed(1)
        # self.reduct_ratio = reduct_ratio
        # self.img_list = np.array(self.img_list)
        # self.img_label = np.array(self.img_label)
        # self.source=np.array(self.source)
        # index = sample(range(len(self.img_list)), len(self.img_list) // reduct_ratio)
        # self.img_list = self.img_list[index]
        # self.img_label = self.img_label[index]
        # self.source = self.source[index]
        # self.img_list = self.img_list.tolist()
        # self.img_label = self.img_label.tolist()
        # self.source=self.source.tolist()
        # index=sample(range(166739), len(self.img_list))
        cnt = -1

        with open(self.path_list_2, "r") as fileDescriptor:
            line = fileDescriptor.readline()
            line = True
            while line:
                line = fileDescriptor.readline()
                cnt += 1
                if line:  # and cnt in index:
                    lineItems = line.strip('\n').split(',')
                    imagePath = os.path.join(self.path_image_2, lineItems[0])
                    imageLabel = lineItems[5:5 + 14]
                    self.img_list.append(imagePath)
                    tmp_label = [-1] * 21
                    for idx, _ in enumerate(imageLabel):
                        # if idx not in [2, 7, 8, 5, 10]:
                        #     continue
                        # if idx in [5,8]:
                        #     imageLabel[idx]=self.dict[0][imageLabel[idx]]
                        # elif idx in [2,6,10]:
                        #     imageLabel[idx]=self.dict[1][imageLabel[idx]]
                        # labels.append(float(imageLabel[idx]))
                        tmp_label[self.dict_chex2combine[idx]] = self.dict[0][imageLabel[idx]]
                    self.img_label.append(tmp_label)
                    self.source.append(1)
        self.img_label = torch.tensor(self.img_label)
        self.source = torch.tensor(self.source)

    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')
        img2 = Image.open(self.img_list[idx]).convert('RGB')

        if self.transform1 is not None:
            img = self.transform1(img)
        # label = torch.zeros((self.num_class),dtype=torch.float)
        #
        # for i in range(0, self.num_class):
        #     label[i] = self.img_label[idx][i]

        return img, self.img_label[idx], self.source[idx], self.transform_semi(img2)

    def __len__(self):
        return len(self.img_list)


class FELIX(Dataset):
    def __init__(self, data_dir, file, augment, whitening=True):
        self.imgs = []
        self.labels = []
        self.normal_slice_indexes = []
        debug_length = 0
        with open(file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    data_path = os.path.join(data_dir, line)[:-1]
                    # print(data_path)
                    data = np.load(data_path).transpose([1, 0, 2, 3])
                    img = data[..., 0]
                    img = np.clip(img, -125, 275)
                    label = data[..., 1]
                    normal_slice_indexes = np.where(np.all(label <= 20, axis=(0, 1)))[0]
                    mean = 20.77
                    std = 102.79

                    img = (img - mean) / std
                    img = (img - img.min()) / (img.max() - img.min()) * 255
                    img = img.astype(np.uint8)
                    self.imgs.append(img)
                    # self.labels.append(label)
                    self.normal_slice_indexes.append(normal_slice_indexes)
                    debug_length += 1

        self.whitening = whitening
        self.augment = augment

        self.data_len = len(self.imgs)

    def get_mid_slice_index(self, normal_indexes):
        index = None
        while index is None:
            index = np.random.choice(normal_indexes)
            if (index - 1) not in normal_indexes or (index + 1) not in normal_indexes:
                index = None
        return index

    def __len__(self):
        return 131072

    def __getitem__(self, index):
        index = index % self.data_len
        img = self.imgs[index]
        # label = self.labels[index]
        normal_slice_indexes = self.normal_slice_indexes[index]
        z = self.get_mid_slice_index(normal_slice_indexes)
        img = img[:, :, z - 1:z + 2]
        img = self.augment(img)
        # label = torch.tensor(label, dtype=torch.float)
        label = torch.zeros(0)
        return img, label


class RetinaDataset(Dataset):
    def __init__(self, data_dir, file, transform=None):
        self.df = pd.read_csv(file)
        self.transform = transform
        self.data_dir = data_dir

        self.imgs = []
        self.labels = []
        self.transform_center_crop = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(256),
        ])
        for i in range(len(self.df)):
            img_path = os.path.join(self.data_dir, self.df.iloc[i].image + ".jpeg")
            img = Image.open(img_path)
            img = self.transform_center_crop(img)
            self.imgs.append(img)
            self.labels.append(self.df.iloc[i].level)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img = self.imgs[index]
        label = self.labels[index]
        if (self.transform):
            img = self.transform(img)

        return img, torch.tensor(label)


class MIMIC(Dataset):
    """
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY, Mark RG, Horng S.
    MIMIC-CXR: A large publicly available database of labeled chest radiographs.
    arXiv preprint arXiv:1901.07042. 2019 Jan 21.

    https://arxiv.org/abs/1901.07042

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    # def __init__(self, imgpath, csvpath,metacsvpath, views=["PA"], transform=None, data_aug=None,
    # flat_dir=True, seed=0, unique_patients=True):
    def __init__(self, path, version="chexpert", split="train", transform=None, views=["AP", "PA"],
                 unique_patients=False, pretraining=False):
        super().__init__()
        splits = pd.read_csv(os.path.join(path, "mimic-cxr-2.0.0-split.csv.gz"))
        imgpath = os.path.join(path, "files_resized_320")
        metacsvpath = os.path.join(path, "mimic-cxr-2.0.0-metadata.csv.gz")
        csvpath = os.path.join(path, f"mimic-cxr-2.0.0-{version}.csv.gz")
        # np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.metacsv = pd.read_csv(self.metacsvpath)

        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        self.metacsv = self.metacsv.set_index(['subject_id', 'study_id'])

        self.csv = self.csv.join(self.metacsv).reset_index()

        # Keep only the desired view
        self.views = views
        if self.views:
            if type(views) is not list:
                views = [views]
            self.views = views

            self.csv["view"] = self.csv["ViewPosition"]
            self.csv = self.csv[self.csv["view"].isin(self.views)]

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        # rename pathologies
        self.pathologies = np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")

        ########## add consistent csv values

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]

        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)

        df = self.csv.copy()
        df["ind"] = np.arange(len(df))
        df = pd.merge(df, splits, on=("dicom_id", "study_id", "subject_id"), how="left")
        df = df[df.split == split]
        self.csv = df
        self.labels = self.labels[df.ind.values]
        self.pretraining = pretraining

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views,
                                                                                       self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])

        img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.pretraining:
            target = -1
        else:
            target = torch.from_numpy(self.labels[idx]).float()
        return img, target
