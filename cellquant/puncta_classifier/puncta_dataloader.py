import torchvision.transforms
from torch.utils.data import Dataset
import os
import torch
from tifffile import imread
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2
import glob


class PuntaDataset(Dataset):
    def __init__(self, path, train):
        self.image_list = []
        label0_count = 0
        label1_count = 0
        label2_count = 0
        for subdir in os.listdir(path):
            for file in os.listdir(os.path.join(path, subdir)):
                if file.endswith('tif') and 'neighbour' not in file:
                    if 'label0' in file:
                        if train:
                            if random.random() <= 0.7:
                                self.image_list.append(os.path.join(path, subdir, file))
                        else:
                            self.image_list.append(os.path.join(path, subdir, file))
                        label0_count += 1
                    elif 'label1' in file:
                        self.image_list.append(os.path.join(path, subdir, file))
                        label1_count += 1
                    else:
                        self.image_list.append(os.path.join(path, subdir, file))
                        label2_count += 1
                        if train:
                            for _ in range(5):
                                self.image_list.append(os.path.join(path, subdir, file))
            self.train = train
        print('label0 count', label0_count, 'label1 count', label1_count, 'label2 count', label2_count)

    def __getitem__(self, item):
        img_path = self.image_list[item]
        img_neighbour = img_path.replace('.tif', '_neighbour.tif')
        label = int(os.path.basename(img_path).split('_')[-1][5:-4])
        img = imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_neighbour = imread(img_neighbour)
        img_neighbour = cv2.cvtColor(img_neighbour, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        img, img_neighbour = self.aug(img, img_neighbour)
        return img, img_neighbour, torch.tensor(label), img_path

    def __len__(self):
        return len(self.image_list)

    def aug(self, img, img_neighbour):
        if self.train:
            if random.random() < 0.5:
                img = np.flip(img, axis=0)
                img_neighbour = np.flip(img_neighbour, axis=0)
            if random.random() < 0.5:
                img = np.rot90(img, k=1)
                img_neighbour = np.rot90(img_neighbour, k=1)
            if random.random() < 0.5:
                img = gaussian_filter(img, sigma=2)
                img_neighbour = gaussian_filter(img_neighbour, sigma=2)
        img = self.normalize(img)
        img_neighbour = self.normalize(img_neighbour)
        return torch.tensor(img).unsqueeze_(dim=0), torch.tensor(img_neighbour).unsqueeze_(dim=0)

    def normalize(self, img):
        img = img.astype(np.float32)
        img = (img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1))
        img[img < 0] = 0
        img[img > 1] = 1
        return img


class PuntaDatasetTest(Dataset):
    def __init__(self, punta_list, img):
        self.punta_list = punta_list
        self.img = img

    def __getitem__(self, item):
        y, x, r = self.punta_list[item]
        img = self.img[max(0, int(y - 1.4 * r)):min(self.img.shape[0], int(y + 1.4 * r)),
              max(0, int(x - 1.4 * r)):min(self.img.shape[1], int(x + 1.4 * r)), :]
        img_neighbour = self.img[max(0, int(y - 1.4 * 3 * r)):min(self.img.shape[0], int(y + 1.4 * 3 * r)),
                        max(0, int(x - 1.4 * 3 * r)):min(self.img.shape[1], int(x + 1.4 * 3 * r)), :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_neighbour = cv2.cvtColor(img_neighbour, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        img_neighbour = cv2.resize(img_neighbour, (48, 48))
        img, img_neighbour = self.aug(img, img_neighbour)
        return img, img_neighbour, torch.tensor(0), ''

    def __len__(self):
        return len(self.punta_list)

    def aug(self, img, img_neighbour):
        img = self.normalize(img)
        img_neighbour = self.normalize(img_neighbour)
        return torch.tensor(img).unsqueeze_(dim=0), torch.tensor(img_neighbour).unsqueeze_(dim=0)

    def normalize(self, img):
        img = img.astype(np.float32)
        img = (img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1))
        img[img < 0] = 0
        img[img > 1] = 1
        return img


class PuntaDatasetRing(Dataset):
    def __init__(self, path, train):
        self.image_list = []
        label0_count = 0
        label1_count = 0
        label2_count = 0
        self.num_neighbour = 5
        for subdir in os.listdir(path):
            # if train and 'thresh0.2' in subdir:
            #     continue
            for file in os.listdir(os.path.join(path, subdir)):
                if 'raw' in file:
                    continue
                label = int(file.split('_')[1])
                if label == 2:
                    for _ in range(1):
                        self.image_list.append(os.path.join(path, subdir, file))
                    label2_count += 1
                elif label == 1:
                    label1_count += 1
                    self.image_list.append(os.path.join(path, subdir, file))
                else:
                    # if '_thresh0.2' in subdir:
                    #     if random.random() > 0.8:
                    #         label0_count += 1
                    #         self.image_list.append(os.path.join(path, subdir, file))
                    # else:
                    label0_count += 1
                    self.image_list.append(os.path.join(path, subdir, file))
        self.train = train
        print('label0 count', label0_count, 'label1 count', label1_count, 'label2 count', label2_count)

    def __getitem__(self, item):
        img_path = self.image_list[item]
        img = cv2.imread(img_path)
        img_raw = cv2.imread(img_path.replace('circle', 'raw'))
        label = int(os.path.basename(img_path).split('_')[1])
        img, img_raw = self.aug(img, img_raw)

        return torch.concat([img, img_raw], dim=0), label, img_path

    def __len__(self):
        return len(self.image_list)

    def aug(self, img, img_neighbour):
        if self.train:
            if random.random() < 0.5:
                img = cv2.flip(img, 0)
                img_neighbour = cv2.flip(img_neighbour, 0)
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                img_neighbour = cv2.flip(img_neighbour, 1)
            if random.random() < 0.5:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img_neighbour = cv2.rotate(img_neighbour, cv2.ROTATE_90_CLOCKWISE)
            if random.random() < 0.5:
                img = gaussian_filter(img, sigma=2)
                img_neighbour = gaussian_filter(img_neighbour, sigma=2)
        img = img.astype(np.float32) / 255.
        img = (img - 0.5) / 0.5
        img = torch.tensor(img)
        img = torch.permute(img, (2, 0, 1))
        img_neighbour = img_neighbour.astype(np.float32) / 255.
        img_neighbour = (img_neighbour - 0.5) / 0.5
        img_neighbour = torch.tensor(img_neighbour)
        img_neighbour = torch.permute(img_neighbour, (2, 0, 1))
        return img, img_neighbour


class PuntaDatasetNeighbour(Dataset):
    def __init__(self, path, train):
        self.image_list = []
        label0_count = 0
        label1_count = 0
        label2_count = 0
        self.num_neighbour = 5
        for subdir in os.listdir(path):
            for puncta_count in os.listdir(os.path.join(path, subdir)):
                file_list = glob.glob(os.path.join(path, subdir, puncta_count, '*_label2.tif'))
                self.image_list.append(os.path.join(path, subdir, puncta_count))
                if len(file_list) != 0:
                    for _ in range(2):
                        self.image_list.append(os.path.join(path, subdir, puncta_count))
                    label2_count += 1
                elif len(glob.glob(os.path.join(path, subdir, puncta_count, '*_label1.tif'))) != 0:
                    label1_count += 1
                else:
                    label0_count += 1
            self.train = train
        print('label0 count', label0_count, 'label1 count', label1_count, 'label2 count', label2_count)

    def __getitem__(self, item):
        img_path = self.image_list[item]
        # img_neighbour = img_path.replace('.tif', '_neighbour.tif')
        puncta_images = []
        puncta_labels = []
        for i in range(self.num_neighbour):
            file = glob.glob(os.path.join(img_path, f'neighbour_{i}*.tif'))
            # print(file)
            assert len(file) == 1
            file = file[0]
            label = int(os.path.basename(file).split('_')[-1][5:-4])
            img = imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            puncta_images.append(img)
            puncta_labels.append(torch.tensor(label))

        img_neighbour = imread(os.path.join(img_path, 'neighbour.tif'))
        img_neighbour = cv2.cvtColor(img_neighbour, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        img, img_neighbour = self.aug(puncta_images, img_neighbour)
        return img, img_neighbour, torch.stack(puncta_labels, dim=0), img_path

    def __len__(self):
        return len(self.image_list)

    def aug(self, img_list, img_neighbour):
        if self.train:
            if random.random() < 0.5:
                for i in range(self.num_neighbour):
                    img_list[i] = np.flip(img_list[i], axis=0)
                img_neighbour = np.flip(img_neighbour, axis=0)
            if random.random() < 0.5:
                for i in range(self.num_neighbour):
                    img_list[i] = np.rot90(img_list[i], k=1)
                img_neighbour = np.rot90(img_neighbour, k=1)
            if random.random() < 0.5:
                for i in range(self.num_neighbour):
                    img_list[i] = gaussian_filter(img_list[i], sigma=2)
                img_neighbour = gaussian_filter(img_neighbour, sigma=2)
        for i in range(self.num_neighbour):
            img_list[i] = self.normalize(img_list[i])
            img_list[i] = torch.tensor(img_list[i])
        img_neighbour = self.normalize(img_neighbour)
        return torch.stack(img_list, dim=0), torch.tensor(img_neighbour).unsqueeze_(dim=0)

    def normalize(self, img):
        img = img.astype(np.float32)
        img = (img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1))
        img[img < 0] = 0
        img[img > 1] = 1
        return img


class PunctaDatasetTest128(Dataset):
    def __init__(self, puncta_list, img, draw_img, patch_size=128):
        self.puncta_list = puncta_list
        self.img = img
        self.draw_img = draw_img
        self.patch_size = patch_size

    def __getitem__(self, item):
        y, x, r = self.puncta_list[item]
        red_image = self.draw_img.copy()
        origin_image = self.img.copy()
        cv2.circle(red_image, (int(x), int(y)), int(r * 1.4), color=(0, 0, 255), thickness=2)
        crop_image_red = red_image[max(0, int(y - self.patch_size // 2)):min(self.img.shape[0], int(y + self.patch_size // 2)),
                         max(0, int(x - self.patch_size // 2)):min(self.img.shape[1], int(x + self.patch_size // 2)), :]
        crop_image_origin = origin_image[max(0, int(y - self.patch_size // 2)):min(self.img.shape[0], int(y + self.patch_size // 2)),
                            max(0, int(x - self.patch_size // 2)):min(self.img.shape[1], int(x + self.patch_size // 2)), :]
        crop_image_red = cv2.resize(crop_image_red, (self.patch_size, self.patch_size))
        crop_image_origin = cv2.resize(crop_image_origin, (self.patch_size, self.patch_size))
        img, img_raw = self.aug(crop_image_red, crop_image_origin)

        return torch.concat([img, img_raw], dim=0), torch.tensor(0), ''

    def __len__(self):
        return len(self.puncta_list)

    def aug(self, img, img_neighbour):
        img = img.astype(np.float32) / 255.
        img = (img - 0.5) / 0.5
        img = torch.tensor(img)
        img = torch.permute(img, (2, 0, 1))
        img_neighbour = img_neighbour.astype(np.float32) / 255.
        img_neighbour = (img_neighbour - 0.5) / 0.5
        img_neighbour = torch.tensor(img_neighbour)
        img_neighbour = torch.permute(img_neighbour, (2, 0, 1))
        return img, img_neighbour
