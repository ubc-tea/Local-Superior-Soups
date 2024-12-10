import numpy as np
import os
import random
import torchvision.transforms as transforms
import torch.utils.data as data
from utils.dataset_utils import split_data, save_file
from os import path
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
from PIL import Image
import os


class RIMONEDataset(Dataset):
    def __init__(self, root, train, transform):
        self.root = root
        self.train = train
        self.transform = transform
        self.img_folder = self.root

        self.img_paths = []
        self.img_labels = []

        if self.train:
            normal_filepath = glob.glob(
                self.img_folder + "partitioned_randomly/training_set/normal/" + "*.png"
            )
            self.img_paths.extend(normal_filepath)
            self.img_labels.extend([0] * len(normal_filepath))

            glaucoma_filepath = glob.glob(
                self.img_folder
                + "partitioned_randomly/training_set/glaucoma/"
                + "*.png"
            )

            self.img_paths.extend(glaucoma_filepath)
            self.img_labels.extend([1] * len(glaucoma_filepath))

        else:
            normal_filepath = glob.glob(
                self.img_folder + "partitioned_randomly/test_set/normal/" + "*.png"
            )
            self.img_paths.extend(normal_filepath)
            self.img_labels.extend([0] * len(normal_filepath))
            glaucoma_filepath = glob.glob(
                self.img_folder + "partitioned_randomly/test_set/glaucoma/" + "*.png"
            )
            self.img_paths.extend(glaucoma_filepath)
            self.img_labels.extend([1] * len(glaucoma_filepath))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.img_labels[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class REFUGEDataset(Dataset):
    def __init__(self, root, train, transform):
        self.root = root
        self.train = train
        self.transform = transform
        self.img_folder = self.root

        self.img_paths = []
        self.img_labels = []

        if self.train:
            normal_filepath = glob.glob(self.img_folder + "Train/NORMAL/" + "*.jpg")
            self.img_paths.extend(normal_filepath)
            self.img_labels.extend([0] * len(normal_filepath))
            glaucoma_filepath = glob.glob(self.img_folder + "Train/GLAUCOMA/" + "*.jpg")
            self.img_paths.extend(glaucoma_filepath)
            self.img_labels.extend([1] * len(glaucoma_filepath))
        else:
            self.label_file_path = self.root + "Fovea_locations.csv"
            self.img_labels_pd = pd.read_csv(self.label_file_path)
            for idx, data in self.img_labels_pd.iterrows():
                img_name = data["ImgName"]
                img_path = os.path.join(self.img_folder, "Validation", img_name)
                self.img_paths.append(img_path)
            self.img_labels.extend(self.img_labels_pd.iloc[:, 2].values)

            self.label_file_path = self.root + "Glaucoma_label_and_Fovea_location.csv"
            self.img_labels_pd = pd.read_csv(self.label_file_path)
            for idx, data in self.img_labels_pd.iterrows():
                img_name = data["ImgName"]
                img_path = os.path.join(self.img_folder, "Test", img_name)
                self.img_paths.append(img_path)
            self.img_labels.extend(self.img_labels_pd.iloc[:, 2].values)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.img_labels[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class KaggleArimaDataset(Dataset):
    def __init__(self, root, train, transform):
        self.root = root
        self.train = train
        self.transform = transform
        self.img_folder = self.root

        self.img_paths = []
        self.img_labels = []

        total_normal_filepath = glob.glob(self.img_folder + "normal/" + "*.jpg")
        total_normal_num = len(total_normal_filepath)
        train_normal_num = int(0.8 * total_normal_num)
        test_normal_num = total_normal_num - train_normal_num
        if self.train:
            self.img_paths.extend(total_normal_filepath[:train_normal_num])
            self.img_labels.extend([0] * train_normal_num)
        else:
            self.img_paths.extend(total_normal_filepath[train_normal_num:])
            self.img_labels.extend([0] * test_normal_num)

        total_glaucoma_filepath = glob.glob(self.img_folder + "glaucoma/" + "*.jpg")
        total_glaucoma_num = len(total_glaucoma_filepath)
        train_glaucoma_num = int(0.8 * total_glaucoma_num)
        test_glaucoma_num = total_glaucoma_num - train_glaucoma_num
        if self.train:
            self.img_paths.extend(total_glaucoma_filepath[:train_glaucoma_num])
            self.img_labels.extend([1] * train_glaucoma_num)
        else:
            self.img_paths.extend(total_glaucoma_filepath[train_glaucoma_num:])
            self.img_labels.extend([1] * test_glaucoma_num)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.img_labels[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class DrishtiDataset(Dataset):
    def __init__(self, root, train, transform):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            self.img_folder = self.root + "Training/"
        else:
            self.img_folder = self.root + "Testing/"

        self.label_file_path = self.root + "Drishti-GS1_diagnosis.csv"
        self.img_labels_pd = pd.read_csv(self.label_file_path)
        self.img_labels = []
        self.img_paths = []

        total_filepath = glob.glob(self.img_folder + "*.png")

        for filepath in total_filepath:
            self.img_paths.append(filepath)
            filename = filepath.split("/")[-1]
            match_filename = filename.replace(".png", "") + "'"
            label = self.img_labels_pd.loc[
                self.img_labels_pd["Drishti-GS_File"] == match_filename
            ]["Total"]
            if label.item() == "Normal":
                self.img_labels.append(0)
            else:
                self.img_labels.append(1)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.img_labels[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class RetinaDataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        super(RetinaDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if img.shape[0] != 1:
            # transpose to Image type, so that the transform function can be used
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # turn the raw image into 3 channels
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        # do transform with PIL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.data.shape[0]


def retina_dataset_read(base_path, domain):
    # define the transform function
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if domain == "Drishti":
        train_dataset = DrishtiDataset(
            root=base_path + "/" + domain + "/", train=True, transform=transform
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset), shuffle=True
        )

        test_dataset = DrishtiDataset(
            root=base_path + "/" + domain + "/", train=False, transform=transform
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=len(test_dataset), shuffle=True
        )

    elif domain == "kaggle_arima":
        train_dataset = KaggleArimaDataset(
            root=base_path + "/" + domain + "/", train=True, transform=transform
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset), shuffle=True
        )

        test_dataset = KaggleArimaDataset(
            root=base_path + "/" + domain + "/", train=False, transform=transform
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=len(test_dataset), shuffle=True
        )

    elif domain == "REFUGE":
        train_dataset = REFUGEDataset(
            root=base_path + "/" + domain + "/", train=True, transform=transform
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset), shuffle=True
        )

        test_dataset = REFUGEDataset(
            root=base_path + "/" + domain + "/", train=False, transform=transform
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=len(test_dataset), shuffle=True
        )

    elif domain == "RIM-ONE_DL_images":
        train_dataset = RIMONEDataset(
            root=base_path + "/" + domain + "/", train=True, transform=transform
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset), shuffle=True
        )

        test_dataset = RIMONEDataset(
            root=base_path + "/" + domain + "/", train=False, transform=transform
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=len(test_dataset), shuffle=True
        )

    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))

    return train_loader, test_loader


random.seed(43)
np.random.seed(43)
ROOT_DIR = "/home/chenmh/projects/rrg-timsbc/chenmh/cmh_proj/dataset/retina/"
data_path = ROOT_DIR
dir_path = ROOT_DIR
# dir_path = "/home/chenmh/projects/rrg-timsbc/chenmh/cmh_proj/dataset/retina_balanced_class/"
# dir_path = "/home/chenmh/projects/rrg-timsbc/chenmh/cmh_proj/dataset/retina_balanced_class_num/"


# Allocate data to usersz``
def generate_retina(dir_path, class_balanced=False, num_balanced=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path + "raw_data"

    X, y = [], []
    domains = ["Drishti", "kaggle_arima", "REFUGE", "RIM-ONE_DL_images"]
    # domains =  ["kaggle_arima", "REFUGE", "RIM-ONE_DL_images"]
    for d in domains:
        print("Domain: ", d)
        train_loader, test_loader = retina_dataset_read(root, d)

        for _, tt in enumerate(train_loader):
            train_data, train_label = tt
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt

        dataset_image = []
        dataset_label = []

        dataset_image.extend(train_data.cpu().detach().numpy())
        dataset_image.extend(test_data.cpu().detach().numpy())
        dataset_label.extend(train_label.cpu().detach().numpy())
        dataset_label.extend(test_label.cpu().detach().numpy())

        if class_balanced:
            one_index_list = [i for i, x in enumerate(dataset_label) if x == 1]
            zero_index_list = [i for i, x in enumerate(dataset_label) if x == 0]
            if num_balanced:
                one_index_list = one_index_list[:30]
                zero_index_list = zero_index_list[:30]
            else:
                if len(zero_index_list) > len(one_index_list):
                    zero_index_list = zero_index_list[: len(one_index_list)]
                else:
                    one_index_list = one_index_list[: len(zero_index_list)]
            new_dataset_image = []
            new_dataset_label = []
            new_dataset_image.extend([dataset_image[i] for i in one_index_list])
            new_dataset_image.extend([dataset_image[i] for i in zero_index_list])
            new_dataset_label.extend([dataset_label[i] for i in one_index_list])
            new_dataset_label.extend([dataset_label[i] for i in zero_index_list])

            X.append(np.array(new_dataset_image))
            y.append(np.array(new_dataset_label))

        else:
            X.append(np.array(dataset_image))
            y.append(np.array(dataset_label))

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f"Number of labels: {labelss}")
    print(f"Number of clients: {num_clients}")

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    train_data, test_data = split_data(X, y)
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        max(labelss),
        statistic,
        None,
        None,
        None,
    )


if __name__ == "__main__":
    generate_retina(dir_path, class_balanced=True, num_balanced=False)
