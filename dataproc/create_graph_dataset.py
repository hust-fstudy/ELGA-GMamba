# -*- coding: utf-8 -*-
# @Time: 2025/3/4
# @File: create_graph_dataset.py
# @Author: fwb
import math
import os
import random
import shutil
import glob2
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torch_geometric.data import Dataset
from utils.read_file import ReadFile
from utils.uniform_event import Event
from e2g.event2voxel import VoxelGenerator


def create_path(path):
    if not (os.path.exists(path)):
        os.makedirs(path)


def create_graph_data(args, events_dict, graph_index_path):
    node_feats, coords = VoxelGenerator(args, events_dict).to_voxel()
    # Graph data info.
    graph_data_dict = {
        'node_feats': np.array(node_feats),
        'coords': np.array(coords),
        'y': np.array(events_dict['label'])
    }
    with h5py.File(graph_index_path, 'w') as f:
        for key, value in graph_data_dict.items():
            if isinstance(value, str):
                value = np.string_(value)
            f.create_dataset(key, data=value)
    return graph_data_dict


def copy_file_to_dir(filepath_list, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file_path in filepath_list:
        if os.path.isfile(file_path):
            shutil.copy2(file_path, target_dir)


def creat_split(root_dir, train_ratio):
    total_dir = os.path.join(root_dir, 'total')
    class_list = os.listdir(total_dir)
    for each_class in class_list:
        each_class_path = os.path.join(total_dir, each_class)
        file_path_list = glob2.glob(os.path.join(each_class_path, '*'))
        train_file_path_index = random.sample(range(0, len(file_path_list)),
                                              math.ceil(len(file_path_list) * train_ratio))
        test_file_path_index = list(set(range(0, len(file_path_list))) - set(train_file_path_index))
        train_file_path = [file_path_list[i] for i in train_file_path_index]
        test_file_path = [file_path_list[i] for i in test_file_path_index]
        copy_file_to_dir(train_file_path, os.path.join(root_dir, 'train', str(each_class)))
        copy_file_to_dir(test_file_path, os.path.join(root_dir, 'test', str(each_class)))


def remove_files_in_dir(directory_path):
    if not os.path.isdir(directory_path):
        print(f"The provided path {directory_path} is not a valid directory.")
        return
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


class CreateGraphDataset(Dataset):
    def __init__(self, args, root_dir, save_graph_dir, mode: str, seed=42, train_ratio=0.8):
        super(CreateGraphDataset, self).__init__()
        random.seed(seed)
        self.args = args
        self.root_dir = root_dir
        self.save_graph_dir = save_graph_dir
        self.mode = mode
        self.dataset_name = Path(root_dir).parent.name
        self.RF = ReadFile()
        if args.is_remove_split:
            remove_files_in_dir(os.path.join(root_dir, 'train'))
            remove_files_in_dir(os.path.join(root_dir, 'test'))
        if args.is_split:
            creat_split(root_dir, train_ratio)
        self.mode_dir = os.path.join(root_dir, mode)
        self.save_graph_dir = os.path.join(save_graph_dir, mode)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_list = os.listdir(self.mode_dir)
        self.label_dict = {class_name: label for label, class_name in enumerate(self.class_list)}
        self.sample_path_list = []
        self.sample_label_list = []
        for each_class in tqdm(self.class_list, desc="Read each class"):  # iterate over all classes
            each_class_path = os.path.join(self.mode_dir, each_class)
            file_path_list = glob2.glob(os.path.join(each_class_path, '*'))
            file_label_list = np.full(len(file_path_list), self.label_dict.get(each_class))
            self.sample_path_list.extend(file_path_list)
            self.sample_label_list.extend(file_label_list)
        self.save_graph_names = [f'{mode}_data_{i}.hdf5' for i in range(len(self.sample_path_list))]
        if not os.path.exists(self.save_graph_dir):
            create_path(self.save_graph_dir)

    def __len__(self):
        return len(self.sample_path_list)

    def __getitem__(self, index):
        graph_index_path = os.path.join(self.save_graph_dir, self.save_graph_names[index])
        sample_path = self.sample_path_list[index]
        if self.dataset_name in ['nmnist', 'ncaltech101']:
            [x, y, t, p] = self.RF.bin_file_reader(sample_path)
        elif self.dataset_name in ['thu', 'ncars']:
            [x, y, t, p] = self.RF.npy_file_reader(sample_path)
        elif self.dataset_name in ['dvsgesture', 'cifar10dvs']:
            [x, y, t, p] = self.RF.npz_file_reader(sample_path)
        elif self.dataset_name in ['paf']:
            [x, y, t, p] = self.RF.aedat_file_reader(sample_path)
        elif self.dataset_name in ['neurohar']:
            [x, y, t, p] = self.RF.txt_file_reader(sample_path)
        elif self.dataset_name in ['ucf101']:
            [x, y, t, p] = self.RF.mat_file_reader(sample_path)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        sample_label = np.array(self.sample_label_list[index])
        events_dict = Event(x, y, t, p, sample_label).to_uniform_format()
        graph_data = create_graph_data(self.args, events_dict, graph_index_path)
        graph_data = list(graph_data)
        return graph_data
