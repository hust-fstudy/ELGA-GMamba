# -*- coding: utf-8 -*-
# @Time: 2024/11/15
# @File: run_recognition.py
# @Author: fwb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import random
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from model.net import Net
from dataproc.create_graph_dataset import CreateGraphDataset, remove_files_in_dir
from dataproc.graph_dataset import GraphDataset
from config.params import DatasetParams
from opt.criterion import build_criterion
from opt.optimizer import build_optimizer
from opt.lr_scheduler import build_scheduler
from utils.model_performance import model_train, model_test


def generate(loader):
    for _, _ in enumerate(loader):
        pass


def create_graph_dataset(args, data_path, save_graph_dir, seed=0):
    batch_train_graph_dataset = CreateGraphDataset(args, data_path, save_graph_dir, 'train', seed=seed)
    batch_test_graph_dataset = CreateGraphDataset(args, data_path, save_graph_dir, 'test', seed=seed)
    batch_train_loader = DataLoader(batch_train_graph_dataset, batch_size=args.batch_size)
    batch_test_loader = DataLoader(batch_test_graph_dataset, batch_size=args.batch_size)
    generate(batch_train_loader)
    generate(batch_test_loader)


def initial_model(args):
    model = Net(
        args=args,
        feats_dim=args.feats_dim,
        in_feats=args.feats_dim * int(args.grid_size[0] * args.grid_size[1]),
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        enc_depth=args.enc_depth,
        enc_chs=args.enc_chs,
        enc_neighbours=args.enc_neighbours,
        enc_radius=args.enc_radius,
        enc_way=args.enc_way,
        pool_ratio=args.pool_ratio,
        is_feats_map=args.is_feats_map,
        is_gnn_mamba=args.is_gnn_mamba
    )
    return model


class RecognitionMain:
    def __init__(self, DP, dataset_name: str):
        if not isinstance(DP, DatasetParams):
            raise TypeError("Instance must be an instance of the DatasetParams class.")
        self.DP = DP
        self.random_seed = self.DP.seed
        self.data_path = os.path.join(self.DP.root_dir, dataset_name.lower(), 'raw')
        # Load the corresponding dataset.
        print(f"The current dataset used is {dataset_name}")
        match dataset_name:
            case 'NMNIST':
                self.args = self.DP.nmnist_params
            case 'NCaltech101':
                self.args = self.DP.ncaltech101_params
            case 'THU':
                self.args = self.DP.thu_params
            case 'NCars':
                self.args = self.DP.ncars_params
            case 'DVSGesture':
                self.args = self.DP.dvsgesture_params
            case 'PAF':
                self.args = self.DP.paf_params
            case 'NeuroHAR':
                self.args = self.DP.neurohar_params
            case 'UCF101':
                self.args = self.DP.ucf101_params
            case 'CIFAR10DVS':
                self.args = self.DP.cifar10dvs_params
            case _:
                print(f"The {dataset_name} dataset does not exist!")
                self.args = None
        # Graph data storage path.
        self.save_graph_dir = os.path.join(self.DP.save_graph_dir,
                                           dataset_name,
                                           f'gs{int("".join(map(str, self.args.grid_size)))}'
                                           f'tr{self.args.coord_range[-1]}'
                                           f'mp{self.args.max_num_points_per_voxel}'
                                           f'Nv{self.args.Nv}')
        # Fixed batch creation graph dataset.
        if self.args.is_remove_graph:
            remove_files_in_dir(self.save_graph_dir)
            create_graph_dataset(self.args, self.data_path, self.save_graph_dir, seed=self.random_seed)

    def run(self):
        # Set the random seed and initialize it.
        print(f"Random seed ID is: {self.random_seed}")
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # Load graph dataset.
        train_graph_dataset = GraphDataset(self.args, self.save_graph_dir, 'train')
        test_graph_dataset = GraphDataset(self.args, self.save_graph_dir, 'test')
        print(f"train len: {len(train_graph_dataset)}, test len: {len(test_graph_dataset)}")
        train_loader = DataLoader(train_graph_dataset, batch_size=self.args.batch_size, shuffle=self.args.shuffle)
        test_loader = DataLoader(test_graph_dataset, batch_size=self.args.batch_size)
        # Initialization Model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = initial_model(self.args)
        model.to(device)
        criterion = build_criterion(self.args)
        optimizer = build_optimizer(self.args, model)
        lr_scheduler = None
        if self.args.is_lr_scheduler:
            lr_scheduler = build_scheduler(self.args, optimizer, len(train_loader))
        # Model training and testing.
        for epoch in range(self.args.epochs):
            with tqdm(desc=f"epoch {epoch + 1}/{self.args.epochs}"):
                train_temp_loss, train_temp_acc = model_train(self.args, model, train_loader, criterion, optimizer,
                                                              lr_scheduler, epoch, self.args.lambda_reg, device)
                print(f"train info for epoch {epoch + 1}: (loss: {train_temp_loss}, acc: {train_temp_acc})")
                test_temp_acc = model_test(model, test_loader, device)
                print(f"test info for epoch {epoch + 1}: (acc: {test_temp_acc}")


if __name__ == '__main__':
    DP = DatasetParams()  # creating an instance of the DatasetParams class
    dataset_dict = {
        1: 'NMNIST',
        2: 'NCaltech101',
        3: 'THU',
        4: 'NCars',
        5: 'DVSGesture',
        6: 'PAF',
        7: 'NeuroHAR',
        8: 'UCF101',
        9: 'CIFAR10DVS'
    }
    RM = RecognitionMain(DP, dataset_name=dataset_dict[5])
    # Run.
    RM.run()
