# -*- coding: utf-8 -*-
# @Time: 2024/12/11
# @File: params.py
# @Author: fwb
import os.path
# from config.nmnist_params import nmnist_params
from config.ncaltech101_params import ncaltech101_params
# from config.thu_params import thu_params
# from config.ncars_params import ncars_params
from config.dvsgesture_params import dvsgesture_params
# from config.paf_params import paf_params
# from config.neurohar_params import neurohar_params
# from config.ucf101_params import ucf101_params
# from config.cifar10dvs_params import cifar10dvs_params


class DatasetParams:
    def __init__(self):
        # Set the random seed
        self.seed = 21
        # Datasets root path.
        self.root_dir = r'./dataset/'
        # Save graph data path.
        self.save_graph_dir = os.path.join(self.root_dir, 'to_graph')
        # Dataset params config.
        # self.nmnist_params = nmnist_params()
        self.ncaltech101_params = ncaltech101_params()
        # self.thu_params = thu_params()
        # self.ncars_params = ncars_params()
        self.dvsgesture_params = dvsgesture_params()
        # self.paf_params = paf_params()
        # self.neurohar_params = neurohar_params()
        # self.ucf101_params = ucf101_params()
        # self.cifar10dvs_params = cifar10dvs_params()
