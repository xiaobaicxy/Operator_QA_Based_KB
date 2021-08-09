# -*- coding: utf-8 -*-
import torch

class Config:
    def __init__(self):
        self.vocab_size = 3000
        self.embed_size = 32
        self.hidden_size = 32
        self.num_classes = 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 1e-3
        self.epochs = 20
        self.batch_size = 256
        self.dev_scale=0. # 切分训练集与验证集（因为数据量很少，所以设置了一个全为训练集的比例）
        