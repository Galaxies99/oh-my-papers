import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
import random
import tarfile
import shutil
from torch.utils.data import Dataset
from utils.preprocessing import split_process_dataset, construct_graph

# pd.options.mode.chained_assignment = None

class PapersDataset(Dataset):
    '''
    Papers dataset
    '''
    def __init__(self, df):
        '''
        Construct paper dataset

        Parameters
        ----------
        df: input DataFrame
        '''
        super(PapersDataset, self).__init__()
        self.left_context = df.LeftString.tolist()
        self.right_context = df.RightString.tolist()
        self.labels = df.TargetID.tolist()

    def __getitem__(self, index):
        return self.left_context[index], self.right_context[index], self.labels[index]

    def __len__(self):
        return len(self.left_context)

def get_bert_dataset(file_path, seq_len=50, year=2016, frequency=5):
    '''
    Get dataset for bert

    Parameters
    ----------
    file_path: path of the whole data
    seq_len: maximal length of the citation context
    year: year of boundary (training and test)
    frequency: only articles that are referenced more than 'frequency' are retained
    '''
    train_df, test_df = split_process_dataset(file_path=file_path, seq_len=seq_len, year=year, frequency=frequency)
    train_dataset = PapersDataset(train_df)
    val_dataset = PapersDataset(test_df)

    return train_dataset, val_dataset

def get_citation_dataset(file_path, seq_len=50, year=2016, frequency=5):
    '''
    Get dataset for citation network

    Parameters
    ----------
    file_path: path of the whole data
    seq_len: maximal length of the citation context
    year: year of boundary (training and test)
    frequency: only articles that are referenced more than 'frequency' are retained
    '''
    train_df, _ = split_process_dataset(file_path=file_path, seq_len=seq_len, year=year, frequency=frequency)
    edge_list, node_info = construct_graph(train_df)
    random.shuffle(edge_list)
    train_edge_num = int(len(edge_list) * 0.9)
    node_num = len(node_info)
    train_edge_list = edge_list[:train_edge_num]
    test_pos_edge_list = edge_list[train_edge_num:]
    test_neg_edge_list = []
    for i in range(len(test_pos_edge_list)):
        valid_sample = False
        while valid_sample == False:
            src = random.randint(0, node_num - 1)
            tar = random.randint(0, node_num - 1)
            if src != tar and [src, tar] not in edge_list and [tar, src] not in edge_list:
                test_neg_edge_list.append([src, tar])
                valid_sample = True    
    return train_edge_list, test_pos_edge_list, test_neg_edge_list, node_info
