from os import system
from sklearn.metrics import average_precision_score
import numpy as np


class ResultRecorder(object):
    def __init__(self, node_num):
        super(ResultRecorder, self).__init__()
        self.node_num = node_num
        self.clear()

    def clear(self):
        self.preds = np.zeros((0, self.node_num))
        self.source_labels = np.zeros(0, dtype = np.int64)
        self.labels = np.zeros(0, dtype = np.int64)

    def add_record(self, pred, label, source_label):
        # pred has the size of batch_size * node_num
        pred = pred.cpu().detach().numpy()
        # label and source_label has the size of batch_size
        label = label.cpu().detach().numpy()    
        source_label = source_label.cpu().detach().numpy()

        self.preds = np.concatenate([self.preds, pred], axis = 0).astype(np.int64)
        self.labels = np.concatenate([self.labels, label], axis = 0).astype(np.int64)
        self.source_labels = np.concatenate([self.source_labels, source_label], axis = 0).astype(np.int64)
    
    def calc_mAP(self):
        assert self.source_labels.shape == self.labels.shape
        num_records = self.labels.shape[0]
        ap_dict_score = {}
        ap_dict_true = {}
        source_dict = {}
        for source, target in set(zip(self.source_labels, self.labels)):
            ap_dict_score[target] = []
            ap_dict_true[target] = []
            source_dict[source] = source_dict.get(source, []) + [target]
        for i in range(num_records):
            source = self.source_labels[i]
            target = self.labels[i]
            target_list = source_dict.get(source, [])
            for potential_target in target_list:
                ap_dict_score[potential_target].append(self.preds[i][potential_target])
                if potential_target == target:
                    ap_dict_true[potential_target].append(1)
                else:
                    ap_dict_true[potential_target].append(0)
        total_AP = 0
        total_cnt = 0
        for key in ap_dict_score.keys():
            y_score = ap_dict_score[key]
            y_true = ap_dict_true[key]
            total_AP = total_AP + average_precision_score(y_true, y_score)
            total_cnt = total_cnt + 1
        return total_AP / total_cnt
    