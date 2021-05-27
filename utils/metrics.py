from sklearn.metrics import average_precision_score
import numpy as np


class ResultRecorder(object):
    def __init__(self, node_num, include_mAP = True, recall_K = [5, 10, 30, 50, 80]):
        super(ResultRecorder, self).__init__()
        self.node_num = node_num
        self.include_mAP = include_mAP
        self.recall_K = recall_K
        self.clear()

    def clear(self):
        self.preds = np.zeros((0, self.node_num))
        self.source_labels = np.zeros(0, dtype = np.int64)
        self.labels = np.zeros(0, dtype = np.int64)
        self.RRtot = 0
        self.RRcnt = 0
        self.recall_cnt = np.zeros(len(self.recall_K))

    def add_record(self, pred, label, source_label):
        # pred has the size of batch_size * node_num
        pred = pred.cpu().detach().numpy()
        # label and source_label has the size of batch_size
        label = label.cpu().detach().numpy()    
        source_label = source_label.cpu().detach().numpy()
        if self.include_mAP:
            # For mAP metric
            self.preds = np.concatenate([self.preds, pred], axis = 0).astype(np.int64)
            self.labels = np.concatenate([self.labels, label], axis = 0).astype(np.int64)
            self.source_labels = np.concatenate([self.source_labels, source_label], axis = 0).astype(np.int64)
        # For MRR metric & Recall@K metrics
        batch_size = label.shape[0]
        for i in range(batch_size):
            rnk = int(np.where(np.argsort(- pred[i]) == label[i])[0]) + 2
            self.RRtot += 1 / rnk
            for j, k in enumerate(self.recall_K):
                if rnk <= k:
                    self.recall_cnt[j] += 1
            self.RRcnt += 1

    def calc_mAP(self):
        if self.include_mAP is False:
            raise AttributeError('Not include mAP in the metric settings.')
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

    def calc_mRR(self):
        return self.RRtot / self.RRcnt
    
    def calc_recall_K(self):
        return self.recall_cnt / self.RRcnt