import os
import json
import yaml
import torch
import logging
import numpy as np
from utils.logger import ColoredLogger
from dataset import get_bert_dataset
from models.models import SimpleBert


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


class BertInferencer(object):
    def __init__(self, **kwargs):
        super(BertInferencer, self).__init__()
        MULTIGPU = kwargs.get('multigpu', False)
        MAX_LENGTH = kwargs.get('max_length', 512)
        SEQ_LEN = kwargs.get('seq_len', 50)
        END_YEAR = kwargs.get('end_year', 2015)
        FREQUENCY = kwargs.get('frequency', 5)
        self.K = kwargs.get('K', 10)
        STATS_DIR = kwargs.get('stats_dir', os.path.join('stats', 'bert'))
        DATA_PATH = kwargs.get('data_path', os.path.join('data', 'citation.csv'))
        if os.path.exists(STATS_DIR) == False:
            os.makedirs(STATS_DIR)
        checkpoint_file = os.path.join(STATS_DIR, 'checkpoint.tar')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load data & Build dataset
        logger.info('Reading bert dataset & citation dataset ...')
        _, _, self.paper_info = get_bert_dataset(DATA_PATH, seq_len = SEQ_LEN, year = END_YEAR, frequency = FREQUENCY)
        self.paper_num = len(self.paper_info)
        logger.info('Finish reading and dividing into training and testing sets.')

        if self.K > self.paper_num:
            self.K = self.paper_num

        # Build model from configs
        self.model = SimpleBert(num_classes = self.paper_num, max_length = MAX_LENGTH)
        self.model.to(self.device)

        # Read checkpoints
        if os.path.isfile(checkpoint_file):
            logger.info('Load checkpoint from {} ...'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file, map_location = self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.info('Checkpoint {} (epoch {}) loaded.'.format(checkpoint_file, start_epoch))
        else:
            raise AttributeError('No checkpoint file!')

        if MULTIGPU is True:
            self.model = torch.nn.DataParallel(self.model)

    def get_top_K_ids(self, res):
        res = res.reshape(self.paper_num)
        return np.argsort(- res)[:self.K].tolist()

    def get_paper_info(self, res_ids):
        res_dict = {}
        res_dict['inference'] = []
        for res in res_ids:
            res_item = {}
            res_item['result'] = []
            for id in res:
                res_item['result'].append(self.paper_info[id])
            res_dict['inference'].append(res_item)
        return res_dict


    def _inference_context(self, context):
        logger.info('Begin inference ...')
        self.model.eval()
        tokens = self.model.convert_tokens([context]).to(self.device)
        with torch.no_grad():
            _, res_softmax = self.model(tokens)
        logger.info('Inference successfully finished!')
        return res_softmax.cpu().detach().numpy()


    def _inference_lr_context(self, left_context, right_context): 
        logger.info('Begin inference ...')
        self.model.eval()
        tokens = self.model.convert_tokens([left_context], [right_context]).to(self.device)
        with torch.no_grad():
            _, res_softmax = self.model(tokens)
        logger.info('Inference successfully finished!')
        return res_softmax.cpu().detach().numpy()


    def inference(self, input_dict):
        if 'inference' not in input_dict.keys():
            raise KeyError('"inference" not in the keys of the input dict.')
        inference_list = input_dict['inference']
        res_ids = []
        for item in inference_list:
            if 'context' in item.keys():
                res_softmax = self._inference_context(item['context'])
            elif 'left_context' in item.keys() and 'right_context' in item.keys():
                res_softmax = self._inference_lr_context(item['left_context'], item['right_context'])
            else:
                raise KeyError('Neither "context" nor both "left_context" and "right_context" is specified in the json input.')
            top_K_ids = self.get_top_K_ids(res_softmax)
            res_ids.append(top_K_ids)
        res_dict = self.get_paper_info(res_ids)
        return res_dict


if __name__ == '__main__':
    with open('configs/bert.yaml', 'r') as cfg_file:
        cfgs = yaml.load(cfg_file, Loader=yaml.FullLoader)
    inferencer = BertInferencer(**cfgs)
    with open('examples/bert.json', 'r') as f:
        input_dict = json.load(f)
    output_dict = inferencer.inference(input_dict)
    with open('examples/bert-res.json', 'w') as f:
        json.dump(output_dict, f)
    