import os
import json
import yaml
import torch
import argparse
import logging
import numpy as np
from utils.logger import ColoredLogger
from dataset import get_citation_dataset
from models.models import SimpleBert


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'bert.yaml'), help = 'Config File', type = str)
parser.add_argument('--input', default = os.path.join('examples', 'bert.json'))
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg
INPUT_FILE = FLAGS.input

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

with open(INPUT_FILE, 'r') as input_file:
    inference_json = json.load(input_file)
    
EMBEDDING_DIM = cfg_dict.get('embedding_dim', 768)
MULTIGPU = cfg_dict.get('multigpu', False)
MAX_LENGTH = cfg_dict.get('max_length', 512)
SEQ_LEN = cfg_dict.get('seq_len', 50)
END_YEAR = cfg_dict.get('end_year', 2015)
FREQUENCY = cfg_dict.get('frequency', 5)
RECALL_K = cfg_dict.get('recall_K', [5, 10, 30, 50, 80])
K = cfg_dict.get('K', 10)
STATS_DIR = cfg_dict.get('stats_dir', os.path.join('stats', 'bert'))
DATA_PATH = cfg_dict.get('data_path', os.path.join('data', 'citation.csv'))
if os.path.exists(STATS_DIR) == False:
    os.makedirs(STATS_DIR)
checkpoint_file = os.path.join(STATS_DIR, 'checkpoint.tar')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data & Build dataset
logger.info('Reading bert dataset & citation dataset ...')
_, _, _, _, node_info = get_citation_dataset(DATA_PATH, seq_len = SEQ_LEN, year = END_YEAR, frequency = FREQUENCY)
node_num = len(node_info)
logger.info('Finish reading and dividing into training and testing sets.')

# Build model from configs
model = SimpleBert(num_classes = node_num, max_length = MAX_LENGTH)
model.to(device)

# Read checkpoints
if os.path.isfile(checkpoint_file):
    logger.info('Load checkpoint from {} ...'.format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    logger.info('Checkpoint {} (epoch {}) loaded.'.format(checkpoint_file, start_epoch))
else:
    raise AttributeError('No checkpoint file!')

if MULTIGPU is True:
    model = torch.nn.DataParallel(model)


def get_top_K_ids(res, K):
    assert K <= node_num
    res = res.reshape(node_num)
    return np.argsort(- res)[:K].tolist()


def _inference_context(context):
    logger.info('Begin inference ...')
    model.eval()
    tokens = model.convert_tokens([context]).to(device)
    with torch.no_grad():
        _, res_softmax = model(tokens)
    logger.info('Inference successfully finished!')
    return res_softmax.cpu().detach().numpy()


def _inference_lr_context(left_context, right_context): 
    logger.info('Begin inference ...')
    model.eval()
    tokens = model.convert_tokens([left_context], [right_context]).to(device)
    with torch.no_grad():
        _, res_softmax = model(tokens)
    logger.info('Inference successfully finished!')
    return res_softmax.cpu().detach().numpy()


def inference(inference_json):
    if 'inference' not in inference_json.keys():
        raise KeyError('"inference" not in the keys of the json input.')
    inference_list = inference_json['inference']
    res_ids = []
    for item in inference_list:
        if 'context' in item.keys():
            res_softmax = _inference_context(item['context'])
        elif 'left_context' in item.keys() and 'right_context' in item.keys():
            res_softmax = _inference_lr_context(item['left_context'], item['right_context'])
        else:
            raise KeyError('Neither "context" nor both "left_context" and "right_context" is specified in the json input.')
        top_K_ids = get_top_K_ids(res_softmax, K)
        res_ids.append(top_K_ids)
    return res_ids


if __name__ == '__main__':
    res_ids = inference(inference_json)
    print(res_ids)
    # TODO: res_ids to paper infomation
    # TODO: create an inference class to call in order to reduce the inference time.