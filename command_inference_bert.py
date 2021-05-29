import os
import json
import yaml
import torch
import argparse
import logging
from utils.logger import ColoredLogger
from dataset import get_bert_dataset
from models.models import SimpleBert


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'bert.yaml'), help = 'Config File', type = str)
parser.add_argument('--input', default = os.path.join('examples', 'bert.json'))
parser.add_argument('--output', default = os.path.join('examples', 'bert-res.json'))
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg
INPUT_FILE = FLAGS.input
OUTPUT_FILE = FLAGS.output

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

if os.path.exists(os.path.dirname(OUTPUT_FILE)) == False:
    os.makedirs(os.path.dirname(OUTPUT_FILE))
    
MULTIGPU = cfg_dict.get('multigpu', False)
BERT_CASED = cfg_dict.get('bert_cased', False)
MAX_LENGTH = cfg_dict.get('max_length', 512)
SEQ_LEN = cfg_dict.get('seq_len', 50)
END_YEAR = cfg_dict.get('end_year', 2015)
FREQUENCY = cfg_dict.get('frequency', 5)
K = cfg_dict.get('K', 10)
STATS_DIR = cfg_dict.get('stats_dir', os.path.join('stats', 'bert'))
DATA_PATH = cfg_dict.get('data_path', os.path.join('data', 'citation.csv'))
if os.path.exists(STATS_DIR) == False:
    os.makedirs(STATS_DIR)
checkpoint_file = os.path.join(STATS_DIR, 'checkpoint.tar')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data & Build dataset
logger.info('Reading bert dataset & citation dataset ...')
_, _, paper_info = get_bert_dataset(DATA_PATH, seq_len = SEQ_LEN, year = END_YEAR, frequency = FREQUENCY)
paper_num = len(paper_info)
logger.info('Finish reading and dividing into training and testing sets.')

# Build model from configs
model = SimpleBert(num_classes = paper_num, max_length = MAX_LENGTH, cased = BERT_CASED)
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

def get_paper_info(paper_info, res_ids):
    res_dict = {}
    res_dict['inference'] = []
    for res in res_ids:
        res_item = {}
        res_item['result'] = []
        for id in res:
            res_item['result'].append(paper_info[id])
        res_dict['inference'].append(res_item)
    return res_dict


def _inference_context(context):
    logger.info('Begin inference ...')
    model.eval()
    tokens = model.convert_tokens([context]).to(device)
    with torch.no_grad():
        _, res_softmax = model(tokens)
    logger.info('Inference successfully finished!')
    return res_softmax


def _inference_lr_context(left_context, right_context): 
    logger.info('Begin inference ...')
    model.eval()
    tokens = model.convert_tokens([left_context], [right_context]).to(device)
    with torch.no_grad():
        _, res_softmax = model(tokens)
    logger.info('Inference successfully finished!')
    return res_softmax


def inference(input_file, output_file):
    with open(input_file, 'r') as f:
        inference_json = json.load(f)
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
        _, top_K_ids = torch.topk(res_softmax, k = K, largest = True, sorted = True)
        top_K_ids = top_K_ids[0].detach().cpu().tolist()
        res_ids.append(top_K_ids)
    res_dict = get_paper_info(paper_info, res_ids)
    with open(output_file, 'w') as f:
        json.dump(res_dict, f)
    return res_dict


if __name__ == '__main__':
    res_ids = inference(INPUT_FILE, OUTPUT_FILE)
    print(res_ids)