import os
import yaml
import torch
import argparse
import pandas as pd
import numpy as np
import logging
from utils.logger import ColoredLogger
from dataset import get_citation_dataset
from models.models import SpecterVGAE


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'vgae.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
EMBEDDING_DIM = cfg_dict.get('embedding_dim', 768)
MULTIGPU = cfg_dict.get('multigpu', False)
SPECTER_BATCH_SIZE = cfg_dict.get('specter_batch_size', 4)
MAX_LENGTH = cfg_dict.get('max_length', 512)
SEQ_LEN = cfg_dict.get('seq_len', 50)
END_YEAR = cfg_dict.get('end_year', 2015)
FREQUENCY = cfg_dict.get('frequency', 5)
STATS_DIR = cfg_dict.get('stats_dir', os.path.join('stats', 'vgae'))
DATA_PATH = cfg_dict.get('data_path', os.path.join('data', 'citation.csv'))
EMBEDDING_FILENAME = cfg_dict.get('embedding_filename', 'embeddings.npy')
SPECTER_EMBEDDING_FILENAME = cfg_dict.get('specter_embedding_filename', 'specter_embeddings.npy')
if os.path.exists(STATS_DIR) == False:
    os.makedirs(STATS_DIR)
checkpoint_file = os.path.join(STATS_DIR, 'checkpoint.tar')
embedding_file = os.path.join(STATS_DIR, EMBEDDING_FILENAME)
specter_embedding_file = os.path.join(STATS_DIR, SPECTER_EMBEDDING_FILENAME)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data & Build dataset
logger.info('Reading citation dataset ...')
edge_list, _, _, _, node_info = get_citation_dataset(DATA_PATH, seq_len = SEQ_LEN, year = END_YEAR, frequency = FREQUENCY)
logger.info('File read successfully. Now reading edge list for training ...')
node_num = len(node_info)
df = pd.read_csv(os.path.join(STATS_DIR, 'train_pos_edge_list.csv'))
train_edge_list = [[row['source'], row['destination']] for _, row in df.iterrows()]
logger.info('File read successfully.')

# Build model from configs
model = SpecterVGAE(embedding_dim = EMBEDDING_DIM, max_length = MAX_LENGTH)
model.process_paper_feature(node_info, use_saved_results = True, filepath = specter_embedding_file, device = device, process_batch_size = SPECTER_BATCH_SIZE)
model.to(device)

if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    logger.info('Load checkpoint {} (epoch {})'.format(checkpoint_file, epoch))
else:
    raise AttributeError('No checkpoint file!')

if MULTIGPU is True:
    model = torch.nn.DataParallel(model)


def inference(edge_list, filename):
    model.eval()
    edge_list = torch.LongTensor(edge_list).to(device).transpose(1, 0)
    logger.info('Fetching embedding results ...')
    with torch.no_grad():
        emb = model.encode(edge_list)
    logger.info('Results fetched. Now saving to {} ...'.format(filename))
    emb = emb.cpu().detach().numpy()
    np.save(filename, emb)
    logger.info('File saved successfully.')


if __name__ == '__main__':
    inference(train_edge_list, embedding_file)