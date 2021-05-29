import os
import yaml
import torch
import argparse
import logging
from utils.logger import ColoredLogger
from dataset import get_bert_dataset
from torch.utils.data import DataLoader
from models.models import SimpleBert
from utils.criterion import CrossEntropyLoss
from utils.metrics import ResultRecorder


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'bert.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader = yaml.FullLoader)

BERT_CASED = cfg_dict.get('bert_cased', False)
MULTIGPU = cfg_dict.get('multigpu', False)
BATCH_SIZE = cfg_dict.get('batch_size', 4)
MAX_LENGTH = cfg_dict.get('max_length', 512)
SEQ_LEN = cfg_dict.get('seq_len', 50)
END_YEAR = cfg_dict.get('end_year', 2015)
FREQUENCY = cfg_dict.get('frequency', 5)
RECALL_K = cfg_dict.get('recall_K', [5, 10, 30, 50, 80])
STATS_DIR = cfg_dict.get('stats_dir', os.path.join('stats', 'bert'))
DATA_PATH = cfg_dict.get('data_path', os.path.join('data', 'citation.csv'))
if os.path.exists(STATS_DIR) == False:
    os.makedirs(STATS_DIR)
checkpoint_file = os.path.join(STATS_DIR, 'checkpoint.tar')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data & Build dataset
logger.info('Reading bert dataset & citation dataset ...')
_, val_dataset, paper_info = get_bert_dataset(DATA_PATH, seq_len = SEQ_LEN, year = END_YEAR, frequency = FREQUENCY)
paper_num = len(paper_info)
logger.info('Finish reading and dividing into training and testing sets.')
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)

# Build model from configs
model = SimpleBert(num_classes = paper_num, max_length = MAX_LENGTH, cased = BERT_CASED)
model.to(device)

# Define criterion
criterion = CrossEntropyLoss()

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

# Result Recorder
recorder = ResultRecorder(paper_num, include_mAP = True, recall_K = RECALL_K)


def evaluate():
    logger.info('Start evaluation process.')
    model.eval()
    recorder.clear()
    total_batches = len(val_dataloader)
    for idx, data in enumerate(val_dataloader):
        left_context, right_context, label, source_label = data
        tokens = model.convert_tokens(list(left_context), list(right_context)).to(device)
        label = torch.LongTensor(label).to(device)
        with torch.no_grad():
            res, res_softmax = model(tokens)
            loss = criterion(res, label)
        logger.info('Val batch {}/{}, loss: {:.6f}'.format(idx + 1, total_batches, loss.item()))
        recorder.add_record(res_softmax, label, source_label)
    logger.info('Finish evaluation process. Now calculating metrics ...')
    mAP = recorder.calc_mAP()
    mRR = recorder.calc_mRR()
    recall_K = recorder.calc_recall_K()
    logger.info('mAP: {:.6f}'.format(mAP))
    logger.info('MRR: {:.6f}'.format(mRR))
    for i, k in enumerate(RECALL_K):
        logger.info('Recall@{}: {:.6f}'.format(k, recall_K[i]))


if __name__ == '__main__':
    evaluate()