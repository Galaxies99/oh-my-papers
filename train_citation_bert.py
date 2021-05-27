import os
import yaml
import torch
import argparse
import logging
from utils.logger import ColoredLogger
from torch.optim import AdamW
from dataset import get_bert_dataset
from torch.utils.data import DataLoader
from models.models import CitationBert
from utils.criterion import CrossEntropyLoss
from utils.metrics import ResultRecorder


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'citation_bert_vgae.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
MAX_EPOCH = cfg_dict.get('max_epoch', 30)
MULTIGPU = cfg_dict.get('multigpu', False)
EMBEDDING_DIM = cfg_dict.get('embedding_dim', 768)
COSINE_SOFTMAX_S = cfg_dict.get('cosine_softmax_S', 4)
ADAM_BETA1 = cfg_dict.get('adam_beta1', 0.9)
ADAM_BETA2 = cfg_dict.get('adam_beta2', 0.999)
ADAM_WEIGHT_DECAY = cfg_dict.get('adam_weight_decay', 0.01)
ADAM_EPS = cfg_dict.get('adam_eps', 1e-6)
LEARNING_RATE = cfg_dict.get('learning_rate', 0.01)
BATCH_SIZE = cfg_dict.get('batch_size', 4)
MAX_LENGTH = cfg_dict.get('max_length', 512)
SEQ_LEN = cfg_dict.get('seq_len', 50)
END_YEAR = cfg_dict.get('end_year', 2015)
FREQUENCY = cfg_dict.get('frequency', 5)
RECALL_K = cfg_dict.get('recall_K', [5, 10, 30, 50, 80])
STATS_DIR = cfg_dict.get('stats_dir', os.path.join('stats', 'citation_bert'))
DATA_PATH = cfg_dict.get('data_path', os.path.join('data', 'citation.csv'))
EMBEDDING_PATH = cfg_dict.get('embedding_path', os.path.join('stats', 'vgae', 'embedding.npy'))
if os.path.exists(STATS_DIR) == False:
    os.makedirs(STATS_DIR)
if os.path.isfile(EMBEDDING_PATH) == False:
    raise AttributeError('No embedding file.')
checkpoint_file = os.path.join(STATS_DIR, 'checkpoint.tar')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data & Build dataset
logger.info('Reading bert dataset & citation dataset ...')
train_dataset, val_dataset, paper_info = get_bert_dataset(DATA_PATH, seq_len = SEQ_LEN, year = END_YEAR, frequency = FREQUENCY)
paper_num = len(paper_info)
logger.info('Finish reading and dividing into training and testing sets.')
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)

# Build model from configs
model = CitationBert(num_classes = paper_num, embedding_dim = EMBEDDING_DIM, max_length = MAX_LENGTH, S = COSINE_SOFTMAX_S)
model.to(device)
model.set_paper_embeddings(filename = EMBEDDING_PATH, device = device)

# Define optimizer
optimizer = AdamW(model.parameters(), betas = (ADAM_BETA1, ADAM_BETA2), lr = LEARNING_RATE, weight_decay = ADAM_WEIGHT_DECAY, eps = ADAM_EPS)

# Define criterion
criterion = CrossEntropyLoss()

# Read checkpoints
start_epoch = 0
if os.path.isfile(checkpoint_file):
    logger.info('Load checkpoint from {} ...'.format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    logger.info('Checkpoint {} (epoch {}) loaded.'.format(checkpoint_file, start_epoch))


if MULTIGPU is True:
    model = torch.nn.DataParallel(model)

# Result Recorder
recorder = ResultRecorder(paper_num, include_mAP = False, recall_K = RECALL_K)


def train_one_epoch(epoch):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    model.train()
    total_batches = len(train_dataloader)
    for idx, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        left_context, right_context, label, _ = data
        tokens = model.convert_tokens(list(left_context), list(right_context)).to(device)
        label = torch.LongTensor(label).to(device)
        res, _ = model(tokens)
        loss = criterion(res, label)
        loss.backward()
        optimizer.step()
        logger.info('Train epoch {}/{} batch {}/{}, loss: {:.6f}'.format(epoch + 1, MAX_EPOCH, idx + 1, total_batches, loss.item()))
    logger.info('Finish training process in epoch {}.'.format(epoch + 1))

def val_one_epoch(epoch):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    model.eval()
    recorder.clear()
    total_batches = len(val_dataloader)
    for idx, data in enumerate(val_dataloader):
        optimizer.zero_grad()
        left_context, right_context, label, source_label = data
        tokens = model.convert_tokens(list(left_context), list(right_context)).to(device)
        label = torch.LongTensor(label).to(device)
        with torch.no_grad():
            res, res_softmax = model(tokens)
            loss = criterion(res, label)
        logger.info('Val epoch {}/{} batch {}/{}, loss: {:.6f}'.format(epoch + 1, MAX_EPOCH, idx + 1, total_batches, loss.item()))
        recorder.add_record(res_softmax, label, source_label)
    logger.info('Finish training process in epoch {}. Now calculating metrics ...'.format(epoch + 1))
    mRR = recorder.calc_mRR()
    recall_K = recorder.calc_recall_K()
    logger.info('MRR: {:.6f}'.format(mRR))
    for i, k in enumerate(RECALL_K):
        logger.info('Recall@{}: {:.6f}'.format(k, recall_K[i]))
    return mRR



def train(start_epoch):
    best_mRR = 0.0
    for epoch in range(start_epoch, MAX_EPOCH):
        train_one_epoch(epoch)
        mRR = val_one_epoch(epoch)
        if mRR > best_mRR:
            best_mRR = mRR
            if MULTIGPU is False:
                save_dict = {
                    'epoch': epoch + 1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict()
                }
            else:
                save_dict = {
                    'epoch': epoch + 1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.module.state_dict()
                }
            torch.save(save_dict, checkpoint_file)


if __name__ == '__main__':
    train(start_epoch)