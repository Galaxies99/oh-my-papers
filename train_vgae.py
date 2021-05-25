import os
import yaml
import torch
import argparse
import pandas as pd
import logging
from utils.logger import ColoredLogger
from torch.optim import Adam
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
    
MAX_EPOCH = cfg_dict.get('max_epoch', 500)
EMBEDDING_DIM = cfg_dict.get('embedding_dim', 768)
MULTIGPU = cfg_dict.get('multigpu', False)
ADAM_BETA1 = cfg_dict.get('adam_beta1', 0.9)
ADAM_BETA2 = cfg_dict.get('adam_beta2', 0.999)
LEARNING_RATE = cfg_dict.get('learning_rate', 0.01)
STATS_DIR = cfg_dict.get('stats_dir', os.path.join('stats', 'vgae'))
DATA_PATH = cfg_dict.get('data_path', os.path.join('data', 'citation.csv'))
if os.path.exists(STATS_DIR) == False:
    os.makedirs(STATS_DIR)
checkpoint_file = os.path.join(STATS_DIR, 'checkpoint.tar')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data & Build dataset
logger.info('Reading citation dataset ...')
_, train_edge_list, test_pos_edge_list, test_neg_edge_list, node_info = get_citation_dataset(DATA_PATH, seq_len = 50, year = 2015, frequency = 5)
logger.info('Finish reading and dividing into training and testing sets. Saving to corresponding files ...')
node_num = len(node_info)
pd.DataFrame(train_edge_list, columns = ['source', 'destination']).to_csv(os.path.join(STATS_DIR, 'train_pos_edge_list.csv'))
pd.DataFrame(test_pos_edge_list, columns = ['source', 'destination']).to_csv(os.path.join(STATS_DIR, 'test_pos_edge_list.csv'))
pd.DataFrame(test_neg_edge_list, columns = ['source', 'destination']).to_csv(os.path.join(STATS_DIR, 'test_neg_edge_list.csv'))
logger.info('File saved. Now you can use these files in inference.')

# Build model from configs
model = SpecterVGAE(embedding_dim = EMBEDDING_DIM)
model.process_paper_feature(node_info, use_saved_results = False, filepath = os.path.join(STATS_DIR, 'specter.npy'), device = device)
model.to(device)

if MULTIGPU is True:
    model = torch.nn.DataParallel(model)

# Define optimizer
optimizer = Adam(model.parameters(), betas = (ADAM_BETA1, ADAM_BETA2), lr = LEARNING_RATE)

def train_one_epoch(edge_list):
    model.train()
    optimizer.zero_grad()
    edge_list = torch.LongTensor(edge_list).to(device).transpose(1, 0)
    z = model.encode(edge_list)
    loss = model.recon_loss(z, edge_list)
    loss = loss + (1 / node_num) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test_one_epoch(edge_list, pos_edge_list, neg_edge_list):
    model.eval()
    edge_list = torch.LongTensor(edge_list).to(device).transpose(1, 0)
    pos_edge_list = torch.LongTensor(pos_edge_list).to(device).transpose(1, 0)
    neg_edge_list = torch.LongTensor(neg_edge_list).to(device).transpose(1, 0)
    with torch.no_grad():
        z = model.encode(edge_list)
    return model.test(z, pos_edge_list, neg_edge_list)


def train(train_edge_list, test_pos_edge_list, test_neg_edge_list):
    global cur_epoch
    max_auc = 0
    for epoch in range(MAX_EPOCH):
        cur_epoch = epoch
        loss = train_one_epoch(train_edge_list)
        auc, ap = test_one_epoch(train_edge_list, test_pos_edge_list, test_neg_edge_list)
        logger.info('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch + 1, loss, auc, ap))
        if auc > max_auc:
            max_auc = auc
            if MULTIGPU is False:
                save_dict = {
                    'epoch': epoch + 1, 'loss': loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict()
                }
            else:
                save_dict = {
                    'epoch': epoch + 1, 'loss': loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.module.state_dict()
                }
            torch.save(save_dict, checkpoint_file)


if __name__ == '__main__':
    train(train_edge_list, test_pos_edge_list, test_neg_edge_list)