import os
import yaml
import json
import torch
import argparse
import pandas as pd
import numpy as np
import logging
import torch.nn.functional as F
from utils.logger import ColoredLogger
from dataset import get_citation_dataset
from models.models import SpecterVGAE


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


class VGAEInferencer(object):
    def __init__(self, **kwargs):
        super(VGAEInferencer, self).__init__()
        EMBEDDING_DIM = kwargs.get('embedding_dim', 768)
        MULTIGPU = kwargs.get('multigpu', False)
        SPECTER_BATCH_SIZE = kwargs.get('specter_batch_size', 4)
        MAX_LENGTH = kwargs.get('max_length', 512)
        SEQ_LEN = kwargs.get('seq_len', 50)
        END_YEAR = kwargs.get('end_year', 2015)
        FREQUENCY = kwargs.get('frequency', 5)
        STATS_DIR = kwargs.get('stats_dir', os.path.join('stats', 'vgae'))
        DATA_PATH = kwargs.get('data_path', os.path.join('data', 'citation.csv'))
        EMBEDDING_FILENAME = kwargs.get('embedding_filename', 'embeddings.npy')
        SPECTER_EMBEDDING_FILENAME = kwargs.get('specter_embedding_filename', 'specter_embeddings.npy')
        if os.path.exists(STATS_DIR) == False:
            os.makedirs(STATS_DIR)
        checkpoint_file = os.path.join(STATS_DIR, 'checkpoint.tar')
        self.embedding_file = os.path.join(STATS_DIR, EMBEDDING_FILENAME)
        specter_embedding_file = os.path.join(STATS_DIR, SPECTER_EMBEDDING_FILENAME)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load data & Build dataset
        logger.info('Reading citation dataset ...')
        self.edge_list, _, _, _, self.node_info = get_citation_dataset(DATA_PATH, seq_len = SEQ_LEN, year = END_YEAR, frequency = FREQUENCY)
        logger.info('File read successfully. Now reading edge list for training ...')
        self.node_num = len(self.node_info)
        df = pd.read_csv(os.path.join(STATS_DIR, 'train_pos_edge_list.csv'))
        self.train_edge_list = [[row['source'], row['destination']] for _, row in df.iterrows()]
        logger.info('File read successfully.')

        # Build model from configs
        self.model = SpecterVGAE(embedding_dim = EMBEDDING_DIM, max_length = MAX_LENGTH)
        self.model.process_paper_feature(self.node_info, use_saved_results = True, filepath = specter_embedding_file, device = self.device, specter_device = self.device, process_batch_size = SPECTER_BATCH_SIZE)
        self.model.to(self.device)

        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location = self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            logger.info('Load checkpoint {} (epoch {})'.format(checkpoint_file, epoch))
        else:
            raise AttributeError('No checkpoint file!')

        if MULTIGPU is True:
            self.model = torch.nn.DataParallel(self.model)
        
        self.prepare_embeddings()

    def prepare_embeddings(self):
        self.model.eval()
        edge_list = torch.LongTensor(self.edge_list).to(self.device).transpose(1, 0)
        logger.info('Fetching embedding results ...')
        with torch.no_grad():
            self.emb = self.model.encode(edge_list)
        logger.info('Results fetched. Now saving to {} ...'.format(self.embedding_file))
        emb = self.emb.cpu().detach().numpy()
        np.save(self.embedding_file, emb)
        logger.info('File saved successfully.')

    def find_topk(self, node_info, k = 10):
        id = node_info['id']
        node_emb = self.emb[id]
        similarity = F.cosine_similarity(node_emb.reshape(1, -1), self.emb, dim = 1)
        _, topkid = torch.topk(similarity, k = k + 1, dim = -1, largest = True, sorted = True)
        topkid = topkid.cpu().detach().numpy()
        res = []
        for i, paper_id in enumerate(topkid.tolist()):
            if i == 0:
                continue
            node_info = self.node_info[paper_id]
            node_info['id'] = paper_id
            res.append(node_info)
        return {"result": res}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default = os.path.join('configs', 'vgae.yaml'), help = 'Config File', type = str)
    parser.add_argument('--input', default = os.path.join('examples', 'relation.json'))
    parser.add_argument('--output', default = os.path.join('examples', 'relation-res.json'))
    FLAGS = parser.parse_args()
    CFG_FILE = FLAGS.cfg
    INPUT_FILE = FLAGS.input
    OUTPUT_FILE = FLAGS.output

    if os.path.exists(os.path.dirname(OUTPUT_FILE)) == False:
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    with open(CFG_FILE, 'r') as cfg_file:
        cfgs = yaml.load(cfg_file, Loader = yaml.FullLoader)

    inferencer = VGAEInferencer(**cfgs)
    
    with open(INPUT_FILE, 'r') as f:
        input_dict = json.load(f)

    output_dict = inferencer.find_topk(input_dict)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_dict, f)