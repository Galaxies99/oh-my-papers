import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
from utils.logger import ColoredLogger
from .bert import Bert, Specter
from .vgae import VariantionalGraphAutoEncoder


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


class SimpleBert(nn.Module):
    '''
    Simple Bert model for context-based citation recommendation.
    '''
    def __init__(self, num_classes, seq_dim = 0, max_length = 512, cased = False):
        '''
        Initialize simple Bert model for context-based citation recommendation.

        Parameters
        ----------
        num_classes: int, the number of categories;
        seq_dim: int in [-1, 0], optional, default: 0, the chosen dim of the bert result;
        cased: bool, optional, default: False, whether use the cased bert pretrained model, or uncased bert pretrained model.
        '''
        super(SimpleBert, self).__init__()
        self.bert = Bert(seq_dim, num_classes, max_length = max_length, cased = cased)
        self.softmax = nn.Softmax(dim = 1)

    def convert_tokens(self, context, right_context = None):
        return self.bert.convert_tokens(context, right_context)
    
    def forward(self, tokens):
        res = self.bert(tokens)
        return res, self.softmax(res)


class CitationBert(nn.Module):
    '''
    Citation-awared Bert model for context-based citation recommendation.
    '''
    def __init__(self, num_classes, embedding_dim, seq_dim = 0, max_length = 512, S = 1, cased = False):
        '''
        Initialize citation-awared Bert model for context-based citation recommendation.

        Parameters
        ----------
        num_classes: int, the number of categories;
        embedding_dim: int, the dimensions of embeddings;
        seq_dim: int in [-1, 0], optional, default: 0, the chosen dim of the bert result;
        max_length: int, the max length of tokens (enable padding / truncation);
        S: int, optional, default: 1, the hyper-parameter of cosine similarity softmax,
           See: https://www.tutorialexample.com/understand-cosine-similarity-softmax-a-beginner-guide-machine-learning/ for details;
        cased: bool, optional, default: False, whether use the cased bert pretrained model, or uncased bert pretrained model.
        '''
        super(CitationBert, self).__init__()
        self.bert = Bert(seq_dim, num_classes, max_length = max_length, cased = cased)
        self.specter = Specter(max_length = max_length)
        self.layernorm1 = nn.LayerNorm([num_classes])
        self.layernorm2 = nn.LayerNorm([num_classes])
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.paper_embeddings = None
        self.softmax = nn.Softmax(dim = 1)
        self.S = S
    
    def set_paper_embeddings(self, filename, device = 'cpu'):
        '''
        Initialize the paper embeddings from file.

        Parameters
        ----------
        filename: str, the .npy file which contains the paper embeddings;
        device: str, optional, default: 'cpu', the device on which the model is located in.
        '''
        self.paper_embeddings = torch.from_numpy(np.load(filename)).to(device).transpose(1, 0)
        assert self.paper_embeddings.shape == (self.embedding_dim, self.num_classes)
        self.paper_embeddings.requires_grad = False

    def convert_tokens(self, context, right_context = None):
        return self.bert.convert_tokens(context, right_context), self.specter.convert_tokens((context, right_context), title_abs = False)
    
    def forward(self, tokens_bert, tokens_specter):
        if self.paper_embeddings is None:
            raise AttributeError('Paper embeddings not initialized, please call "set_paper_embeddings" to initialize the paper embeddings.')
        context_class = self.bert(tokens_bert)
        context_embeddings = self.specter(tokens_specter)
        batch_size = context_embeddings.shape[0]
        similarity = F.cosine_similarity(
            self.paper_embeddings.reshape(1, self.embedding_dim, self.num_classes),
            context_embeddings.reshape(batch_size, self.embedding_dim, 1),
            dim = 1
        )
        similarity = self.layernorm1(similarity)
        context_class = self.layernorm2(context_class)
        res = self.S * similarity + context_class
        return res, self.softmax(res)


class SpecterVGAE(nn.Module):
    '''
    Variantional Graph Auto-encoder with Specter features.
    '''
    def __init__(self, embedding_dim, max_length = 512):
        '''
        Initialize citation-awared Bert model for context-based citation recommendation.

        Parameters
        ----------
        embedding_dim: int, the dimensions of embeddings;
        max_length: int, the max length of tokens (enable padding / truncation).
        '''
        super(SpecterVGAE, self).__init__()
        self.specter = Specter(max_length = max_length)
        specter_dim = 768
        self.vgae = VariantionalGraphAutoEncoder(specter_dim, embedding_dim)
    
    def process_paper_feature(self, papers, use_saved_results, filepath, specter_device = 'cpu', device = 'cpu', process_batch_size = 16):
        '''
        Process the paper features using Specter or previous inference results.

        Parameters
        ----------
        papers: list of dict, including the title and abstract of each paper;
        use_saved_results: bool, whether to use the saved result;
        filepath: str, the filepath of the result file;
        specter_device: str, optional, default: 'cpu', the device on which the specter model is located in;
        device: str, optional, default: 'cpu', the device on which the model is located in;
        process_batch_size: int, optional, default: 16, the batch size of the Specter inference process.
        '''
        self.specter.to(specter_device)
        if type(use_saved_results) is not bool:
            raise TypeError('The type of attribute use_saved_results should be bool.')
        if use_saved_results is False:
            logger.info('Processing paper features using Specter ...')
            self.paper_features = None
            for i in tqdm(range(0, len(papers), process_batch_size)):
                tokens = self.specter.convert_tokens(papers[i: min(i + process_batch_size, len(papers))]).to(specter_device)
                feature = self.specter(tokens)
                if self.paper_features is None:
                    self.paper_features = feature.cpu().detach().numpy()
                else:
                    self.paper_features = np.concatenate([self.paper_features, feature.cpu().detach().numpy()], axis = 0)
            logger.info('Saving Specter paper embedding into {} ...'.format(filepath))
            sav_res = self.paper_features
            np.save(filepath, sav_res)
            logger.info('File saved, next time you can set use_saved_results=True to read the features.')
            self.paper_features = torch.from_numpy(sav_res)
        else:
            logger.info('Reading saved paper features from {} ...'.format(filepath))
            sav_res = np.load(filepath)
            if sav_res.shape[0] != len(papers):
                raise AttributeError('The length of the saving results is incompatible with the number of given papers.')
            self.paper_features = torch.from_numpy(sav_res)
            logger.info('File read successfully.')
        self.paper_features = self.paper_features.to(device)

    def forward(self, edge_index):
        return self.vgae(self.paper_features, edge_index)
    
    def encode(self, edge_index):
        return self.vgae.encode(self.paper_features, edge_index)

    def test(self, z, pos_edge_index, neg_edge_index):
        return self.vgae.test(z, pos_edge_index, neg_edge_index)

    def recon_loss(self, z, edge_index):
        return self.vgae.recon_loss(z, edge_index)

    def kl_loss(self):
        return self.vgae.kl_loss()