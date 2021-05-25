import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert import Bert, Specter
from .vgae import VariantionalGraphAutoEncoder


class SimpleBert(nn.Module):
    '''
    Simple Bert model for context-based citation recommendation.
    '''
    def __init__(self, num_classes, seq_dim = 0):
        '''
        Initialize simple Bert model for context-based citation recommendation.

        Parameters
        ----------
        num_classes: int, the number of categories;
        seq_dim: int in [-1, 0], optional, default: 0, the chosen dim of the bert result.
        '''
        super(SimpleBert, self).__init__()
        self.bert = Bert(seq_dim, num_classes)
        self.softmax = nn.Softmax(num_classes)
    
    def forward(self, context):
        return self.softmax(self.bert(context))


class CitationBert(nn.Module):
    '''
    Citation-awared Bert model for context-based citation recommendation.
    '''
    def __init__(self, num_classes, embedding_dim, seq_dim = 0, S = 4):
        '''
        Initialize citation-awared Bert model for context-based citation recommendation.

        Parameters
        ----------
        num_classes: int, the number of categories;
        embedding_dim: int, the dimensions of embeddings;
        seq_dim: int in [-1, 0], optional, default: 0, the chosen dim of the bert result;
        S: int, optional, default: 4, the hyper-parameter of cosine similarity softmax,
           See: https://www.tutorialexample.com/understand-cosine-similarity-softmax-a-beginner-guide-machine-learning/ for details.
        '''
        super(CitationBert, self).__init__()
        self.bert = Bert(seq_dim, embedding_dim)
        self.softmax = nn.Softmax(num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.S = S

    def forward(self, context, paper_embeddings):
        batch_size = len(context)
        context_embeddings = self.bert(context)
        assert paper_embeddings.shape == torch.size([self.embedding_dim, self.num_classes])
        sim = F.cosine_similarity(paper_embeddings.repeat(batch_size, self.embedding_dim, self.num_classes), context_embeddings, dim = 1)
        return self.softmax(self.S * sim)


class SpecterVGAE(nn.Module):
    '''
    Variantional Graph Auto-encoder with Specter features.
    '''
    def __init__(self, feature_dim, embedding_dim):
        '''
        Initialize citation-awared Bert model for context-based citation recommendation.

        Parameters
        ----------
        feature_dim: int, the dimensions of features;
        embedding_dim: int, the dimensions of embeddings.
        '''
        self.specter = Specter()
        specter_dim = 768
        self.linear = nn.Linear(specter_dim, feature_dim)
        self.vgae = VariantionalGraphAutoEncoder(feature_dim, embedding_dim)

    def forward(self, papers, edge_index):
        paper_features = self.linear(self.specter(papers))
        return self.vgae(paper_features, edge_index)
    
    def encode(self, papers):
        paper_features = self.linear(self.specter(papers))
        return self.vgae.encode(paper_features)    
